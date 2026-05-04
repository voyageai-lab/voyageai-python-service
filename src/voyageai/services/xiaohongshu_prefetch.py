"""Xiaohongshu (小红书) pre-fetch pipeline.

Searches Xiaohongshu for travel content about a destination, fetches
post details, and builds a formatted context block for the LLM.  Also
handles login-recovery via QR code when the session has expired.

All functions are standalone (no class state) so they can be called from
the agent orchestrator or tested independently.  The only external
dependencies are the tool registry and a progress-callback function.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from typing import Any

from voyageai.schemas.itinerary import SourceLink, StructuredItinerary
from voyageai.services.agent_types import ProgressCallback
from voyageai.tools.registry import tool_registry

logger = logging.getLogger(__name__)


# ── Destination extraction ─────────────────────────────────────────


def extract_destination(requirements: str) -> str:
    """Extract the destination name from a planning requirements string.

    Handles English patterns ("trip to X", "visit X", "going to X"),
    Chinese patterns ("去北京", "北京5日游"), and mixed-language inputs.
    """
    first_line = requirements.split("\n")[0].strip()

    # English: "trip to/on/in X", "visit X", "going to X", etc.
    for pattern in [
        r"trip\s+(?:to|on|in)\s+(.+?)(?:\s+from\s|\s+with\s|\s+for\s|\s+in\s+\w+\s+\d|\s*[,.]|\s*$)",
        r"(?:visit|travel(?:ing)?\s+to|going\s+to|heading\s+to|fly(?:ing)?\s+to)\s+(.+?)"
        r"(?:\s+from\s|\s+with\s|\s+for\s|\s+in\s+\w+\s+\d|\s*[,.]|\s*$)",
    ]:
        m = re.search(pattern, first_line, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".,;:!?")

    # Chinese: "去X", "到X旅游", "X N日游", "X旅行"
    for pattern in [
        r"[去到]([^\s,，。!！?？]{2,10}?)(?:旅[游行]|玩|度假|自由行|\d|[,，。]|$)",
        r"([\u4e00-\u9fff]{2,8}?)\d+日[游行]",
        r"([\u4e00-\u9fff]{2,8}?)(?:旅[游行]|攻略|行程)",
    ]:
        m = re.search(pattern, first_line)
        if m:
            return m.group(1).strip()

    # Fallback: if the line is short, use it as-is; otherwise truncate
    return first_line[:50]


# ── Progress-callback helper (mirrors ResponsesAgentService._emit) ─


async def _emit(
    callback: ProgressCallback | None,
    event_type: str,
    data: dict[str, Any],
) -> None:
    if callback is None:
        return
    try:
        await callback(event_type, data)
    except Exception:
        logger.warning("Progress callback error for event %s", event_type, exc_info=True)


# ── Auth recovery via QR code ──────────────────────────────────────


async def _get_qr_code(
    progress_callback: ProgressCallback | None = None,
) -> bool:
    """Get XHS login QR code and push it to the frontend via SSE."""
    qr_tool = tool_registry.get("xiaohongshu__get_login_qrcode")
    if not qr_tool:
        logger.warning("XHS get_login_qrcode tool not available")
        return False

    try:
        qr_result = await asyncio.wait_for(qr_tool.execute(), timeout=60)
    except asyncio.TimeoutError:
        logger.warning("XHS get_login_qrcode timed out after 60s")
        return False
    except Exception as e:
        logger.warning("XHS get_login_qrcode failed: %s", e)
        return False

    if not qr_result.success or not qr_result.output:
        logger.warning("XHS QR code retrieval failed: %s", qr_result.error)
        return False

    # The MCP tool returns a string like:
    #   "请用小红书 App 在 2026-03-04 06:50:00 前扫码登录 👇\niVBORw0KGgo..."
    raw_output = qr_result.output
    logger.info("XHS QR raw output type=%s, len=%d",
                type(raw_output).__name__,
                len(str(raw_output)) if raw_output else 0)

    qr_base64 = ""
    expiry_iso = None
    raw_str = ""

    if isinstance(raw_output, dict):
        for key in ("qrcode", "image", "qr_code", "data", "base64", "url"):
            val = raw_output.get(key, "")
            if val and isinstance(val, str) and len(val) > 50:
                raw_str = val
                break
    elif isinstance(raw_output, str):
        raw_str = raw_output
    elif isinstance(raw_output, bytes):
        qr_base64 = base64.b64encode(raw_output).decode()

    if raw_str and not qr_base64:
        ts_match = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", raw_str)
        if ts_match:
            expiry_iso = ts_match.group(1).replace(" ", "T") + "Z"

        png_match = re.search(r"(iVBORw0KGgo[A-Za-z0-9+/=\s]+)", raw_str)
        if png_match:
            qr_base64 = png_match.group(1).replace("\n", "").replace(" ", "")
        elif raw_str.startswith("data:image"):
            qr_base64 = raw_str

    if not qr_base64:
        logger.warning("XHS QR code: could not extract image data")
        return False

    logger.info("XHS auth: pushing QR code to frontend (base64_len=%d, expires=%s)",
                len(qr_base64), expiry_iso or "unknown")
    await _emit(progress_callback, "auth_required", {
        "service": "xiaohongshu",
        "serviceName": "小红书",
        "qrCodeBase64": qr_base64,
        "expiresAt": expiry_iso,
        "message": "请用小红书 App 扫码登录",
    })
    return True


# ── Context builder (search result → formatted text) ───────────────


async def _build_xhs_context(
    search_result: Any,
    destination: str,
    progress_callback: ProgressCallback | None = None,
) -> str | None:
    """Build formatted context text from a successful XHS search result."""
    detail_tool = tool_registry.get("xiaohongshu__get_feed_detail")

    feeds = search_result.output if isinstance(search_result.output, dict) else {}
    feed_list = feeds.get("feeds", [])
    if not feed_list:
        return None

    top_feeds = feed_list[:5]

    async def _fetch_one(feed: dict) -> str | None:
        feed_id = feed.get("id", "")
        xsec_token = feed.get("xsecToken", "")
        title = feed.get("noteCard", {}).get("displayTitle", "Unknown")
        interact = feed.get("noteCard", {}).get("interactInfo", {})
        likes = interact.get("likedCount", "?")
        collects = interact.get("collectedCount", "?")
        author = feed.get("noteCard", {}).get("user", {}).get("nickname", "Unknown")
        note_url = (
            f"https://www.xiaohongshu.com/explore/{feed_id}"
            f"?xsec_token={xsec_token}&xsec_source=pc_search"
            if xsec_token else
            f"https://www.xiaohongshu.com/explore/{feed_id}"
        )

        desc_text = ""
        if detail_tool and feed_id and xsec_token:
            try:
                detail_result = await asyncio.wait_for(
                    detail_tool.execute(feed_id=feed_id, xsec_token=xsec_token),
                    timeout=30,
                )
                if detail_result.success and detail_result.output:
                    data = detail_result.output if isinstance(detail_result.output, dict) else {}
                    note = data.get("data", {}).get("note", data)
                    desc_text = note.get("desc", "")
            except Exception as e:
                logger.warning("Xiaohongshu get_feed_detail failed for %s: %s", feed_id, e)

        section = (
            f"### {title}\n"
            f"Author: {author} | Likes: {likes} | Collects: {collects}\n"
            f"URL: {note_url}\n"
        )
        if desc_text:
            truncated = desc_text[:1500] + ("..." if len(desc_text) > 1500 else "")
            section += f"Content:\n{truncated}\n"
        return section

    results = await asyncio.gather(*[_fetch_one(f) for f in top_feeds], return_exceptions=True)
    sections = [r for r in results if isinstance(r, str)]

    if not sections:
        return None

    result_text = (
        "=== XIAOHONGSHU (小红书) TRAVEL RECOMMENDATIONS ===\n"
        "The following are popular travel posts from Xiaohongshu. "
        "Use this information to enrich your itinerary with local tips, "
        "restaurant recommendations, and insider knowledge. "
        "Include the Xiaohongshu URLs in source_links (source: 'xiaohongshu').\n\n"
        + "\n---\n".join(sections)
    )

    logger.info(
        "Xiaohongshu pre-fetch: %d posts fetched, %d chars of content",
        len(sections), len(result_text),
    )
    await _emit(progress_callback, "thinking", {
        "text": f"Found {len(sections)} Xiaohongshu travel posts with tips and recommendations.",
    })
    return result_text


# ── Main pre-fetch entry point ─────────────────────────────────────


async def prefetch_xiaohongshu(
    destination: str,
    progress_callback: ProgressCallback | None = None,
) -> str | None:
    """Pre-fetch Xiaohongshu travel content for *destination*.

    Runs ``search_feeds`` → ``get_feed_detail`` (top 5 posts, concurrent)
    programmatically.  If the initial search fails, attempts automatic
    auth recovery via QR code.

    Returns a formatted text block to inject into the LLM context, or
    ``None`` if unavailable.
    """
    try:
        search_tool = tool_registry.get("xiaohongshu__search_feeds")
        if not search_tool:
            return None

        keyword = f"{destination}旅游攻略"

        # Step 1: Try search directly (fast path if already logged in)
        logger.info("Xiaohongshu pre-fetch: trying search for '%s'", destination)
        await _emit(progress_callback, "thinking", {
            "text": f"Searching Xiaohongshu for '{destination}' travel tips...",
        })

        try:
            search_result = await asyncio.wait_for(
                search_tool.execute(keyword=keyword), timeout=45,
            )
            if search_result.success and search_result.output:
                logger.info("XHS search succeeded on first try (logged in)")
                return await _build_xhs_context(search_result, destination, progress_callback)
            logger.info("XHS search returned no results: %s", search_result.error)
        except asyncio.TimeoutError:
            logger.info("XHS search timed out (likely not logged in)")
        except Exception as e:
            logger.info("XHS search failed: %s", e)

        # Step 2: Search failed — get QR code and push to frontend
        logger.info("XHS search failed, starting QR auth recovery")
        qr_ok = await _get_qr_code(progress_callback)
        if not qr_ok:
            return None

        # Step 3: Poll by retrying search (avoids unreliable check_login_status)
        poll_interval = 15
        max_polls = 8
        for attempt in range(max_polls):
            await asyncio.sleep(poll_interval)
            try:
                retry = await asyncio.wait_for(
                    search_tool.execute(keyword=keyword), timeout=45,
                )
                if retry.success and retry.output:
                    logger.info("XHS search succeeded after QR scan (poll %d)", attempt + 1)
                    await _emit(progress_callback, "auth_success", {
                        "service": "xiaohongshu",
                        "serviceName": "小红书",
                    })
                    return await _build_xhs_context(retry, destination, progress_callback)
                logger.info("XHS search poll %d: no results yet", attempt + 1)
            except Exception as e:
                logger.info("XHS search poll %d failed: %s", attempt + 1, e)

        logger.warning("XHS auth: login recovery timed out")
        await _emit(progress_callback, "auth_expired", {
            "service": "xiaohongshu",
            "serviceName": "小红书",
            "message": "登录超时，小红书推荐内容将不可用",
        })
        return None

    except Exception as e:
        logger.warning("Xiaohongshu pre-fetch failed: %s", e)
        return None


# ── Post-processing: inject XHS source links into itinerary ────────


def inject_xhs_source_links(
    itinerary: StructuredItinerary,
    xhs_context: str,
) -> None:
    """Add Xiaohongshu links to itinerary ``source_links``."""
    xhs_posts: list[dict[str, str]] = []
    for block in xhs_context.split("---"):
        title_m = re.search(r"###\s*(.+)", block)
        url_m = re.search(r"URL:\s*(https://\S+)", block)
        author_m = re.search(r"Author:\s*(.+?)\s*\|", block)
        likes_m = re.search(r"Likes:\s*(\S+)", block)
        if title_m and url_m:
            xhs_posts.append({
                "title": title_m.group(1).strip(),
                "url": url_m.group(1).strip(),
                "snippet": (
                    f"by {author_m.group(1).strip() if author_m else '?'}, "
                    f"{likes_m.group(1) if likes_m else '?'} likes"
                ),
            })

    if not xhs_posts:
        return

    added = 0
    for day in itinerary.days:
        if not day.activities:
            continue
        act = day.activities[0]
        if act.source_links is None:
            act.source_links = []
        existing_urls = {sl.url for sl in act.source_links}
        xhs_links = []
        for post in xhs_posts:
            if post["url"] not in existing_urls:
                xhs_links.append(SourceLink(
                    title=post["title"],
                    url=post["url"],
                    source="xiaohongshu",
                    snippet=post["snippet"],
                ))
        act.source_links = xhs_links + list(act.source_links)
        added += len(xhs_links)
    if added > 0:
        logger.info("Post-processed: added %d Xiaohongshu links to source_links", added)

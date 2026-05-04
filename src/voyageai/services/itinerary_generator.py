"""Itinerary generation, validation, plan outline, and edit pipeline.

Contains all logic for converting tool-call findings into a structured
``StructuredItinerary`` JSON, including:

- ``call_json_model``  — wrapper for o-series / standard Chat Completions
- ``validate_itinerary`` / ``parse_itinerary`` — JSON → Pydantic
- ``generate_plan_outline`` — quick outline from tool findings
- ``generate_structured_final`` — full generation with day-count retry
- ``edit_itinerary`` — lightweight edit of an existing itinerary

These are standalone async functions so they can be reused by any
orchestrator without coupling to a specific agent class.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI
from pydantic import ValidationError

from voyageai.config import settings
from voyageai.schemas.itinerary import StructuredItinerary
from voyageai.services.agent_types import (
    AgentResponse,
    LLMCallRecord,
    ProgressCallback,
    _calc_cost,
)

logger = logging.getLogger(__name__)


# ── JSON structure example (shared between generation & editing) ───

ITINERARY_JSON_EXAMPLE = """{
  "metadata": {"destination":"Tokyo, Japan","start_date":"2024-04-01","end_date":"2024-04-03","total_days":3,"budget":"Medium ($100-200/day)","interests":["culture","food"],"best_season":"Spring","currency":"JPY","language":"Japanese"},
  "days": [
    {
      "day_number": 1, "date": "2024-04-01", "theme": "Arrival and City Exploration",
      "activities": [
        {"activity_id":"act-day1-001","time":"09:00-11:00","title":"Visit Senso-ji Temple","description":"Explore Tokyo's oldest Buddhist temple.","location":{"name":"Senso-ji Temple","latitude":35.7148,"longitude":139.7967,"address":"2-3-1 Asakusa, Taito City"},"estimated_cost":"Free","duration_minutes":120,"notes":["Visit early to avoid crowds"],"rating":4.7,"website_url":"https://www.senso-ji.jp/"}
      ]
    }
  ],
  "tips": ["Get a Suica card for easy transit"],
  "travel_tips": [{"category":"booking","message":"Book guided tour 3 days ahead","priority":"high","applies_to":"act-day1-001","advance_days":3}]
}"""


# ── Progress-callback helper ──────────────────────────────────────


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


# ── JSON model call (supports o-series reasoning models) ──────────


def _is_o_series(model: str) -> bool:
    return model.startswith("o1") or model.startswith("o3") or model.startswith("o4")


async def call_json_model(
    client: AsyncOpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.7,
) -> tuple[str, str, int, int, int, str]:
    """Call the LLM with ``json_object`` response format.

    Returns ``(content, finish_reason, total_tokens, input_tokens,
    output_tokens, reasoning_content)``.
    """
    is_reasoning = _is_o_series(model)
    system_role = "developer" if is_reasoning else "system"

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": system_role, "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    }

    if is_reasoning:
        kwargs["max_completion_tokens"] = settings.max_tokens
    else:
        kwargs["max_tokens"] = settings.max_tokens
        kwargs["temperature"] = temperature

    response = await client.chat.completions.create(**kwargs)

    _usage = response.usage
    tokens = _usage.total_tokens if _usage else 0
    input_tokens = _usage.prompt_tokens if _usage else 0
    output_tokens = _usage.completion_tokens if _usage else 0
    reasoning_tokens = 0
    if _usage and hasattr(_usage, "completion_tokens_details"):
        details = _usage.completion_tokens_details
        if details and hasattr(details, "reasoning_tokens"):
            reasoning_tokens = details.reasoning_tokens or 0

    finish_reason = response.choices[0].finish_reason or "unknown"
    content = response.choices[0].message.content or ""

    reasoning_content = ""
    msg = response.choices[0].message
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        reasoning_content = msg.reasoning_content

    logger.info(
        "JSON model call: model=%s, finish_reason=%s, tokens=%d "
        "(in=%d, out=%d, reasoning=%d), output_len=%d",
        model, finish_reason, tokens, input_tokens, output_tokens,
        reasoning_tokens, len(content),
    )
    return content, finish_reason, tokens, input_tokens, output_tokens, reasoning_content


# ── Pre-validation fixups ─────────────────────────────────────────


def _fix_alternatives(data: dict) -> None:
    """Normalize ``alternatives`` in each day to ``list[list[dict]]``.

    LLMs sometimes produce ``alternatives`` as a flat list of activity
    dicts instead of the expected list-of-lists.  This mutates *data*
    in-place so Pydantic validation succeeds without a retry.
    """
    for day in data.get("days", []):
        alts = day.get("alternatives")
        if not alts or not isinstance(alts, list):
            continue
        if alts and isinstance(alts[0], dict):
            day["alternatives"] = [alts]


# ── Itinerary validation ──────────────────────────────────────────


def validate_itinerary(
    content: str,
) -> tuple[StructuredItinerary | None, str | None]:
    """Parse and validate itinerary JSON into a ``StructuredItinerary``."""
    if not content or not content.strip():
        return None, "Empty response"

    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"

    if not isinstance(data, dict):
        return None, f"Expected JSON object, got {type(data).__name__}"

    _fix_alternatives(data)

    try:
        itinerary = StructuredItinerary.model_validate(data)
        return itinerary, None
    except ValidationError as e:
        return None, f"Pydantic validation: {e}"


def parse_itinerary(text: str) -> StructuredItinerary | None:
    """Try to extract a ``StructuredItinerary`` from text (JSON block)."""
    if not text:
        return None
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            _fix_alternatives(data)
            return StructuredItinerary.model_validate(data)
    except Exception:
        pass
    return None


# ── Plan outline (quick summary before full generation) ────────────


async def generate_plan_outline(
    client: AsyncOpenAI,
    requirements: str,
    input_items: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Generate a quick plan outline from tool findings."""
    tool_summaries = []
    assistant_text = ""
    for item in input_items:
        if isinstance(item, dict):
            role = item.get("role", "")
            content = item.get("content", "")
            if item.get("type") == "function_call_output":
                tool_summaries.append(str(item.get("output", ""))[:400])
            elif role == "user" and content:
                assistant_text = content
        elif hasattr(item, "type") and item.type == "message":
            for part in (item.content or []):
                if hasattr(part, "text"):
                    assistant_text = part.text

    findings = "\n".join(tool_summaries[-6:]) if tool_summaries else ""
    reasoning = assistant_text[:1500] if assistant_text else ""

    outline_prompt = (
        f"Based on this travel request and research findings, generate a brief plan outline.\n\n"
        f"User request: {requirements}\n\n"
        f"Research findings:\n{findings}\n\n"
        f"Agent analysis:\n{reasoning}\n\n"
        '"summary": One-sentence trip summary\n'
        '"daily_themes": Array of {"day": N, "theme": "Theme text", "highlight": "Key attraction"}\n'
        '"estimated_budget": Budget range string\n'
        '"weather_summary": Brief weather note (if known)\n\n'
        "Keep it concise. Return ONLY JSON."
    )

    try:
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "Generate a brief JSON travel plan outline. Return only valid JSON."},
                {"role": "user", "content": outline_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=600,
            temperature=0.4,
        )
        content = response.choices[0].message.content or "{}"
        outline = json.loads(content)
        logger.info("Plan outline generated: %d daily themes", len(outline.get("daily_themes", [])))
        return outline
    except Exception as e:
        logger.warning("Plan outline generation failed (non-blocking): %s", e)
        return None


# ── Full structured itinerary generation (with day-count retry) ────


async def generate_structured_final(
    client: AsyncOpenAI,
    requirements: str,
    input_items: list[dict[str, Any]],
    progress_callback: ProgressCallback | None,
    llm_calls: list[LLMCallRecord],
    temperature: float = 0.7,
    max_retries: int = 2,
    model_override: str | None = None,
) -> StructuredItinerary | None:
    """Generate a structured itinerary with prompt-guided JSON generation.

    Uses the Chat Completions API with ``json_object`` format for reliable
    structured output, with retry logic for incomplete days.
    """
    final_model = model_override or settings.openai_final_model
    call_records: list[LLMCallRecord] = []

    # Condense tool findings from input_items
    tool_summary_parts: list[str] = []
    supplementary_context_parts: list[str] = []
    assistant_reasoning = ""
    for item in input_items:
        if isinstance(item, dict):
            if item.get("type") == "function_call_output":
                out = str(item.get("output", ""))
                tool_summary_parts.append(out[:800] if len(out) > 800 else out)
            elif item.get("role") == "user" and item.get("content"):
                assistant_reasoning = item["content"]
            elif item.get("role") == "system" and item.get("content"):
                content = item["content"]
                if "XIAOHONGSHU" in content.upper() or "小红书" in content:
                    supplementary_context_parts.append(content)
        elif hasattr(item, "type"):
            if item.type == "message":
                for part in (item.content or []):
                    if hasattr(part, "text"):
                        assistant_reasoning = part.text

    tool_findings = "\n".join(tool_summary_parts) if tool_summary_parts else "No tools were called."
    reasoning_snippet = assistant_reasoning[:2000] if assistant_reasoning else ""
    supplementary_context = "\n\n".join(supplementary_context_parts)

    xhs_instruction = ""
    if supplementary_context:
        xhs_instruction = f"""

## Supplementary Travel Recommendations
{supplementary_context}

IMPORTANT: When an activity's recommendation comes from the Xiaohongshu posts above,
you MUST include the Xiaohongshu URL in that activity's source_links with source: "xiaohongshu".
Only add Xiaohongshu links to activities where the content is genuinely relevant."""

    final_prompt = f"""Generate a COMPLETE travel itinerary as JSON.

## User Request
{requirements}

## Research Findings (from tools)
{tool_findings}

## Agent Analysis
{reasoning_snippet}{xhs_instruction}

## JSON Structure
Required: metadata (destination, start_date, end_date, total_days, budget, interests),
days[] (day_number, date, theme, activities[]),
activities (activity_id, time, title, description, location with lat/lng, estimated_cost).

Compact example:
```json
{ITINERARY_JSON_EXAMPLE}
```

## HARD RULES
1. ALL dates MUST use the current year or later. NEVER use past dates like 2024.
2. "days" array MUST have one entry for EVERY day. Do NOT skip or combine days.
3. Each day MUST have 3-5 activities. Fewer than 3 is NOT acceptable.
4. ALL user interests MUST be covered across the trip. Do NOT drop any.
5. activity_id: "act-dayN-NNN". time: "HH:MM-HH:MM". date: "YYYY-MM-DD".
6. Include duration_minutes, notes, rating, website_url where available.
7. "tips": 5-10 destination-specific, actionable tips (NOT generic like "book in advance").
8. Include "travel_tips" array with at least 5 structured tips.
9. Mix iconic landmarks with local favorites — avoid tourist-trap-only itineraries.

Return ONLY the JSON object."""

    system_msg = (
        "You are a travel itinerary generator. Output a single JSON object. "
        "Include ALL days with 3-5 activities each. Use ONLY current/future dates. "
        "Include real GPS coordinates for every location. "
        "Tips must be specific and actionable, not generic platitudes. "
        "Respond with ONLY valid JSON."
    )

    content, finish_reason, tokens, _in, _out, reasoning = await call_json_model(
        client, model=final_model, system=system_msg, user=final_prompt,
        temperature=temperature,
    )
    call_records.append(LLMCallRecord(
        label="final_generation",
        model=final_model,
        input_tokens=_in,
        output_tokens=_out,
        cost_usd=_calc_cost(final_model, _in, _out),
    ))

    if reasoning and progress_callback:
        await _emit(progress_callback, "thinking", {
            "message": reasoning,
            "source": "reasoning_model",
            "model": final_model,
        })

    itinerary, errors = validate_itinerary(content)

    if itinerary:
        expected_days = itinerary.metadata.total_days
        actual_days = len(itinerary.days)

        if actual_days >= expected_days:
            logger.info("Itinerary generated: %d days, model=%s", actual_days, final_model)
            llm_calls.extend(call_records)
            return itinerary

        logger.warning("Itinerary has %d/%d days. Retrying...", actual_days, expected_days)

        for retry in range(max_retries):
            retry_content, _, _, r_in, r_out, retry_reasoning = await call_json_model(
                client,
                model=final_model,
                system=f"Generate a COMPLETE {expected_days}-day travel itinerary as JSON. You MUST include ALL {expected_days} days.",
                user=(
                    f"The previous attempt only produced {actual_days} out of {expected_days} days.\n\n"
                    f"User request: {requirements}\n\n"
                    f"Please generate the COMPLETE itinerary with ALL {expected_days} days "
                    f"(day_number 1 through {expected_days}). Each day needs 2-4 activities "
                    "with real coordinates. Do NOT stop after day 1.\n\n"
                    f"JSON structure example:\n```json\n{ITINERARY_JSON_EXAMPLE}\n```\n\n"
                    "Return ONLY the JSON object."
                ),
                temperature=temperature,
            )
            call_records.append(LLMCallRecord(
                label=f"final_generation_retry_{retry + 1}",
                model=final_model,
                input_tokens=r_in,
                output_tokens=r_out,
                cost_usd=_calc_cost(final_model, r_in, r_out),
            ))

            if retry_reasoning and progress_callback:
                await _emit(progress_callback, "thinking", {
                    "message": retry_reasoning,
                    "source": "reasoning_model",
                    "model": final_model,
                })

            retry_itinerary, _ = validate_itinerary(retry_content)
            if retry_itinerary and len(retry_itinerary.days) > actual_days:
                logger.info("Retry %d succeeded: %d days", retry + 1, len(retry_itinerary.days))
                llm_calls.extend(call_records)
                return retry_itinerary

        logger.warning("All retries exhausted, returning %d-day itinerary", actual_days)
        llm_calls.extend(call_records)
        return itinerary

    logger.warning("Itinerary validation failed: %s", errors[:300] if errors else "unknown")

    for retry in range(max_retries):
        retry_content, _, _, r_in, r_out, retry_reasoning = await call_json_model(
            client, model=final_model, system=system_msg,
            user=(
                f"Your previous JSON was invalid: {errors[:500]}\n\n"
                f"Please fix and regenerate the complete itinerary.\n\n"
                f"User request: {requirements}\n\n"
                f"JSON structure:\n```json\n{ITINERARY_JSON_EXAMPLE}\n```\n\n"
                "Return ONLY valid JSON."
            ),
            temperature=temperature,
        )
        call_records.append(LLMCallRecord(
            label=f"validation_retry_{retry + 1}",
            model=final_model,
            input_tokens=r_in,
            output_tokens=r_out,
            cost_usd=_calc_cost(final_model, r_in, r_out),
        ))

        if retry_reasoning and progress_callback:
            await _emit(progress_callback, "thinking", {
                "message": retry_reasoning,
                "source": "reasoning_model",
                "model": final_model,
            })

        retry_itinerary, _ = validate_itinerary(retry_content)
        if retry_itinerary:
            logger.info("Validation retry %d succeeded: %d days", retry + 1, len(retry_itinerary.days))
            llm_calls.extend(call_records)
            return retry_itinerary

    llm_calls.extend(call_records)
    return None


# ── Edit pipeline (lightweight modification of existing itinerary) ─

import re as _re

_ALTERNATIVE_KEYWORDS = _re.compile(
    r"(?i)(alternative|替代|备选|其他方案|不同方案|换一个|换一种|另一个|别的选择|other\s*option|plan\s*b)",
)


def _is_alternative_request(instruction: str) -> bool:
    """Return True if the user is asking for alternative options."""
    return bool(_ALTERNATIVE_KEYWORDS.search(instruction))


# ── System prompts ────────────────────────────────────────────────

_EDIT_SYSTEM = (
    "You are a travel itinerary editor. You receive an existing itinerary "
    "as JSON and a user's edit instruction. Apply the MINIMUM change "
    "necessary and return the COMPLETE modified itinerary as JSON.\n\n"
    "Rules:\n"
    "1. Return the FULL itinerary (all days), not just the changed parts.\n"
    "2. Preserve all existing fields/data that are not affected by the edit.\n"
    "3. When adding new activities, include realistic coordinates, times, "
    "and cost estimates.\n"
    "4. When removing activities, adjust the remaining schedule times.\n"
    "5. When replacing, keep the same time slot but swap the content.\n"
    "6. When optimizing, reorder for geographic efficiency.\n"
    "7. Do NOT generate \"alternatives\" arrays — omit them entirely.\n"
    "8. Respond with ONLY valid JSON, no markdown, no commentary."
)

_ALTERNATIVES_SYSTEM = (
    "You are a travel itinerary editor that generates alternative options.\n"
    "You receive an existing itinerary as JSON and a request for alternatives.\n\n"
    "Rules:\n"
    "1. Return the FULL itinerary (all days) — keep unchanged days as-is.\n"
    "2. For each day the user wants alternatives for, populate the "
    "\"alternatives\" field with 2-3 DIFFERENT complete day schedules.\n"
    "3. Each alternative is a list of activities (same structure as the "
    "main \"activities\" array) covering the full day.\n"
    "4. Alternatives should offer genuinely different experiences "
    "(e.g., cultural vs outdoor vs food-focused).\n"
    "5. Use activity_id format \"alt-dayN-optM-NNN\" (e.g., \"alt-day2-opt1-001\").\n"
    "6. Include realistic coordinates, times, costs for every activity.\n"
    "7. Keep the original \"activities\" unchanged — only add \"alternatives\".\n"
    "8. \"alternatives\" structure: [[activities for option 1], [activities for option 2], ...].\n"
    "9. Respond with ONLY valid JSON, no markdown, no commentary."
)


async def edit_itinerary(
    client: AsyncOpenAI,
    edit_instruction: str,
    existing_itinerary_json: str,
    progress_callback: ProgressCallback | None = None,
    temperature: float = 0.7,
) -> AgentResponse:
    """Apply a targeted edit or generate alternatives for an existing itinerary.

    Detects whether the user wants alternatives or a regular edit
    and uses the appropriate prompt.  Typically completes in a single
    LLM call — much faster than the full agent pipeline.
    """
    import time

    start_time = time.time()
    llm_calls: list[LLMCallRecord] = []
    final_model = settings.openai_final_model
    is_alt = _is_alternative_request(edit_instruction)

    stage_msg = (
        "Generating alternative options for your itinerary..."
        if is_alt
        else "Applying your edit to the itinerary..."
    )
    await _emit(progress_callback, "stage_change", {
        "stage": "EDITING",
        "message": stage_msg,
    })

    system_msg = _ALTERNATIVES_SYSTEM if is_alt else _EDIT_SYSTEM

    if is_alt:
        user_prompt = f"""## Request
{edit_instruction}

## Current Itinerary
```json
{existing_itinerary_json}
```

Generate 2-3 alternative day schedules for the days mentioned above.
Put them in each day's "alternatives" array (each element is a complete list of activities).
Return the COMPLETE itinerary as JSON with alternatives included."""
    else:
        user_prompt = f"""## Edit Instruction
{edit_instruction}

## Current Itinerary
```json
{existing_itinerary_json}
```

Apply the edit instruction to the itinerary above.
Return the COMPLETE modified itinerary as JSON (all days, not just changes)."""

    label = "alternatives_generation" if is_alt else "edit_generation"

    await _emit(progress_callback, "thinking", {
        "text": f"{'Generating alternatives' if is_alt else 'Editing itinerary'}: {edit_instruction[:200]}",
    })

    content, finish_reason, tokens, _in, _out, reasoning = await call_json_model(
        client, model=final_model, system=system_msg, user=user_prompt,
        temperature=temperature,
    )
    llm_calls.append(LLMCallRecord(
        label=label,
        model=final_model,
        input_tokens=_in,
        output_tokens=_out,
        cost_usd=_calc_cost(final_model, _in, _out),
    ))

    if reasoning and progress_callback:
        await _emit(progress_callback, "thinking", {
            "message": reasoning,
            "source": "reasoning_model",
            "model": final_model,
        })

    itinerary, errors = validate_itinerary(content)

    if not itinerary:
        logger.warning("Edit validation failed: %s — retrying once", errors[:300] if errors else "unknown")
        retry_content, _, _, r_in, r_out, retry_reasoning = await call_json_model(
            client, model=final_model, system=system_msg,
            user=(
                f"Your previous JSON was invalid: {errors[:500]}\n\n"
                f"Please fix and return the complete edited itinerary.\n\n"
                f"Edit instruction: {edit_instruction}\n\n"
                f"Original itinerary:\n```json\n{existing_itinerary_json}\n```\n\n"
                "Return ONLY valid JSON."
            ),
            temperature=temperature,
        )
        llm_calls.append(LLMCallRecord(
            label=f"{label}_retry",
            model=final_model,
            input_tokens=r_in,
            output_tokens=r_out,
            cost_usd=_calc_cost(final_model, r_in, r_out),
        ))
        itinerary, _ = validate_itinerary(retry_content)

    processing_time = int((time.time() - start_time) * 1000)
    total_cost = round(sum(c.cost_usd for c in llm_calls), 6)
    total_tokens = sum(c.input_tokens + c.output_tokens for c in llm_calls)

    await _emit(progress_callback, "cost_summary", {
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
        "llm_calls": len(llm_calls),
        "tool_calls": 0,
        "processing_time_ms": processing_time,
        "api": "alternatives" if is_alt else "edit",
    })

    if itinerary:
        alt_count = sum(len(d.alternatives) for d in itinerary.days)
        logger.info(
            "Edit completed (alt=%s): %d days, %d alternatives, %dms, $%.6f",
            is_alt, len(itinerary.days), alt_count, processing_time, total_cost,
        )

    return AgentResponse(
        itinerary=itinerary,
        tool_trace=[],
        raw_response=content,
        success=itinerary is not None,
        error=errors if not itinerary else None,
        total_tokens=total_tokens,
        processing_time_ms=processing_time,
        llm_calls=llm_calls,
        total_cost_usd=total_cost,
    )

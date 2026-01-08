# Module 7: Python AI Service — FastAPI + Structured Output

A deep-dive into building AI generation services with Python. This guide covers not just *how* to implement, but *why* Python is the right choice for AI workloads, and the engineering trade-offs that drive real-world decisions.

---

## Table of Contents

1. [Learning Objectives](#1-learning-objectives)
2. [The Philosophy of Polyglot Architecture](#2-the-philosophy-of-polyglot-architecture)
3. [Python Runtime Internals](#3-python-runtime-internals)
4. [FastAPI Deep Dive](#4-fastapi-deep-dive)
5. [Pydantic v2 Internals](#5-pydantic-v2-internals)
6. [OpenAI Structured Output](#6-openai-structured-output)
7. [Implementation Deep Dive](#7-implementation-deep-dive)
8. [Testing Async Services](#8-testing-async-services)
9. [Hands-On Lab](#9-hands-on-lab)
10. [Summary & Key Takeaways](#10-summary--key-takeaways)

**Appendices**
- [Appendix A: Project Structure](#appendix-a-project-structure)
- [Appendix B: CLI Quick Reference](#appendix-b-cli-quick-reference)
- [Appendix C: Troubleshooting](#appendix-c-troubleshooting)

---

## 1. Learning Objectives

After completing this module, you will understand:

- **Why polyglot architecture**: When to use Python vs Java in the same system
- **Python's async model**: How `asyncio` achieves concurrency on a single thread
- **FastAPI's ASGI stack**: Uvicorn → Starlette → FastAPI internals
- **Pydantic v2's Rust core**: Why validation is 10-20x faster than v1
- **Constrained decoding**: How OpenAI guarantees valid JSON output
- **Testing patterns**: Mocking async code and validating AI outputs

---

## 2. The Philosophy of Polyglot Architecture

### 2.1 Why Not Just Use Java for Everything?

The uncomfortable truth of AI development: **Java's ecosystem lags behind Python for LLM workloads**.

```
┌─────────────────────────────────────────────────────────────────┐
│                    VoyageAI Architecture                         │
│                                                                  │
│  Java Backend                    Python AI Service               │
│  ┌─────────────────────┐        ┌─────────────────────┐        │
│  │ Spring Boot         │        │ FastAPI             │        │
│  │ - Auth/JWT          │  HTTP  │ - Structured Output │        │
│  │ - MySQL/MongoDB     │◄──────►│ - Prompt Engineering│        │
│  │ - Redis/SSE         │        │ - Model Selection   │        │
│  │ - Task Orchestration│        │ - Retry Logic       │        │
│  └─────────────────────┘        └─────────────────────┘        │
│                                                                  │
│  User-facing, auth-critical     AI generation, I/O-bound        │
│  Transactions, consistency      Flexible, experimental          │
└─────────────────────────────────────────────────────────────────┘
```

**SDK Quality Comparison:**

| Feature | Java (OpenAI SDK) | Python (OpenAI SDK) |
|---------|-------------------|---------------------|
| Structured Output | Manual JSON Schema | Native Pydantic support |
| Async Support | CompletableFuture (verbose) | Native async/await |
| Streaming | Iterator-based | Generator-based |
| Type Hints | Runtime annotations | First-class types |
| Update Frequency | Monthly | Weekly |

**Real Example: Structured Output**

```java
// Java: 40+ lines to define schema and parse response
Map<String, Object> schema = new HashMap<>();
schema.put("type", "object");
schema.put("properties", Map.of(
    "destination", Map.of("type", "string"),
    "days", Map.of("type", "integer")
));
// ... manually build entire schema tree
// ... parse JSON response manually
// ... handle validation errors manually
```

```python
# Python: 10 lines with Pydantic
class Itinerary(BaseModel):
    destination: str
    days: int = Field(..., ge=1, le=30)

response = await client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_schema", "json_schema": {...}},
)
itinerary = Itinerary.model_validate_json(response.choices[0].message.content)
```

### 2.2 The Cost of Wrong Language Choice

**Case Study**: A startup built their RAG pipeline in Java because "we're a Java shop":
- 3 months to build what Python does in 3 weeks
- Constant battles with async handling (CompletableFuture hell)
- Couldn't use LangChain, ChromaDB, or most embedding libraries
- Eventually rewrote in Python, kept Java for auth/transactions

**Lesson**: Choose languages based on **ecosystem fit**, not organizational inertia.

### 2.3 When to Keep It in Java

Java is still the right choice for:

| Use Case | Why Java |
|----------|----------|
| Authentication | Mature security libraries (Spring Security) |
| Transactions | JPA, Hibernate, ACID guarantees |
| Task Orchestration | Thread pools, schedulers, resilience |
| User-facing APIs | Well-understood REST patterns |
| Data Validation | Bean Validation (JSR-380) |

**Our architecture decision**: Java orchestrates, Python generates.

---

## 3. Python Runtime Internals

### 3.1 The Global Interpreter Lock (GIL)

Python has a **Global Interpreter Lock** that prevents true parallelism:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CPython Runtime                               │
│                                                                  │
│  Thread 1 ──────►│              │◄────── Thread 2               │
│  (waiting)       │     GIL      │        (running)              │
│                  │   (mutex)    │                               │
│  Thread 3 ──────►│              │◄────── Thread 4               │
│  (waiting)       │              │        (waiting)              │
│                                                                  │
│  Only ONE thread executes Python bytecode at a time!            │
└─────────────────────────────────────────────────────────────────┘
```

**Why doesn't this kill performance for AI services?**

Because AI services are **I/O-bound**, not CPU-bound:

```python
# CPU-bound: GIL hurts
def fibonacci(n):
    # Pure Python computation - blocked by GIL
    if n < 2: return n
    return fibonacci(n-1) + fibonacci(n-2)

# I/O-bound: GIL doesn't matter
async def call_openai(prompt):
    # While waiting for network, other tasks run
    response = await client.chat.completions.create(...)
    return response
```

### 3.2 How Async I/O Works

Python's `asyncio` uses **cooperative multitasking**:

```
Event Loop (single thread):
                                                           
  ┌────────────────────────────────────────────────────────┐
  │  Task A: await openai.create()                         │
  │          │                                             │
  │          └──► Registers callback, YIELDS control       │
  │                                                        │
  │  Task B: await db.query()                              │
  │          │                                             │
  │          └──► Registers callback, YIELDS control       │
  │                                                        │
  │  Event Loop: Polls for I/O completion                  │
  │          │                                             │
  │          ├──► OpenAI responds! Resume Task A           │
  │          └──► DB responds! Resume Task B               │
  └────────────────────────────────────────────────────────┘
```

**The key insight**: `await` is a **yield point**. When you `await`, you're saying "I'm waiting for I/O, run something else."

**Practical impact:**

```python
# Sequential: 3 API calls = 3 × latency
async def sequential():
    r1 = await call_api()  # 1 second
    r2 = await call_api()  # 1 second
    r3 = await call_api()  # 1 second
    # Total: 3 seconds

# Concurrent: 3 API calls = 1 × latency
async def concurrent():
    r1, r2, r3 = await asyncio.gather(
        call_api(),  # Start immediately
        call_api(),  # Start immediately
        call_api(),  # Start immediately
    )
    # Total: 1 second (parallel I/O)
```

### 3.3 uvloop: The Fast Event Loop

Standard `asyncio` uses Python's event loop. `uvloop` replaces it with a Cython wrapper around `libuv` (Node.js's event loop):

```
Performance comparison (requests/second):
                                                           
  Standard asyncio:  ████████████████ 15,000 req/s
  uvloop:            ████████████████████████████ 25,000 req/s
                                                           
  ~70% faster for I/O-bound workloads!
```

FastAPI uses uvloop by default when running under Uvicorn.

---

## 4. FastAPI Deep Dive

### 4.1 The ASGI Stack

FastAPI is built on layers of abstraction:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Your Code (FastAPI)                         │
│  @app.post("/generate")                                          │
│  async def generate(request: GenerateRequest) -> GenerateResponse│
├─────────────────────────────────────────────────────────────────┤
│                      FastAPI Framework                           │
│  - Dependency injection                                          │
│  - Request body validation (Pydantic)                           │
│  - Response serialization                                        │
│  - OpenAPI schema generation                                     │
├─────────────────────────────────────────────────────────────────┤
│                      Starlette (ASGI Framework)                  │
│  - Routing (path → handler mapping)                             │
│  - Middleware pipeline                                           │
│  - Request/Response objects                                      │
│  - WebSocket/SSE support                                         │
├─────────────────────────────────────────────────────────────────┤
│                      Uvicorn (ASGI Server)                       │
│  - HTTP parsing (httptools, written in C)                       │
│  - Event loop (uvloop, wraps libuv)                             │
│  - Process management (multiple workers)                        │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Request Lifecycle

What happens when a request arrives:

```
1. TCP connection received by Uvicorn
2. HTTP bytes parsed by httptools (C extension)
3. ASGI scope dict created:
   {
     "type": "http",
     "method": "POST",
     "path": "/api/v1/generate",
     "headers": [...],
   }

4. Starlette matches route → FastAPI endpoint
5. FastAPI parses request body:
   - Reads JSON bytes
   - Pydantic validates against GenerateRequest
   - If invalid: Returns 422 with validation errors

6. Dependency injection resolves:
   - Depends(get_current_user) → injects user
   - Depends(get_db) → injects DB session

7. Your async function executes:
   async def generate(request: GenerateRequest):
       return await ai_service.generate(...)

8. Response handling:
   - Pydantic serializes GenerateResponse to JSON
   - Content-Type: application/json header added
   - HTTP bytes sent to client
```

### 4.3 Why FastAPI Over Flask?

| Feature | Flask | FastAPI |
|---------|-------|---------|
| Async Support | Requires gevent/eventlet hacks | Native async/await |
| Type Hints | Ignored at runtime | Pydantic validation |
| OpenAPI | Requires flask-apispec | Automatic generation |
| Performance | ~2,000 req/s | ~15,000 req/s |
| Learning Curve | Simpler | Slightly steeper |

**For AI services**, FastAPI wins because:
1. Native async for non-blocking OpenAI calls
2. Pydantic integration for structured output validation
3. Automatic OpenAPI docs for debugging

---

## 5. Pydantic v2 Internals

### 5.1 The Rust Core (pydantic-core)

Pydantic v2 was rewritten with validation logic in Rust:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pydantic v2 Architecture                      │
│                                                                  │
│  Python Layer                                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ class Itinerary(BaseModel):                                 ││
│  │     destination: str                                        ││
│  │     days: int = Field(..., ge=1, le=30)                     ││
│  │                                                              ││
│  │ At import time: Python → Rust schema compilation            ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  Rust Layer (pydantic-core)                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Compiled Validator:                                         ││
│  │ - Type checks in Rust (fast!)                               ││
│  │ - Regex matching in Rust                                    ││
│  │ - Number range checks in Rust                               ││
│  │ - Coercion (str → int) in Rust                              ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

**Performance impact:**

| Operation | Pydantic v1 | Pydantic v2 | Speedup |
|-----------|-------------|-------------|---------|
| Simple validation | 1.0x | 5-10x | Rust |
| Nested objects | 1.0x | 10-20x | Rust |
| JSON parsing | 1.0x | 5-10x | orjson + Rust |
| Regex patterns | 1.0x | 3-5x | Rust regex crate |

### 5.2 Validation Pipeline

When you call `model.model_validate(data)`:

```python
class Activity(BaseModel):
    activity_id: str = Field(pattern=r"^act-day\d+-\d{3}$")
    time: str = Field(pattern=r"^\d{2}:\d{2}-\d{2}:\d{2}$")
    estimated_cost: str | None = None
```

**Conceptual Rust validation (simplified):**

```rust
fn validate_activity(input: &Value) -> Result<Activity, ValidationError> {
    let obj = input.as_object()
        .ok_or(ValidationError::type_error("expected object"))?;
    
    // Validate activity_id (required, pattern)
    let activity_id = obj.get("activity_id")
        .ok_or(ValidationError::missing_field("activity_id"))?
        .as_str()
        .ok_or(ValidationError::type_error("expected string"))?;
    
    if !REGEX_CACHE.get("act-day").is_match(activity_id) {
        return Err(ValidationError::pattern_mismatch("activity_id"));
    }
    
    // Validate estimated_cost (optional)
    let estimated_cost = match obj.get("estimated_cost") {
        Some(v) if !v.is_null() => Some(v.as_str()?),
        _ => None,  // Missing or null → None
    };
    
    Ok(Activity { activity_id, time, estimated_cost })
}
```

### 5.3 JSON Schema Generation

Pydantic generates JSON Schema for OpenAPI and structured output:

```python
StructuredItinerary.model_json_schema()
```

Output:
```json
{
  "$defs": {
    "Activity": {
      "type": "object",
      "properties": {
        "activity_id": {
          "type": "string",
          "pattern": "^act-day\\d+-\\d{3}$"
        },
        "estimated_cost": {
          "anyOf": [{"type": "string"}, {"type": "null"}],
          "default": null
        }
      },
      "required": ["activity_id", "time", "title", "description", "location"]
    }
  },
  "type": "object",
  "properties": {
    "metadata": {"$ref": "#/$defs/ItineraryMetadata"},
    "days": {"type": "array", "items": {"$ref": "#/$defs/DailyItinerary"}}
  }
}
```

**Why this matters**: This schema is passed directly to OpenAI for constrained generation.

---

## 6. OpenAI Structured Output

### 6.1 How LLMs Generate Text

Traditional LLM generation is probabilistic:

```
Prompt: "Generate a travel plan for Tokyo"

Token probabilities at each step:
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: "Here" (0.15), "Day" (0.12), "{" (0.08), "I" (0.10)    │
│          Model samples → "Here"                                  │
│                                                                  │
│  Step 2: "'s" (0.20), " is" (0.15), "'re" (0.05)                │
│          Model samples → "'s"                                    │
│                                                                  │
│  Result: "Here's your travel plan:\n\n1. Day one: Visit..."    │
└─────────────────────────────────────────────────────────────────┘

Problem: Output is natural language, not parseable JSON!
```

### 6.2 Constrained Decoding with CFG

OpenAI's Structured Output uses **Context-Free Grammar (CFG)** constraints:

```
JSON Schema → CFG Rules → Token Filtering
                                                           
┌─────────────────────────────────────────────────────────────────┐
│  Schema: {"type": "object", "properties": {"name": {"type": "string"}}}
│                                                                  │
│  CFG Rules Generated:                                           │
│  S → '{' pairs '}'                                              │
│  pairs → pair (',' pair)*                                       │
│  pair → '"name"' ':' string                                     │
│  string → '"' chars '"'                                         │
│                                                                  │
│  At each generation step:                                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ State: AFTER_OPEN_BRACE                                     ││
│  │ Valid next tokens: ['"name"']   (only property name!)       ││
│  │ Invalid tokens: ["here", "is", "the", ...]                  ││
│  │                                                              ││
│  │ Token probabilities MASKED to only valid tokens!            ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

**Key insight**: The model can still generate creative *content*, but the *structure* is guaranteed.

### 6.3 Why `strict: true` Requires Schema Modifications

OpenAI's strict mode has requirements that Pydantic doesn't enforce by default:

**Requirement 1: `additionalProperties: false`**
```json
// Pydantic generates:
{"type": "object", "properties": {...}}

// OpenAI strict needs:
{"type": "object", "properties": {...}, "additionalProperties": false}

// Why? Prevents model from hallucinating extra fields:
// Bad: {"name": "Tokyo", "randomField": "unexpected"}
// Good: {"name": "Tokyo"}  (only defined properties)
```

**Requirement 2: All properties in `required`**
```json
// Pydantic generates (for optional fields):
{"properties": {"cost": {"type": "string"}}, "required": []}

// OpenAI strict needs:
{"properties": {"cost": {"anyOf": [{"type": "string"}, {"type": "null"}]}}, 
 "required": ["cost"]}

// Why? CFG needs to know ALL tokens to expect
```

**Our solution:**

```python
def make_strict_schema(schema: dict) -> dict:
    """Transform Pydantic schema for OpenAI strict mode."""
    if schema.get("type") == "object":
        # Prevent hallucinated fields
        schema["additionalProperties"] = False
        
        # All properties must be in required (nullable handled by anyOf)
        if "properties" in schema:
            schema["required"] = list(schema["properties"].keys())
    
    # Recursively process nested schemas
    for key in ["properties", "items", "$defs"]:
        if key in schema:
            if isinstance(schema[key], dict):
                for k, v in schema[key].items():
                    schema[key][k] = make_strict_schema(v)
    
    return schema
```

### 6.4 Performance and Reliability Trade-offs

| Mode | Latency | Token Efficiency | Reliability |
|------|---------|------------------|-------------|
| No format | 1.0x | 1.0x | ~70% valid JSON |
| `json_object` | 1.05x | 1.02x | ~95% valid JSON |
| `json_schema` strict | 1.1x | 1.05x | ~99.9% valid |

**Why the slight overhead?**
- CFG parsing adds computation per token
- More constrained generation = fewer "creative" paths
- Worth it for guaranteed parseability

---

## 7. Implementation Deep Dive

### 7.1 Configuration with Pydantic Settings

```python
"""
Pydantic Settings provides:
1. Type-safe environment variable loading
2. Automatic .env file reading
3. Validation at startup (fail fast!)
"""
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )
    
    # Required: Will fail at startup if missing
    openai_api_key: str
    
    # Optional with defaults
    openai_model: str = "gpt-4o-mini"
    max_retries: int = 3
    timeout_seconds: int = 60
    temperature: float = 0.7

# Singleton instance - loaded once at import
settings = Settings()

# Usage: settings.openai_api_key
```

**Why not just `os.environ`?**
- Type conversion: `MAX_RETRIES=3` → `int(3)`
- Validation: `TEMPERATURE=abc` → Error at startup
- Defaults: Missing vars get fallback values
- Documentation: Type hints document expected config

### 7.2 Schema Design Principles

```python
"""
Design principles for AI output schemas:

1. Pattern validation: Enforce machine-parseable formats
2. Length limits: Prevent runaway generation
3. Semantic types: Use domain-specific types
4. Optional with defaults: Everything the model might skip
"""

class Location(BaseModel):
    name: str = Field(..., description="Location name")
    
    # Geographic bounds validation
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    
    # Optional metadata
    address: str | None = None
    place_type: str | None = None

class Activity(BaseModel):
    # Pattern enforces parseable ID format
    activity_id: str = Field(
        ...,
        pattern=r"^act-day\d+-\d{3}$",  # act-day1-001
        description="Unique ID for referencing"
    )
    
    # Pattern enforces 24-hour time format
    time: str = Field(
        ...,
        pattern=r"^\d{2}:\d{2}-\d{2}:\d{2}$",  # 09:00-11:00
        description="Time range in 24-hour format"
    )
    
    # Length limits prevent token explosion
    title: str = Field(..., max_length=100)
    description: str = Field(..., max_length=500)
    
    # Nested object with full validation
    location: Location
    
    # Optional fields model might not generate
    estimated_cost: str | None = None
    notes: list[str] = Field(default_factory=list)
```

### 7.3 AI Service with Error Handling

```python
"""
AI service design principles:

1. Retry with exponential backoff for transient failures
2. Structured logging for debugging
3. Clean separation of prompt construction
4. Explicit error handling for each failure mode
"""
from tenacity import retry, stop_after_attempt, wait_exponential

class AIService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
    
    @retry(
        stop=stop_after_attempt(3),  # Max 3 attempts
        wait=wait_exponential(min=2, max=10),  # 2s, 4s, 8s delays
        reraise=True,  # Propagate final error
    )
    async def generate_itinerary(self, requirements: str) -> StructuredItinerary:
        """
        Generate structured itinerary using OpenAI.
        
        Retry logic handles:
        - Rate limiting (429): Backoff and retry
        - Server errors (500): Backoff and retry
        - Timeout: Backoff and retry
        
        Fails immediately on:
        - Invalid API key (401): No retry
        - Invalid schema (400): No retry
        """
        # Construct prompt with current context
        user_prompt = TRAVEL_ITINERARY_USER_PROMPT.format(
            requirements=requirements,
            current_date=datetime.now().strftime("%Y-%m-%d"),
        )
        
        logger.info(f"Generating itinerary for: {requirements[:100]}...")
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Example:\n{json.dumps(EXAMPLE)}"},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "travel_itinerary",
                        "schema": make_strict_schema(
                            StructuredItinerary.model_json_schema()
                        ),
                        "strict": True,
                    },
                },
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
            
            # Extract content (may be None on some errors)
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from OpenAI")
            
            # Parse and validate with Pydantic
            # This catches any schema violations the model made
            itinerary = StructuredItinerary.model_validate_json(content)
            
            logger.info(f"Generated {itinerary.metadata.total_days}-day itinerary")
            return itinerary
            
        except json.JSONDecodeError as e:
            # Should be rare with strict mode, but handle it
            logger.error(f"Invalid JSON from OpenAI: {e}")
            raise
        except ValidationError as e:
            # Pydantic validation failed - schema mismatch
            logger.error(f"Schema validation failed: {e}")
            raise
        except Exception as e:
            # Log and re-raise for retry decorator
            logger.error(f"AI generation failed: {e}")
            raise
```

### 7.4 Prompt Engineering

```python
"""
Prompt design principles:

1. Role: Define who the AI is
2. Guidelines: Numbered, actionable rules
3. Format constraints: Explicit patterns
4. Example: Show expected output format
"""

SYSTEM_PROMPT = """You are an expert travel planner with extensive knowledge.

Guidelines:
1. Create realistic schedules with appropriate travel times
2. Include a mix of popular attractions and local experiences
3. Consider meal times and rest periods
4. Provide accurate GPS coordinates for all locations
5. Include cost estimates in local currency
6. Add practical tips specific to each activity

Output must be valid JSON matching the provided schema exactly."""

USER_PROMPT = """Create a travel itinerary based on these requirements:

{requirements}

Additional context:
- Current date: {current_date}
- Response language: English
- Activity IDs: act-day1-001, act-day1-002, etc.
- Time format: 24-hour (e.g., 09:00-11:00)
- Date format: YYYY-MM-DD"""

# Few-shot example improves output quality
EXAMPLE = {
    "metadata": {
        "destination": "Tokyo, Japan",
        "start_date": "2026-01-10",
        "total_days": 2,
        # ... complete example
    },
    "days": [...],
    "tips": [...]
}
```

---

## 8. Testing Async Services

### 8.1 The Challenge of Mocking Async Code

```python
"""
Common mistake: AsyncMock for everything

AsyncMock is for async methods that need to be awaited.
MagicMock is for sync attributes on the response.
"""

# ❌ WRONG: Will fail with "can't use await expression"
@pytest.mark.asyncio
async def test_wrong(ai_service):
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]  # Wrong!
    
    with patch.object(ai_service.client.chat.completions, "create", 
                      return_value=mock_response):  # Wrong!
        await ai_service.generate()  # TypeError!

# ✅ CORRECT: AsyncMock for method, MagicMock for response
@pytest.mark.asyncio  
async def test_correct(ai_service):
    # Response object is accessed synchronously
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"valid": "json"}'
    
    # The create() method is awaited, so use AsyncMock
    mock_create = AsyncMock(return_value=mock_response)
    
    with patch.object(ai_service.client.chat.completions, "create", mock_create):
        result = await ai_service.generate()
        assert result is not None
```

### 8.2 Testing Pydantic Validation

```python
"""
Schema tests verify:
1. Valid data passes
2. Invalid data fails with specific errors
3. Edge cases (boundaries, patterns)
"""

class TestActivity:
    def test_valid_activity(self):
        activity = Activity(
            activity_id="act-day1-001",
            time="09:00-11:00",
            title="Temple Visit",
            description="Explore the ancient temple",
            location=Location(name="Temple", latitude=35.0, longitude=139.0),
        )
        assert activity.activity_id == "act-day1-001"
    
    def test_invalid_activity_id_pattern(self):
        """Pattern validation catches malformed IDs."""
        with pytest.raises(ValidationError) as exc_info:
            Activity(
                activity_id="wrong-format",  # Doesn't match pattern!
                time="09:00-11:00",
                # ...
            )
        
        # Verify specific error
        errors = exc_info.value.errors()
        assert errors[0]["loc"] == ("activity_id",)
        assert "pattern" in errors[0]["type"]
    
    def test_latitude_bounds(self):
        """Range validation catches impossible coordinates."""
        with pytest.raises(ValidationError):
            Location(name="Invalid", latitude=999, longitude=0)
```

### 8.3 Integration Tests with HTTPX

```python
"""
Integration tests verify:
1. Endpoints return correct status codes
2. Request validation works
3. Response format matches schema
"""
from httpx import AsyncClient, ASGITransport

@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/api/v1/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "UP"
    assert "timestamp" in data

@pytest.mark.asyncio
async def test_generate_validation_error(client):
    """Missing required fields return 422."""
    response = await client.post("/api/v1/generate", json={
        "task_id": "test",
        # Missing: user_id, project_id, requirements
    })
    
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(e["loc"] == ["body", "user_id"] for e in errors)
```

---

## 9. Hands-On Lab

### Lab 1: Run and Verify the Service

```bash
# 1. Navigate to project
cd voyageai-python-service

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Create .env file
cp .env.example .env
# Edit .env to add your OPENAI_API_KEY

# 5. Start service
uvicorn voyageai.main:app --reload --port 8000

# 6. Verify health
curl http://localhost:8000/api/v1/health | jq .
```

### Lab 2: Test Generation Endpoint

```bash
# Generate an itinerary
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test-001",
    "user_id": "user-001", 
    "project_id": "proj-001",
    "requirements": "Plan a 2-day trip to Tokyo focusing on temples and local food"
  }' | jq .

# Check processing time
curl -s -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{...}' | jq '.processing_time_ms'
```

### Lab 3: Run Tests

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=voyageai --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Lab 4: Explore the Schema

```python
# In Python REPL
from voyageai.schemas.itinerary import StructuredItinerary
import json

# View the JSON Schema
schema = StructuredItinerary.model_json_schema()
print(json.dumps(schema, indent=2))

# Test validation
from pydantic import ValidationError

try:
    StructuredItinerary.model_validate({"invalid": "data"})
except ValidationError as e:
    print(e.errors())
```

---

## 10. Summary & Key Takeaways

### What We Built

```
┌─────────────────────────────────────────────────────────────────┐
│                  VoyageAI Python Service                         │
│                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │  FastAPI   │  │  Pydantic  │  │  OpenAI    │                │
│  │  (ASGI)    │  │  (Rust)    │  │  (Strict)  │                │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘                │
│         │               │               │                        │
│         ▼               ▼               ▼                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  /api/v1/generate                           │ │
│  │                                                              │ │
│  │  Request (JSON) → Pydantic Validation → AI Generation      │ │
│  │                           ↓                                  │ │
│  │                 Structured Output (CFG)                     │ │
│  │                           ↓                                  │ │
│  │                Pydantic Validation → Response (JSON)        │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Key Engineering Lessons

1. **Choose languages by ecosystem, not inertia**
   - Python for AI workloads: Better SDKs, faster iteration
   - Java for business logic: Transactions, security, enterprise patterns

2. **Understand async I/O**
   - `await` is a yield point, not a blocking call
   - One thread can handle thousands of I/O-bound requests

3. **Pydantic is your contract**
   - Define strict schemas with patterns and bounds
   - Validate at boundaries (input and output)

4. **Structured output > hoping for JSON**
   - CFG-based generation guarantees valid output
   - Worth the 10% latency overhead

5. **Test async code correctly**
   - AsyncMock for methods, MagicMock for response objects
   - HTTPX ASGITransport for integration tests

### Decision Framework

```
When to add Python to your Java stack:

├── Need LLM integration?
│   └── Yes → Python (better SDKs, structured output)
│
├── Need ML model inference?
│   └── Yes → Python (transformers, PyTorch, TensorFlow)
│
├── Need vector embeddings?
│   └── Yes → Python (sentence-transformers, OpenAI embeddings)
│
├── Need complex auth/transactions?
│   └── Yes → Stay in Java
│
└── Need rapid prototyping?
    └── Yes → Python, then port critical paths to Java
```

---

## Appendix A: Project Structure

```
voyageai-python-service/
├── pyproject.toml              # Dependencies and project metadata
├── .env.example                # Environment variable template
├── .gitignore                  # Excludes .env, .venv, __pycache__
├── README.md                   # Quick start guide
│
├── src/
│   └── voyageai/
│       ├── __init__.py         # Package marker
│       ├── config.py           # Pydantic Settings configuration
│       ├── main.py             # FastAPI application entry
│       │
│       ├── schemas/
│       │   ├── __init__.py     # Re-exports all schemas
│       │   ├── itinerary.py    # StructuredItinerary, Activity, Location
│       │   └── task.py         # GenerateRequest, GenerateResponse
│       │
│       ├── services/
│       │   ├── __init__.py
│       │   └── ai_service.py   # OpenAI integration with retry logic
│       │
│       ├── routers/
│       │   ├── __init__.py
│       │   ├── health.py       # GET /api/v1/health
│       │   └── planning.py     # POST /api/v1/generate
│       │
│       └── prompts/
│           ├── __init__.py
│           └── templates.py    # System/user prompts, examples
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures (async client)
│   ├── test_schemas.py         # Pydantic validation tests
│   ├── test_ai_service.py      # AI service with mocked OpenAI
│   └── test_integration.py     # Endpoint integration tests
│
└── docs/
    └── 07-python-ai-service.md # This document
```

---

## Appendix B: CLI Quick Reference

### Service Management

```bash
# Development (auto-reload)
uvicorn voyageai.main:app --reload --port 8000

# Production (multiple workers)
uvicorn voyageai.main:app --host 0.0.0.0 --port 8000 --workers 4

# With specific log level
uvicorn voyageai.main:app --log-level debug
```

### API Testing

```bash
# Health check
curl http://localhost:8000/api/v1/health | jq .

# Generate itinerary
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test-001",
    "user_id": "user-001",
    "project_id": "proj-001",
    "requirements": "3 day trip to Paris"
  }' | jq .

# OpenAPI docs
open http://localhost:8000/docs
```

### Testing

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Run specific test file
pytest tests/test_schemas.py

# Run specific test
pytest tests/test_schemas.py::TestActivity::test_valid_activity

# With coverage
pytest --cov=voyageai --cov-report=html
open htmlcov/index.html
```

### Dependency Management

```bash
# Install dependencies
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Upgrade dependencies
pip install --upgrade -e ".[dev]"
```

---

## Appendix C: Troubleshooting

### C.1 OpenAI API Errors

**401 Unauthorized:**
```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Check .env file
cat .env | grep OPENAI

# Verify key works
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

**400 Bad Request (Schema Error):**
```
Error: "additionalProperties is required to be false"
```
Solution: Ensure `make_strict_schema()` is applied to Pydantic schema.

```
Error: "required array must include all properties"  
```
Solution: All properties must be in `required` array for strict mode.

**429 Rate Limited:**
```bash
# Check your usage at
open https://platform.openai.com/usage

# Increase retry delays in ai_service.py
@retry(wait=wait_exponential(min=5, max=30))  # Longer waits
```

### C.2 Pydantic Validation Errors

**Pattern Mismatch:**
```python
ValidationError: activity_id - String should match pattern '^act-day\d+-\d{3}$'
```
Solution: Ensure AI prompt includes pattern examples.

**Missing Required Field:**
```python
ValidationError: metadata - Field required
```
Solution: Check if OpenAI response is complete, increase `max_tokens` if truncated.

### C.3 Async Test Issues

**TypeError: object AsyncMock can't be used in 'await' expression:**
```python
# Wrong
mock_create = AsyncMock()
with patch.object(client, "create", return_value=mock_create):
    await service.generate()  # Fails!

# Correct
mock_create = AsyncMock(return_value=mock_response)  # return_value is sync
with patch.object(client, "create", mock_create):
    await service.generate()  # Works!
```

### C.4 Import Errors

**ModuleNotFoundError: No module named 'voyageai':**
```bash
# Ensure package is installed in editable mode
pip install -e .

# Verify installation
pip show voyageai-python-service
```

**ImportError: cannot import name 'Settings':**
```bash
# Check for circular imports
# Settings should only depend on pydantic_settings, nothing else

# Verify import order in __init__.py files
```

---

**Module Version**: 7.0  
**Last Updated**: January 2026  
**Prerequisites**: Module 1-6 (Java backend running)  
**Next Module**: Module 8 - Tool Calling

# Module 8: Tool Calling - Building AI Agents with Real-World Capabilities

## Table of Contents

1. [Overview](#1-overview)
2. [Theoretical Foundations](#2-theoretical-foundations)
3. [Architecture Design](#3-architecture-design)
4. [Implementation Deep Dive](#4-implementation-deep-dive)
5. [OpenAI Function Calling Internals](#5-openai-function-calling-internals)
6. [Tool Implementation Patterns](#6-tool-implementation-patterns)
7. [Security Considerations](#7-security-considerations)
8. [Testing Strategies](#8-testing-strategies)
9. [Production Considerations](#9-production-considerations)
10. [Hands-On Exercises](#10-hands-on-exercises)

---

## 1. Overview

### What is Tool Calling?

Tool calling (also known as "function calling") is a mechanism that allows Large Language Models (LLMs) to interact with external systems, APIs, and data sources. Instead of relying solely on their training data, LLMs can:

1. **Recognize** when external data is needed
2. **Decide** which tool(s) to call
3. **Generate** structured arguments for the tool
4. **Interpret** results and incorporate them into responses

### Why Tool Calling Matters

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Without Tools                             │
│                                                                  │
│  User: "What's the weather in Tokyo tomorrow?"                  │
│  LLM: "I don't have access to real-time weather data..."        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    LLM With Tools                                │
│                                                                  │
│  User: "What's the weather in Tokyo tomorrow?"                  │
│  LLM: [calls get_weather_forecast tool]                         │
│  LLM: "Tomorrow in Tokyo will be 13°C with partly cloudy        │
│        skies and a 15% chance of precipitation."                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Module Learning Objectives

By the end of this module, you will:

- Understand the ReAct (Reasoning + Acting) pattern for AI agents
- Implement a tool abstraction layer with proper typing
- Build real tools using free external APIs
- Create a tool registry for dynamic tool discovery
- Implement an agent loop with multi-turn tool execution
- Apply security best practices for tool-based systems

---

## 2. Theoretical Foundations

### 2.1 The ReAct Pattern

ReAct (Reason + Act) is a paradigm for building AI agents that interleave reasoning and acting:

```
┌─────────────────────────────────────────────────────────────────┐
│                     ReAct Loop                                   │
│                                                                  │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│   │ Thought │───▶│ Action  │───▶│ Observe │───▶│ Thought │──▶...│
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘     │
│                                                                  │
│   Example:                                                       │
│   Thought: "I need weather data for Tokyo to plan activities"   │
│   Action:  get_weather_forecast(lat=35.68, lon=139.76, ...)    │
│   Observe: {"temp_max": 13.3, "condition": "Mainly clear", ...} │
│   Thought: "Good weather, I can plan outdoor activities"        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight**: The LLM doesn't just execute a fixed plan—it reasons about each observation and decides the next action dynamically.

### 2.2 Tool Calling vs. RAG

| Aspect | Tool Calling | RAG |
|--------|-------------|-----|
| **Data Source** | External APIs, calculations | Vector database |
| **Data Freshness** | Real-time | Pre-indexed |
| **Use Case** | Actions, live data | Knowledge retrieval |
| **Output** | Structured data | Text passages |
| **Latency** | API-dependent | Database query |

**When to Use Both**: Modern AI systems often combine both approaches:
- Use **RAG** for historical knowledge (travel guides, reviews)
- Use **Tools** for real-time data (weather, prices, availability)

### 2.3 Grounding and Hallucination Reduction

Tool calling helps **ground** LLM responses in reality:

```
Without Grounding:
  LLM might say: "Tokyo weather is usually around 20°C in January"
  (Hallucinated based on general knowledge, actually ~5-10°C)

With Tool Grounding:
  LLM calls: get_weather_forecast(...)
  LLM says: "Based on the forecast, Tokyo will be 0.8-13.3°C"
  (Accurate, verifiable data)
```

---

## 3. Architecture Design

### 3.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Agent Service                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Tool Calling Loop                     │    │
│  │                                                          │    │
│  │  ┌──────────┐   ┌────────────┐   ┌────────────────┐    │    │
│  │  │  OpenAI  │──▶│   Parse    │──▶│  Tool Registry │    │    │
│  │  │   API    │   │ Tool Calls │   │                │    │    │
│  │  └──────────┘   └────────────┘   └────────────────┘    │    │
│  │       ▲                                   │             │    │
│  │       │                                   ▼             │    │
│  │       │              ┌────────────────────────────┐    │    │
│  │       │              │      Execute Tools         │    │    │
│  │       │              │  ┌─────┐ ┌─────┐ ┌─────┐  │    │    │
│  │       │              │  │Geo- │ │Wea- │ │Curr-│  │    │    │
│  │       │              │  │code │ │ther │ │ency │  │    │    │
│  │       │              │  └─────┘ └─────┘ └─────┘  │    │    │
│  │       │              └────────────────────────────┘    │    │
│  │       │                           │                    │    │
│  │       └───────────────────────────┘                    │    │
│  │                Tool Results                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  External APIs:                                                  │
│  ├── Open-Meteo (weather)                                       │
│  ├── Nominatim (geocoding)                                      │
│  ├── Frankfurter (currency)                                     │
│  └── Nager.Date (holidays)                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Tool Abstraction Layer

```python
# Design Principles:
# 1. Single Responsibility: Each tool does one thing well
# 2. Stateless: Tools don't maintain state between calls
# 3. Async: All tools are async for non-blocking I/O
# 4. Structured Output: Tools return typed ToolResult objects

class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    name: str                    # Unique identifier
    description: str             # For LLM to understand usage
    parameters_schema: dict      # JSON Schema for input validation
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool and return structured result."""
        pass
```

### 3.3 Message Flow Diagram

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  User   │    │  Agent  │    │ OpenAI  │    │  Tools  │
└────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘
     │              │              │              │
     │  Request     │              │              │
     │─────────────▶│              │              │
     │              │              │              │
     │              │  Messages +  │              │
     │              │  Tools Def   │              │
     │              │─────────────▶│              │
     │              │              │              │
     │              │ Tool Calls   │              │
     │              │◀─────────────│              │
     │              │              │              │
     │              │  Execute     │              │
     │              │─────────────────────────────▶
     │              │              │              │
     │              │  Results     │              │
     │              │◀─────────────────────────────
     │              │              │              │
     │              │  Messages +  │              │
     │              │  Tool Results│              │
     │              │─────────────▶│              │
     │              │              │              │
     │              │   (repeat if more tools)   │
     │              │              │              │
     │              │ Final Answer │              │
     │              │◀─────────────│              │
     │              │              │              │
     │  Response    │              │              │
     │◀─────────────│              │              │
     │              │              │              │
```

---

## 4. Implementation Deep Dive

### 4.1 Tool Result Model

The `ToolResult` provides a consistent contract for all tool outputs:

```python
class ToolResult(BaseModel):
    """
    Structured result from tool execution.
    
    Why this design?
    1. Consistent error handling across all tools
    2. Performance tracking via latency_ms
    3. Debugging with input_args preservation
    4. Type safety with Pydantic validation
    """
    
    tool_name: str                    # Which tool was called
    input_args: dict = {}             # What arguments were passed
    output: Any = None                # The actual result (varies by tool)
    success: bool                     # Did execution succeed?
    error: str | None = None          # Error message if failed
    latency_ms: int = 0               # Execution time for monitoring
```

**Design Decision**: Why `output: Any` instead of generics?

```python
# Option A: Generic (more type-safe but complex)
class ToolResult[T](BaseModel):
    output: T | None = None

# Option B: Any (simpler, works with JSON serialization)
class ToolResult(BaseModel):
    output: Any = None

# We chose Option B because:
# 1. Tool outputs are ultimately serialized to JSON for LLM
# 2. Different tools have completely different output shapes
# 3. Pydantic handles JSON serialization of any nested structure
# 4. Simplicity wins for teaching purposes
```

### 4.2 Base Tool Implementation

```python
class BaseTool(ABC):
    """
    Abstract base class for all VoyageAI tools.
    
    Inheritance vs Composition:
    - We use inheritance here because all tools share the same interface
    - The LLM expects a consistent format (name, description, parameters)
    - Template Method pattern: execute() is the template, subclasses fill in
    """
    
    name: str
    description: str
    parameters_schema: dict
    
    def to_openai_function(self) -> dict:
        """
        Convert tool to OpenAI function calling format.
        
        OpenAI expects this exact structure:
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "...",
                "parameters": { JSON Schema }
            }
        }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            }
        }
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with the given arguments.
        
        Why async?
        - Most tools make HTTP requests (I/O bound)
        - Async allows concurrent tool execution
        - FastAPI is async-native, no thread pool needed
        
        Why **kwargs?
        - Different tools have different parameters
        - JSON Schema validates the actual arguments
        - Subclasses define their specific parameters
        """
        pass
```

### 4.3 Real Tool Example: WeatherTool

```python
class WeatherTool(BaseTool):
    """
    Get weather forecast using Open-Meteo API.
    
    API Choice: Open-Meteo (https://open-meteo.com/)
    - Completely free, no API key required
    - 16-day forecast available
    - Based on national weather services
    - No rate limits for reasonable usage
    
    Why not OpenWeatherMap?
    - Requires API key registration
    - Free tier has limitations
    - For teaching, zero-config is better
    """
    
    name = "get_weather_forecast"
    description = (
        "Get weather forecast for a destination. "
        "Requires latitude and longitude coordinates. "
        "Returns daily temperature, precipitation chance, and conditions. "
        "Use geocode_location first to get coordinates."
    )
    
    # JSON Schema for input validation
    parameters_schema = {
        "type": "object",
        "properties": {
            "latitude": {
                "type": "number",
                "description": "Latitude coordinate (-90 to 90)"
            },
            "longitude": {
                "type": "number",
                "description": "Longitude coordinate (-180 to 180)"
            },
            "start_date": {
                "type": "string",
                "description": "Start date (YYYY-MM-DD format)"
            },
            "end_date": {
                "type": "string",
                "description": "End date (YYYY-MM-DD format)"
            }
        },
        "required": ["latitude", "longitude", "start_date", "end_date"],
        "additionalProperties": False  # Required for OpenAI strict mode
    }
```

### 4.4 WMO Weather Codes

The Open-Meteo API uses WMO (World Meteorological Organization) standard weather codes:

```python
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}
```

**Why Translate Codes?**

```python
# Raw API Response:
{"weathercode": 61}

# After Translation (for LLM):
{"condition": "Slight rain"}

# The LLM can now reason about weather conditions in natural language
```

### 4.5 Distance Calculation: Haversine Formula

The `DistanceTool` uses the Haversine formula for calculating great-circle distances:

```python
def _haversine_distance(self, lat1, lon1, lat2, lon2) -> float:
    """
    Calculate the great-circle distance between two points.
    
    The Haversine formula:
    a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    c = 2 × atan2(√a, √(1−a))
    d = R × c
    
    Where:
    - R = Earth's radius (6,371 km)
    - Δlat = lat2 − lat1
    - Δlon = lon2 − lon1
    
    Why Haversine?
    - Accurate for any distance on Earth
    - Simple to implement (no external API needed)
    - Gives "as-the-crow-flies" distance
    
    Limitation:
    - Doesn't account for actual road/rail routes
    - For that, you'd need a routing API (Google Maps, etc.)
    """
    R = 6371.0  # Earth's radius in km
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = (
        math.sin(delta_lat / 2) ** 2 +
        math.cos(lat1_rad) * math.cos(lat2_rad) * 
        math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c
```

**Visual Representation:**

```
             North Pole
                 │
                 │
    Tokyo ●──────│──────● Osaka
    (35.68°N)    │      (34.69°N)
                 │
                 │
         Great Circle Path
         (shortest distance on sphere)
         
    Result: 402.5 km
```

---

## 5. OpenAI Function Calling Internals

### 5.1 How Function Calling Works

```
┌─────────────────────────────────────────────────────────────────┐
│                OpenAI Function Calling Pipeline                  │
│                                                                  │
│  1. Tool Definitions Sent                                       │
│     ┌─────────────────────────────────────────┐                │
│     │ tools: [                                │                │
│     │   { name: "get_weather", params: {...}},│                │
│     │   { name: "geocode", params: {...}}     │                │
│     │ ]                                       │                │
│     └─────────────────────────────────────────┘                │
│                         │                                       │
│                         ▼                                       │
│  2. LLM Reasoning (Internal)                                   │
│     ┌─────────────────────────────────────────┐                │
│     │ "User wants weather in Tokyo..."        │                │
│     │ "I should call get_weather tool..."     │                │
│     │ "But I need coordinates first..."       │                │
│     │ "I'll call geocode first"               │                │
│     └─────────────────────────────────────────┘                │
│                         │                                       │
│                         ▼                                       │
│  3. Tool Call Generated                                        │
│     ┌─────────────────────────────────────────┐                │
│     │ tool_calls: [{                          │                │
│     │   id: "call_abc123",                    │                │
│     │   function: {                           │                │
│     │     name: "geocode_location",           │                │
│     │     arguments: '{"location": "Tokyo"}'  │                │
│     │   }                                     │                │
│     │ }]                                      │                │
│     └─────────────────────────────────────────┘                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 JSON Schema Requirements

OpenAI's function calling uses JSON Schema for parameter validation:

```python
parameters_schema = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "City name or address"  # Crucial for LLM
        },
        "format": {
            "type": "string",
            "enum": ["json", "text"],              # Constrained values
            "description": "Output format"
        }
    },
    "required": ["location"],                      # Mandatory fields
    "additionalProperties": False                  # Strict mode requirement
}
```

**Key Points:**

1. **Descriptions are Critical**: The LLM uses descriptions to understand when and how to use each parameter
2. **`additionalProperties: false`**: Required for OpenAI's strict mode to prevent hallucinated parameters
3. **`enum` for Constraints**: When parameters have fixed options, use `enum` to prevent invalid values

### 5.3 Multi-Turn Tool Calling

A single user request can result in multiple tool calls:

```python
# Turn 1: User Request
messages = [
    {"role": "user", "content": "Plan a trip to Tokyo"}
]

# Turn 2: LLM decides to call geocode
response = await client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tool_definitions,
)
# response.choices[0].message.tool_calls = [{name: "geocode_location", ...}]

# Turn 3: We execute the tool and add result
messages.append(response.choices[0].message)  # Assistant's decision
messages.append({
    "role": "tool",
    "tool_call_id": "call_abc123",
    "content": '{"latitude": 35.68, "longitude": 139.76}'
})

# Turn 4: LLM calls another tool (weather)
response = await client.chat.completions.create(...)
# response.choices[0].message.tool_calls = [{name: "get_weather_forecast", ...}]

# ... and so on until finish_reason == "stop"
```

### 5.4 The Tool Calling Loop

```python
async def generate_with_tools(self, requirements: str) -> AgentResponse:
    """
    The core agent loop implementing ReAct pattern.
    
    Key Design Decisions:
    1. Max iterations to prevent infinite loops
    2. All tool calls tracked for observability
    3. Final output uses structured JSON schema
    """
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": requirements}
    ]
    
    tool_trace = []
    
    for iteration in range(self.max_iterations):
        # Call LLM with tools
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tool_registry.get_openai_tools(),
            tool_choice="auto",  # Let LLM decide
        )
        
        message = response.choices[0].message
        
        # Check if LLM wants to call tools
        if message.tool_calls:
            # Execute each tool call
            for tool_call in message.tool_calls:
                result = await tool_registry.execute(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )
                
                # Track for observability
                tool_trace.append(ToolCallTrace(
                    call_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    arguments=arguments,
                    result=result.output,
                    latency_ms=result.latency_ms
                ))
                
                # Add result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result.output)
                })
            
            continue  # Next iteration
        
        # No tool calls - LLM is done, get structured output
        return self._generate_final_response(messages)
```

---

## 6. Tool Implementation Patterns

### 6.1 Pattern: External API Tool

```python
class ExternalAPITool(BaseTool):
    """Pattern for tools that call external APIs."""
    
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
    
    async def execute(self, **kwargs) -> ToolResult:
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.api_url,
                    params=self._build_params(kwargs),
                    headers=self._get_headers(),
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
            
            return ToolResult(
                tool_name=self.name,
                input_args=kwargs,
                output=self._parse_response(data),
                success=True,
                latency_ms=int((time.time() - start_time) * 1000)
            )
            
        except httpx.TimeoutException:
            return ToolResult(
                tool_name=self.name,
                input_args=kwargs,
                output=None,
                success=False,
                error="API request timed out",
                latency_ms=int((time.time() - start_time) * 1000)
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                input_args=kwargs,
                output=None,
                success=False,
                error=str(e),
                latency_ms=int((time.time() - start_time) * 1000)
            )
```

### 6.2 Pattern: Local Computation Tool

```python
class LocalComputationTool(BaseTool):
    """Pattern for tools that compute locally without external calls."""
    
    async def execute(self, **kwargs) -> ToolResult:
        start_time = time.time()
        
        try:
            # Pure Python computation
            result = self._compute(**kwargs)
            
            return ToolResult(
                tool_name=self.name,
                input_args=kwargs,
                output=result,
                success=True,
                latency_ms=int((time.time() - start_time) * 1000)
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                input_args=kwargs,
                output=None,
                success=False,
                error=str(e),
                latency_ms=int((time.time() - start_time) * 1000)
            )
    
    def _compute(self, **kwargs) -> dict:
        """Override in subclass with actual computation."""
        raise NotImplementedError
```

### 6.3 Pattern: Rate-Limited Tool

```python
class RateLimitedTool(BaseTool):
    """Pattern for tools that must respect API rate limits."""
    
    def __init__(self, requests_per_second: float = 1.0):
        self.min_interval = 1.0 / requests_per_second
        self._last_request_time = 0.0
    
    async def _respect_rate_limit(self) -> None:
        """Wait if necessary to respect rate limit."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()
    
    async def execute(self, **kwargs) -> ToolResult:
        await self._respect_rate_limit()
        return await self._execute_impl(**kwargs)
```

**Example**: The `GeocodeTool` implements this pattern because Nominatim requires max 1 request/second.

### 6.4 Pattern: Tool Registry

```python
class ToolRegistry:
    """
    Centralized registry for tool discovery and execution.
    
    Design Patterns Used:
    1. Registry Pattern: Central catalog of available tools
    2. Factory Pattern: Tools are instantiated and registered at startup
    3. Strategy Pattern: Each tool implements the same interface
    """
    
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool
    
    def get_openai_tools(self) -> list[dict]:
        """Get all tools in OpenAI format for API calls."""
        return [tool.to_openai_function() for tool in self._tools.values()]
    
    async def execute(self, name: str, arguments: dict) -> ToolResult:
        """Execute a tool by name."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Unknown tool: {name}"
            )
        return await tool.execute(**arguments)

# Global singleton
tool_registry = ToolRegistry()
tool_registry.register(GeocodeTool())
tool_registry.register(WeatherTool())
# ... etc
```

---

## 7. Security Considerations

### 7.1 Input Validation

**Always validate inputs before processing:**

```python
async def execute(self, latitude: float, longitude: float, ...) -> ToolResult:
    # Validate coordinate ranges
    if not (-90 <= latitude <= 90):
        return ToolResult(
            tool_name=self.name,
            success=False,
            error="Latitude must be between -90 and 90"
        )
    
    if not (-180 <= longitude <= 180):
        return ToolResult(
            tool_name=self.name,
            success=False,
            error="Longitude must be between -180 and 180"
        )
    
    # Safe to proceed...
```

### 7.2 API Key Management

```python
# ❌ BAD: Hardcoded API keys
class WeatherTool:
    API_KEY = "sk-abc123..."

# ✅ GOOD: Environment variables via pydantic-settings
class Settings(BaseSettings):
    openai_api_key: str
    # weatherapi_key: str  # If we used a paid weather API
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )
```

### 7.3 Service-to-Service Authentication

When the Python service is called by Java backend:

```python
# Option 1: API Key Header
@router.post("/generate")
async def generate(
    request: GenerateRequest,
    x_api_key: str = Header(..., alias="X-API-Key")
):
    if x_api_key != settings.internal_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    # ...

# Option 2: Network Isolation (preferred in production)
# - Run Python service in same VPC/network as Java
# - Only expose internal port, not public
# - Use service mesh (Istio) for mTLS

# Option 3: JWT validation
# - Java includes its JWT in request
# - Python validates signature (shared secret or JWKS)
```

### 7.4 Rate Limiting and Abuse Prevention

```python
# At FastAPI level
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.get("/tools/geocode")
@limiter.limit("10/minute")  # Prevent abuse
async def geocode_location(request: Request, location: str):
    # ...
```

### 7.5 Prompt Injection Prevention

Be cautious about passing user input directly to tool descriptions:

```python
# ❌ VULNERABLE: User input in tool context
tool_description = f"Search for: {user_query}"  # Could contain injection

# ✅ SAFER: Structured separation
messages = [
    {"role": "system", "content": FIXED_SYSTEM_PROMPT},
    {"role": "user", "content": user_query}  # Clearly marked as user input
]
```

---

## 8. Testing Strategies

### 8.1 Unit Testing Individual Tools

```python
class TestWeatherTool:
    @pytest.fixture
    def tool(self):
        return WeatherTool()
    
    @pytest.mark.asyncio
    async def test_weather_valid_coordinates(self, tool):
        """Test with valid coordinates returns forecast."""
        today = datetime.now()
        result = await tool.execute(
            latitude=35.68,
            longitude=139.76,
            start_date=today.strftime("%Y-%m-%d"),
            end_date=(today + timedelta(days=2)).strftime("%Y-%m-%d")
        )
        
        assert result.success is True
        assert "forecast" in result.output
        assert len(result.output["forecast"]) >= 1
    
    @pytest.mark.asyncio
    async def test_weather_invalid_latitude(self, tool):
        """Test invalid latitude returns error."""
        result = await tool.execute(
            latitude=999,  # Invalid
            longitude=139.76,
            start_date="2026-01-10",
            end_date="2026-01-12"
        )
        
        assert result.success is False
        assert "latitude" in result.error.lower()
```

### 8.2 Integration Testing with Real APIs

```python
@pytest.mark.asyncio
async def test_geocode_to_weather_pipeline():
    """Test realistic tool chain: geocode → weather."""
    
    # Step 1: Geocode
    geocode_result = await tool_registry.execute(
        "geocode_location",
        {"location": "Paris, France"}
    )
    assert geocode_result.success
    
    # Step 2: Use coordinates for weather
    weather_result = await tool_registry.execute(
        "get_weather_forecast",
        {
            "latitude": geocode_result.output["latitude"],
            "longitude": geocode_result.output["longitude"],
            "start_date": "2026-01-10",
            "end_date": "2026-01-12"
        }
    )
    assert weather_result.success
    assert weather_result.output["forecast"][0]["condition"]
```

### 8.3 Mocking for Agent Tests

```python
@pytest.mark.asyncio
async def test_agent_tool_execution():
    """Test agent correctly executes tool calls."""
    
    # Mock OpenAI response with tool call
    mock_message = MagicMock()
    mock_message.tool_calls = [MagicMock(
        id="call_123",
        function=MagicMock(
            name="geocode_location",
            arguments='{"location": "Tokyo"}'
        )
    )]
    
    with patch.object(agent.client.chat.completions, 'create', 
                      return_value=mock_response):
        response = await agent.generate_with_tools("Plan trip to Tokyo")
    
    # Verify geocode was called
    assert any(t.tool_name == "geocode_location" for t in response.tool_trace)
```

---

## 9. Production Considerations

### 9.1 Observability

```python
# Structured logging for tool calls
logger.info(
    "Tool execution",
    extra={
        "tool_name": result.tool_name,
        "success": result.success,
        "latency_ms": result.latency_ms,
        "input_args": result.input_args,
        "request_id": request_id,  # Correlation ID
    }
)

# Metrics (with Prometheus)
TOOL_CALLS = Counter(
    'tool_calls_total',
    'Total tool calls',
    ['tool_name', 'status']
)

TOOL_LATENCY = Histogram(
    'tool_latency_seconds',
    'Tool execution latency',
    ['tool_name']
)
```

### 9.2 Error Recovery

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(httpx.TimeoutException)
)
async def _call_api(self, **kwargs):
    """Retry transient failures with exponential backoff."""
    async with httpx.AsyncClient() as client:
        response = await client.get(...)
        response.raise_for_status()
        return response.json()
```

### 9.3 Caching Tool Results

```python
import hashlib
from functools import lru_cache

class CachedGeocodeTool(GeocodeTool):
    """Cache geocoding results to reduce API calls."""
    
    def __init__(self):
        super().__init__()
        self._cache: dict[str, dict] = {}
    
    async def execute(self, location: str) -> ToolResult:
        # Normalize cache key
        cache_key = location.lower().strip()
        
        if cache_key in self._cache:
            return ToolResult(
                tool_name=self.name,
                input_args={"location": location},
                output=self._cache[cache_key],
                success=True,
                latency_ms=0  # Cached!
            )
        
        result = await super().execute(location=location)
        
        if result.success:
            self._cache[cache_key] = result.output
        
        return result
```

### 9.4 Graceful Degradation

```python
async def generate_with_tools(self, requirements: str) -> AgentResponse:
    """Generate itinerary with graceful degradation."""
    
    try:
        # Try with tools
        return await self._generate_with_tools_impl(requirements)
    except Exception as e:
        logger.warning(f"Tool-enhanced generation failed: {e}")
        
        # Fallback to basic generation without tools
        return await self._generate_basic(requirements)
```

---

## 10. Hands-On Exercises

### Exercise 1: Add a New Tool

Implement a `TranslationTool` that translates text:

```python
class TranslationTool(BaseTool):
    """
    Requirements:
    1. Use LibreTranslate API (free, no key)
       https://libretranslate.com/
    2. Support source and target language parameters
    3. Handle API errors gracefully
    """
    
    name = "translate_text"
    description = "Translate text from one language to another"
    parameters_schema = {
        # Define your schema here
    }
    
    async def execute(self, ...) -> ToolResult:
        # Your implementation
        pass
```

### Exercise 2: Implement Tool Chaining

Write a function that automatically chains tools:

```python
async def get_weather_for_city(city_name: str) -> dict:
    """
    Chain geocode → weather tools automatically.
    
    Steps:
    1. Call geocode_location with city_name
    2. Extract latitude, longitude from result
    3. Call get_weather_forecast with coordinates
    4. Return weather data
    """
    pass
```

### Exercise 3: Add Tool Call Tracing to MongoDB

Extend the agent to store tool traces in MongoDB:

```python
async def save_tool_trace(task_id: str, trace: ToolCallTrace):
    """
    Save tool call trace to MongoDB for observability.
    
    Document structure:
    {
        "task_id": "task-123",
        "call_id": "call-abc",
        "tool_name": "geocode_location",
        "arguments": {...},
        "result": {...},
        "latency_ms": 150,
        "timestamp": ISODate("...")
    }
    """
    pass
```

### Exercise 4: Implement Parallel Tool Execution

When multiple tools don't depend on each other, execute them in parallel:

```python
async def execute_tools_parallel(
    tool_calls: list[ToolCall]
) -> list[ToolResult]:
    """
    Execute multiple tool calls in parallel.
    
    Use asyncio.gather() to run independent tools concurrently.
    """
    pass
```

---

## Summary

In this module, you learned:

1. **Tool Calling Fundamentals**: How LLMs use tools to access real-world data
2. **ReAct Pattern**: Interleaving reasoning and acting in an agent loop
3. **Tool Abstraction**: Building reusable, testable tool components
4. **OpenAI Integration**: Converting tools to function calling format
5. **Real APIs**: Using free APIs (Open-Meteo, Nominatim, Frankfurter, Nager.Date)
6. **Security**: Input validation, API key management, rate limiting
7. **Testing**: Unit tests, integration tests, mocking strategies
8. **Production**: Observability, error recovery, caching

### Tool Summary

| Tool | API | Key? | Use Case |
|------|-----|------|----------|
| `geocode_location` | OpenStreetMap Nominatim | No | City → Coordinates |
| `get_weather_forecast` | Open-Meteo | No | Weather for dates |
| `convert_currency` | Frankfurter | No | Budget planning |
| `convert_timezone` | Python zoneinfo | Local | Flight planning |
| `calculate_distance` | Haversine | Local | Travel time estimates |
| `get_public_holidays` | Nager.Date | No | Plan around closures |

### Next Steps

In **Module 9 (RAG + Hybrid Search)**, you will:
- Combine tool calling with retrieval-augmented generation
- Implement vector search with ChromaDB
- Add BM25 keyword search with Elasticsearch
- Use RRF (Reciprocal Rank Fusion) for hybrid results


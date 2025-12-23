# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Development server (HTTP/1.1 with hot reload)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Production server (HTTP/2 with SSL)
python run.py
```

## Architecture Overview

FastAPI-based LLM REST API server with SSE streaming, multi-model support, and MCP agent capabilities.

### Core Components

```
main.py                    # Entry point - lifespan mgmt, router registration
├── app/core/
│   ├── config.py          # Pydantic settings (DB, app config)
│   ├── database.py        # Async SQLAlchemy setup
│   └── request_logger.py  # Request/response logging middleware
├── app/services/
│   ├── llm_service.py     # Multi-model router (singleton)
│   ├── bllossom_service.py # Bllossom/llama-3.2-3B model wrapper
│   ├── kanana_service.py  # Kakao Kanana-2-30B model wrapper
│   └── agent_service.py   # MCP-based tool calling agent loop
├── app/mcp/
│   ├── server.py          # MCP server (web_search, database_query, calculator)
│   └── client.py          # In-process MCP client for tool invocation
└── app/routers/
    ├── chat.py            # Chat sessions & SSE streaming (primary router)
    ├── agent.py           # Agent endpoints with tool execution
    ├── sse.py             # Schedule extraction & text summarization
    ├── monitor.py         # Live request monitor UI
    ├── terminal.py        # Web terminal UI
    └── mysql_viewer.py    # Database viewer UI
```

### Key Patterns

- **All SSE streaming endpoints use GET method** with query parameters (not POST)
- Singleton pattern for LLM service instances
- Async generators for SSE streaming responses
- 4-bit model quantization via BitsAndBytes for memory efficiency

### SSE Event Types
- `start` - Begin streaming
- `message` - Token generation (intermediate)
- `complete` - End with stats
- `error` - Error occurred

### Supported Models
- `Bllossom/llama-3.2-Korean-Bllossom-3B` - 3B params, lightweight, fast
- `kakaocorp/kanana-2-30b-a3b-instruct` - 30B params, high quality

### LLM Parameters
- `temperature`: 0.1-2.0 (default 0.7, <0.3 uses greedy decoding)
- `max_tokens`: 1-4096 (default 1024)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/sse/chat/sessions` | Create chat session |
| GET | `/sse/chat/sessions` | List sessions |
| GET | `/sse/chat/send` | **Stream LLM response (SSE)** |
| GET | `/sse/agent/ask` | Agent chat with MCP tool calling |
| GET | `/sse/extract-schedule` | Extract schedules from text |
| GET | `/sse/summarize` | Summarize text |
| GET | `/monitor` | Live request monitor UI |
| GET | `/mysql` | Database viewer UI |
| GET | `/docs` | Swagger UI |

## Database Schema

MySQL with async SQLAlchemy. Key tables:
- `chat_sessions` - Conversation metadata (model, system_prompt, temperature)
- `chat_messages` - Messages with role (user/assistant/system) and token counts
- `tool_logs` - MCP tool execution history

## MCP Tool Calling

Agent service iterates up to 5 times, calling tools via MCP client:
1. LLM generates tool call request
2. MCP client executes tool
3. Tool result fed back to LLM
4. Loop until LLM provides final answer

## Router Description Guidelines (for MCP)

Write detailed docstrings and Query parameter descriptions so LLM can auto-invoke APIs via tool_calling:

```python
@router.get("/endpoint")
async def endpoint_name(
    message: str = Query(
        ...,
        description="LLM에게 전달할 사용자 메시지. 질문, 요청, 대화 내용 등 자유 형식의 텍스트를 입력합니다."
    ),
):
    """
    [One-line summary]

    [When to use this API]
    [Expected results]

    Args:
        param: Purpose and allowed values/format

    Returns:
        Response structure and field meanings
    """
```

Required description elements:
- **What**: What the API does
- **When**: When to use it
- **How**: How to configure parameters
- **Result**: What to expect
- **Error**: Possible errors

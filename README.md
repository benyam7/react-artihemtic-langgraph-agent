# Arithmetic Agent API

## What it does

- Exposes a FastAPI endpoint (`POST /invoke`) that runs a LangGraph + DeepSeek arithmetic agent.
- Supports tool-based arithmetic: add, subtract, multiply, divide. With error handling. (example, division by zero)
- Keeps in-memory thread state via LangGraph `MemorySaver`.
- Sends monitoring data and traces to Langfuse.
- Enforces per-client cooldown between `/invoke` calls (default: 10 seconds).

## Built with

- Python
- FastAPI
- LangGraph
- LangChain + `langchain-deepseek`
- Langfuse

## Run locally

1. Install dependencies after creating a virtual environment:
    ```bash
    pip install -r requirements.txt
    ```
2. Set environment variables in .env see .env.example
3. Start server with `fastapi dev`

## Sample request

```bash
curl -X POST https://arthmetic-agent.fastapicloud.dev/invoke \
	-H "Content-Type: application/json" \
	-d '{"message":"divide 4 by 0 and 7 multiplied by 2"}'
```

## Sample response (200)

```json
{
    "reply": "Division by zero is undefined. 7 multiplied by 2 is 14."
}
```

## Sample cooldown response (429)

```json
{
    "detail": "Cooldown active. Try again in 10 second(s)."
}
```

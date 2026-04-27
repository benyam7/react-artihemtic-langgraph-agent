from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import os
from threading import Lock
from time import monotonic
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=False)

if not os.getenv("DEEPSEEK_API_KEY"):
    raise RuntimeError("DEEPSEEK_API_KEY must be set in the environment or .env file.")


from arithmetic_agent import react_graph_memory
from arithmetic_agent import config

class InvokeRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User input for the arithmetic agent")


class InvokeResponse(BaseModel):
    reply: str


app = FastAPI(title="Arithmetic LangGraph Agent", version="0.1.0")

COOLDOWN_SECONDS = float(os.getenv("INVOKE_COOLDOWN_SECONDS", "10"))
_last_invoke_by_client: dict[str, float] = {}
_cooldown_lock = Lock()


def _client_identifier(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown-client"




@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/invoke", response_model=InvokeResponse)
def invoke_agent(payload: InvokeRequest, request: Request) -> InvokeResponse:
    try:
        client_id = _client_identifier(request)
        now = monotonic()

        with _cooldown_lock:
            last_invoke = _last_invoke_by_client.get(client_id)
            if last_invoke is not None:
                elapsed = now - last_invoke
                if elapsed < COOLDOWN_SECONDS:
                    retry_after_seconds = max(1, int(COOLDOWN_SECONDS - elapsed + 0.999))
                    raise HTTPException(
                        status_code=429,
                        detail=f"Cooldown active. Try again in {retry_after_seconds} second(s).",
                        headers={"Retry-After": str(retry_after_seconds)},
                    )
            _last_invoke_by_client[client_id] = now

        result = react_graph_memory.invoke({"messages": [("user", payload.message)]}, config=config)
        messages = result.get("messages", [])
        if not messages:
            raise HTTPException(status_code=500, detail="Agent returned no messages")

        last_message = messages[-1]
        content = getattr(last_message, "content", "")

        if isinstance(content, str):
            reply = content
        elif isinstance(content, list):
            # Some providers return structured content blocks.
            reply = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ).strip()
        else:
            reply = str(content)

        if not reply:
            reply = "Agent returned an empty response."
        print("Agent reply:", reply)
        return InvokeResponse(reply=reply)
    except HTTPException:
        raise
    except Exception as exc:
        print("Error during agent invocation:", exc)
        raise HTTPException(status_code=500, detail=f"Agent invocation failed: {exc}") from exc

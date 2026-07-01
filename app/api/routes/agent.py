from fastapi import APIRouter, Depends, HTTPException
from app.core.security import require_api_key
from app.models.requests import AgentRequest, AgentResponse
from app.services.agent.react_loop import run_react_agent

router = APIRouter()


@router.post("/agent/run", response_model=AgentResponse, tags=["Agent"])
async def run_agent(
    request: AgentRequest,
    _: str = Depends(require_api_key),
):
    """
    ReAct agent: multi-step reasoning over the knowledge base.
    The agent autonomously decides which tools to call and in what order.
    Returns the final answer + a full trace of its reasoning steps.
    """
    try:
        result = await run_react_agent(
            task=request.task,
            max_steps=request.max_steps,
        )
        return AgentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

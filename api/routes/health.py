from fastapi import APIRouter

from api import __version__
from api.dependencies import ChatServiceDep
from api.schemas import ReadinessResponse
from api.services.resilience import CircuitState

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def live() -> dict[str, str]:
    """Liveness: is the process alive? Never check dependencies here, or an
    LLM outage would make the orchestrator restart healthy pods in a loop."""
    return {"status": "ok"}


@router.get("/ready", response_model=ReadinessResponse)
async def ready(service: ChatServiceDep) -> ReadinessResponse:
    """Readiness reports 'degraded' but stays 200 when the LLM circuit is open:
    the LLM is a SHARED dependency, so failing readiness would pull every pod
    out of the load balancer at once and turn a partial outage into a total one."""
    circuit = service.breaker.state
    return ReadinessResponse(
        status="degraded" if circuit is CircuitState.OPEN else "ok",
        version=__version__,
        llm_provider=service.llm.name,
        circuit=circuit.value,
        llm_slots_in_use=service.limiter.in_use,
    )

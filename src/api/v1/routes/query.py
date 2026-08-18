from fastapi import APIRouter, HTTPException
from src.api.v1.schemas.query_schema import QueryRequest, QueryResponse
from src.api.v1.services.query_service import query_documents, query_documents_stream
from fastapi.responses import StreamingResponse
from src.core.guardrails import GuardrailViolation

router = APIRouter(prefix="/api/v1/query")


@router.post("/")
def query_endpoint(request: QueryRequest) -> QueryResponse:
    try:
        docs = query_documents(request.query, "user_id")
    except GuardrailViolation as violation:
        raise HTTPException(
            status_code=400,
            detail={"guardrail": violation.guard, "message": violation.message},
        )
    return docs


@router.post("/stream")
async def stream_query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    endpoint that return an SSE steam of the agent's response
    """
    generator = await query_documents_stream(request.query)
    return StreamingResponse(generator, media_type="text/event-stream")

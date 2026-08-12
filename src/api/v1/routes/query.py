from fastapi import APIRouter
from src.api.v1.schemas.query_schema import QueryRequest,QueryResponse
from src.api.v1.services.query_service import query_documents,query_documents_stream
from fastapi.responses import StreamingResponse



router = APIRouter(prefix="/api/v1/query")




@router.post("/")
def query_endpoint(request: QueryRequest):
   docs = query_documents(request.query)
   return docs

@router.post("/stream")
async def stream_query_endpoint(request: QueryRequest) -> QueryResponse:
   """
   endpoint that return an SSE steam of the agent's response
   """
   generator = await query_documents_stream(request.query)
   return StreamingResponse(generator, media_type="text/event-stream")
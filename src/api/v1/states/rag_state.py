from typing import TypedDict, List
from langchain_core.documents import Document

# state for the agent
class RAGState(TypedDict):
   query: str
   retrieved_docs: List[Document]
   reranked_docs: List[Document]
   response: dict
   generated_sql: str
   attempts: int
   query_feedback: str

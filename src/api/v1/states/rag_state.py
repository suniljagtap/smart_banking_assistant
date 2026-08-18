from typing import TypedDict, List
from langchain_core.documents import Document


# state for the agent
class RAGState(TypedDict):
    query: str
    chat_history: List[dict]
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    db_retrieved_docs: List[Document]
    response: dict
    generated_sql: str
    attempts: int
    retry_count: int
    query_feedback: str
    route: str
    top_score: float

from src.api.v1.states.rag_state import RAGState
from src.core.db import get_vector_store, get_db_connection_fts, get_db_connection
import os
import psycopg
import psycopg2
from openai import OpenAI
from typing import List, Dict, Any, Optional
from psycopg.rows import dict_row
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from langchain_core.tools import tool
from sentence_transformers import CrossEncoder
from langchain_openai import OpenAIEmbeddings

load_dotenv()

#  select id,content,page_number,ts_rank_cd(fts_vector,websearch_to_tsquery('english',%s))
#         as score
#         from multimodal_chunks
#         where fts_vector @@ websearch_to_tsquery('english',%s)
#         order by score DESC
#         LIMIT %s

_API_KEY = os.getenv("OPENAI_API_KEY")

_raw_conn = os.getenv("PG_CONNECTION_STRING_FTS")
_EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

_embeddings = OpenAIEmbeddings(
    model=_EMBED_MODEL,
    api_key=_API_KEY,
    dimensions=1536,  # default is 1536, when you not set this
)


def get_embedding(texts: list[str]) -> list[list[float]]:
    """Embed a batch of text strings with OpenAI text-embedding-3-small.

    OpenAIEmbeddings handles request batching internally, so we pass the whole
    list and get back one 1536-dimensional vector per input string.
    """
    return _embeddings.embed_documents(texts)


@tool
def fts_search_tool(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    use this tool for exact keyword matching, some technical term or document IDs,
    policy codes (e.g. POL-XX), RBI guidelines, acronyms (e.g. CIBIL, DTI, LTV),
    or borrower reference numbers.
    """
    print("Executing Full-Text Search [BM25] for: " + query)
    sql = """
        select id,content,page_number,ts_rank_cd(fts_vector,websearch_to_tsquery('english',%s)) 
        as score
        from multimodal_chunks
        where fts_vector @@ websearch_to_tsquery('english',%s)
        order by score DESC
        LIMIT %s
            """
    with get_db_connection_fts() as conn:
        conn.rollback()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (str(query), str(query), int(k)))
            results = cur.fetchall()
            # print(f"DB RAW RESLTS COUNT: {len(results)}")
            # print(f"DB SAMPLE OUTPUT: {results[:1]}")
            return results


@tool
def vector_search_tool(query: str, k: int = 5):
    """
    use this tool for conceptual or semantic questions
    without specific codes, IDs, or exact keywords.
    """
    print("Executing Vector Search for: " + query)
    query_vector = get_embedding(query)
    if isinstance(query_vector[0], list):
        query_vector = query_vector[0]

    query_vector_str = str(query_vector)

    sql = """ 
            Select id, content,page_number,(1- (embedding <=> %s::vector)) as score
            from multimodal_chunks
            order by embedding <=> %s::vector asc
            LIMIT %s"""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (str(query_vector_str), str(query_vector_str), int(k)))
            results = cur.fetchall()
            # print(f"DB RAW RESLTS COUNT: {len(results)}")
            # print(f"DB SAMPLE OUTPUT: {results[:1]}")
            return results


@tool
def hybrid_search_tool(query: str, k: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
    """
    use this tool for hybrid questions which has conceptual or semantic questions
    with specific codes, IDs, or exact keywords.
    """
    query_vector = get_embedding(query)

    print("Executing Hybrid Search for: " + query)

    if isinstance(query_vector[0], list):
        query_vector = query_vector[0]

    query_vector_str = str(query_vector)

    sql = """
        with vector_ranks as (
        select id, content, page_number,ROW_NUMBER() over (order by embedding <=> %s::vector ASC) as rank
                    from multimodal_chunks
                    LIMIT %s
        ),
        fts_ranks as (
        select id,content, ROW_NUMBER() over 
        (order by ts_rank_cd(fts_vector,websearch_to_tsquery('english',%s))DESC) as rank
        from multimodal_chunks
        where fts_vector @@ websearch_to_tsquery('english',%s)
        LIMIT %s
        )
        select coalesce(v.id,f.id) as id,
        coalesce(v.content,f.content) as content,
        coalesce(1.0/ (60 + v.rank),0.0) + coalesce(1.0/ (60 + f.rank),0.0) as  hybrid_score
        from vector_ranks v
        full outer join
        fts_ranks f  on v.id = f.id
        order by hybrid_score desc
        limit %s;
        """
    fetch_limit = k * 3

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                sql, (query_vector_str, fetch_limit, query, query, fetch_limit, k)
            )

            results = cur.fetchall()
            # print(f"DB RAW RESLTS COUNT: {len(results)}")
            # print(f"DB SAMPLE OUTPUT: {results[:1]}")
            return results

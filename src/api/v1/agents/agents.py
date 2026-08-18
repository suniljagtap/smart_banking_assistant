import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Literal
from src.api.v1.states.rag_state import RAGState
from src.api.v1.schemas.query_schema import AIResponse
from src.core.db import get_sql_database
from src.api.v1.tools.tools import (
    vector_search_tool,
    fts_search_tool,
    hybrid_search_tool,
)
import cohere

load_dotenv()

RELEVANCE_THRESHOLD = 0.35
MAX_RETRIES = 2


def _get_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL"), api_key=os.getenv("OPENAI_API_KEY")
    )


class RouteDecision(BaseModel):
    route: Literal["CHIT_CHAT", "VECTOR_DB", "RDBMS", "HYBRID"]
    reason: str  # for debugging


class SearchDecision(BaseModel):
    tools_name: Literal["fts_search_tool", "vector_search_tool", "hybrid_search_tool"]
    reason: str  # for debugging


def rag_retriever(state: RAGState) -> RAGState:
    llm = _get_llm()

    tools = [fts_search_tool, vector_search_tool, hybrid_search_tool]
    tools_by_name = {tool.name: tool for tool in tools}

    llm_with_tools = llm.bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                      You are an expert Smart banking Agent for assiting Relationship Managers, 
                      Customer Service Officers, and Compliance Staff.

                      Your main task is to evaluate the user's input query and select the SINGLE most effective
                      search tool to retrieve relevant policy documents and contextual data

                      ### AVAILABLE SEARCH TOOLS

            1 **fts_search_tool: 
            --**use case: ** Exact keyword lookups, specific alphanumeric codes, structured identifiers,
             banking abbreviations and proper nouns
            --**Triggers:**financial acronyms (RBI,NSBR,LTV,PAN,ITR etc), document section/clause numbers.
            2 **vector_search_tool: 
            --**use case: ** Use for broad conceptual queries or open-ended policy explanations, abstract
            guidance or intent based searches
            --**Triggers:** "How do I..", " Explan the procedure for..", generic risk guidelines
            3 **hybrid_search_tool: 
            --** use case:** complete queries containing both strict metadata constraints and conceptual
            explanations.
            --**Triggers:** queries combining numeric thresholds with policy terms or multip part 
            compliance checks

        -The response should be solely based on the avaialble embeddings data and the raw chunks in the DB and no additional 
        details or suggestion should be provided.
        - Select only one tool per invocation.
        Mandatory citation:
        - Every rule check, threshold evaluation, or decision should include an inline citation.
        - If a retrieved chunk is missing a page or clause number, use the document source name.
                   """,
            ),
            (
                "human",
                """
                   Question:
                   {query}
                """,
            ),
        ]
    )

    chain = prompt | llm_with_tools
    decision = chain.invoke({"query": state["query"]})

    retrieved_docs = []
    if decision.tool_calls:
        for tool_call in decision.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name in tools_by_name:
                selected_tool = tools_by_name[tool_name]
                retrieved_docs = selected_tool.invoke(tool_args)

    return {**state, "retrieved_docs": retrieved_docs}


def query_classifier(state: RAGState) -> RAGState:
    llm = _get_llm()
    structured_llm = llm.with_structured_output(RouteDecision)

    # 'CHIT_CHAT' - the query uses conversational small talk, greetings, pleasentries or casual
    #                      questions with no data retrieval needs.

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                      You are a high precision query router for an Agentic RAG System.
                      Your sole responsibility is  to anlayze incoming user queries and route them 
                      to appropriate underlying system components.

                      'RBDMS' - the query asks about accounts, transactions,loans, fixed deposits, credit card,
                       or anything answerable from a structrured banking database tables:
                      accounts, transactions, loan_accounts, fixed_deposits,credit_cards,card_transactions

                      'VECTOR_DB' -  the query asks about product terms, interest rates, eligibility criteria, 
                      charges,regulatory disclosures for all retail banking products offered by NorthStar Bank
                        or any topic that requires reading text documents
                      'CHIT_CHAT' - the query uses conversational small talk, greetings, pleasentries or casual
                        questions with no data retrieval needs.
                    
                      'HYBRID' - Complex queries that combine elements of two or more of the above
                      categories (e.g., combining small talk with a data query or combining small talk 
                      with semantic search or combining data query with semantic search)

                      Reply with the route and one sentence of reason.
                   """,
            ),
            (
                "human",
                """
                   Question:
                   {query}
                """,
            ),
        ]
    )

    chain = prompt | structured_llm
    decision = chain.invoke({"query": state["query"]})
    print(f"[router_node's decision]: {decision.route} and reason: {decision.reason}")
    retry_count = 0
    return {**state, "route": decision.route, "retry_count": retry_count}


def sql_generator_node(state: RAGState) -> RAGState:
    print("About to generate nl2sql")

    feedback = state.get("feedback", "")

    prompt = f"{state["query"]}\n"

    if feedback:
        prompt += f"Previous attempt failed with error/feedback: {feedback}. Please fix the query."

    # connect to LLM
    llm = _get_llm()
    # connect to rdbms
    db = get_sql_database()
    # get the tables' live schema
    schema_info = db.get_table_info()
    # write the system prompt and pass on the schema to get only sql query
    sql_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                   You are a PostgreSQL expert. Given the database schema below,
                   write a single valid SELECT query that answers the user's question.


                   Rules:
                   - Return ONLY the raw SQL — no explanation, no summary, no markdown fences, no backticks.
                   - Use only the tables and columns present in the schema.
                   - Do NOT generate INSERT, UPDATE, DELETE, DROP, or any DML/DDL statements.
                   - Always add a LIMIT clause (max 50 rows) unless the question asks for aggregates.
                   - For product or text searches: NEVER search for the full multi-word phrase as one
                       ILIKE pattern. Instead, split the search into individual meaningful keywords
                       and OR them together across both name and description columns.
                       Example — user asks "wireless headset":
                           WHERE (name ILIKE '%wireless%' OR description ILIKE '%wireless%')
                           OR (name ILIKE '%headset%'  OR description ILIKE '%headset%')
                           OR (name ILIKE '%headphones%' OR description ILIKE '%headphones%')
                       Use your knowledge of synonyms (headset/headphones, laptop/notebook, etc.)
                       to cast a wider net when the exact term may not match.
                  
                   Database schema:
                   {schema}
               """,
            ),
            (
                "human",
                """
                   Question:
                   {question}
               """,
            ),
        ]
    )
    # preprare the chain and invoke with a query
    sql_chain = sql_prompt | llm
    # look for sql query only
    raw_sql = sql_chain.invoke({"schema": schema_info, "question": state["query"]})
    print("========GENERATED raw_sql query is: =====")
    print(raw_sql.content)
    state["generated_sql"] = raw_sql.content
    state["feedback"] = feedback

    return {
        **state,
    }


def sql_executor_node(state: RAGState) -> RAGState:

    print("About to execute sql")

    current_retries = state.get("attempts", 0)
    generated_sql = state["generated_sql"]

    llm = _get_llm()
    db = get_sql_database()
    schema_info = db.get_table_info()

    # execute the generated sql query  to get the outout from RDMBS
    try:
        sql_result = db.run(generated_sql)
        is_error = None
        print(sql_result)
    except Exception as err:
        sql_result = f"Generated SQL execution error: {err}"
        is_error = f"{err}"

    return {
        **state,
        "db_retrieved_docs": sql_result,
        "attempts": current_retries + 1,
        "query_feedback": is_error,
    }


def route_after_execution(state: RAGState) -> str:
    if state.get("query_feedback") and state.get("attempts", 0) < 3:
        return "sql_generator"
    return "resposne_generator"


def evaluate_quality(state: RAGState) -> str:
    ranked = state.get("reranked_docs", [])
    top_score = state.get("top_score", 0.0)
    retry_count = state.get("retry_count", 0)

    print(f"[Router Check] Retry count:{retry_count} | Top Score : {top_score:.4f}")

    if retry_count >= 2:
        print("Maximum retries reached")
        return "continue"

    if ranked and top_score >= RELEVANCE_THRESHOLD:
        print(f"Good Quality achieved. Moving to generation")
        return "continue"

    return "rewrite"


def reranker(state: RAGState):
    # establish connection with the cohere reranking model
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    # send the query and the retrieved_docs to the reranking model

    docs = state["retrieved_docs"]

    print("=======3. INSIDE rerank_node. Before calling reranker =========")
    if not docs:
        print("[rerank_node] No documents returned from search. Skipping cohere")
        return {**state, "reranked_docs": docs, "top_score": 0.0}

    document_texts = [doc["content"] for doc in docs if doc.get("content")]

    if not document_texts:
        print(" Document content string were empty")
        return {**state, "reranked_docs": docs, "top_score": 0.0}

    print("=======3. INSIDE rerank_node. calling cohere =========")

    rerank_response = co.rerank(
        model="rerank-v3.5",
        query=state["query"],
        documents=[doc["content"] for doc in docs],
        top_n=5,
    )
    # Map Cohere result indices back to LangChain Document objects
    reranked_docs = [docs[r.index] for r in rerank_response.results]
    print(f"[rerank_node] Top {len(reranked_docs)} chunks after reranking:")
    top_score = (
        rerank_response.results[0].relevance_score if rerank_response.results else 0.0
    )
    for i, r in enumerate(rerank_response.results):
        print(
            f"  Rank {i+1} | Cohere score: {r.relevance_score:.4f} | original index: {r.index}"
        )

    return {**state, "reranked_docs": reranked_docs, "top_score": top_score}


def query_rewriter(state: RAGState):
    current_query = state["query"]
    current_retries = state.get("retry_count", 0)

    if current_retries >= 2:
        print(f"Hard limit reached inside rewriter.Halting loop")
        return state

    llm = _get_llm()

    rewriting_prompt = f"""
the following search yielded no results: '{current_query}
rephrase or broaden this query to improve search retrieval across banking documents
return only the new query string without qoutes or explanation.
Do not list excessive synonyms or repeat words
"""
    new_query = llm.invoke(rewriting_prompt).content.strip()
    new_count = current_retries + 1
    print(f"Retrying search ({current_retries+1}/2). New query: '{new_query}' ")

    return {"query": new_query, "retry_count": new_count}


def response_generator_node(state: RAGState) -> RAGState:

    # connect to LLM to get the natural language response
    query = state.get("query")
    sql = state.get("generated_sql")
    sql_result = state.get("db_retrieved_docs")
    search_result = state.get("reranked_docs")

    llm = _get_llm()
    structured_llm = llm.with_structured_output(AIResponse)
    nl_answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful data analyst. Answer the user's question using
               the SQL query results below. Be concise and format numbers/lists clearly.
               Set policy_citations to empty string,
               page_no to 'N/A', and document_name to 'smart_banking_assistant_DB'.""",
            ),
            (
                "human",
                "Question: {query}\n\n"
                "SQL Used:\n{sql}\n\n"
                "sql_result:\n{sql_result}"
                "search_result:\n{search_result}",
            ),
        ]
    )

    nl_chain = nl_answer_prompt | structured_llm
    answer = nl_chain.invoke(
        {
            "query": state["query"],
            "sql": query,
            "sql_result": sql_result,
            "search_result": search_result,
        }
    )
    print("[nl2sql_node] Answer generated.")
    response = answer.model_dump()
    response["policy_citations"] = "N/A"
    response["sql_query_executed"] = query
    response["doc_query_executed"] = search_result
    print(response)
    # return the sql query is RAGState
    # and also the output in sql_result of RAGState
    return {
        **state,
        "generated_sql": query,
        "sql_result": str(query),
        "response": response,
    }


def build_rag_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("router", query_classifier)
    workflow.add_node("sql_generator", sql_generator_node)
    workflow.add_node("sql_execution", sql_executor_node)
    workflow.add_node("response_generator", response_generator_node)
    workflow.add_node("rag_retriever", rag_retriever)
    workflow.add_node("reranker", reranker)
    workflow.add_node("rewrite", query_rewriter)

    # the following is the starting point
    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "VECTOR_DB": "rag_retriever",
            "RDBMS": "sql_generator",
            "CHIT_CHAT": "response_generator",
        },
    )

    workflow.add_edge("rag_retriever", "reranker")

    workflow.add_edge("sql_generator", "sql_execution")

    workflow.add_edge("rewrite", "rag_retriever")

    workflow.add_conditional_edges(
        "sql_execution",
        route_after_execution,
        {"sql_generator": "sql_generator", "resposne_generator": "response_generator"},
    )

    workflow.add_conditional_edges(
        "reranker",
        evaluate_quality,
        {"rewrite": "rewrite", "continue": "response_generator"},
    )

    # workflow.add_edge("reranker","resposne_generator")

    workflow.add_edge("response_generator", END)

    search_agent = workflow.compile()

    # generating and saving the graph visualization
    graph_image = search_agent.get_graph().draw_mermaid_png()
    with open("search_agent.png", "wb") as f:
        f.write(graph_image)

    return search_agent


rag_graph = build_rag_graph()


def run_search_agent(query: str, user_id: str = None):
    print("============1. INSIDE run_search_agent ")
    initial_state = {
        "query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
    }

    final_state = rag_graph.invoke(initial_state)
    return final_state["response"]


async def run_search_agent_stream(query: str):
    print("============1. INSIDE run_search_agent ")
    initial_state = {
        "query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
    }

    async for event in rag_graph.astream_events(initial_state, version="v1"):
        kind = event["event"]
        print(kind)

        # if it is a token generated by the chat model
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # format as an Server Side Event data straem payload
                yield f"data: {json.dumps({'token': content})}\n\n"

    yield "data: [DONE]\n\n"

# nodes we want
# 1. vector_search (top-k=20)
# 2. rerank
# 3. generate_answer


import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Literal
from src.api.v1.states.rag_state import RAGState
from src.api.v1.schemas.query_schema import AIResponse
from src.core.db import get_sql_database


load_dotenv()




def _get_llm():
   return ChatOpenAI(
       model=os.getenv("OPENAI_CHAT_MODEL"), api_key=os.getenv("OPENAI_API_KEY")
   )




class RouteDecision(BaseModel):
   route: Literal["VECTOR_DB", "RDBMS"]
   reason: str  # for debugging




def router_node(state: RAGState) -> RAGState:
   llm = _get_llm()
   structured_llm = llm.with_structured_output(RouteDecision)


   prompt = ChatPromptTemplate.from_messages(
       [
           (
               "system",
               """
                      You are a query router for an Agentic RAG System.
                      'RBDMS - the query asks about accounts, transactions,loans, fixed deposits, credit card,
                       or anything answerable from a structrured banking database tables:
                      accounts, transactions, loan_accounts, fixed_deposits,credit_cards,card_transactions

                      Reply with the RDBMS route and one sentence of reason.
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


   return {**state, "route": decision.route}




def nl2sql_node(state: RAGState) -> RAGState:
   print("About to generate nl2sql")
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
   generated_sql = raw_sql.content


   # execute the generated sql query  to get the outout from RDMBS
   try:
       sql_result = db.run(generated_sql)
   except Exception as err:
       sql_result = f"Generated SQL execution error: {err}"


   # connect to LLM to get the natural language response
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
               "Query Results:\n{result}",
           ),
       ]
   )


   nl_chain = nl_answer_prompt | structured_llm
   answer = nl_chain.invoke(
       {"query": state["query"], "sql": generated_sql, "result": sql_result}
   )
   print("[nl2sql_node] Answer generated.")
   response = answer.model_dump()
   response["policy_citations"] = "N/A"
   response["sql_query_executed"] = generated_sql
   # return the sql query is RAGState
   # and also the output in sql_result of RAGState
   return {
       **state,
       "generated_sql": generated_sql,
       "sql_result": str(sql_result),
       "response": response,
   }


def build_rag_graph():
   workflow = StateGraph(RAGState)


   workflow.add_node("router", router_node)
   workflow.add_node("nl2sql", nl2sql_node)
  
   # the following is the starting point
   workflow.set_entry_point("router")



   workflow.add_edge("router","nl2sql")


   workflow.add_edge("nl2sql", END)


   search_agent = workflow.compile()


   # generating and saving the graph visualization
   graph_image = search_agent.get_graph().draw_mermaid_png()
   with open("search_agent.png", "wb") as f:
       f.write(graph_image)


   return search_agent




rag_graph = build_rag_graph()




def run_search_agent(query: str):
   print("============1. INSIDE run_search_agent ")
   initial_state = {
       "query": query,
       "retrieved_docs": [],
       "reranked_docs": [],
       "response": {},
   }


   final_state = rag_graph.invoke(initial_state)
   return final_state["response"]

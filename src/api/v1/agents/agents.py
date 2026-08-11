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

def sql_generator_node(state: RAGState) -> RAGState:
   print("About to generate nl2sql")


   feedback = state.get("feedback","")

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
    generated_sql= state["generated_sql"]

    llm = _get_llm()
    db = get_sql_database()
    schema_info = db.get_table_info()

   # execute the generated sql query  to get the outout from RDMBS
    try:
       sql_result = db.run(generated_sql)
       is_error = None
    except Exception as err:
       sql_result = f"Generated SQL execution error: {err}"
       is_error = f"{err}"
       
    return {
       **state,
       "response": sql_result,
       "attempts": current_retries + 1,
       "query_feedback": is_error,
       
    }
 
def route_after_execution(state: RAGState) -> str:
   if state.get("query_feedback") and state.get("attempts",0) < 3:
      return "sql_generator"
   return "resposne_generator"



def response_generator_node(state: RAGState) -> RAGState:

   # connect to LLM to get the natural language response
   query = state.get("query")
   sql = state.get("generated_sql")
   result = state.get("response")
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
               "Query Results:\n{result}",
           ),
       ]
   )


   nl_chain = nl_answer_prompt | structured_llm
   answer = nl_chain.invoke(
       {"query": state["query"], "sql": query, "result": result}
   )
   print("[nl2sql_node] Answer generated.")
   response = answer.model_dump()
   response["policy_citations"] = "N/A"
   response["sql_query_executed"] = query
   # return the sql query is RAGState
   # and also the output in sql_result of RAGState
   return {
       **state,
       "generated_sql": query,
       "sql_result": str(result),
       "response": response,
   }



def build_rag_graph():
   workflow = StateGraph(RAGState)


   workflow.add_node("router", router_node)
   workflow.add_node("sql_generator", sql_generator_node)
   workflow.add_node("sql_execution", sql_executor_node)
   workflow.add_node("resposne_generator", response_generator_node)

    
   # the following is the starting point
   workflow.set_entry_point("router")

   workflow.add_edge("router","sql_generator")

   workflow.add_edge("sql_generator","sql_execution")

   workflow.add_conditional_edges("sql_execution",
                                  route_after_execution,
                                  {
                                     "sql_generator" : "sql_generator",
                                     "resposne_generator" : "resposne_generator"
                                  })


   workflow.add_edge("resposne_generator", END)

  
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
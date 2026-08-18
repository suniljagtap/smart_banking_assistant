from src.core.guardrails import guard_input, guard_output
from src.api.v1.agents.agents import run_search_agent
from src.api.v1.agents.agents import run_search_agent_stream, run_search_agent
from src.api.v1.schemas.query_schema import QueryResponse


def query_documents_dprct(query: str):
    print(query)
    # returned_state = QueryResponse()
    returned_state = run_search_agent(query)
    print(returned_state)
    return {
        "answer": returned_state.get("answer", ""),
        "query_type": returned_state.get("query_type", ""),
        "citations": returned_state.get("policy_citations", []),
        # "images": returned_state.get("response_sources", []),
        # "confidence_score": returned_state.get("confidence_score", 0),
    }


def query_documents(query: str, user_id: str):
    print(query)
    guard_input(query)
    result = run_search_agent(query, user_id)

    if isinstance(result, dict) and result.get("response"):
        result["response"] = guard_output(result["response"])

    return result


# method for streaming response
async def query_documents_stream(query: str):
    # just return async generator
    return run_search_agent_stream(query)

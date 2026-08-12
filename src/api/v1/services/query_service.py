from src.api.v1.agents.agents import run_search_agent
from src.api.v1.agents.agents import run_search_agent_stream, run_search_agent




def query_documents(query: str):
   print(query)
   return run_search_agent(query)

# method for streaming response
async def query_documents_stream(query: str):
   # just return async generator
   return run_search_agent_stream(query)

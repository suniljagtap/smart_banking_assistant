import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase


load_dotenv()


model = os.getenv("OPENAI_EMBEDDING_MODEL")
api_key = os.getenv("OPENAI_API_KEY")
pg_vector_connection = os.getenv("PG_CONNECTION_STRING")
pg_rdbms_connection = os.getenv("PG_RDBMS_CONNECTION_STRING")




def get_sql_database() -> SQLDatabase:
   """
   uses read only credentials and connect to rdbms.
   and targets specific tables our agent can access
   """
   if not pg_rdbms_connection:
       raise ValueError("PG_RDBMS_CONNECTION_STRING is not set. Check your .env")
   else:
       return SQLDatabase.from_uri(
           pg_rdbms_connection,
           include_tables=["accounts", "transactions", "loan_accounts", "fixed_deposits","credit_cards","card_transactions"],
           # TODO: sample rows in table info
       )

import streamlit as st

st.title("SQL demo")


'''
This demonstration illustrates building an LLM-based agent that performs tasks by generating and executing code based on plain-text queries. In this example, we use a custom system prompt to instruct the LLM to generate SQL code which is then executed against the parquet data we generated in Module 3.  Note that SQL query itself is shown as well as the table produced by the query.  


'''

## dockerized streamlit app wants to read from os.getenv(), otherwise use st.secrets
import os
api_key = os.getenv("LITELLM_KEY")
if api_key is None:
    api_key = st.secrets["LITELLM_KEY"]
        

parquet = st.text_input("parquet file:", "https://espm-157-f24.github.io/spatial-carl-amanda-tyler/new_haven_stats.parquet")

# create sharable low-level connection, see: https://github.com/Mause/duckdb_engine
import sqlalchemy
eng = sqlalchemy.create_engine("duckdb:///:memory:")

# ibis can talk to this connection and create the VIEW
import ibis
from ibis import _
con = ibis.duckdb.from_connection(eng.raw_connection())
tbl = con.read_parquet(parquet, "mydata")

# langchain can also talk to this connection and see the table:
from langchain_community.utilities import SQLDatabase
db = SQLDatabase(eng, view_support=True)

#db.run(f"create or replace view mydata as select * from read_parquet('{parquet}');")
#print(db.get_usable_table_names()) 

# Build the template for system prompt
template = '''
You are a {dialect} expert. Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer to the input question.
Always return all columns from a query (select *) unless otherwise instructed.
Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. 
Be careful to not query for columns that do not exist. 
Also, pay attention to which column is in which table.
Pay attention to use today() function to get the current date, if the question involves "today".
Respond with only the SQL query to run.  Do not repeat the question or explanation. Just the raw SQL query.
Only use the following tables:
{table_info}
Question: {input}    
'''

with st.sidebar:
    model = st.selectbox("LLM:", ["gorilla", "llama3", "olmo"])


from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate.from_template(template, partial_variables = {"dialect": "duckdb", "top_k": 10})

# Now we are ready to create our model and start querying!
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gorilla", # Try: llama3, gorilla, or groq-tools, or other models
                 temperature=0, 
                 api_key=api_key, 
                 base_url = "https://llm.nrp-nautilus.io")


from langchain.chains import create_sql_query_chain
chain = create_sql_query_chain(llm, db, prompt)

prompt = st.chat_input("What is the mean ndvi by grade?")

if prompt:
    response = chain.invoke({"question": prompt})
    with st.chat_message("ai"):
        st.write(response)
        df = tbl.sql(response).head(10).execute()
        df





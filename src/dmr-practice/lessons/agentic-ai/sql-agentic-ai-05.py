import os
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits import create_sql_agent

HOME=os.environ["HOME"]
db_uri = "sqlite:///" + HOME + "/repo/playground-ai-ml/data/sql-agentic-ai.db"
db = SQLDatabase.from_uri(db_uri)

api_url = "http://localhost:12434/engines/v1"
api_key = "docker"
llm_model = "ai/gpt-oss:latest"

llm = ChatOpenAI(
    model=llm_model,
    temperature=0,
    base_url=api_url,
    api_key=api_key,
)

# from langchain_community.agent_toolkits.sql.prompt import SQL_FUNCTIONS_SUFFIX
# from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
# custom_prefix = """
#     You are a highly skilled and accurate SQL database agent.
#     Your primary goal is to answer user questions by interacting with the provided SQL database.
#     Always prioritize retrieving accurate data and formatting the output clearly.
#     If a question cannot be answered from the database, you can answer using your training data.
# """
# prompt = ChatPromptTemplate.from_messages(
#     [
#         SystemMessagePromptTemplate.from_template(custom_prefix),
#         HumanMessagePromptTemplate.from_template("{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#         SystemMessagePromptTemplate.from_template(SQL_FUNCTIONS_SUFFIX),
#     ]
# )

sql_agent = create_sql_agent(llm=llm, db=db, agent_type="tool-calling", verbose=False)
response = sql_agent.invoke({ "input": "How many departments are there?" })
print(f"response:\n{response}\n")

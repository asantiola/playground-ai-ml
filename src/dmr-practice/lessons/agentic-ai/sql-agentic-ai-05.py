import os
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits import create_sql_agent
from langchain.tools import Tool
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough

HOME=os.environ["HOME"]
db_uri = "sqlite:///" + HOME + "/repo/playground-ai-ml/data/sql-agentic-ai.db"
db = SQLDatabase.from_uri(db_uri)

api_url = "http://localhost:12434/engines/v1"
api_key = "docker"
llm_model = "ai/gpt-oss:latest"
embeddings_model = "ai/mxbai-embed-large"

llm = ChatOpenAI(
    model=llm_model,
    temperature=0,
    base_url=api_url,
    api_key=api_key,
)

oa_embeddings = OpenAIEmbeddings(
    model=embeddings_model,
    base_url=api_url,
    api_key=api_key,
    # disable check_embedding_ctx_length if your local model has different constraints
    check_embedding_ctx_length=False,
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
# sql_agent = create_sql_agent(llm=llm, db=db, agent_type="tool-calling", prompt=prompt, verbose=False)

sql_agent = create_sql_agent(llm=llm, db=db, agent_type="tool-calling", verbose=False)

# answer = sql_agent.invoke({ "input": "How many departments are there?" })
# print(f"answer: {answer}\n")

def sql_agent_wrapper(input: dict[str, any]):
    response = sql_agent.invoke(input)
    return response["output"]

sql_query_tool = Tool(
    name="sql_agent",
    description="This tool is an agent that queries the Albatross company database for a given input.",
    func=sql_agent_wrapper,
)

def get_retriever(oa_embeddings: OpenAIEmbeddings, store_path: str):
    vector_store = Chroma(
        embedding_function=oa_embeddings,
        persist_directory=store_path
    )
    return vector_store.as_retriever()

db_billiards = HOME + "/repo/playground-ai-ml/data/billiards.db"
db_guitars = HOME + "/repo/playground-ai-ml/data/guitars.db"

retriever_billiards = get_retriever(oa_embeddings, db_billiards)
retriever_guitars = get_retriever(oa_embeddings, db_guitars)

rag_system = "You are a helpful assistant that answers question. Use the documents and your training to answer the input question." 
rag_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(rag_system),
        HumanMessagePromptTemplate.from_template("Documents: {documents}. Input: {input}"),
    ]
)

def query_billiards_rag(input: str):
    rag_chain = (rag_prompt | llm)
    response = rag_chain.invoke({
        "input": input,
        "documents": retriever_billiards.invoke(input),
    })
    return response.content

billiards_query_tool = Tool(
    name="billiards_rag_llm_agent",
    description="This tool is an agent that does RAG llm query the database for a given input related to Lex's billiards.",
    func=query_billiards_rag,
)

def query_guitars_rag(input: str):
    rag_chain = (rag_prompt | llm)
    response = rag_chain.invoke({
        "input": input,
        "documents": retriever_guitars.invoke(input),
    })
    return response.content

guitars_query_tool = Tool(
    name="guitars_rag_llm_agent",
    description="This tool is an agent that does RAG llm query the database for a given input related to Lex's guitars.",
    func=query_guitars_rag,
)

expert_tools = [sql_query_tool, billiards_query_tool, guitars_query_tool]
expert_tools_map = {
    "sql_agent": sql_query_tool,
    "billiards_rag_llm_agent": billiards_query_tool,
    "guitars_rag_llm_agent": guitars_query_tool,
}

expert_system = """
You are an intelligent AI assistant that calls tools depending on the input.
If input is related to Lex's billiards, call the billiards RAG query tool.
If input is related to Lex's guitars, call the guitars RAG query tool.
If input is related to Albatross company and could be answered by the database information, call the SQL agent tool.
Otherwise, answer the question based on your training data.
"""

expert_human = """
Input: {input}
Database information: {db_info}
"""

expert_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(expert_system),
        HumanMessagePromptTemplate.from_template(expert_human),
    ]
)

expert_chain = (expert_prompt | llm.bind_tools(expert_tools))

def expert_agent(input: str):
    response = expert_chain.invoke({
        "input": input,
        "db_info": db.get_table_info(),
    })

    if len(response.content) > 0:
        return  response.content
    
    info = ""
    for tool_call in response.tool_calls:
        # print(f"tool_call:\n{tool_call}\n")
        if function_to_call := expert_tools_map.get(tool_call["name"]):
            tool_response = function_to_call.invoke(tool_call["args"])
            # print(f"tool_response:\n{tool_response}\n")
            info += " " + tool_response
            
    return info

question = "How many break cues does Lex have?"
answer = expert_agent(question)
print(f"question:\n{question}\nanswer:\n{answer}\n")

question = "Does Lex play the guitar?"
answer = expert_agent(question)
print(f"question:\n{question}\nanswer:\n{answer}\n")

question = "What is the largest bone in the human body?"
answer = expert_agent(question)
print(f"question:\n{question}\nanswer:\n{answer}\n")

question = "How many departments are there in Albatross company?"
answer = expert_agent(question)
print(f"question:\n{question}\nanswer:\n{answer}\n")


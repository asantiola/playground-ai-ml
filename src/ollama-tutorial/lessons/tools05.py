from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import os
import json


ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
ollama_url = f"http://{ollama_host}:11434"

ollama_config_file = "/workspace/data/ollama_conf.json"
with open(ollama_config_file) as file:
    ollama_config = json.load(file)

llm_model = ollama_config.get("llm_model", "llama3.1")
llm_temp = ollama_config.get("llm_temp", 0.0)

print(f"Using LLM model: {llm_model}")
print(f"Using LLM temp: {llm_temp}")
print("\n", "-------------------------\n")

ollama_embeddings_model = ollama_config.get("ollama_embeddings_model", "mxbai-embed-large")
print(f"Embedding model '{ollama_embeddings_model}'")
print("\n", "-------------------------\n")

oe_embeddings = OllamaEmbeddings(
    base_url=ollama_url,
    model=ollama_embeddings_model
)

# SKLearn persist path
persist_path = "/workspace/data/sklearn-oe"

vector_store = SKLearnVectorStore(
    persist_path=persist_path,
    embedding=oe_embeddings
)
retriever = vector_store.as_retriever(k=4)

class Veagles(BaseModel):
    name: str = Field(
        description="Topic about Veagles."
    )
    confidence_score: float = Field(
        description="Confidence score on the scale of 0 to 1 on your answer."
    )
    info: str = Field(
        description="Information about the Veagles."
    )

@tool
def get_veagles_info(query: str) -> Veagles:
    """
    Get more information on Veagles related query.

    Args:
        query (str): the query.
    """
    documents = retriever.invoke(query)
    prompt = PromptTemplate(
        template="Query: {query}. Documents={documents}. Provide a confidence score on the scale of 0 to 1 on your answer.",
        input_variables=["query", "documents"]
    )

    print(f"debug prompt: {prompt.template.format(query=query, documents=documents)}")
    
    llm = ChatOllama(
        base_url=ollama_url,
        model=llm_model,
        temperature=llm_temp,
    )
    
    chain = prompt | llm.with_structured_output(Veagles)
    result = chain.invoke({
        "query": query,
        "documents": documents,
    })

    return result

class Place(BaseModel):
    name: str = Field(
        description="Name of the city or country."
    )
    confidence_score: float = Field(
        description="Confidence score on the scale of 0 to 1 on your answer."
    )
    info: str = Field(
        description="Information about the city or country"
    )

@tool
def get_place_info(query: str) -> Place:
    """
    Get more information about a query on city or country

    Args:
        query (str): The query related to city or country.
    """
    prompt = PromptTemplate(
        template="Query: {query}. Provide a confidence score on the scale of 0 to 1 on your answer.",
        input_variables=["query"]
    )
    
    print(f"debug prompt: {prompt.template.format(query=query)}")

    llm = ChatOllama(
        base_url=ollama_url,
        model=llm_model,
        temperature=llm_temp,
    )
    
    chain = prompt | llm.with_structured_output(Place)
    result = chain.invoke({
        "query": query
    })

    return result

class Animal(BaseModel):
    name: str = Field(
        description="Name of the animal."
    )
    confidence_score: float = Field(
        description="Confidence score on the scale of 0 to 1 on your answer."
    )
    info: str = Field(
        description="Information about the animal"
    )

@tool
def get_animal_info(query: str) -> Animal:
    """
    Get more information about a query on an animal.

    Args:
        query (str): The query related to an animal.
    """
    prompt = PromptTemplate(
        template="Query: {query}. Provide a confidence score on the scale of 0 to 1 on your answer.",
        input_variables=["query"]
    )
    
    print(f"debug prompt: {prompt.template.format(query=query)}")

    llm = ChatOllama(
        base_url=ollama_url,
        model=llm_model,
        temperature=llm_temp,
    )
    
    chain = prompt | llm.with_structured_output(Animal)
    result = chain.invoke({
        "query": query
    })

    return result

def catch_all():
    """
    Fallback function for messages not about animals, Veagles, or places."
    """
    pass

available_functions = {
    "get_animal_info": get_animal_info,
    "get_place_info": get_place_info,
    "get_veagles_info": get_veagles_info,
}

inputs = [
    "Give me more information about aardvarks.",
    "Who is Lex from the Veagles?",
    "Tell me something about Manila.",
    "Give me yesterday's run logs.",
]

tools = [
    get_animal_info,
    get_veagles_info,
    get_place_info,
    catch_all,
]

for input in inputs:
    print(f"processing input for tool calling: {input}")
    
    prompt = PromptTemplate(
        template=input,
        input_variables=[]
    )

    llm = ChatOllama(
        base_url=ollama_url,
        model=llm_model,
        temperature=llm_temp,
    )

    chain = prompt | llm.bind_tools(tools)
    result = chain.invoke({})

    for tool in result.tool_calls:
        if function_to_call := available_functions.get(tool["name"]):
            print(f"calling {tool["name"]}")
            output = function_to_call.invoke(tool["args"])
            print(f"output: {output}")
        else:
            print(f"unhandled function {tool["name"]}")
        print("\n", "-------------------------\n")

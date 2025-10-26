from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
from pydantic import BaseModel, Field
from enum import StrEnum

api_url = "http://localhost:12434/engines/v1"
api_key = "docker"
llm_model = "ai/gpt-oss:latest"

llm = ChatOpenAI(
    model=llm_model,
    temperature=0,
    base_url=api_url,
    api_key=api_key,
)

class BuildTypes(StrEnum):
    EXE = "executable"
    LIB = "static library"
    DLL = "shared libary"

class CppCode(BaseModel):
    declarations: str = Field(
        description="The contents for C++ header file. This only contains class and functions declarations."
    )
    definitions: str = Field(
        description="The contents for the C++ file. This contains code class and functions definitions."
    )
    hpp_filename: str = Field(
        description="Appropriate filename for the C++ header file. Should have .hpp suffix."
    )
    cpp_filename: str = Field(
        description="Appropriate filename for the C++ cpp file. Should have .cpp suffix."
    )

class UnittestCode(BaseModel):
    filename: str = Field(
        description="Appropriate filename for the Googletest file. Should have .cpp suffix."
    )
    unittest: str = Field(
        description="The Googletest unit-test code."
    )

class State(TypedDict):
    description: str
    cpp_code: CppCode
    unittest_code: UnittestCode

def generate_code(state: State):
    structured_llm = llm.with_structured_output(CppCode)

    prompt = PromptTemplate.from_template(
        """
        You are a coding assistant expert in C++.
        Generate a C++ code for the provided description.
        Use C++14 as minimum standard.
        Add error checks.
        Add brief but meaningful comments to describe the classes and methods.
        Separate the declarations from defintions.
        This code will be compiled using clang++.
        Description: {description}
        """
    )

    chain = prompt | structured_llm
    response = chain.invoke({ "description": state["description"] })

    return { "cpp_code": response }

def generate_unittest(state: State):
    structured_llm = llm.with_structured_output(UnittestCode)

    prompt = PromptTemplate.from_template(
        """
        You are a coding assistant expert in Googletest and C++.
        Generate a Googletest unit-test codes for the provided files.
        The files are pairs of filenames and their contents.
        No need to put in quotes. Add brief but meaningful comments.
        This code will be compiled using clang++.
        No need to provide a main function.
        Files: {files}
        """
    )

    chain = prompt | structured_llm
    files = [
        (state["cpp_code"].hpp_filename, state["cpp_code"].declarations),
        (state["cpp_code"].cpp_filename, state["cpp_code"].definitions),
    ]
    unittest = chain.invoke({ "files": files })
    return { "unittest_code": unittest }

# # TODO
# def generate_cmakelist(state: State):
#     prompt = PromptTemplate.from_template(
#         """
#         You are a coding assistant expert in creating cmake build files for C++.
#         Use C++14 as minimum standard.
#         Create a CMakelists.txt for the provided files and build type.
#         The files are pairs of filenames and their contents.
#         Just provide the CMakelists.txt contents, no need for explanations or quotes.
#         Build type: {build}
#         Files: {files}
#         """
#     )

#     chain = prompt | llm | StrOutputParser()
#     files = [
#         (state["cpp_code"].hpp_filename, state["cpp_code"].declarations),
#         (state["cpp_code"].cpp_filename, state["cpp_code"].definitions),
#         (state["unittest_code"].filename, state["unittest_code"].unittest),
#     ]
#     cmakelists = chain.invoke({ "build": BuildTypes.DLL, "files": files })
    

graph_builder = StateGraph(State)
graph_builder.add_node("generate_code", generate_code)
graph_builder.add_node("generate_unittest", generate_unittest)
graph_builder.add_edge(START, "generate_code")
graph_builder.add_edge("generate_code", "generate_unittest")
graph_builder.add_edge("generate_unittest", END)

memory = MemorySaver()

config = {"configurable": {"thread_id": "my_conversation_1"}}
app = graph_builder.compile(checkpointer=memory)

description="Write a class that will do addition, subtraction, multipication and division of real numbers."
code = app.invoke({ "description": description }, config=config)

print(f"codes:\n{code["cpp_code"].hpp_filename}\n{code["cpp_code"].declarations}\n")
print(f"codes:\n{code["cpp_code"].cpp_filename}\n{code["cpp_code"].definitions}\n")
print(f"codes:\n{code["unittest_code"].filename}\n{code["unittest_code"].unittest}\n")

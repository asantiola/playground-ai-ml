from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(
    model="ai/gpt-oss",
    temperature=0,
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
)

def agent_cpp_coder(input, context):
    prompt = PromptTemplate(
        template="""You are a helpful assistant expert int writing C++14 code.
            Codes are compiled using g++.
            Generate the requested input. No need to put the response in quotes.
            You can add comments.
            You can use the context if it is provided.
            Input: {input}
            Context: {context}
            """,
        input_variables=["input", "context", ],
    )
    chain = (prompt | llm)
    response = chain.invoke({ "input": input, "context": context })
    return response.content

input_declarations = """Generate a C++ header file content.
    No need to quote your response. You can add comments.
    This header will contain the class declarations only for an Accumulator.
    The class will have a private integer variable to keep the accumulated sum.
    The class will have a a default constructor.
    The class will have a constructor that can initialize the member variable to the passed integer.
    Declare a default destructor, copy constructor and copy operator.
    Declare an add member function that accepts an integer.
    """

output_declarations = agent_cpp_coder(input=input_declarations, context="")
print(f"\ndeclarations: {output_declarations}\n")

input_definitions = """Generate a C++ file content for the provided header in the context.
    No need to create a main function.
    """

output_definitions = agent_cpp_coder(input=input_definitions, context=output_declarations)
print(f"\ndefinitions: {output_definitions}\n")

input_testing = "Generate a C++ unit test code for the provided declarations & definitions in the context."
context_testing=f"declaration: {output_declarations}\ndefintions: {output_definitions}\n"
output_testing = agent_cpp_coder(input=input_testing, context=context_testing)
print(f"\ntesting: {output_testing}\n")

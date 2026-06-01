from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
import os
import operator

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://model-runner.docker.internal/engines/v1"
)

api_key = os.environ.get(
    "OPENAI_API_KEY",
    "your-default-key"
)

llm = ChatOpenAI(
    model="ai/gemma4:E4B",
    base_url=openai_base_url,
    api_key=api_key,
)

class AgentState(TypedDict):
    question: str
    secret_word: str
    secret_words: Annotated[Sequence[str], operator.add]
    guesses: int

def choose_word_node(state: AgentState) -> AgentState:    
    system_prompt = """
    You are a helpful AI assistant keeping the secret word for 20 questions game.
    """

    secret_words = []
    if "secret_words" in state:
        secret_words = state["secret_words"]
    
    prompt = f"""
    Think of a random, specific noun that can be used for 20 questions game.
    The word should not be in the previous words: {secret_words}
    Just respond with the word.
    """
    
    response = llm.invoke(prompt)
    print(f"DEBUG: secret word is '{response.content}'")
    return {
        "secret_word": response.content,
        "secret_words": [response.content],
        "guesses": 5
    }

def route_have_guesses(state: AgentState) -> bool:
    print(f"You have {state["guesses"]} guesses left.")
    return state["guesses"] > 0

def ask_question_node(state: AgentState) -> AgentState:
    question = input("Question: ")
    return {
        "question": question,
        "guesses": state["guesses"] - 1,
    }

def route_answer(state: AgentState) -> str:
    system_prompt = """
    You are a helpful AI assistant keeping the secret word for 20 questions game.
    """

    human_prompt = f"""
    Analyze the user's question: '{state["question"]}' regarding the secret word '{state["secret_word"]}'.
    
    You must evaluate this in strict order of priority:
    1. FIRST, check if the user is guessing the secret word. 
       If the question explicitly names or identifies '{{state["secret_word"]}}' (ignoring capitalization or punctuation), 
       you MUST reply exactly with: 'Solved'.
    2. SECOND, if it is not a guess, check if the question can be answered with a Yes or No. Reply exactly with 'Yes' or 'No'.
    3. THIRD, if the question cannot be answered with a simple Yes or No, reply exactly with 'Invalid'.

    Do not include any other text, punctuation, or explanation. Only return one of the four words: Solved, Yes, No, or Invalid.
    """
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    answer = response.content
    if answer == "Yes" or answer == "No":
        print(f"Answer: {answer}")
    elif answer == "Invalid":
        print("That is not a valid question!")
    return answer

def winner_node(state: AgentState) -> AgentState:
    print(f"Congratulations! You have solved the secret word: '{state["secret_word"]}'.")
    return {}

def loser_node(state: AgentState) -> AgentState:
    print(f"You did not guess the secret word: '{state["secret_word"]}'. Better luck next time.")
    return {}

if __name__ == "__main__":
    graph = StateGraph(AgentState)
    
    graph.add_node("choose_word_node", choose_word_node)
    graph.add_node("check_have_guesses", lambda x: x)
    graph.add_node("ask_question_node", ask_question_node)
    graph.add_node("loser_node", loser_node)
    graph.add_node("winner_node", winner_node)
    
    graph.add_edge(START, "choose_word_node")
    graph.add_edge("choose_word_node", "check_have_guesses")
    graph.add_conditional_edges(
        "check_have_guesses",
        route_have_guesses,
        {
            True: "ask_question_node",
            False: "loser_node"
        }
    )
    graph.add_conditional_edges(
        "ask_question_node",
        route_answer,
        {
            "Solved": "winner_node",
            "Yes": "check_have_guesses",
            "No": "check_have_guesses",
            "Invalid": "check_have_guesses",
        }
    )
    graph.add_edge("loser_node", END)
    graph.add_edge("winner_node", END)

    app = graph.compile()

    drawing_filename = "/workspaces/playground-ai-ml/data/drawing.png"
    app.get_graph().draw_mermaid_png(output_file_path=drawing_filename)

    app.invoke({})

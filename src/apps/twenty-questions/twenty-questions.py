from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, Optional
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
    question: Optional[str] = None
    answer: Optional[str] = None
    secret_word: Optional[str] = None
    secret_words: Annotated[Sequence[str], operator.add]
    guesses: int = 0

def choose_word_node(state: AgentState) -> dict:    
    system_prompt = """
    You are a helpful AI assistant keeping the secret word for 20 questions game.
    """

    secret_words = state.get("secret_words", [])
    
    prompt = f"""
    Think of a random, specific noun that can be used for 20 questions game.
    The word should not be in the previous words: {secret_words}
    Just respond with the word.
    """
    
    response = llm.invoke(prompt)
    chosen_word = response.content.strip().lower()

    print(f"DEBUG: secret word is '{chosen_word}'")
    return {
        "secret_word": chosen_word,
        "secret_words": [chosen_word],
        "guesses": 5
    }

def ask_question_node(state: AgentState) -> dict:
    print(f"You have {state['guesses']} guesses left.")
    question = input("Question: ")
    new_guesses = state["guesses"] - 1
    return {
        "question": question,
        "guesses": new_guesses,
    }

def evaluate_answer_node(state: AgentState) -> dict:
    system_prompt = """
    You are a helpful AI assistant keeping the secret word for 20 questions game.
    """

    question = state.get("question", "")
    secret_word = state.get("secret_word", "")
    human_prompt = f"""
    Analyze the user's question: '{question}' regarding the secret word '{secret_word}'.
    
    You must evaluate this in strict order of priority:
    1. FIRST, check if the user is guessing the secret word. 
       If the question explicitly names or identifies '{secret_word}' (ignoring capitalization or punctuation), 
       you MUST reply exactly with: 'Solved'.
    2. SECOND, if it is not a guess, check if the question can be answered with a Yes or No. Reply exactly with 'Yes' or 'No'.
    3. THIRD, if the question cannot be answered with a simple Yes or No, reply exactly with 'Invalid'.

    Do not include any other text, punctuation, or explanation. Only return one of the four words: Solved, Yes, No, or Invalid.
    """
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    answer = response.content.strip().capitalize()
    if answer == "Yes" or answer == "No":
        print(f"Answer: {answer}")
    elif answer == "Invalid":
        print("That is not a valid question!")
    return {
        "answer": answer
    }

def game_router(state: AgentState) -> str:
    answer = state.get("answer", "Invalid")
    guesses = state.get("guesses", 0)
    if guesses == 0:
        return "loser"
    elif answer == "Solved":
        return "winner" 
    return "continue"
    

def winner_node(state: AgentState) -> dict:
    print(f"Congratulations! You have solved the secret word: '{state['secret_word']}'.")
    return {}

def loser_node(state: AgentState) -> dict:
    print(f"You did not guess the secret word: '{state['secret_word']}'. Better luck next time.")
    return {}

if __name__ == "__main__":
    graph = StateGraph(AgentState)
    
    graph.add_node("choose_word_node", choose_word_node)
    graph.add_node("ask_question_node", ask_question_node)
    graph.add_node("evaluate_answer_node", evaluate_answer_node)
    graph.add_node("loser_node", loser_node)
    graph.add_node("winner_node", winner_node)
    
    graph.add_edge(START, "choose_word_node")
    graph.add_edge("choose_word_node", "ask_question_node")
    graph.add_edge("ask_question_node", "evaluate_answer_node")
    graph.add_conditional_edges(
        "evaluate_answer_node",
        game_router,
        {
            "continue": "ask_question_node",
            "winner": "winner_node",
            "loser": "loser_node",
        }
    )
    graph.add_edge("loser_node", END)
    graph.add_edge("winner_node", END)

    app = graph.compile()

    drawing_filename = "/workspaces/playground-ai-ml/data/drawing.png"
    app.get_graph().draw_mermaid_png(output_file_path=drawing_filename)

    app.invoke({})

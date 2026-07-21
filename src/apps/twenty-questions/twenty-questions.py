from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field
from wonderwords import RandomWord
import os

workspaces = os.environ.get(
    "WORKSPACES",
    "/workspaces"
)

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://localhost:12434/v1"
)

api_key = os.environ.get(
    "OPENAI_API_KEY",
    "your-default-key"
)

llm = ChatOpenAI(
    model="mlx-community/gemma-4-12B-it-qat-6bit",
    base_url=openai_base_url,
    api_key=api_key,
    temperature=1.0,
    extra_body={
        "top_p": 0.95,
        "top_k": 64,
    },
)

max_guesses = 20
rw = RandomWord()

def safe_append_words(old: Optional[list[str]], new: list[str]) -> list[str]:
    return (old or []) + list(new)

class AgentState(TypedDict):
    question: Optional[str]
    answer: Optional[str]
    secret_word: Optional[str]
    secret_words: Annotated[list[str], safe_append_words]
    guesses: int
    play_again: Optional[bool]

class TangibleCheck(BaseModel):
    is_tangible: bool = Field(
        description="True if the word is a concrete, physical object you can touch. False if it is a concept, feeling, or abstract idea."
    )

llm_tangible_check = llm.with_structured_output(schema=TangibleCheck, method="json_schema")

def choose_word_node(state: AgentState) -> dict:    
    secret_words = state.get("secret_words", [])
    system_prompt = "You are a linguistics expert that classifies nouns as concrete/tangible or abstract."

    while True:
        chosen_word = rw.word(
            include_parts_of_speech=["nouns"],
            word_min_length=3,
            word_max_length=10
        ).lower()

        if chosen_word in secret_words:
            continue

        human_prompt = f"Is the noun '{chosen_word}' a tangible, physical object that a human can see and touch? Respond with true or false."
        
        response = llm_tangible_check.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])

        if response.is_tangible:
            break
    
    print(f"\n--- NEW GAME STARTED ---")
    print(f"DEBUG: secret word is '{chosen_word}'")
    
    return {
        "secret_word": chosen_word,
        "secret_words": [chosen_word],
        "guesses": max_guesses,
    }

def ask_question_node(state: AgentState) -> dict:
    guesses = state.get("guesses", max_guesses)

    user_input = interrupt({"action": "get_question"})

    return {
        "question": user_input,
    }

class EvaluatedAnswer(BaseModel):
    answer: Literal["Yes", "No", "Solved", "Invalid"] = Field(
        description="The evaluation of the user's question. Must be 'Yes', 'No', 'Solved', or 'Invalid'."
    )

llm_evaluated_answer = llm.with_structured_output(schema=EvaluatedAnswer, method="json_schema")

def evaluate_answer_node(state: AgentState) -> dict:
    system_prompt = """
    You are a helpful AI assistant keeping the secret word for 20 questions game.
    """

    question = state.get("question", "")
    secret_word = state.get("secret_word", "")
    guesses = state.get("guesses", max_guesses)

    if not question or not secret_word:
        return {
            "answer": "Invalid",
            "guesses": guesses,
        }
    
    human_prompt = f"""
    Analyze the user's question: '{question}' regarding the secret word '{secret_word}'.
Evaluate this in strict order of priority:
1. FIRST, check if the user is guessing the secret word. 
   If the question explicitly names or identifies '{secret_word}' (ignoring capitalization or punctuation), 
   you MUST reply exactly with: 'Solved'.
2. SECOND, if it is not a guess, check if the question can be answered with a Yes or No. Reply exactly with 'Yes' or 'No'.
3. THIRD, if the question cannot be answered with a simple Yes or No, reply exactly with 'Invalid'."""
    
    response = llm_evaluated_answer.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    answer = response.answer
    new_guesses = guesses - 1 if answer in ["Yes", "No", "Solved"] else guesses

    if answer == "Yes" or answer == "No":
        print(f"Answer: {answer}")
    elif answer == "Invalid":
        print("That is not a valid question!")
    
    return {
        "answer": answer,
        "guesses": new_guesses
    }

def game_router(state: AgentState) -> str:
    answer = state.get("answer", "Invalid")
    guesses = state.get("guesses", 0)

    if answer == "Solved":
        return "winner" 
    elif guesses <= 0:
        return "loser"
    return "continue"
    

def winner_node(state: AgentState) -> dict:
    print(f"Congratulations! You have solved the secret word: '{state['secret_word']}'.")
    return {}

def loser_node(state: AgentState) -> dict:
    print(f"You did not guess the secret word: '{state['secret_word']}'. Better luck next time.")
    return {}

def play_again_node(state: AgentState) -> dict:
    user_input = interrupt({"action": "get_replay_choice"})
    choice = str(user_input).strip().lower()
    return {
        "play_again": choice in ["y", "yes"]
    }

def replay_router(state: AgentState) -> str:
    if state.get("play_again", False):
        return "replay"
    return "exit"

if __name__ == "__main__":
    graph = StateGraph(AgentState)
    
    graph.add_node("choose_word", choose_word_node)
    graph.add_node("ask_question", ask_question_node)
    graph.add_node("evaluate_answer", evaluate_answer_node)
    graph.add_node("loser", loser_node)
    graph.add_node("winner", winner_node)
    graph.add_node("play_again", play_again_node)
    
    graph.add_edge(START, "choose_word")
    graph.add_edge("choose_word", "ask_question")
    graph.add_edge("ask_question", "evaluate_answer")
    graph.add_conditional_edges(
        "evaluate_answer",
        game_router,
        {
            "continue": "ask_question",
            "winner": "winner",
            "loser": "loser",
        }
    )
    graph.add_edge("loser", "play_again")
    graph.add_edge("winner", "play_again")
    graph.add_conditional_edges(
        "play_again",
        replay_router,
        {
            "replay": "choose_word",
            "exit": END
        }
    )

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    # drawing_filename = os.path.join(workspaces, "playground-ai-ml/shared-data/python/drawing.png")
    # app.get_graph().draw_mermaid_png(output_file_path=drawing_filename)

    config = {"configurable": {"thread_id": "session_abc123"}}
    state = {"secret_words": []}
    events = app.stream(input=state, config=config)

    # Standard runtime execution runner
    active_state = None
    for event in events:
        active_state = event
    
    # Continue running as long as the state is paused on an interrupt
    while True:
        state_snapshot = app.get_state(config)

        # if there are tasks waiting, means we hit an interrupt
        if (state_snapshot.tasks):
            current_interrupt = state_snapshot.tasks[0].interrupts[0]
            action_needed = current_interrupt.value.get("action")

            # User inputs external from the engine core
            if action_needed == "get_question":
                guesses_left = state_snapshot.values.get("guesses", max_guesses)
                print(f"You have {guesses_left} guesses left.")
                user_response = input("Question: ")
            elif action_needed == "get_replay_choice":
                user_response = input("\nDo you want to play again? (yes/no): ")
            else:
                user_response = None
        
            # Resume graph using Command by passing user response
            events = app.stream(Command(resume=user_response), config=config, stream_mode="values")
            for event in events:
                active_state = event
        else:
            # No tasks, graph reached END safely
            break

import os
from typing import Annotated, Literal, Optional, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
import streamlit as st
from pydantic import BaseModel, Field
from wonderwords import RandomWord

# --- 1. CONFIGURATION & STATE INITIALIZATION ---
st.set_page_config(page_title="20 Questions Game", page_icon="🎮")
st.title("🎮 The 20 Questions Game")

# Setup API and Environment Variables
openai_base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:12434/v1")
api_key = os.environ.get("OPENAI_API_KEY", "your-default-key")

max_guess = 20

@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="mlx-community/gemma-4-12B-it-qat-6bit",
        base_url=openai_base_url,
        api_key=api_key,
        temperature=1.0,
        extra_body={
            "top_p": 0.95,
            "top_k": 64,
        },
    )

# --- 2. LANGGRAPH ENGINE DEFINITIONS ---
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

def choose_word_node(state: AgentState) -> dict:
    secret_words = state.get("secret_words", [])

    llm_tangible_check = get_llm().with_structured_output(schema=TangibleCheck, method="json_schema")
    rw = RandomWord()
    
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
       
    return {
        "secret_word": chosen_word,
        "secret_words": [chosen_word],
        "guesses": max_guess,
    }

def ask_question_node(state: AgentState) -> dict:
    # Interrupt triggers a pause here, shifting execution out to Streamlit
    user_input = interrupt({"action": "get_question"})
    return {"question": user_input}

class EvaluatedAnswer(BaseModel):
    answer: Literal["Yes", "No", "Solved", "Invalid"] = Field(
        description="The evaluation of the user's question. Must be 'Yes', 'No', 'Solved', or 'Invalid'."
    )

def evaluate_answer_node(state: AgentState) -> dict:
    llm = get_llm()
    llm_evaluated_answer = llm.with_structured_output(schema=EvaluatedAnswer, method="json_schema")
    
    system_prompt = "You are a helpful AI assistant keeping the secret word for 20 questions game."
    question = state.get("question", "")
    secret_word = state.get("secret_word", "")
    guesses = state.get("guesses", max_guess)

    if not question or not secret_word:
        return {"answer": "Invalid", "guesses": guesses}
    
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
    
    return {"answer": answer, "guesses": new_guesses}

def game_router(state: AgentState) -> str:
    answer = state.get("answer", "Invalid")
    guesses = state.get("guesses", 0)

    if answer == "Solved":
        return "winner" 
    elif guesses <= 0:
        return "loser"
    return "continue"

def winner_node(state: AgentState) -> dict:
    return {}

def loser_node(state: AgentState) -> dict:
    return {}

def play_again_node(state: AgentState) -> dict:
    user_input = interrupt({"action": "get_replay_choice"})
    return {"play_again": str(user_input).strip().lower() in ["y", "yes"]}

def replay_router(state: AgentState) -> str:
    if state.get("play_again", False):
        return "replay"
    return "exit"

# --- 3. COMPILING GRAPH INTO CACHE ---
@st.cache_resource
def compile_graph():
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
        "evaluate_answer", game_router,
        {"continue": "ask_question", "winner": "winner", "loser": "loser"}
    )
    graph.add_edge("loser", "play_again")
    graph.add_edge("winner", "play_again")
    graph.add_conditional_edges(
        "play_again", replay_router,
        {"replay": "choose_word", "exit": END}
    )
    return graph.compile(checkpointer=MemorySaver())

# --- 4. PERSISTENT APP STATE MANAGEMENT ---
if "game_app" not in st.session_state:
    st.session_state.game_app = compile_graph()
    st.session_state.config = {"configurable": {"thread_id": "streamlit_session_001"}}
    # Added spinner for the initial session setup
    with st.spinner("Generating game universe..."):
        for event in st.session_state.game_app.stream({"secret_words": []}, config=st.session_state.config, stream_mode="values"):
            pass

# UI Event Callbacks to handle streaming resume commands
def submit_question():
    if st.session_state.user_q_input.strip():
        q_text = st.session_state.user_q_input
        # Added spinner while the LLM processes the question evaluation
        with st.spinner("Consulting the AI Oracle..."):
            for event in st.session_state.game_app.stream(Command(resume=q_text), config=st.session_state.config, stream_mode="values"):
                pass
        st.session_state.user_q_input = "" # Clear box

def submit_replay(choice: str):
    # Added spinner for setting up a brand new round
    with st.spinner("Resetting game board..."):
        for event in st.session_state.game_app.stream(Command(resume=choice), config=st.session_state.config, stream_mode="values"):
            pass

# --- 5. RENDER THE CURRENT INTERFACE LAYER ---
app = st.session_state.game_app
config = st.session_state.config
state_snapshot = app.get_state(config)
current_state = state_snapshot.values

# Debug panel for internal testing
with st.sidebar:
    st.subheader("🛠️ Engine Inspection Panel")
    if current_state.get("secret_word"):
        st.write(f"**Secret Word:** `{current_state['secret_word']}`")
    st.write("**History Logs:**", current_state.get("secret_words", []))

# Main UI layout conditional processing
if state_snapshot.tasks:
    # FIX: Extract the first interrupt object safely from the task stack tuples
    active_task = state_snapshot.tasks[0]
    
    if active_task.interrupts:
        current_interrupt = active_task.interrupts[0]
        action_needed = current_interrupt.value.get("action")
    else:
        action_needed = None
    
    if action_needed == "get_question":
        # Game Stats Context Card
        st.metric(label="Guesses Remaining", value=current_state.get("guesses", max_guess))
        
        # Display previous response assessment validation if it exists
        last_ans = current_state.get("answer")
        if last_ans == "Invalid":
            st.warning("⚠️ That was not a valid question!")
        elif last_ans in ["Yes", "No"]:
            st.info(f"💡 AI response to your last question: **{last_ans}**")

        # Chat-style entry input box
        st.text_input(
            "Ask a question to guess the word (or type the word directly):", 
            key="user_q_input", 
            on_change=submit_question
        )
        
    elif action_needed == "get_replay_choice":
        # Determine ending context from last node evaluation
        last_ans = current_state.get("answer")
        if last_ans == "Solved":
            st.success(f"🎉 Congratulations! You solved the secret word: **{current_state.get('secret_word')}**!")
        else:
            st.error(f"💀 Game Over! You ran out of guesses. The word was: **{current_state.get('secret_word')}**.")
            
        st.write("### Would you like to play another match?")
        col1, col2 = st.columns(2)
        with col1:
            st.button("Yes, deal me in!", on_click=submit_replay, args=("yes",), use_container_width=True, type="primary")
        with col2:
            st.button("No, I'm done", on_click=submit_replay, args=("no",), use_container_width=True)
else:
    # No active interrupt tasks means graph finished via END route cleanly
    st.balloons()
    st.success("Thanks for playing! The engine session has safely exited.")
    if st.button("Start Fresh Application Session"):
        st.session_state.clear()
        st.rerun()

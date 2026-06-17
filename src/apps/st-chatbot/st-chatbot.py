import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://localhost:12434/v1"
)

api_key = os.environ.get(
    "OPENAI_API_KEY",
    "your-default-key"
)

# ----------------------------------------------------------------------
# Page Configuration & Styling
# ----------------------------------------------------------------------
st.set_page_config(page_title="Custom LLM Sandbox", page_icon="🤖", layout="wide")

# Modern UI Styling
st.markdown(
    """
    <style>
    .main { background-color: #0f1117; color: #ffffff; }
    .stButton>button { width: 100%; border-radius: 8px; }
    div.stActionButton { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# State Management Initialization
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "loaded_model_name" not in st.session_state:
    st.session_state.loaded_model_name = None

if "llm_instance" not in st.session_state:
    st.session_state.llm_instance = None


# ----------------------------------------------------------------------
# Sidebar - Model Controls & Management
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Model Control Panel")
    st.caption("Manage your active LLM lifecycle and history here.")
    st.divider()

    # Model Selection dropdown
    model_option = st.selectbox(
        "Select a Model Architecture",
        options=[
            "mlx-community/gemma-4-12B-it-6bit",
            "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"
        ],
        index=0,
    )

    # Autoload logic: Check if the selection differs from what's currently loaded
    if model_option != st.session_state.loaded_model_name and api_key:
        try:
            # Initialize LangChain ChatOpenAI object with your custom base URL automatically
            st.session_state.llm_instance = ChatOpenAI(
                base_url=openai_base_url,
                model=model_option,
                openai_api_key=api_key,
                temperature=0.1
            )
            st.session_state.loaded_model_name = model_option
            st.toast(f"Successfully loaded {model_option}!", icon="✅")
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")

    st.divider()

    # Conversation Management
    st.subheader("Session Management")
    reset_chat = st.button("🧹 Clear Chat History", type="secondary")
    if reset_chat:
        st.session_state.messages = []
        st.toast("Chat history wiped clean.", icon="🧼")
        st.rerun()


# ----------------------------------------------------------------------
# Main Chat Interface UI
# ----------------------------------------------------------------------
st.title("💬 Modern AI Chat Environment")

# Active Model Status Banner
if st.session_state.loaded_model_name:
    st.success(
        f"**Active Model:** `{st.session_state.loaded_model_name}` is automatically loaded and ready.",
        icon="🟢",
    )
else:
    st.warning(
        "**No Active Model:** Please ensure an API Key is set to automatically load the model.",
        icon="🔴",
    )

st.divider()

# Display Chat History using Streamlit's native chat elements
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Chat Input Block
if prompt := st.chat_input(
    "Type a message...", disabled=(st.session_state.llm_instance is None)
):

    # 1. Append and display user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call Langchain model and stream/display response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Gather history to pass back to LangChain to maintain context
            context_history = st.session_state.messages[-10:]

            # Stream the response chunk by chunk for a modern feel
            for chunk in st.session_state.llm_instance.stream(context_history):
                full_response += chunk.content or ""
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            # Append the AI reply to history
            st.session_state.messages.append(AIMessage(content=full_response))

        except Exception as e:
            st.error(f"An error occurred while generating a response: {e}")

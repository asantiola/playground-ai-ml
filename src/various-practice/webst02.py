import streamlit as st

# 1. Force wide mode for side-by-side layout
st.set_page_config(layout="wide")

st.title("My Side-by-Side Streamlit App")

# 2. Create the two "windows"
col1, col2 = st.columns(2)

# 3. Populate the Left Window
with col1:
    st.header("Left Window (Inputs)")
    user_input = st.text_input("Enter some data:")
    clicked = st.button("Submit")

# 4. Populate the Right Window
with col2:
    st.header("Right Window (Outputs)")
    if clicked and user_input:
        st.success(f"Received from the left window: {user_input}")
    else:
        st.info("Awaiting input from the left window...")

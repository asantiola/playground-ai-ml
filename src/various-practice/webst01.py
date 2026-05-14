import streamlit as st

def main():
    """Main function to run the Simple Counter App."""
    
    # 1. Initialize session state for the counter
    if 'counter' not in st.session_state:
        st.session_state.counter = 0

    # 2. Set up the header
    st.title('Simple Counter App')
    
    # 3. Display the current value
    st.metric(label="Current Count", value=st.session_state.counter)

    # 4. Define the callback functions for buttons
    def increment_counter():
        st.session_state.counter += 1

    def decrement_counter():
        st.session_state.counter -= 1

    # 5. Create buttons side-by-side
    col1, col2 = st.columns(2)
    
    with col1:
        st.button('Add', on_click=increment_counter)
        
    with col2:
        st.button('Subtract', on_click=decrement_counter)

if __name__ == "__main__":
    main()
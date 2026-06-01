from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os

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

def choose_secret_item():
    """Chooses a random word from a predefined list of possible items."""
    return "aardvark"

def validate_question(question, secret_item):
    system_prompt = f"""
    You are validating a guess for a 20 question game, and the secret item is '{secret_item}'.
    You can only answer 'Yes', 'No', or 'Invalid question' if question is not answerable with Yes/No.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    response = llm.invoke(messages)
    return response.content

def play_twenty_questions():
    """Runs the 20 Questions game console application."""
    secret_item = choose_secret_item()
    max_guesses = 3
    guesses_taken = 0

    print("=========================================")
    print("        WELCOME TO TWENTY QUESTIONS     ")
    print("=========================================")
    print("I am thinking of something. You have 20 'Yes' or 'No' questions to guess what it is.")
    print("Start guessing!")

    while guesses_taken < max_guesses:
        print(f"\n--- Guess Attempt {guesses_taken + 1} of {max_guesses} ---")
        
        if guesses_taken < max_guesses:
            guess = input("Enter your guess (or type 'hint'): ").strip().lower()

            if guess == secret_item:
                print(f"\n*** Congratulations! You guessed it! The item was '{secret_item}' in {guesses_taken + 1} attempts. ***")
                return True

            answer = validate_question(guess, secret_item=secret_item)
            print(f"Response to your question is: {answer}")
        
        guesses_taken += 1

    print("\n=========================================")
    print("             GAME OVER!                 ")
    print(f"You ran out of guesses. The item I was thinking of was '{secret_item}'.")
    print("=========================================")
    return False

if __name__ == "__main__":
    play_twenty_questions()
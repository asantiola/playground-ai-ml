from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(
    model="ai/gemma4:4B-128k",
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
)

system_prompt = "You are an expert assistant good in solving logic problems."

puzzle_prompt_einstein = """
There are five houses of different colors adjacent to one another on a road. 
In each house lives a man of different nationality. 
Each man has a favorite drink, a favorite brand of cigarettes, and keeps a different kind of pet.

The Englishman lives in the red house.
The Swede keeps dogs.
The Dane drinks tea.
The green house is just to the left of the white house.
The owner of the green house drinks coffee.
The Pall Mall smoker keeps birds.
The owner of the yellow house smokes Dunhills.
The man in the center house drinks milk.
The Norwegian lives in the first house.
The Blend Smoker has a neighbor who keeps cats.
The man who smokes Blue Masters drinks bier.
The man who keeps horses lives next to the Dunhill smoker.
The German smokes Prince.
The Norwegian lives next to the blue house.
The Blend smoker has a neighbor who drinks water.

Who owns the fish?
"""

puzzle_prompt_drunk = """
The Drunk Passenger Problem (Expected Value): A plane has 50 seats, and 50 passengers are boarding. 
The first 10 passengers are drunk and sit in random unoccupied seats.
Every subsequent passenger sits in their assigned seat if it's available; otherwise, they choose a random seat.
What is the expected number of passengers who sit in their assigned seats?
"""

puzzle_prompt_floors = """
A building has 10 floors above the basement.
If 12 people get into an elevator at the basement, and each chooses a floor at random to get out, 
independently of the others,
at how many floors do you expect the elevator to make a stop to let out one or more of these 12 people?
"""

messages = [SystemMessage(content=system_prompt), HumanMessage(content=puzzle_prompt_floors)]
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)

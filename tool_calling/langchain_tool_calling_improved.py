import os
from dotenv import load_dotenv
import requests
from langchain_groq import ChatGroq
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
from langchain.agents import initialize_agent, AgentType

# --- Step 1: Load API Key ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")  # set this in your .env file

# --- Step 2: Define Tools ---

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Fetch currency conversion rate between base and target currency."""
    url = f'https://v6.exchangerate-api.com/v6/e4c905300819f45bc98cbe03/pair/{base_currency}/{target_currency}'
    response = requests.get(url)
    data = response.json()
    return data['conversion_rate']

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """Convert a value using the provided currency conversion rate."""
    return base_currency_value * conversion_rate

# --- Step 3: Initialize LLM ---
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.4,
    api_key=api_key
)

# --- Step 4: Bind Tools to Agent ---
tools = [multiply, subtract, get_conversion_factor, convert]

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  # prints step-by-step reasoning
)

# --- Step 5: Try Sample Queries ---

# Math Example
response1 = agent_executor.invoke({"input": "Multiply 4 with 5 and subtract 6 from the result"})
print("\n✅ Math Result:\n", response1)

# Currency Example
response2 = agent_executor.invoke({"input": "What is the USD to INR conversion rate, and convert 100 USD to INR?"})
print("\n💱 Currency Result:\n", response2)

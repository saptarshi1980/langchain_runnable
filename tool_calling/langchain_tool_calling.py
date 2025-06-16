
import os



#!pip install -q langchain-groq langchain-core requests

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv

# tool create

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

@tool
def multiply(a: int, b: int) -> int:
  """Given 2 numbers a and b this tool returns their product"""
  return a * b

print(multiply.invoke({'a':3, 'b':4}))

@tool
def subtract(a:int,b:int)->int:
  """ Given two numbers it returns the subtraction of second number from first number"""
  return a-b

multiply.name

subtract.name

multiply.description

subtract.description

multiply.args

subtract.args

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.4,
    api_key=api_key
)

llm.invoke('hi')

llm_with_tools = llm.bind_tools([multiply,subtract])

llm_with_tools.invoke('Hi how are you')

query2 = HumanMessage('can you subtract 300 from 1000')
query1 = HumanMessage('can you multiply 3 with 1000')

messages = [query2,query1]

messages

result = llm_with_tools.invoke(messages)

messages.append(result)

messages

result.tool_calls[0]

from langchain_core.messages import HumanMessage

messages = [HumanMessage(content="can you subtract 300 from 1000")]
result1 = llm_with_tools.invoke(messages)
print(result1)

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


# Initialize LLM with tools
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.4,
    api_key=api_key
)
llm_with_tools = llm.bind_tools([multiply, subtract])

# Step 1: Provide query
messages = [HumanMessage(content="Can you subtract 300 from 1000?")]

# Step 2: Let LLM decide tool to call
ai_response = llm_with_tools.invoke(messages)

# Debug: Show what tool was called
print("🧠 Tool Call Response:\n", ai_response)

# Step 3: Extract tool call from LLM response
if not ai_response.tool_calls:
    print("❌ No tool calls made by the LLM.")
    exit()

tool_call = ai_response.tool_calls[0]
tool_name = tool_call["name"]
tool_args = tool_call["args"]
tool_id = tool_call["id"]

print(f"🔧 Tool Selected: {tool_name}")
print(f"📦 Arguments: {tool_args}")
print(f"🆔 Tool Call ID: {tool_id}")

# Step 4: Execute the selected tool
if tool_name == 'subtract':
    tool_result = subtract.invoke(tool_args)
elif tool_name == 'multiply':
    tool_result = multiply.invoke(tool_args)
else:
    print(f"❌ Unknown tool: {tool_name}")
    exit()

print(f"✅ Tool Result: {tool_result}")

# Step 5: Pass tool result back to the LLM
final_response = llm_with_tools.invoke(
    messages + [ai_response, ToolMessage(tool_call_id=tool_id, content=str(tool_result))]
)

# Step 6: Final Output
print("\n🎉 Final Answer from LLM:")
print(final_response.content)



# tool create
from langchain_core.tools import InjectedToolArg
from typing import Annotated

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
  """
  This function fetches the currency conversion factor between a given base currency and a target currency
  """
  url = f'https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/pair/{base_currency}/{target_currency}'

  response = requests.get(url)

  return response.json()

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
  """
  given a currency conversion rate this function calculates the target currency value from a given base currency value
  """

  return base_currency_value * conversion_rate

convert.args

get_conversion_factor.invoke({'base_currency':'USD','target_currency':'INR'})

convert.invoke({'base_currency_value':10, 'conversion_rate':85.16})

llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [HumanMessage('What is the conversion factor between INR and USD, and based on that can you convert 10 inr to usd')]

messages

ai_message = llm_with_tools.invoke(messages)

messages.append(ai_message)

ai_message.tool_calls

import json

for tool_call in ai_message.tool_calls:
  # execute the 1st tool and get the value of conversion rate
  if tool_call['name'] == 'get_conversion_factor':
    tool_message1 = get_conversion_factor.invoke(tool_call)
    # fetch this conversion rate
    conversion_rate = json.loads(tool_message1.content)['conversion_rate']
    # append this tool message to messages list
    messages.append(tool_message1)
  # execute the 2nd tool using the conversion rate from tool 1
  if tool_call['name'] == 'convert':
    # fetch the current arg
    tool_call['args']['conversion_rate'] = conversion_rate
    tool_message2 = convert.invoke(tool_call)
    messages.append(tool_message2)

messages

llm_with_tools.invoke(messages).content

from langchain.agents import initialize_agent, AgentType

# Step 5: Initialize the Agent ---
agent_executor = initialize_agent(
    tools=[get_conversion_factor, convert],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # using ReAct pattern
    verbose=True  # shows internal thinking
)

user_query = "Hi how are you?"

response = agent_executor.invoke({"input": user_query})


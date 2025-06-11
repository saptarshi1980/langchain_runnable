from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda,RunnableSequence
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal,Optional
from langchain.schema.runnable import RunnableMap


load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

parser = StrOutputParser()

model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.4,
    api_key=api_key
)

prompt1=PromptTemplate(
    template="Write a poem on the {topic}",
    input_variables="topic"    
)
prompt2=PromptTemplate(
    template="Explain the joke {joke}",
    input_variables="joke"   
)

chain = RunnableSequence(prompt1|model|parser|prompt2|model|parser)

result = chain.invoke({"topic":"Himalaya"})

print(result)






from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os
from langchain.schema.runnable import RunnableBranch,RunnableSequence,RunnablePassthrough
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

parser=StrOutputParser()

model = ChatGroq(
    
    model='meta-llama/llama-4-scout-17b-16e-instruct',
    temperature=0.5,
    api_key=api_key
)

prompt1 = PromptTemplate(
    template="Write a report on the following topic \n {topic}",
    input_variables="topic"
)

prompt2 = PromptTemplate(
    template="Summarize the following report {report}",
    input_variables="report"
    
)

chain = RunnableSequence(prompt1|model|parser)

branch_chain =RunnableBranch(
    (lambda x:len(x.split())>300,prompt2|model|parser),
    RunnablePassthrough()
    
)

final_chain = RunnableSequence(chain|branch_chain)

print(final_chain.invoke({"topic":"How to teach paksitan a tough lesson for killing innocent tourists in Indian Kashmir?"}))



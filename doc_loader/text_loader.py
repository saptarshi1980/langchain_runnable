from langchain_community import document_loaders
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
import os
from langchain_community.document_loaders import TextLoader
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda,RunnableSequence,RunnablePassthrough



load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

parser = StrOutputParser()

model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.4,
    api_key=api_key
)

prompt1 = PromptTemplate(
    
    template="Can you summarize the follwoing poem {poem}",
    input_variables="poem"
    
)

text_loader = TextLoader(file_path="football.txt",encoding='utf-8')

docs = text_loader.load()
print(type(docs))
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)


print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
chain = RunnableSequence(prompt1|model|parser)




result = chain.invoke({"poem",docs[0].page_content})

print(result)
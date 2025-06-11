from langchain_community import document_loaders
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
import os
from langchain_community.document_loaders import PyPDFLoader
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
    
    template="Can you summarize the follwoing resume {resume}",
    input_variables="resume"
    
)

prompt2 = PromptTemplate(
    
    template="Can you Generate 20 interview question and expected answers based on the profile summary {profile}",
    input_variables="profile"
    
)

pdf_loader = PyPDFLoader(file_path="resume.pdf")

docs = pdf_loader.load()    
print(type(docs))
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)


summary_chain=RunnableSequence(prompt1|model|parser)


parallel_chain = RunnableParallel(
    
    {'summary':RunnablePassthrough(),
    'interview':RunnableSequence(prompt2|model|parser)
    }
)

final_chain = RunnableSequence(summary_chain|parallel_chain)



result = final_chain.invoke({"resume",docs[0].page_content})



print(result['summary'])
print(result['interview'])

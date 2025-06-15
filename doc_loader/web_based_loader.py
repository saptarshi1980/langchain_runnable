from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize model
api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.4,
    api_key=api_key
)

# Define prompt template
prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

# Output parser
parser = StrOutputParser()

# Ask user for URL input
url = input("Enter the product page URL: ")

# Load web content
loader = WebBaseLoader(url)
docs = loader.load()

# Define the chain
chain = prompt | model | parser

# Invoke the chain with the loaded content
result = chain.invoke(
    {
        "question": "What is the product details?",
        "text": docs[0].page_content
    }
)

# Print the result
print("\n--- Product Details ---")
print(result)

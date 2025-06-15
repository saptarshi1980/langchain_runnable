# %%
print("Hi")

# %% Imports
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

import os

# %% Load environment
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# %% Get transcript from YouTube
video_id = "8XYO-mj1ugg"  # Just the ID

try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print("Transcript loaded.")
except TranscriptsDisabled:
    print("No captions available for this video.")
    transcript = ""

# %% Text Splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# %% Embedding Model
model_dir = "D:/huggingface_models/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=model_dir,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# %% FAISS Vector Store
vector_store = FAISS.from_documents(chunks, embeddings)

# %% Retriever with Scoring
question = "is the anything related to Air crash?"
docs_with_scores = vector_store.similarity_search_with_score(question, k=4)

# Filter irrelevant docs based on similarity score
similarity_threshold = 0.6  # lower means more similar
filtered_docs = [doc for doc, score in docs_with_scores if score < similarity_threshold]

# Build the context
context_text = "\n\n".join(doc.page_content for doc in filtered_docs)
if not context_text:
    context_text = "No relevant content found in the document."

# %% Initialize LLM
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.4,
    api_key=api_key
)

# %% Prompt
prompt = PromptTemplate.from_template("""
Analyze if this question relates to the context's domain (aviation accidents).
If unrelated, respond ONLY with: "This query is unrelated to the content."

Context: {context}
Question: {question}
""")

# %% LLM Chain
chain = prompt | llm | StrOutputParser()

# %% Final LLM Call
response = chain.invoke({
    "context": context_text,
    "question": question
})

print("\n--- Final Answer ---\n")
print(response)

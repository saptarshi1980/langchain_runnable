# %%
print("Hi")

# %%
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel
import os
from langchain_huggingface import HuggingFaceEmbeddings

# %%
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# %%
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# %%
video_id = "8XYO-mj1ugg" # only the ID, not full URL
try:
    # If you don’t care which language, this returns the “best” one
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

    # Flatten it to plain text
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

# %%
transcript_list

# %% [markdown]
# ## Step 1b - Indexing (Text Splitting)

# %%
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# %%
len(chunks)

# %%
model_dir = "D:/huggingface_models/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=model_dir,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# %% [markdown]
# ## Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)

# %%
vector_store = FAISS.from_documents(chunks, embeddings)

# %%
vector_store.index_to_docstore_id

# %%
vector_store.get_by_ids(['be2b23b5-2837-4316-b0f7-7aa0bde7ca49'])

# %%
vector_store.get_by_ids(['be2b23b5-2837-4316-b0f7-7aa0bde7ca49'])

# %% [markdown]
# ## Step 2 - Retrieval

# %%
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# %%
retriever

# %%
retriever.invoke('What is deepmind')

# %% [markdown]
# ### Step 3 - Augmentation

# %%
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.4,
    api_key=api_key
    
)


# %%
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

# %%
question          = "is the topic of cricket? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)

# %%
retrieved_docs

# %%


# %%




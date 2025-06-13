from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain.schema import Document
import shutil, os

# Step 1: Clean previous DB
if os.path.exists("my_chroma_db"):
    shutil.rmtree("my_chroma_db")

# Step 2: Load local embeddings
model_dir = "D:/huggingface_models/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=model_dir,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Step 3: Create documents with STRING metadata (NOT lists)
docs = [
    Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history.",
        metadata={"team": "Royal Challengers Bangalore", "role": "batter"}  
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history.",
        metadata={"team": "Mumbai Indians", "role": "batter, captain"}  
    ),
    Document(
        page_content="MS Dhoni has led Chennai Super Kings to multiple IPL titles.",
        metadata={"team": "Chennai Super Kings", "role": "batter, captain, wicket keeper"}  
    ),
    Document(
        page_content="Jasprit Bumrah is one of the best fast bowlers in T20 cricket.",
        metadata={"team": "Mumbai Indians", "role": "bowler"} 
    ),
    Document(
        page_content="Ravindra Jadeja contributes with both bat and ball.",
        metadata={"team": "Chennai Super Kings", "role": "bowler, batter, all rounder"} 
    )
]


# Step 5: Create vector store
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="my_chroma_db",
    collection_name="sample"
)

# Step 6: Add documents - this will work now!
vector_store.add_documents(docs)

# Step 7: Search
results = vector_store.similarity_search("Who are the bowlers?", k=5)

# Step 8: DEBUG – Print raw metadata
print("\n--- RAW RESULTS ---")
for doc in results:
    print(f"Role: {doc.metadata.get('role')}, Content: {doc.page_content}")

# Step 9: Filter on "bowler" - now works with string containment!
filtered_results = [
    doc for doc in results if "bowler" in doc.metadata.get("role", "").lower()
]

print("\n--- FILTERED BOWLERS ---")
for doc in filtered_results:
    print(f"\nTeam: {doc.metadata['team']}")
    print(f"Role: {doc.metadata['role']}")
    print(f"Content: {doc.page_content}")
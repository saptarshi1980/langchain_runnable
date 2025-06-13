from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain.schema import Document
import shutil, os

# Load embeddings (same as your setup)
model_dir = "D:/huggingface_models/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=model_dir,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Load existing vector store
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="my_chroma_db",
    collection_name="sample"
)

# =============================================================================
# METHOD 1: GET ALL DOCUMENT IDs AND DATA
# =============================================================================
print("METHOD 1: Get all documents with their IDs")

# This is the main method to get all data including IDs
all_data = vector_store.get()

print(f"Total documents: {len(all_data['documents'])}")
print(f"Available keys: {all_data.keys()}")

# Display all documents with their IDs
print("\n--- ALL DOCUMENTS WITH IDs ---")
for i, (doc_id, content, metadata) in enumerate(zip(
    all_data['ids'], 
    all_data['documents'], 
    all_data['metadatas']
)):
    print(f"\n{i+1}. ID: {doc_id}")
    print(f"   Content: {content[:60]}...")
    print(f"   Metadata: {metadata}")

# =============================================================================
# METHOD 2: GET SPECIFIC DOCUMENTS BY FILTERING
# =============================================================================
print("\n\nMETHOD 2: Filter documents and get their IDs")

# You can filter by metadata
filtered_data = vector_store.get(
    where={"team": "Mumbai Indians"}  # Filter by team
)

print(f"\nMumbai Indians players:")
for doc_id, content, metadata in zip(
    filtered_data['ids'], 
    filtered_data['documents'], 
    filtered_data['metadatas']
):
    print(f"ID: {doc_id}")
    print(f"Player info: {content[:50]}...")
    print(f"Role: {metadata.get('role')}")
    print("-" * 50)

# =============================================================================
# METHOD 3: SEARCH AND GET IDs (Similarity search doesn't return IDs directly)
# =============================================================================
print("\n\nMETHOD 3: Working around similarity search limitations")

def get_ids_for_similar_docs(query, k=5):
    """Get IDs for documents returned by similarity search"""
    
    # Step 1: Get similarity search results
    search_results = vector_store.similarity_search(query, k=k)
    
    # Step 2: Get all documents with IDs
    all_data = vector_store.get()
    
    # Step 3: Match content to find IDs
    matched_ids = []
    for search_doc in search_results:
        for i, stored_content in enumerate(all_data['documents']):
            if search_doc.page_content == stored_content:
                matched_ids.append(all_data['ids'][i])
                break
    
    return list(zip(matched_ids, search_results))

# Example usage
query_results = get_ids_for_similar_docs("Who are the bowlers?", k=5)

print("Search results with IDs:")
for doc_id, doc in query_results:
    print(f"ID: {doc_id}")
    print(f"Content: {doc.page_content[:50]}...")
    print(f"Team: {doc.metadata.get('team')}")
    print("-" * 40)

# =============================================================================
# METHOD 4: ADD DOCUMENTS WITH CUSTOM IDs (RECOMMENDED)
# =============================================================================
print("\n\nMETHOD 4: Adding documents with custom IDs")

# When adding new documents, specify your own IDs
custom_docs = [
    Document(
        page_content="KL Rahul is a versatile wicket-keeper batsman.",
        metadata={"team": "Lucknow Super Giants", "role": "wicket keeper, batter"}
    ),
    Document(
        page_content="Suryakumar Yadav is known for his innovative batting.",
        metadata={"team": "Mumbai Indians", "role": "batter"}
    )
]

# Add with custom IDs
custom_ids = ["kl_rahul", "suryakumar_yadav"]
vector_store.add_documents(custom_docs, ids=custom_ids)

print("Added documents with custom IDs:")
print(f"IDs: {custom_ids}")

# Verify they were added
new_data = vector_store.get(ids=custom_ids)
for doc_id, content in zip(new_data['ids'], new_data['documents']):
    print(f"ID: {doc_id} -> {content[:50]}...")

# =============================================================================
# METHOD 5: UPDATE/DELETE USING IDs
# =============================================================================
print("\n\nMETHOD 5: Update and delete using IDs")

# Get a specific document by ID to update it
doc_to_update_id = custom_ids[0]  # "kl_rahul"
print(f"Updating document with ID: {doc_to_update_id}")

# Create updated version
updated_doc = Document(
    page_content="KL Rahul is the captain and wicket-keeper of LSG.",
    metadata={"team": "Lucknow Super Giants", "role": "captain, wicket keeper, batter"}
)

# Update by adding with the same ID (this replaces the document)
vector_store.add_documents([updated_doc], ids=[doc_to_update_id])

# Verify update
updated_data = vector_store.get(ids=[doc_to_update_id])
print(f"Updated content: {updated_data['documents'][0]}")

# Delete a document using its ID
doc_to_delete_id = custom_ids[1]  # "suryakumar_yadav"
print(f"\nDeleting document with ID: {doc_to_delete_id}")
vector_store.delete(ids=[doc_to_delete_id])

# Verify deletion
try:
    deleted_check = vector_store.get(ids=[doc_to_delete_id])
    print(f"Documents found after deletion: {len(deleted_check['documents'])}")
except:
    print("Document successfully deleted")

# =============================================================================
# METHOD 6: CREATE A MAPPING SYSTEM
# =============================================================================
print("\n\nMETHOD 6: Creating a mapping system for easy ID management")

def create_id_mapping(vector_store):
    """Create a mapping of meaningful names to document IDs"""
    all_data = vector_store.get()
    
    id_mapping = {}
    for doc_id, content, metadata in zip(
        all_data['ids'], 
        all_data['documents'], 
        all_data['metadatas']
    ):
        # Create a meaningful key based on content/metadata
        if 'team' in metadata:
            # Extract player name from content (simple approach)
            words = content.split()
            if len(words) >= 2:
                player_name = f"{words[0]}_{words[1]}".lower().replace(".", "")
                team_short = metadata['team'].replace(" ", "_").lower()
                key = f"{player_name}_{team_short}"
                id_mapping[key] = doc_id
    
    return id_mapping

# Create and display the mapping
id_mapping = create_id_mapping(vector_store)

print("ID Mapping:")
for key, doc_id in id_mapping.items():
    print(f"{key} -> {doc_id}")

# Use the mapping to update a specific player
if 'virat_kohli_royal_challengers_bangalore' in id_mapping:
    virat_id = id_mapping['virat_kohli_royal_challengers_bangalore']
    print(f"\nFound Virat Kohli's ID: {virat_id}")
    
    # Now you can easily update Virat's document
    updated_virat = Document(
        page_content="Virat Kohli is RCB's legendary batsman and former captain.",
        metadata={"team": "Royal Challengers Bangalore", "role": "batter, legend"}
    )
    
    vector_store.add_documents([updated_virat], ids=[virat_id])
    print("Updated Virat Kohli's document using mapped ID")

# =============================================================================
# METHOD 7: UTILITY FUNCTIONS FOR ID MANAGEMENT
# =============================================================================
print("\n\nMETHOD 7: Utility functions")

def find_document_id_by_content(vector_store, search_text):
    """Find document ID by searching for specific text in content"""
    all_data = vector_store.get()
    
    for doc_id, content in zip(all_data['ids'], all_data['documents']):
        if search_text.lower() in content.lower():
            return doc_id, content
    return None, None

def find_document_ids_by_metadata(vector_store, metadata_filter):
    """Find document IDs by metadata criteria"""
    all_data = vector_store.get(where=metadata_filter)
    return list(zip(all_data['ids'], all_data['documents']))

# Example usage
doc_id, content = find_document_id_by_content(vector_store, "Jasprit Bumrah")
if doc_id:
    print(f"Found Bumrah's ID: {doc_id}")
    print(f"Content: {content}")

# Find all bowlers
bowler_docs = find_document_ids_by_metadata(vector_store, {"role": "bowler"})
print(f"\nFound {len(bowler_docs)} documents with role 'bowler':")
for doc_id, content in bowler_docs:
    print(f"ID: {doc_id} -> {content[:40]}...")

print("\n=== SUMMARY ===")
print("Key takeaways:")
print("1. Use vector_store.get() to retrieve all documents with IDs")
print("2. Add documents with custom IDs using ids parameter")
print("3. Use where clause to filter documents by metadata")
print("4. Create mapping systems for easier ID management")
print("5. Update documents by adding with the same ID")
print("6. Delete documents using vector_store.delete(ids=[...])")
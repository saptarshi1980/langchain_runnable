from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import shutil, os

# Setup
if os.path.exists("working_filters_db"):
    shutil.rmtree("working_filters_db")

model_dir = "D:/huggingface_models/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=model_dir,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Sample data
docs = [
    Document(
        page_content="Virat Kohli is one of the most successful batsmen in IPL history.",
        metadata={
            "player": "Virat Kohli",
            "team": "Royal Challengers Bangalore",
            "role": "batter",
            "age": 35,
            "matches_played": 237,
            "is_captain": False,
            "batting_average": 37.25,
            "country": "India"
        }
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history.",
        metadata={
            "player": "Rohit Sharma",
            "team": "Mumbai Indians",
            "role": "batter, captain",
            "age": 37,
            "matches_played": 258,
            "is_captain": True,
            "batting_average": 31.17,
            "country": "India"
        }
    ),
    Document(
        page_content="MS Dhoni has led Chennai Super Kings to multiple IPL titles.",
        metadata={
            "player": "MS Dhoni",
            "team": "Chennai Super Kings",
            "role": "batter, captain, wicket keeper",
            "age": 42,
            "matches_played": 264,
            "is_captain": True,
            "batting_average": 39.13,
            "country": "India"
        }
    ),
    Document(
        page_content="Jasprit Bumrah is one of the best fast bowlers in T20 cricket.",
        metadata={
            "player": "Jasprit Bumrah",
            "team": "Mumbai Indians",
            "role": "bowler",
            "age": 30,
            "matches_played": 133,
            "is_captain": False,
            "batting_average": 8.24,
            "country": "India"
        }
    ),
    Document(
        page_content="David Warner is an explosive opener for Delhi Capitals.",
        metadata={
            "player": "David Warner",
            "team": "Delhi Capitals",
            "role": "batter",
            "age": 37,
            "matches_played": 184,
            "is_captain": False,
            "batting_average": 41.59,
            "country": "Australia"
        }
    )
]

# Create vector store
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="working_filters_db",
    collection_name="players"
)

vector_store.add_documents(docs)

print("=" * 70)
print("CHROMADB FILTERS THAT ACTUALLY WORK")
print("=" * 70)

# ============================================================================
# ✅ WORKING FILTERS
# ============================================================================

print("\n1. ✅ EXACT MATCH FILTERS")
print("-" * 40)

mumbai_players = vector_store.similarity_search(
    "Mumbai players",
    k=10,
    filter={"where": {"team": {"$eq": "Mumbai Indians"}}}
)
print("Mumbai Indians players:")
for doc in mumbai_players:
    print(f"  - {doc.metadata['player']}")

captains = vector_store.similarity_search(
    "Captains",
    k=10,
    filter={"where": {"is_captain": {"$eq": True}}}
)
print("\nCaptains:")
for doc in captains:
    print(f"  - {doc.metadata['player']}")

print("\n2. ✅ SINGLE NUMERICAL COMPARISONS")
print("-" * 40)

experienced = vector_store.similarity_search(
    "Experienced players",
    k=10,
    filter={"where": {"matches_played": {"$gt": 200}}}
)
print("Players with >200 matches:")
for doc in experienced:
    print(f"  - {doc.metadata['player']}: {doc.metadata['matches_played']}")

young_players = vector_store.similarity_search(
    "Young players",
    k=10,
    filter={"where": {"age": {"$lt": 35}}}
)
print("\nPlayers under 35:")
for doc in young_players:
    print(f"  - {doc.metadata['player']}: {doc.metadata['age']}")

print("\n3. ✅ STRING CONTAINS (if supported)")
print("-" * 40)

try:
    multi_role = vector_store.similarity_search(
        "Multi-role players",
        k=10,
        filter={"where": {"role": {"$regex": ".*,.*"}}}
    )
    print("Multi-role players:")
    for doc in multi_role:
        print(f"  - {doc.metadata['player']}: {doc.metadata['role']}")
except Exception as e:
    print(f"Regex not supported: {e}")
    all_results = vector_store.similarity_search("players", k=10)
    multi_role = [doc for doc in all_results if "," in doc.metadata.get("role", "")]
    print("Multi-role players (post-processed):")
    for doc in multi_role:
        print(f"  - {doc.metadata['player']}: {doc.metadata['role']}")

print("\n4. ✅ LIST MEMBERSHIP")
print("-" * 40)

specific_teams = vector_store.similarity_search(
    "Specific teams",
    k=10,
    filter={"where": {"team": {"$in": ["Mumbai Indians", "Chennai Super Kings"]}}}
)
print("MI or CSK players:")
for doc in specific_teams:
    print(f"  - {doc.metadata['player']} ({doc.metadata['team']})")

print("\n5. ✅ SIMPLE AND CONDITIONS")
print("-" * 40)

indian_captains = vector_store.similarity_search(
    "Indian captains",
    k=10,
    filter={"where": {"country": {"$eq": "India"}, "is_captain": {"$eq": True}}}
)
print("Indian captains:")
for doc in indian_captains:
    print(f"  - {doc.metadata['player']}")

# ============================================================================
# ❌ PROBLEMATIC FILTERS (with workarounds)
# ============================================================================

print("\n" + "=" * 70)
print("WORKAROUNDS FOR UNSUPPORTED FILTERS")
print("=" * 70)

print("\n❌ RANGE QUERIES (150 <= matches <= 250)")
print("✅ WORKAROUND: Use post-processing")
print("-" * 50)

broad_results = vector_store.similarity_search("players", k=10)
range_filtered = [doc for doc in broad_results if 150 <= doc.metadata.get("matches_played", 0) <= 250]
print("Players with 150-250 matches:")
for doc in range_filtered:
    print(f"  - {doc.metadata['player']}: {doc.metadata['matches_played']}")

print("\n❌ OR CONDITIONS")
print("✅ WORKAROUND: Multiple queries + merge")
print("-" * 50)

try:
    or_results = vector_store.similarity_search(
        "Young or foreign",
        k=10,
        filter={"where": {"$or": [{"age": {"$lt": 32}}, {"country": "Australia"}]}}
    )
    print("Young or foreign players (native $or):")
    for doc in or_results:
        print(f"  - {doc.metadata['player']}: {doc.metadata['age']} yrs, {doc.metadata['country']}")
except Exception as e:
    print(f"Native $or not supported: {e}")
    young = vector_store.similarity_search("young", k=10, filter={"where": {"age": {"$lt": 32}}})
    foreign = vector_store.similarity_search("foreign", k=10, filter={"where": {"country":{"$eq":"Australia"}}})
    seen_players = set()
    or_results = []
    for doc in young + foreign:
        player = doc.metadata['player']
        if player not in seen_players:
            seen_players.add(player)
            or_results.append(doc)
    print("Young or foreign players (merged queries):")
    for doc in or_results:
        print(f"  - {doc.metadata['player']}: {doc.metadata['age']} yrs, {doc.metadata['country']}")

print("\n❌ COMPLEX CONDITIONS")
print("✅ WORKAROUND: Post-processing with custom logic")
print("-" * 50)

all_results = vector_store.similarity_search("cricket players", k=10)

def complex_filter(doc):
    meta = doc.metadata
    return (
        meta.get('country') == 'India' and
        ',' in meta.get('role', '') and
        meta.get('matches_played', 0) > 180 and
        meta.get('age', 0) > 32
    )

complex_results = [doc for doc in all_results if complex_filter(doc)]
print("Complex filtered players:")
for doc in complex_results:
    print(f"  - {doc.metadata['player']}: {doc.metadata['role']} ({doc.metadata['matches_played']} matches)")

# ============================================================================
# 📋 TEMPLATE FOR SAFE FILTERING
# ============================================================================

def safe_filter_search(vector_store, query, simple_filters=None, complex_filter_func=None, k=20):
    if simple_filters:
        results = vector_store.similarity_search(query, k=k, filter={"where": simple_filters})
    else:
        results = vector_store.similarity_search(query, k=k)
    if complex_filter_func:
        results = [doc for doc in results if complex_filter_func(doc)]
    return results

def my_complex_filter(doc):
    meta = doc.metadata
    return meta.get('age', 0) > 30 and meta.get('batting_average', 0) > 25

final_results = safe_filter_search(
    vector_store=vector_store,
    query="good players",
    simple_filters={"country": "India"},
    complex_filter_func=my_complex_filter,
    k=10
)

print("\nSafe filtered results:")
for doc in final_results:
    print(f"  - {doc.metadata['player']}: {doc.metadata['age']} yrs, {doc.metadata['batting_average']} avg")

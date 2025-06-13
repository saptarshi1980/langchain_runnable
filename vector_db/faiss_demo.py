import sys
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

def verify_faiss_installation():
    """Verify FAISS is properly installed without circular imports"""
    try:
        # Check if there's any local file named faiss.py that might cause conflicts
        if Path('faiss.py').exists():
            raise ImportError("Found 'faiss.py' in working directory - this conflicts with the FAISS package. Please rename it.")
        
        # Test basic FAISS functionality
        import faiss
        try:
            # Test creating a simple index
            index = faiss.IndexFlatL2(128)  # Test with dummy dimension
            print("✓ FAISS installed correctly")
            return True
        except AttributeError as e:
            print(f"FAISS installation problem: {e}")
            print("Try: pip uninstall faiss faiss-cpu faiss-gpu --yes && pip install faiss-cpu --no-cache-dir")
            return False
    except ImportError as e:
        print(f"FAISS not installed: {e}")
        print("Install with: pip install faiss-cpu")
        return False

# Verify FAISS before proceeding
if not verify_faiss_installation():
    sys.exit(1)

# Initialize embeddings
model_dir = "D:/huggingface_models/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=model_dir,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Document collection with enhanced metadata
ipl_players = [
    {
        "name": "Virat Kohli",
        "content": "Virat Kohli is one of the most successful and consistent batsmen in IPL history.",
        "team": "Royal Challengers Bangalore",
        "role": "batsman",
        "stats": {"matches": 223, "runs": 6624, "avg": 36.20, "titles": 0}
    },
    {
        "name": "Rohit Sharma", 
        "content": "Rohit Sharma is the most successful captain in IPL history with 5 titles.",
        "team": "Mumbai Indians",
        "role": "batsman, captain",
        "stats": {"matches": 227, "runs": 5879, "avg": 30.30, "titles": 5}
    },
    {
        "name": "MS Dhoni",
        "content": "MS Dhoni has led Chennai Super Kings to 4 IPL titles with his calm captaincy.",
        "team": "Chennai Super Kings",
        "role": "wicket-keeper, captain",
        "stats": {"matches": 234, "runs": 4978, "avg": 39.20, "titles": 4}
    },
    {
        "name": "Jasprit Bumrah",
        "content": "Jasprit Bumrah is the most lethal fast bowler in IPL with his yorkers.",
        "team": "Mumbai Indians",
        "role": "bowler",
        "stats": {"matches": 120, "wickets": 145, "econ": 7.39, "titles": 5}
    },
    {
        "name": "Ravindra Jadeja",
        "content": "Ravindra Jadeja is a complete all-rounder contributing with bat, ball and fielding.",
        "team": "Chennai Super Kings",
        "role": "all-rounder",
        "stats": {"matches": 210, "runs": 2502, "wickets": 132, "titles": 4}
    }
]

# Convert to LangChain Documents
documents = [
    Document(
        page_content=player["content"],
        metadata={
            "player": player["name"],
            "team": player["team"],
            "role": player["role"],
            "matches": player["stats"]["matches"],
            "performance": f"{player['stats'].get('runs', 'N/A')} runs" if 'runs' in player['stats'] else f"{player['stats'].get('wickets', 'N/A')} wickets"
        }
    ) for player in ipl_players
]

class IPLPlayerSearch:
    def __init__(self, documents, embeddings, index_name="ipl_players_faiss"):
        self.index_name = index_name
        self.embeddings = embeddings
        self.db = self._initialize_db(documents)
    
    def _initialize_db(self, documents):
        """Initialize the FAISS vector store with error handling"""
        try:
            if Path(f"{self.index_name}/index.faiss").exists():
                print(f"Loading existing index from {self.index_name}")
                return FAISS.load_local(self.index_name, self.embeddings)
            else:
                print("Creating new FAISS index")
                db = FAISS.from_documents(documents, self.embeddings)
                db.save_local(self.index_name)
                return db
        except Exception as e:
            print(f"Error initializing FAISS: {e}")
            raise
    
    def search(self, query, team=None, role=None, min_matches=None, k=3):
        """Search with multiple filters"""
        # First get vector similarity results
        results = self.db.similarity_search(query, k=k)
        
        # Apply filters
        filtered = []
        for doc in results:
            match = True
            
            # Team filter (exact match)
            if team and doc.metadata["team"].lower() != team.lower():
                match = False
                
            # Role filter (substring match)
            if role and role.lower() not in doc.metadata["role"].lower():
                match = False
                
            # Minimum matches filter
            if min_matches and doc.metadata.get("matches", 0) < min_matches:
                match = False
                
            if match:
                filtered.append(doc)
        
        return filtered or results  # Return filtered results or all if no filters
    
    def display_results(self, results, query):
        """Display results in a formatted way"""
        print(f"\n🔍 Search results for '{query}':")
        print("=" * 80)
        for i, doc in enumerate(results, 1):
            print(f"\n🏏 Player #{i}: {doc.metadata['player']}")
            print(f"🏷️ Team: {doc.metadata['team']}")
            print(f"👤 Role: {doc.metadata['role']}")
            print(f"📊 Stats: {doc.metadata['performance']} in {doc.metadata['matches']} matches")
            print(f"📝 Description: {doc.page_content}")
        print("=" * 80)

# Main execution
if __name__ == "__main__":
    try:
        search_system = IPLPlayerSearch(documents, embeddings)
        
        # Example searches
        searches = [
            ("most successful players", None, None, None),
            ("best bowlers", None, "bowler", None),
            ("experienced CSK players", "Chennai Super Kings", None, 200),
            ("Mumbai Indians captains", "Mumbai Indians", "captain", None),
            ("top all-rounders", None, "all-rounder", 150)
        ]
        
        for query, team, role, matches in searches:
            results = search_system.search(
                query=query,
                team=team,
                role=role,
                min_matches=matches
            )
            search_system.display_results(results, query)
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure no file named 'faiss.py' exists in your directory")
        print("2. Try: pip uninstall faiss faiss-cpu faiss-gpu")
        print("3. Then: pip install faiss-cpu --no-cache-dir")
        print("4. Restart your Python environment")
from document_indexer import DocumentIndexer
from vector_search_agent import VectorSearchAgent
import os
from typing import List, Dict, Any

def print_introduction():
    """Print the RAG agent's introduction."""
    intro = """
🤖 Welcome to the Research Papers RAG Agent!
==========================================
I'm your research assistant, powered by the multilingual-e5-large-instruct model.
I can help you find relevant research papers from our collection about AI, LLMs, and related topics.

I use FAISS (Facebook AI Similarity Search) for efficient vector similarity search,
and I'll show you:
- Top 3 most relevant papers
- Similarity scores for each match
- Detailed paper information

Example queries you might try:
- "What papers discuss LLM evaluation?"
- "Show me research about AI in education"
- "Tell me about fact-checking systems"
- "Find papers about prompt engineering"

Let's get started! 
"""
    print(intro)

def print_search_process(query: str, results: List[Dict[str, Any]]):
    """Print detailed information about the search process."""
    print("\n🔍 Search Process")
    print("===============")
    print(f"Query: '{query}'")
    print("Using model: intfloat/multilingual-e5-large-instruct")
    print("Vector similarity metric: L2 distance (converted to similarity score)")
    
    print("\n📊 Similarity Scores")
    print("=================")
    for i, result in enumerate(results, 1):
        score = result['score']
        title = result['metadata']['title']
        print(f"{i}. Score: {score:.4f} - {title[:60]}...")

def create_index():
    """Create the search index from the research papers."""
    indexer = DocumentIndexer()
    indexer.process_and_index_documents(
        json_path="../data/research_papers.json",
        save_dir="../data/index"
    )
    print("✅ Index created successfully!")

def test_search():
    """Test the search functionality."""
    agent = VectorSearchAgent()
    agent.load_index("../data/index")
    
    print_introduction()
    
    while True:
        query = input("\n🤔 What would you like to know about? (type 'quit' to exit): ")
        if query.lower() == 'quit':
            print("\n👋 Thanks for using the Research Papers RAG Agent! Goodbye!")
            break
            
        print("\n🔄 Processing your query...")
        results = agent.search(query, top_k=3)
        
        # Print search process details
        print_search_process(query, results)
        
        # Print detailed results
        print("\n📑 Detailed Results")
        print("================")
        for i, result in enumerate(results, 1):
            print(f"\nResult #{i}")
            print("=" * 80)
            formatted = agent.format_result(result)
            print(formatted)

if __name__ == "__main__":
    # Create index directory if it doesn't exist
    os.makedirs("../data/index", exist_ok=True)
    
    # Check if index exists
    if not os.path.exists("../data/index/papers.index"):
        print("⚙️ First-time setup: Creating search index...")
        create_index()
    
    # Run search interface
    test_search() 
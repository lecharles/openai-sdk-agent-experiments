import json
import numpy as np
from typing import List, Dict, Any, Tuple
import sentence_transformers
import faiss
from pathlib import Path

class VectorSearchAgent:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct"):
        """Initialize the vector search agent.
        
        Args:
            model_name (str): Name of the sentence-transformer model to use
        """
        self.model = sentence_transformers.SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.metadata = []
        
    def load_index(self, index_dir: str) -> None:
        """Load the FAISS index and metadata from disk.
        
        Args:
            index_dir (str): Directory containing the index and metadata
        """
        index_dir = Path(index_dir)
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_dir / "papers.index"))
        
        # Load metadata
        with open(index_dir / "metadata.json", 'r') as f:
            data = json.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for documents similar to the query.
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of top_k most similar documents with scores
        """
        # Generate query embedding
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding.astype(np.float32)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for score, index in zip(distances[0], indices[0]):
            results.append({
                'score': 1 / (1 + score),  # Convert distance to similarity score
                'document': self.documents[index],
                'metadata': self.metadata[index]
            })
            
        return results
    
    def format_result(self, result: Dict[str, Any]) -> str:
        """Format a search result for display.
        
        Args:
            result (Dict[str, Any]): Search result to format
            
        Returns:
            str: Formatted result string
        """
        metadata = result['metadata']
        confidence = f"{result['score']:.2%}"
        
        formatted = f"Confidence: {confidence}\n"
        formatted += f"Title: {metadata['title']}\n"
        formatted += f"Authors: {', '.join(metadata['authors'])}\n"
        formatted += f"Year: {metadata['year']}"
        if metadata['month']:
            formatted += f", Month: {metadata['month']}"
        formatted += f"\nTechnique: {metadata['technique_type']}\n"
        if metadata['technique_description']:
            formatted += f"Technique Description: {metadata['technique_description']}\n"
        formatted += f"Summary: {metadata['summary']}\n"
        
        return formatted

    def query_and_format(self, query: str, top_k: int = 3) -> List[str]:
        """Search for documents and return formatted results.
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            List[str]: List of formatted result strings
        """
        results = self.search(query, top_k)
        return [self.format_result(result) for result in results] 
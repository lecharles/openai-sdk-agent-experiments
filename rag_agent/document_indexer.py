import json
import numpy as np
from typing import List, Dict, Any
import sentence_transformers
import faiss
import os
from pathlib import Path

class DocumentIndexer:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct"):
        """Initialize the document indexer with a specific embedding model.
        
        Args:
            model_name (str): Name of the sentence-transformer model to use
        """
        self.model = sentence_transformers.SentenceTransformer(model_name)
        self.documents = []
        self.metadata = []
        self.index = None
        
    def load_documents(self, json_path: str) -> None:
        """Load documents from a JSON file.
        
        Args:
            json_path (str): Path to the JSON file containing the documents
        """
        with open(json_path, 'r') as file:
            data = json.load(file)
            self.documents = []
            self.metadata = []
            
            for paper in data['top_papers']:
                # Create a text representation of the paper
                text = f"Title: {paper['title']}\n"
                text += f"Authors: {', '.join(paper['authors'])}\n"
                text += f"Year: {paper['year']}, Month: {paper['month']}\n"
                text += f"Technique: {paper['technique_type']}\n"
                if paper['technique_description']:
                    text += f"Technique Description: {paper['technique_description']}\n"
                text += f"Summary: {paper['summary']}"
                
                self.documents.append(text)
                self.metadata.append(paper)
    
    def generate_embeddings(self) -> np.ndarray:
        """Generate embeddings for all loaded documents.
        
        Returns:
            np.ndarray: Document embeddings
        """
        embeddings = self.model.encode(self.documents, show_progress_bar=True)
        return embeddings.astype(np.float32)
    
    def create_index(self, embeddings: np.ndarray) -> None:
        """Create a FAISS index from document embeddings.
        
        Args:
            embeddings (np.ndarray): Document embeddings
        """
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
    
    def save_index(self, save_dir: str) -> None:
        """Save the FAISS index and metadata to disk.
        
        Args:
            save_dir (str): Directory to save the index and metadata
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_dir / "papers.index"))
        
        # Save metadata
        with open(save_dir / "metadata.json", 'w') as f:
            json.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f, indent=2)
    
    def process_and_index_documents(self, json_path: str, save_dir: str) -> None:
        """Process documents and create searchable index.
        
        Args:
            json_path (str): Path to the JSON file containing the documents
            save_dir (str): Directory to save the index and metadata
        """
        self.load_documents(json_path)
        embeddings = self.generate_embeddings()
        self.create_index(embeddings)
        self.save_index(save_dir) 
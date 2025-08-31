"""Vector database operations for RAG system."""

import os
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import faiss
import pickle
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorStore:
    """Base class for vector database operations."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
    
    def embed_text(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text list."""
        return self.embedding_model.encode(texts)
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to vector store."""
        raise NotImplementedError
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    """ChromaDB-based vector store implementation."""
    
    def __init__(self, persist_directory: str, collection_name: str = "argo_summaries", 
                 model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized ChromaDB at {persist_directory}")
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to ChromaDB collection.
        
        Args:
            documents: List of dicts with 'id', 'text', and 'metadata' keys
        """
        if not documents:
            return
        
        texts = [doc['text'] for doc in documents]
        embeddings = self.embed_text(texts)
        
        ids = [doc['id'] for doc in documents]
        metadatas = [doc.get('metadata', {}) for doc in documents]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    def search(self, query: str, k: int = 5, where: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents in ChromaDB.
        
        Args:
            query: Search query text
            k: Number of results to return
            where: Optional metadata filter
            
        Returns:
            List of matching documents with scores
        """
        query_embedding = self.embed_text([query])
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k,
            where=where
        )
        
        documents = []
        for i in range(len(results['ids'][0])):
            documents.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': results['distances'][0][i]
            })
        
        return documents
    
    def delete_collection(self):
        """Delete the collection."""
        self.client.delete_collection(self.collection.name)
        logger.info(f"Deleted collection {self.collection.name}")


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store implementation."""
    
    def __init__(self, index_path: str, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.index_path / "faiss.index"
        self.metadata_file = self.index_path / "metadata.pkl"
        
        # Initialize or load FAISS index
        if self.index_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            with open(self.metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            self.metadata = {}
        
        logger.info(f"Initialized FAISS index at {index_path}")
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to FAISS index.
        
        Args:
            documents: List of dicts with 'id', 'text', and 'metadata' keys
        """
        if not documents:
            return
        
        texts = [doc['text'] for doc in documents]
        embeddings = self.embed_text(texts)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        start_idx = self.index.ntotal
        self.index.add(embeddings)
        
        # Store metadata
        for i, doc in enumerate(documents):
            self.metadata[start_idx + i] = {
                'id': doc['id'],
                'text': doc['text'],
                'metadata': doc.get('metadata', {})
            }
        
        # Save index and metadata
        self._save_index()
        
        logger.info(f"Added {len(documents)} documents to FAISS index")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents in FAISS index.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of matching documents with scores
        """
        if self.index.ntotal == 0:
            return []
        
        query_embedding = self.embed_text([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        documents = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx in self.metadata:
                doc_data = self.metadata[idx]
                documents.append({
                    'id': doc_data['id'],
                    'text': doc_data['text'],
                    'metadata': doc_data['metadata'],
                    'score': float(score)
                })
        
        return documents
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_file))
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)


class ArgoVectorManager:
    """Manager for ARGO data vector operations."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def index_float_summaries(self, summaries: List[Dict]):
        """Index float summary data for RAG retrieval.
        
        Args:
            summaries: List of float summary dictionaries
        """
        documents = []
        
        for summary in summaries:
            # Create comprehensive text for embedding
            text_parts = [
                f"ARGO Float {summary['platform_number']}",
                summary.get('summary_text', ''),
                f"Location: {summary['spatial_range']['latitude'][0]:.2f}°N to {summary['spatial_range']['latitude'][1]:.2f}°N, "
                f"{summary['spatial_range']['longitude'][0]:.2f}°E to {summary['spatial_range']['longitude'][1]:.2f}°E",
                f"Time period: {summary['temporal_range']['start']} to {summary['temporal_range']['end']}",
                f"Total profiles: {summary['total_profiles']}"
            ]
            
            if summary.get('depth_range'):
                text_parts.append(f"Depth: {summary['depth_range'][0]:.1f}m to {summary['depth_range'][1]:.1f}m")
            
            if summary.get('temperature_range'):
                text_parts.append(f"Temperature: {summary['temperature_range'][0]:.2f}°C to {summary['temperature_range'][1]:.2f}°C")
            
            if summary.get('salinity_range'):
                text_parts.append(f"Salinity: {summary['salinity_range'][0]:.2f} to {summary['salinity_range'][1]:.2f} PSU")
            
            full_text = " | ".join(text_parts)
            
            documents.append({
                'id': f"float_{summary['platform_number']}",
                'text': full_text,
                'metadata': {
                    'platform_number': summary['platform_number'],
                    'type': 'float_summary',
                    'total_profiles': summary['total_profiles'],
                    'spatial_bounds': summary['spatial_range'],
                    'temporal_bounds': summary['temporal_range']
                }
            })
        
        self.vector_store.add_documents(documents)
        logger.info(f"Indexed {len(documents)} float summaries")
    
    def search_relevant_floats(self, query: str, k: int = 5) -> List[Dict]:
        """Search for floats relevant to a natural language query.
        
        Args:
            query: Natural language query
            k: Number of results to return
            
        Returns:
            List of relevant float information
        """
        results = self.vector_store.search(query, k)
        
        # Extract platform numbers and metadata
        relevant_floats = []
        for result in results:
            relevant_floats.append({
                'platform_number': result['metadata']['platform_number'],
                'relevance_score': result['score'],
                'summary_text': result['text'],
                'spatial_bounds': result['metadata'].get('spatial_bounds'),
                'temporal_bounds': result['metadata'].get('temporal_bounds'),
                'total_profiles': result['metadata'].get('total_profiles')
            })
        
        return relevant_floats
    
    def add_region_summary(self, region_name: str, bounds: Dict, description: str):
        """Add a regional summary for geographic queries.
        
        Args:
            region_name: Name of the region (e.g., "Arabian Sea", "Equatorial Pacific")
            bounds: Dictionary with lat/lon bounds
            description: Descriptive text about the region
        """
        text = f"Region: {region_name} | {description} | "
        text += f"Bounds: {bounds['min_lat']:.2f}°N to {bounds['max_lat']:.2f}°N, "
        text += f"{bounds['min_lon']:.2f}°E to {bounds['max_lon']:.2f}°E"
        
        document = {
            'id': f"region_{region_name.lower().replace(' ', '_')}",
            'text': text,
            'metadata': {
                'type': 'region',
                'region_name': region_name,
                'bounds': bounds
            }
        }
        
        self.vector_store.add_documents([document])
        logger.info(f"Added region summary for {region_name}")


def create_vector_store(store_type: str = "chroma", **kwargs) -> VectorStore:
    """Factory function to create vector store instances.
    
    Args:
        store_type: Type of vector store ("chroma" or "faiss")
        **kwargs: Additional arguments for store initialization
        
    Returns:
        VectorStore instance
    """
    if store_type.lower() == "chroma":
        return ChromaVectorStore(**kwargs)
    elif store_type.lower() == "faiss":
        return FAISSVectorStore(**kwargs)
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")

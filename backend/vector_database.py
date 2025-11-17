"""
Vector Database Module
=======================
This module manages the vector database using ChromaDB for storing and retrieving
image embeddings along with metadata.

Learning Points:
- What is a vector database and why we need it
- How ChromaDB stores and indexes vectors
- Similarity search with cosine distance
- Metadata filtering and hybrid search
- CRUD operations for vector databases
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

from backend.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    Vector database manager using ChromaDB.
    
    Learning Notes:
    - Vector databases are optimized for similarity search
    - Unlike traditional databases that match exact values,
      vector databases find "similar" items based on distance metrics
    - Common distance metrics: Cosine, Euclidean, Dot Product
    """
    
    def __init__(self):
        """
        Initialize ChromaDB client and collection.
        
        Learning Points:
        - ChromaDB persists data to disk automatically
        - Collections are like tables in traditional databases
        - Each item has: id, embedding, metadata, document (optional text)
        """
        logger.info("Initializing Vector Database (ChromaDB)...")
        
        # Create persistent client
        # This saves the database to disk so data persists between runs
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        # Collections store related vectors together
        try:
            self.collection = self.client.get_collection(
                name=settings.collection_name
            )
            logger.info(f"✓ Loaded existing collection: {settings.collection_name}")
            logger.info(f"  Current items: {self.collection.count()}")
        except Exception:
            self.collection = self.client.create_collection(
                name=settings.collection_name,
                metadata={
                    "description": "Image embeddings with OCR, objects, and captions",
                    "hnsw:space": "cosine"  # Use cosine similarity
                }
            )
            logger.info(f"✓ Created new collection: {settings.collection_name}")
    
    def add_image(
        self, 
        image_id: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
        combined_text: str
    ) -> bool:
        """
        Add an image embedding to the database.
        
        Args:
            image_id: Unique identifier for the image
            embedding: Image embedding vector (512-dim from CLIP)
            metadata: Dictionary containing OCR text, objects, caption, etc.
            combined_text: Combined text from all sources (for text search)
            
        Returns:
            True if successful, False otherwise
            
        Learning Points:
        - embeddings: The actual vector (list of floats)
        - metadatas: Structured information (must be JSON-serializable)
        - documents: Optional text for text-based search
        - ids: Unique identifier for retrieval
        """
        try:
            # Convert numpy array to list for ChromaDB
            embedding_list = embedding.tolist()
            
            # Prepare metadata (ChromaDB requires JSON-serializable values)
            clean_metadata = self._clean_metadata(metadata)
            clean_metadata['indexed_at'] = datetime.now().isoformat()
            
            # Add to collection
            self.collection.add(
                embeddings=[embedding_list],
                metadatas=[clean_metadata],
                documents=[combined_text],
                ids=[image_id]
            )
            
            logger.info(f"✓ Added image to database: {image_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding image to database: {e}")
            return False
    
    def add_images_batch(
        self,
        image_ids: List[str],
        embeddings: List[np.ndarray],
        metadatas: List[Dict[str, Any]],
        combined_texts: List[str]
    ) -> bool:
        """
        Add multiple images in a batch (more efficient).
        
        Learning Points:
        - Batch operations are much faster than individual adds
        - ChromaDB handles transactions internally
        - Important for large datasets
        """
        try:
            # Convert embeddings to lists
            embedding_lists = [emb.tolist() for emb in embeddings]
            
            # Clean metadatas
            clean_metadatas = [self._clean_metadata(meta) for meta in metadatas]
            for meta in clean_metadatas:
                meta['indexed_at'] = datetime.now().isoformat()
            
            # Batch add
            self.collection.add(
                embeddings=embedding_lists,
                metadatas=clean_metadatas,
                documents=combined_texts,
                ids=image_ids
            )
            
            logger.info(f"✓ Added {len(image_ids)} images to database in batch")
            return True
            
        except Exception as e:
            logger.error(f"Error in batch add: {e}")
            return False
    
    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar images using vector similarity.
        
        Args:
            query_embedding: Query vector (from text or image)
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"has_text": True})
            
        Returns:
            List of similar images with scores and metadata
            
        Learning Points:
        - Cosine similarity: Measures angle between vectors (0-1, higher = more similar)
        - ChromaDB uses HNSW index for fast approximate nearest neighbor search
        - Filter helps narrow down results based on metadata
        """
        try:
            if top_k is None:
                top_k = settings.top_k_results
            
            # Convert embedding to list
            query_list = query_embedding.tolist()
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_list],
                n_results=top_k,
                where=filter_metadata,  # Optional metadata filtering
                include=['embeddings', 'metadatas', 'documents', 'distances']
            )
            
            # Parse results
            search_results = []
            if results['ids'][0]:  # Check if any results
                for i in range(len(results['ids'][0])):
                    search_results.append({
                        'id': results['ids'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'document': results['documents'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity_score': 1 - results['distances'][0][i]  # Convert distance to similarity
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching database: {e}")
            return []
    
    def search_by_text(
        self,
        query_text: str,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search using text query (leverages ChromaDB's built-in text search).
        
        Learning Points:
        - Text search uses the 'documents' field
        - Can be combined with vector search for hybrid approach
        - Good for exact keyword matching
        """
        try:
            if top_k is None:
                top_k = settings.top_k_results
            
            # ChromaDB doesn't have built-in text search, so we'll use
            # the embedding search with a text embedding instead
            # This should be done via the ImageProcessor's text embedding
            logger.warning("Use search_by_embedding with text_embedding for semantic search")
            return []
            
        except Exception as e:
            logger.error(f"Error in text search: {e}")
            return []
    
    def get_by_id(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific image by ID.
        
        Args:
            image_id: Image identifier
            
        Returns:
            Image data or None if not found
        """
        try:
            result = self.collection.get(
                ids=[image_id],
                include=['embeddings', 'metadatas', 'documents']
            )
            
            if result['ids']:
                return {
                    'id': result['ids'][0],
                    'metadata': result['metadatas'][0],
                    'document': result['documents'][0],
                    'embedding': result['embeddings'][0]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving image: {e}")
            return None
    
    def delete_image(self, image_id: str) -> bool:
        """
        Delete an image from the database.
        
        Args:
            image_id: Image identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[image_id])
            logger.info(f"✓ Deleted image: {image_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting image: {e}")
            return False
    
    def get_all_ids(self) -> List[str]:
        """
        Get all image IDs in the database.
        
        Returns:
            List of image IDs
        """
        try:
            result = self.collection.get(include=[])
            return result['ids']
        except Exception as e:
            logger.error(f"Error getting IDs: {e}")
            return []
    
    def count(self) -> int:
        """
        Get the total number of images in the database.
        
        Returns:
            Count of images
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error counting images: {e}")
            return 0
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            total_images = self.count()
            
            # Get sample of metadata to analyze
            sample = self.collection.get(
                limit=min(100, total_images),
                include=['metadatas']
            )
            
            # Count images with text, objects, captions
            has_text = sum(1 for m in sample['metadatas'] if m.get('has_text', False))
            has_objects = sum(1 for m in sample['metadatas'] if m.get('object_count', 0) > 0)
            has_caption = sum(1 for m in sample['metadatas'] if m.get('has_caption', False))
            
            sample_size = len(sample['metadatas'])
            
            return {
                'total_images': total_images,
                'sample_size': sample_size,
                'images_with_text_pct': (has_text / sample_size * 100) if sample_size > 0 else 0,
                'images_with_objects_pct': (has_objects / sample_size * 100) if sample_size > 0 else 0,
                'images_with_caption_pct': (has_caption / sample_size * 100) if sample_size > 0 else 0,
                'collection_name': settings.collection_name,
                'persist_directory': settings.chroma_persist_directory
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}
    
    def reset_database(self) -> bool:
        """
        Delete all data and reset the database (use with caution!).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(name=settings.collection_name)
            self.collection = self.client.create_collection(
                name=settings.collection_name,
                metadata={
                    "description": "Image embeddings with OCR, objects, and captions",
                    "hnsw:space": "cosine"
                }
            )
            logger.info("✓ Database reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            return False
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata to ensure it's JSON-serializable for ChromaDB.
        
        Learning Points:
        - ChromaDB metadata must be: str, int, float, or bool
        - Complex objects need to be converted to strings
        - Nested dicts need to be flattened or stringified
        """
        clean = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, (list, dict)):
                # Convert to JSON string
                clean[key] = json.dumps(value)
            elif value is None:
                clean[key] = ""
            else:
                clean[key] = str(value)
        
        return clean


# Global instance
_database_instance = None

def get_vector_database() -> VectorDatabase:
    """
    Get or create the global VectorDatabase instance.
    
    Learning Note:
    - Singleton pattern ensures single database connection
    - Improves performance and prevents connection issues
    """
    global _database_instance
    if _database_instance is None:
        _database_instance = VectorDatabase()
    return _database_instance


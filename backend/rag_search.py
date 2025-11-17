"""
RAG Search Module
=================
This module implements Retrieval-Augmented Generation (RAG) for intelligent
image search using LangChain and NVIDIA AI Endpoints.

Learning Points:
- What is RAG and why it's powerful for search
- How to combine vector search with LLMs
- Query understanding and expansion
- Re-ranking search results
- Context-aware responses
"""

from typing import List, Dict, Any, Optional
import logging
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import numpy as np

from backend.config import settings
from backend.vector_database import get_vector_database
from backend.image_processor import get_image_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSearchEngine:
    """
    RAG-based search engine for intelligent image retrieval.
    
    Learning Notes:
    RAG (Retrieval-Augmented Generation) combines:
    1. Retrieval: Find relevant information from a knowledge base
    2. Generation: Use an LLM to understand queries and provide intelligent responses
    
    Benefits:
    - Better query understanding (handles typos, synonyms, intent)
    - Query expansion (search for related concepts)
    - Context-aware results
    - Natural language interaction
    """
    
    def __init__(self):
        """
        Initialize the RAG search engine with NVIDIA LLM.
        
        Learning Points:
        - LangChain abstracts LLM interactions
        - NVIDIA AI Endpoints provide fast inference
        - Temperature controls randomness (0=deterministic, 1=creative)
        """
        logger.info("Initializing RAG Search Engine...")
        
        # Initialize components
        self.vector_db = get_vector_database()
        self.image_processor = get_image_processor()
        
        # Initialize NVIDIA LLM if API key is provided
        self.llm = None
        if settings.nvidia_api_key:
            try:
                self.llm = ChatNVIDIA(
                    model=settings.nvidia_model_name,
                    api_key=settings.nvidia_api_key,
                    temperature=0.2,  # Low temperature for consistent results
                    max_tokens=500
                )
                logger.info(f"✓ NVIDIA LLM initialized with model: {settings.nvidia_model_name}")
            except Exception as e:
                logger.warning(f"Could not initialize NVIDIA LLM: {e}")
                logger.warning("Falling back to simple vector search")
        else:
            logger.warning("No NVIDIA API key provided. Using simple vector search.")
    
    def search(
        self,
        query: str,
        top_k: int = None,
        use_llm_enhancement: bool = True,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main search function with RAG capabilities.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            use_llm_enhancement: Whether to use LLM for query understanding
            filter_metadata: Optional metadata filters
            
        Returns:
            Search results with images and explanations
            
        Learning Points:
        - Query understanding: LLM interprets user intent
        - Embedding-based search: Finds semantically similar images
        - Re-ranking: LLM can re-order results based on relevance
        """
        try:
            if top_k is None:
                top_k = settings.top_k_results
            
            logger.info(f"Searching for: '{query}'")
            
            # Step 1: Query Enhancement (if LLM available)
            enhanced_query = query
            search_keywords = []
            
            if use_llm_enhancement and self.llm:
                enhancement_result = self._enhance_query(query)
                enhanced_query = enhancement_result.get('enhanced_query', query)
                search_keywords = enhancement_result.get('keywords', [])
                logger.info(f"Enhanced query: '{enhanced_query}'")
                logger.info(f"Keywords: {search_keywords}")
            
            # Step 2: Generate embeddings for both original and enhanced query
            # This provides better semantic matching
            original_embedding = self.image_processor.generate_text_embedding(query)
            enhanced_embedding = self.image_processor.generate_text_embedding(enhanced_query)
            
            # Blend both embeddings for more robust search (70% enhanced, 30% original)
            import numpy as np
            query_embedding = 0.7 * enhanced_embedding + 0.3 * original_embedding
            
            # Normalize the blended embedding
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Step 3: Vector search (get more for filtering)
            search_results = self.vector_db.search_by_embedding(
                query_embedding=query_embedding,
                top_k=top_k * 3,  # Get more results for filtering and re-ranking
                filter_metadata=filter_metadata
            )
            
            if not search_results:
                return {
                    'query': query,
                    'enhanced_query': enhanced_query,
                    'keywords': search_keywords,
                    'results': [],
                    'message': 'No matching images found'
                }
            
            # Multi-stage filtering for better results
            from backend.config import settings
            
            # Stage 1: Filter by similarity threshold
            filtered_results = [
                r for r in search_results 
                if r.get('similarity_score', 0) >= settings.similarity_threshold
            ]
            
            # Stage 2: Boost results that match multiple criteria
            for result in filtered_results:
                metadata = result.get('metadata', {})
                boost_score = 0.0
                
                # Check if query keywords appear in OCR text
                ocr_text = metadata.get('ocr_text', '').lower()
                if ocr_text:
                    for keyword in search_keywords:
                        if keyword.lower() in ocr_text:
                            boost_score += 0.05
                
                # Check if query keywords appear in caption
                caption = metadata.get('caption', '').lower()
                if caption:
                    for keyword in search_keywords:
                        if keyword.lower() in caption:
                            boost_score += 0.03
                
                # Check if query keywords appear in object labels
                try:
                    import json
                    object_labels_str = metadata.get('object_labels', '[]')
                    object_labels = json.loads(object_labels_str) if isinstance(object_labels_str, str) else object_labels_str
                    if isinstance(object_labels, list):
                        for keyword in search_keywords:
                            for obj in object_labels:
                                if keyword.lower() in str(obj).lower():
                                    boost_score += 0.04
                except:
                    pass
                
                # Apply boost (capped at 0.2 = 20% increase)
                result['similarity_score'] = min(1.0, result['similarity_score'] + min(boost_score, 0.2))
                result['boosted'] = boost_score > 0
            
            # Re-sort by boosted scores
            filtered_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            
            if not filtered_results:
                return {
                    'query': query,
                    'enhanced_query': enhanced_query,
                    'keywords': search_keywords,
                    'results': [],
                    'message': f'No images found with similarity >= {settings.similarity_threshold:.0%}'
                }
            
            search_results = filtered_results
            
            # Step 4: Re-rank results (if LLM available)
            if use_llm_enhancement and self.llm and len(search_results) > 0:
                search_results = self._rerank_results(
                    query, 
                    enhanced_query,
                    search_results
                )
            
            # Step 5: Take top K after re-ranking
            final_results = search_results[:top_k]
            
            # Step 6: Generate explanation (if LLM available)
            explanation = ""
            if use_llm_enhancement and self.llm:
                explanation = self._generate_explanation(query, final_results)
            
            return {
                'query': query,
                'enhanced_query': enhanced_query,
                'keywords': search_keywords,
                'results': final_results,
                'explanation': explanation,
                'total_found': len(search_results)
            }
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {
                'query': query,
                'results': [],
                'error': str(e)
            }
    
    def _enhance_query(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to enhance and expand the search query.
        
        Learning Points:
        - LLMs can understand intent and add related concepts
        - Query expansion improves recall (finding more relevant items)
        - Handles typos, synonyms, and implicit meaning
        """
        try:
            # Create improved prompt for better query enhancement
            prompt = f"""You are an expert image search query optimizer.

User is searching for: "{query}"

Your task:
1. Fix any typos or spelling errors
2. Expand with visual attributes, synonyms, and related concepts
3. Think about what the image would LOOK like (colors, shapes, text, objects, scenes)
4. Generate an enhanced query that describes visual elements
5. Extract key searchable terms

Format your response as:
ENHANCED_QUERY: [detailed visual description in one sentence]
KEYWORDS: [keyword1, keyword2, keyword3, keyword4, keyword5]

Focus on: colors, objects, text content, logos, UI elements, scenes, actions.
Be specific about visual appearance.
"""
            
            # Get LLM response
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse response
            enhanced_query = query  # Default to original
            keywords = []
            
            for line in response_text.split('\n'):
                if line.startswith('ENHANCED_QUERY:'):
                    enhanced_query = line.replace('ENHANCED_QUERY:', '').strip()
                elif line.startswith('KEYWORDS:'):
                    keywords_str = line.replace('KEYWORDS:', '').strip()
                    keywords = [k.strip() for k in keywords_str.split(',')]
            
            return {
                'enhanced_query': enhanced_query,
                'keywords': keywords
            }
            
        except Exception as e:
            logger.error(f"Query enhancement error: {e}")
            return {
                'enhanced_query': query,
                'keywords': []
            }
    
    def _rerank_results(
        self,
        original_query: str,
        enhanced_query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to re-rank search results based on relevance.
        
        Learning Points:
        - Initial vector search is fast but may not be perfect
        - LLM can understand nuanced relevance
        - Re-ranking improves precision (accuracy of top results)
        """
        try:
            # Prepare results summary for LLM
            results_summary = []
            for i, result in enumerate(results[:10]):  # Only re-rank top 10
                metadata = result.get('metadata', {})
                summary = {
                    'index': i,
                    'caption': metadata.get('caption', ''),
                    'objects': metadata.get('object_labels', ''),
                    'text': metadata.get('ocr_text', ''),
                    'score': result.get('similarity_score', 0)
                }
                results_summary.append(summary)
            
            # Create re-ranking prompt
            prompt = f"""You are a search result ranker for an image search system.
User is searching for: "{original_query}"

Here are the top search results with their content:
{self._format_results_for_llm(results_summary)}

Task: Rank these results by relevance to the user's query.
Output only the indices in order of relevance (most relevant first).
Format: 0,2,1,4,3,...

Ranking:"""
            
            # Get LLM response
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse ranking
            ranking_str = response_text.strip().split('\n')[0]
            ranked_indices = [int(idx.strip()) for idx in ranking_str.split(',') if idx.strip().isdigit()]
            
            # Reorder results
            reranked = []
            used_indices = set()
            
            for idx in ranked_indices:
                if idx < len(results) and idx not in used_indices:
                    reranked.append(results[idx])
                    used_indices.add(idx)
            
            # Add any remaining results
            for i, result in enumerate(results):
                if i not in used_indices:
                    reranked.append(result)
            
            logger.info(f"Re-ranked {len(reranked)} results")
            return reranked
            
        except Exception as e:
            logger.error(f"Re-ranking error: {e}")
            return results
    
    def _generate_explanation(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate natural language explanation of search results.
        
        Learning Points:
        - Helps users understand why results were returned
        - Builds trust in the search system
        - Can suggest refinements
        """
        try:
            if not results:
                return "No images found matching your query."
            
            # Prepare results summary
            top_results = []
            for i, result in enumerate(results[:3]):
                metadata = result.get('metadata', {})
                top_results.append({
                    'rank': i + 1,
                    'caption': metadata.get('caption', 'No caption'),
                    'objects': metadata.get('object_labels', 'No objects detected'),
                    'score': f"{result.get('similarity_score', 0):.2f}"
                })
            
            prompt = f"""You are an image search assistant helping users understand their search results.

User searched for: "{query}"

Top 3 results:
{self._format_results_for_llm(top_results)}

Task: Write a brief (2-3 sentences) explanation of why these images match the query.
Be conversational and helpful.

Explanation:"""
            
            response = self.llm.invoke(prompt)
            explanation = response.content if hasattr(response, 'content') else str(response)
            
            return explanation.strip()
            
        except Exception as e:
            logger.error(f"Explanation generation error: {e}")
            return ""
    
    def _format_results_for_llm(self, results: List[Dict[str, Any]]) -> str:
        """Format results for LLM consumption."""
        formatted = []
        for r in results:
            formatted.append(f"[{r.get('index', r.get('rank', 0))}] "
                           f"Caption: {r.get('caption', 'N/A')}, "
                           f"Objects: {r.get('objects', 'N/A')}, "
                           f"Score: {r.get('score', 'N/A')}")
        return '\n'.join(formatted)
    
    def search_by_image(
        self,
        image_path: str,
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        Search for similar images using an image as query.
        
        Args:
            image_path: Path to query image
            top_k: Number of results to return
            
        Returns:
            Similar images
            
        Learning Points:
        - Image-to-image search uses image embeddings
        - Useful for "find similar" functionality
        - Can find duplicates or variations
        """
        try:
            if top_k is None:
                top_k = settings.top_k_results
            
            logger.info(f"Searching by image: {image_path}")
            
            # Generate embedding for query image
            query_embedding = self.image_processor.generate_image_embedding(image_path)
            
            # Search
            results = self.vector_db.search_by_embedding(
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            return {
                'query_image': image_path,
                'results': results,
                'total_found': len(results)
            }
            
        except Exception as e:
            logger.error(f"Image search error: {e}")
            return {
                'query_image': image_path,
                'results': [],
                'error': str(e)
            }
    
    def hybrid_search(
        self,
        query: str,
        required_objects: Optional[List[str]] = None,
        must_have_text: bool = False,
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        Advanced hybrid search with filters.
        
        Args:
            query: Text query
            required_objects: List of objects that must be in the image
            must_have_text: Whether image must contain text
            top_k: Number of results
            
        Returns:
            Filtered search results
            
        Learning Points:
        - Hybrid search combines vector search with filters
        - Useful for specific requirements
        - Balances semantic similarity with constraints
        """
        # Build metadata filters
        filters = {}
        if must_have_text:
            filters['has_text'] = True
        
        # Perform search
        results = self.search(
            query=query,
            top_k=top_k,
            filter_metadata=filters if filters else None
        )
        
        # Post-filter for required objects if specified
        if required_objects and results.get('results'):
            filtered_results = []
            for result in results['results']:
                metadata = result.get('metadata', {})
                object_labels_str = metadata.get('object_labels', '')
                
                # Parse object labels (it's stored as JSON string)
                try:
                    import json
                    object_labels = json.loads(object_labels_str) if isinstance(object_labels_str, str) else object_labels_str
                    if not isinstance(object_labels, list):
                        object_labels = []
                except:
                    object_labels = []
                
                # Check if all required objects are present
                if all(obj.lower() in [o.lower() for o in object_labels] for obj in required_objects):
                    filtered_results.append(result)
            
            results['results'] = filtered_results[:top_k if top_k else settings.top_k_results]
            results['message'] = f"Filtered by required objects: {required_objects}"
        
        return results


# Global instance
_search_engine_instance = None

def get_rag_search_engine() -> RAGSearchEngine:
    """Get or create the global RAG search engine instance."""
    global _search_engine_instance
    if _search_engine_instance is None:
        _search_engine_instance = RAGSearchEngine()
    return _search_engine_instance


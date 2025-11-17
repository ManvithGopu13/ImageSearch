"""
System Testing Script
=====================
Test the image search system with sample queries and operations.

Usage:
    python test_system.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.image_processor import get_image_processor
from backend.vector_database import get_vector_database
from backend.rag_search import get_rag_search_engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_image_processor():
    """
    Test the image processing components.
    
    Learning Points:
    - Component initialization
    - Model loading verification
    - Basic functionality testing
    """
    print("\n" + "="*70)
    print("🧪 Testing Image Processor")
    print("="*70)
    
    try:
        processor = get_image_processor()
        print("✅ Image Processor initialized successfully")
        print(f"   Device: {processor.device}")
        
        # Test that models can be loaded (lazy loading)
        print("\n📦 Testing model components:")
        
        # OCR
        _ = processor.ocr_reader
        print("✅ EasyOCR loaded")
        
        # YOLO
        _ = processor.yolo_model
        print("✅ YOLO loaded")
        
        # CLIP
        _ = processor.clip_model
        print("✅ CLIP loaded")
        
        # BLIP (can be slow to load)
        print("⏳ Loading BLIP-2 (this may take a moment)...")
        _ = processor.blip_model
        print("✅ BLIP-2 loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Image Processor test failed: {e}")
        return False


def test_vector_database():
    """
    Test the vector database operations.
    
    Learning Points:
    - Database initialization
    - CRUD operations
    - Statistics retrieval
    """
    print("\n" + "="*70)
    print("🧪 Testing Vector Database")
    print("="*70)
    
    try:
        db = get_vector_database()
        print("✅ Vector Database initialized successfully")
        
        # Get stats
        stats = db.get_database_stats()
        print(f"\n📊 Database Statistics:")
        print(f"   Total Images: {stats['total_images']}")
        print(f"   Collection: {stats['collection_name']}")
        print(f"   Location: {stats['persist_directory']}")
        
        if stats['total_images'] > 0:
            print(f"   Images with text: {stats['images_with_text_pct']:.1f}%")
            print(f"   Images with objects: {stats['images_with_objects_pct']:.1f}%")
            print(f"   Images with captions: {stats['images_with_caption_pct']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector Database test failed: {e}")
        return False


def test_search_engine():
    """
    Test the RAG search engine.
    
    Learning Points:
    - Search engine initialization
    - Component integration
    - LLM availability
    """
    print("\n" + "="*70)
    print("🧪 Testing Search Engine")
    print("="*70)
    
    try:
        search_engine = get_rag_search_engine()
        print("✅ Search Engine initialized successfully")
        
        if search_engine.llm:
            print("✅ NVIDIA LLM connected")
            print("   RAG enhancement available")
        else:
            print("⚠️  NVIDIA LLM not available (basic search only)")
            print("   Set NVIDIA_API_KEY in .env to enable RAG features")
        
        # Test if database has images
        db = get_vector_database()
        count = db.count()
        
        if count > 0:
            print(f"\n✅ Database ready with {count} images")
            print("   You can start searching!")
        else:
            print("\n⚠️  Database is empty")
            print("   Upload some images first to test search")
        
        return True
        
    except Exception as e:
        print(f"❌ Search Engine test failed: {e}")
        return False


def run_sample_search():
    """
    Run a sample search if database has images.
    """
    db = get_vector_database()
    count = db.count()
    
    if count == 0:
        print("\n⚠️  Skipping sample search - no images in database")
        return True
    
    print("\n" + "="*70)
    print("🧪 Running Sample Search")
    print("="*70)
    
    try:
        search_engine = get_rag_search_engine()
        
        # Sample queries
        queries = [
            "image with text",
            "person",
            "outdoor scene"
        ]
        
        for query in queries:
            print(f"\n🔍 Query: '{query}'")
            results = search_engine.search(
                query=query,
                top_k=3,
                use_llm_enhancement=False  # Skip LLM for faster testing
            )
            
            print(f"   Found: {results.get('total_found', 0)} matching images")
            
            if results.get('results'):
                for i, result in enumerate(results['results'][:3], 1):
                    score = result.get('similarity_score', 0) * 100
                    caption = result.get('metadata', {}).get('caption', 'N/A')
                    print(f"   {i}. Score: {score:.1f}% - {caption[:60]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample search failed: {e}")
        return False


def test_embedding_generation():
    """
    Test embedding generation for text and images.
    """
    print("\n" + "="*70)
    print("🧪 Testing Embedding Generation")
    print("="*70)
    
    try:
        processor = get_image_processor()
        
        # Test text embedding
        text_embedding = processor.generate_text_embedding("cat on a sunny day")
        print(f"✅ Text embedding generated: shape {text_embedding.shape}")
        
        print("\nℹ️  Image embedding test requires an actual image file")
        print("   Upload images through the API to test image processing")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding generation test failed: {e}")
        return False


def main():
    """
    Run all system tests.
    """
    print("\n" + "="*70)
    print("🧪 IMAGE SEARCH RAG SYSTEM - COMPREHENSIVE TEST")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Image Processor", test_image_processor()))
    results.append(("Vector Database", test_vector_database()))
    results.append(("Search Engine", test_search_engine()))
    results.append(("Embedding Generation", test_embedding_generation()))
    results.append(("Sample Search", run_sample_search()))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready.")
        print("\n📝 Next steps:")
        print("   1. Start the server: python run_server.py")
        print("   2. Open frontend/index.html in your browser")
        print("   3. Upload images and start searching!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        print("   Common issues:")
        print("   - Missing dependencies: pip install -r requirements.txt")
        print("   - NVIDIA API key: Set NVIDIA_API_KEY in .env (optional)")
        print("   - GPU not detected: Install PyTorch with CUDA")


if __name__ == "__main__":
    main()


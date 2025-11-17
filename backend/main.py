"""
FastAPI Backend Application
============================
Main API server for the image search system.

Endpoints:
- POST /upload: Upload and process images
- POST /search: Search for images using text query
- POST /search/image: Search using an image
- GET /visualize: Get visualization of vector space
- GET /stats: Get database statistics
- GET /image/{image_id}: Get specific image details
- DELETE /image/{image_id}: Delete an image

Learning Points:
- RESTful API design
- Asynchronous processing with FastAPI
- File upload handling
- Error handling and validation
- CORS for frontend integration
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import shutil
from pathlib import Path
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid

from backend.config import settings
from backend.image_processor import get_image_processor
from backend.vector_database import get_vector_database
from backend.rag_search import get_rag_search_engine
from backend.visualizer import get_visualizer
from backend.visualizer_3d import get_visualizer_3d

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Image Search RAG System",
    description="AI-powered image search with OCR, object detection, and natural language queries",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
# Learning Note: CORS allows frontend (different origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images as static files
app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

# Initialize components (lazy loading)
image_processor = None
vector_db = None
search_engine = None
visualizer = None


def get_components():
    """
    Lazy initialization of components.
    
    Learning Note: 
    - Components are initialized on first request
    - Reduces startup time
    - Allows graceful error handling
    """
    global image_processor, vector_db, search_engine, visualizer
    
    if image_processor is None:
        logger.info("Initializing components...")
        image_processor = get_image_processor()
        vector_db = get_vector_database()
        search_engine = get_rag_search_engine()
        visualizer = get_visualizer()
        logger.info("✓ All components initialized")
    
    return image_processor, vector_db, search_engine, visualizer


# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class SearchRequest(BaseModel):
    """
    Schema for search requests.
    
    Learning Note:
    - Pydantic provides automatic validation
    - Type hints ensure data correctness
    - Generates OpenAPI schema automatically
    """
    query: str = Field(..., description="Search query in natural language", min_length=1)
    top_k: Optional[int] = Field(5, description="Number of results to return", ge=1, le=50)
    use_llm: Optional[bool] = Field(True, description="Use LLM for query enhancement")
    filter_has_text: Optional[bool] = Field(None, description="Filter images with text")
    required_objects: Optional[List[str]] = Field(None, description="Objects that must be present")


class SearchResponse(BaseModel):
    """Schema for search responses."""
    query: str
    enhanced_query: Optional[str]
    keywords: Optional[List[str]]
    results: List[Dict[str, Any]]
    explanation: Optional[str]
    total_found: int
    processing_time_ms: float


class ImageUploadResponse(BaseModel):
    """Schema for upload responses."""
    success: bool
    image_id: str
    filename: str
    message: str
    processing_details: Optional[Dict[str, Any]]


class BatchUploadResponse(BaseModel):
    """Schema for batch upload responses."""
    total_uploaded: int
    successful: int
    failed: int
    results: List[ImageUploadResponse]


class StatsResponse(BaseModel):
    """Schema for database statistics."""
    total_images: int
    sample_size: int
    images_with_text_pct: float
    images_with_objects_pct: float
    images_with_caption_pct: float
    collection_name: str
    persist_directory: str


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """
    Root endpoint - API information.
    
    Learning Note:
    - Health check endpoint
    - Provides API overview
    """
    return {
        "name": "Image Search RAG System",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "upload": "/upload",
            "search": "/search",
            "visualize": "/visualize",
            "stats": "/stats",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        proc, db, search, vis = get_components()
        db_count = db.count()
        
        return {
            "status": "healthy",
            "components": {
                "image_processor": "ok",
                "vector_database": "ok",
                "search_engine": "ok",
                "visualizer": "ok"
            },
            "database_images": db_count
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/upload", response_model=ImageUploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and process a single image.
    
    Learning Points:
    - File upload handling in FastAPI
    - Background processing for long tasks
    - Unique ID generation
    - Error handling
    """
    try:
        proc, db, _, _ = get_components()
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Must be an image."
            )
        
        # Generate unique ID
        image_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        filename = f"{image_id}{file_extension}"
        file_path = Path(settings.upload_dir) / filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Saved file: {filename}")
        
        # Process image (CPU-intensive, run in thread pool)
        loop = asyncio.get_event_loop()
        processing_result = await loop.run_in_executor(
            executor,
            proc.process_image_complete,
            str(file_path)
        )
        
        # Add to vector database
        db.add_image(
            image_id=image_id,
            embedding=processing_result['embedding'],
            metadata={
                'filename': file.filename,
                'uploaded_at': datetime.now().isoformat(),
                'file_path': str(file_path),
                'ocr_text': processing_result['ocr']['full_text'],
                'object_labels': processing_result['objects']['object_labels'],
                'object_count': processing_result['objects']['total_objects'],
                'caption': processing_result['caption'].get('caption', ''),
                'has_text': processing_result['metadata']['has_text'],
                'has_caption': processing_result['metadata']['has_caption']
            },
            combined_text=processing_result['combined_text']
        )
        
        return ImageUploadResponse(
            success=True,
            image_id=image_id,
            filename=file.filename,
            message="Image uploaded and processed successfully",
            processing_details={
                'ocr_found': processing_result['metadata']['has_text'],
                'objects_detected': processing_result['objects']['total_objects'],
                'caption_generated': processing_result['metadata']['has_caption']
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/upload/batch", response_model=BatchUploadResponse)
async def upload_images_batch(files: List[UploadFile] = File(...)):
    """
    Upload and process multiple images in batch.
    
    Learning Points:
    - Batch processing for efficiency
    - Progress tracking
    - Partial failure handling
    """
    try:
        proc, db, _, _ = get_components()
        
        results = []
        successful = 0
        failed = 0
        
        for file in files:
            try:
                # Validate
                if not file.content_type.startswith('image/'):
                    results.append(ImageUploadResponse(
                        success=False,
                        image_id="",
                        filename=file.filename,
                        message=f"Invalid file type: {file.content_type}"
                    ))
                    failed += 1
                    continue
                
                # Save and process
                image_id = str(uuid.uuid4())
                file_extension = Path(file.filename).suffix
                filename = f"{image_id}{file_extension}"
                file_path = Path(settings.upload_dir) / filename
                
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # Process
                processing_result = proc.process_image_complete(str(file_path))
                
                # Add to DB
                db.add_image(
                    image_id=image_id,
                    embedding=processing_result['embedding'],
                    metadata={
                        'filename': file.filename,
                        'uploaded_at': datetime.now().isoformat(),
                        'file_path': str(file_path),
                        'ocr_text': processing_result['ocr']['full_text'],
                        'object_labels': processing_result['objects']['object_labels'],
                        'object_count': processing_result['objects']['total_objects'],
                        'caption': processing_result['caption'].get('caption', ''),
                        'has_text': processing_result['metadata']['has_text'],
                        'has_caption': processing_result['metadata']['has_caption']
                    },
                    combined_text=processing_result['combined_text']
                )
                
                results.append(ImageUploadResponse(
                    success=True,
                    image_id=image_id,
                    filename=file.filename,
                    message="Success",
                    processing_details={
                        'ocr_found': processing_result['metadata']['has_text'],
                        'objects_detected': processing_result['objects']['total_objects']
                    }
                ))
                successful += 1
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                results.append(ImageUploadResponse(
                    success=False,
                    image_id="",
                    filename=file.filename,
                    message=str(e)
                ))
                failed += 1
        
        return BatchUploadResponse(
            total_uploaded=len(files),
            successful=successful,
            failed=failed,
            results=results
        )
        
    except Exception as e:
        logger.error(f"Batch upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_images(request: SearchRequest):
    """
    Search for images using natural language query.
    
    Learning Points:
    - POST for search (allows complex request bodies)
    - RAG-enhanced search
    - Performance tracking
    """
    try:
        import time
        start_time = time.time()
        
        _, db, search_engine, _ = get_components()
        
        # Check if database has images
        if db.count() == 0:
            raise HTTPException(
                status_code=404,
                detail="No images in database. Please upload images first."
            )
        
        # Perform search
        filter_metadata = None
        if request.filter_has_text is not None:
            filter_metadata = {'has_text': request.filter_has_text}
        
        if request.required_objects:
            # Use hybrid search
            results = search_engine.hybrid_search(
                query=request.query,
                required_objects=request.required_objects,
                must_have_text=request.filter_has_text or False,
                top_k=request.top_k
            )
        else:
            # Regular search
            results = search_engine.search(
                query=request.query,
                top_k=request.top_k,
                use_llm_enhancement=request.use_llm,
                filter_metadata=filter_metadata
            )
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        return SearchResponse(
            query=results.get('query', request.query),
            enhanced_query=results.get('enhanced_query'),
            keywords=results.get('keywords', []),
            results=results.get('results', []),
            explanation=results.get('explanation', ''),
            total_found=results.get('total_found', 0),
            processing_time_ms=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...)):
    """
    Search for similar images using an image as query.
    
    Learning Points:
    - Image-to-image search
    - Temporary file handling
    """
    try:
        proc, db, search_engine, _ = get_components()
        
        # Validate
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # Save temporarily
        temp_path = Path(settings.upload_dir) / f"temp_{uuid.uuid4()}{Path(file.filename).suffix}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            # Search
            results = search_engine.search_by_image(
                image_path=str(temp_path),
                top_k=5
            )
            
            return results
            
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/visualize")
async def visualize_vectors(
    method: str = Query("pca", description="Reduction method: tsne, pca"),
    dimensions: int = Query(2, description="2 or 3 dimensions", ge=2, le=3),
    sample_size: Optional[int] = Query(None, description="Number of images to visualize")
):
    """
    Get visualization of vector space.
    
    Learning Points:
    - Data visualization API
    - Dimensionality reduction
    - Interactive plots with Plotly
    """
    try:
        _, db, _, visualizer = get_components()
        
        if db.count() == 0:
            raise HTTPException(status_code=404, detail="No images in database")
        
        if dimensions == 2:
            result = visualizer.create_2d_visualization(
                sample_size=sample_size,
                method=method
            )
        else:
            result = visualizer.create_3d_visualization(
                sample_size=sample_size,
                method=method
            )
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/visualize/3d")
async def visualize_3d_advanced(
    method: str = Query("pca", description="Reduction method: tsne, pca (pca is faster)"),
    sample_size: Optional[int] = Query(50, description="Number of images (max 100 for performance)"),
    n_clusters: int = Query(5, description="Number of clusters to show", ge=2, le=10),
    show_connections: bool = Query(True, description="Show connection lines between similar images")
):
    """
    Advanced 3D visualization with clusters and connections.
    """
    try:
        _, db, _, _ = get_components()

        if db.count() == 0:
            raise HTTPException(status_code=404, detail="No images in database")

        visualizer_3d = get_visualizer_3d()

        result = visualizer_3d.create_3d_visualization(
            sample_size=sample_size,
            method=method,
            n_clusters=n_clusters,
            show_connections=show_connections
        )

        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"3D visualization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/visualize/network")
async def visualize_network_map(
    method: str = Query("pca", description="Reduction method: pca, tsne"),
    sample_size: Optional[int] = Query(50, description="Number of images", ge=10, le=100),
    n_clusters: int = Query(6, description="Number of clusters", ge=3, le=10),
    connection_threshold: float = Query(0.75, description="Similarity threshold for connections", ge=0.5, le=0.95)
):
    """
    Create interactive 2D network map visualization (like NetworkX).
    Shows nodes with labels, connections, and hover image previews.
    """
    try:
        _, db, _, _ = get_components()

        if db.count() == 0:
            raise HTTPException(status_code=404, detail="No images in database")

        visualizer_3d = get_visualizer_3d()
        
        logger.info(f"Creating network map: {sample_size} images, {n_clusters} clusters")

        result = visualizer_3d.create_network_map(
            sample_size=sample_size,
            method=method,
            n_clusters=n_clusters,
            connection_threshold=connection_threshold
        )

        if 'error' in result:
            logger.error(f"Network map returned error: {result['error']}")
            raise HTTPException(status_code=500, detail=result['error'])
        
        logger.info(f"Network map created successfully: {result.get('sample_size')} nodes")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Network map error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_statistics():
    """
    Get database statistics.
    
    Learning Points:
    - Analytics and monitoring
    - Database health metrics
    """
    try:
        _, db, _, _ = get_components()
        
        stats = db.get_database_stats()
        
        if 'error' in stats:
            raise HTTPException(status_code=500, detail=stats['error'])
        
        return StatsResponse(**stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/image/{image_id}")
async def get_image_details(image_id: str):
    """
    Get details of a specific image.
    """
    try:
        _, db, _, _ = get_components()
        
        image_data = db.get_by_id(image_id)
        
        if image_data is None:
            raise HTTPException(status_code=404, detail="Image not found")
        
        return image_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get image error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/image/{image_id}")
async def delete_image(image_id: str):
    """
    Delete an image from the database.
    """
    try:
        _, db, _, _ = get_components()
        
        # Get image details first
        image_data = db.get_by_id(image_id)
        
        if image_data is None:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Delete from database
        success = db.delete_image(image_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete image")
        
        # Delete file if exists
        try:
            metadata = image_data.get('metadata', {})
            file_path = metadata.get('file_path')
            if file_path and Path(file_path).exists():
                Path(file_path).unlink()
        except Exception as e:
            logger.warning(f"Could not delete file: {e}")
        
        return {"success": True, "message": "Image deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/images")
async def list_images(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """
    List all images in the database.
    """
    try:
        _, db, _, _ = get_components()
        
        all_ids = db.get_all_ids()
        
        # Pagination
        paginated_ids = all_ids[offset:offset + limit]
        
        # Get details for paginated IDs
        images = []
        for image_id in paginated_ids:
            image_data = db.get_by_id(image_id)
            if image_data:
                # Remove embedding from response (too large)
                image_data.pop('embedding', None)
                images.append(image_data)
        
        return {
            'total': len(all_ids),
            'limit': limit,
            'offset': offset,
            'images': images
        }
        
    except Exception as e:
        logger.error(f"List images error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Initialize application on startup.
    
    Learning Note:
    - Startup events for initialization
    - Logging configuration
    """
    logger.info("="* 60)
    logger.info("🚀 Image Search RAG System Starting...")
    logger.info("="* 60)
    logger.info(f"Upload directory: {settings.upload_dir}")
    logger.info(f"Database directory: {settings.chroma_persist_directory}")
    logger.info(f"Device: {settings.device}")
    logger.info("="* 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down...")
    executor.shutdown(wait=True)


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,  # Auto-reload on code changes (development only)
        log_level="info"
    )


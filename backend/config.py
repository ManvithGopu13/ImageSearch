"""
Configuration Module
====================
This module handles all configuration settings for the image search system.
It uses pydantic-settings for type-safe configuration management.

Learning Points:
- Environment variable management
- Type-safe configuration with Pydantic
- Default values and validation
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Pydantic automatically loads values from .env file and validates types.
    """
    
    # NVIDIA API Configuration
    nvidia_api_key: str = Field(
        default="",
        description="NVIDIA API key for LLM access"
    )
    
    nvidia_model_name: str = Field(
        default="meta/llama-3.1-8b-instruct",
        description="NVIDIA model name to use for LLM"
    )
    
    # Model Configuration
    clip_model: str = Field(
        default="ViT-B/32",
        description="CLIP model for image-text embeddings"
    )
    
    blip_model: str = Field(
        default="Salesforce/blip-image-captioning-base",
        description="BLIP model for image captioning (using base for lower memory)"
    )
    
    yolo_model: str = Field(
        default="yolov8n.pt",
        description="YOLO model for object detection"
    )
    
    # ChromaDB Configuration
    chroma_persist_directory: str = Field(
        default="./data/chroma_db",
        description="Directory to persist ChromaDB data"
    )
    
    collection_name: str = Field(
        default="image_collection",
        description="Name of the ChromaDB collection"
    )
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    upload_dir: str = Field(default="./uploads")
    
    # Processing Configuration
    batch_size: int = Field(
        default=32,
        description="Batch size for processing images"
    )
    
    max_image_size: int = Field(
        default=1024,
        description="Maximum image dimension (width/height) before resizing"
    )
    
    top_k_results: int = Field(
        default=10,
        description="Number of top results to return for searches"
    )
    
    similarity_threshold: float = Field(
        default=0.22,
        description="Minimum similarity score for search results (0-1, higher = stricter)"
    )
    
    enable_query_blending: bool = Field(
        default=True,
        description="Blend original and enhanced queries for better results"
    )
    
    enable_multi_field_boost: bool = Field(
        default=True,
        description="Boost results that match keywords in multiple fields (OCR, caption, objects)"
    )
    
    # Device Configuration (auto-detect GPU)
    device: str = Field(
        default="cpu",
        description="Device to use for model inference (cpu/cuda)"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Auto-detect GPU availability
try:
    import torch
    if torch.cuda.is_available():
        settings.device = "cuda"
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 Using CPU (GPU not available)")
except ImportError:
    print("⚠️  PyTorch not installed, using CPU")

# Create necessary directories
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.chroma_persist_directory, exist_ok=True)
os.makedirs("./models", exist_ok=True)


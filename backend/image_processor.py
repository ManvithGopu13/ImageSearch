"""
Image Processing Module
========================
This module handles all image processing tasks including:
1. OCR (Optical Character Recognition) - Extract text from images
2. Object Detection - Detect and label objects in images
3. Image Captioning - Generate natural language descriptions of images
4. Feature Extraction - Generate embeddings for similarity search

Learning Points:
- How OCR works with EasyOCR
- Object detection with YOLOv8
- Image captioning with BLIP-2
- Image-text embeddings with CLIP
- Multi-modal AI models
"""

import easyocr
import torch
import warnings
import os

# Suppress all warnings including torch.load security warnings
warnings.filterwarnings('ignore')
os.environ['TORCH_LOAD_WARNING'] = '0'

from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Any
from pathlib import Path
import cv2
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import logging

from backend.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Comprehensive image processor that extracts multiple types of information
    from images for building a rich, searchable vector database.
    """
    
    def __init__(self):
        """
        Initialize all models required for image processing.
        
        Learning Note:
        - Models are loaded once during initialization for efficiency
        - We use lazy loading to save memory
        """
        self.device = settings.device
        logger.info(f"Initializing ImageProcessor on device: {self.device}")
        
        # Initialize models as None (lazy loading)
        self._ocr_reader = None
        self._yolo_model = None
        self._clip_model = None
        self._clip_processor = None
        self._blip_model = None
        self._blip_processor = None
        
    @property
    def ocr_reader(self):
        """
        Lazy load EasyOCR reader.
        
        EasyOCR: Deep learning-based OCR that supports 80+ languages.
        It uses CRAFT for text detection and a custom recognition network.
        
        Learning Points:
        - EasyOCR is more accurate than Tesseract for natural scenes
        - Supports GPU acceleration
        - Can handle multiple languages simultaneously
        """
        if self._ocr_reader is None:
            logger.info("Loading EasyOCR model...")
            # Languages: English ('en') - add more as needed ['en', 'ch_sim', 'es', etc.]
            self._ocr_reader = easyocr.Reader(
                ['en'], 
                gpu=self.device == 'cuda',
                verbose=False
            )
            logger.info("✓ EasyOCR loaded successfully")
        return self._ocr_reader
    
    @property
    def yolo_model(self):
        """
        Lazy load YOLO model for object detection.
        
        YOLOv8: State-of-the-art object detection model.
        - Real-time detection (very fast)
        - 80 pre-trained object classes (COCO dataset)
        - Excellent accuracy-speed tradeoff
        
        Learning Points:
        - YOLO divides image into grid and predicts bounding boxes
        - Single-stage detector (faster than R-CNN family)
        - Available in different sizes: n(nano), s(small), m(medium), l(large), x(extra large)
        """
        if self._yolo_model is None:
            logger.info("Loading YOLO model...")
            self._yolo_model = YOLO(settings.yolo_model)
            logger.info(f"✓ YOLO model loaded: {settings.yolo_model}")
        return self._yolo_model
    
    @property
    def clip_model(self):
        """
        Lazy load CLIP model for image-text embeddings.
        
        CLIP (Contrastive Language-Image Pre-training):
        - Trained on 400M image-text pairs
        - Maps images and text to same embedding space
        - Perfect for semantic image search
        
        Learning Points:
        - CLIP learns visual concepts from natural language descriptions
        - Zero-shot learning capability
        - Can understand complex queries like "a photo of a cat on a sunny day"
        """
        if self._clip_model is None:
            logger.info("Loading CLIP model...")
            self._clip_model = CLIPModel.from_pretrained(
                f"openai/clip-vit-base-patch32"
            ).to(self.device)
            self._clip_processor = CLIPProcessor.from_pretrained(
                f"openai/clip-vit-base-patch32"
            )
            logger.info("✓ CLIP model loaded")
        return self._clip_model
    
    @property
    def clip_processor(self):
        """CLIP processor for preprocessing images and text."""
        if self._clip_processor is None:
            # Trigger loading via clip_model property
            _ = self.clip_model
        return self._clip_processor
    
    @property
    def blip_model(self):
        """
        Lazy load BLIP model for image captioning.
        
        BLIP (Bootstrapped Language-Image Pre-training):
        - Generates natural language descriptions of images
        - Understands scene context, objects, actions, and relationships
        - Uses a vision encoder + language model architecture
        
        Learning Points:
        - BLIP bridges vision and language understanding
        - Generates rich, contextual descriptions
        - Base model uses less memory than BLIP-2
        """
        if self._blip_model is None:
            logger.info("Loading BLIP model (this may take a moment)...")
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                
                self._blip_processor = BlipProcessor.from_pretrained(
                    settings.blip_model
                )
                self._blip_model = BlipForConditionalGeneration.from_pretrained(
                    settings.blip_model,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
                ).to(self.device)
                logger.info("✓ BLIP model loaded")
            except Exception as e:
                logger.error(f"Failed to load BLIP model: {e}")
                logger.info("Captioning will be disabled")
                self._blip_model = None
                self._blip_processor = None
        return self._blip_model
    
    @property
    def blip_processor(self):
        """BLIP-2 processor for preprocessing images."""
        if self._blip_processor is None:
            # Trigger loading via blip_model property
            _ = self.blip_model
        return self._blip_processor
    
    def preprocess_image(self, image_path: str) -> Image.Image:
        """
        Load and preprocess an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
            
        Learning Points:
        - Always convert to RGB (some images are RGBA or grayscale)
        - Resize large images to save memory and processing time
        - Maintain aspect ratio to avoid distortion
        """
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (maintain aspect ratio)
        max_size = settings.max_image_size
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        return image
    
    def extract_text_ocr(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text from image using OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing extracted text and metadata
            
        Learning Points:
        - OCR works best on high-contrast, clear text
        - Can handle rotated text and different fonts
        - Returns bounding boxes and confidence scores
        """
        try:
            # Read image using OpenCV for proper preprocessing
            img = cv2.imread(str(image_path))
            
            if img is None:
                logger.error(f"Failed to read image: {image_path}")
                return {
                    'full_text': '',
                    'detailed_results': [],
                    'text_found': False,
                    'error': 'Failed to read image'
                }
            
            # Convert BGR to RGB (OpenCV loads as BGR, EasyOCR expects RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # EasyOCR accepts numpy array
            results = self.ocr_reader.readtext(img_rgb, detail=1)
            
            # Parse results - EasyOCR returns list where each element is [bbox, text, confidence]
            # bbox is a list of 4 corner points: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]]
            extracted_texts = []
            total_text = []
            
            # Handle empty results
            if not results:
                return {
                    'full_text': '',
                    'detailed_results': [],
                    'text_found': False
                }
            
            for i, result in enumerate(results):
                try:
                    # Robust parsing for any format
                    bbox = None
                    text = None
                    confidence = 0.0
                    
                    # Check if result is a list or tuple
                    if not isinstance(result, (list, tuple)):
                        logger.warning(f"Unexpected result type at index {i}: {type(result)}")
                        continue
                    
                    # Get the actual length
                    result_length = len(result)
                    
                    # Parse based on length
                    if result_length >= 3:
                        # Standard format: [bbox, text, confidence]
                        bbox = result[0]
                        text = result[1]
                        confidence = result[2]
                    elif result_length == 2:
                        # Format without confidence: [bbox, text]
                        bbox = result[0]
                        text = result[1]
                        confidence = 1.0
                    elif result_length == 1:
                        # Only text
                        text = result[0]
                        bbox = []
                        confidence = 1.0
                    else:
                        # Empty or invalid
                        logger.warning(f"Invalid result length at index {i}: {result_length}")
                        continue
                    
                    # Validate and convert text
                    if text is None or text == '':
                        continue
                    
                    text_str = str(text).strip()
                    if not text_str:
                        continue
                    
                    # Store the result
                    extracted_texts.append({
                        'text': text_str,
                        'confidence': float(confidence) if confidence else 0.0,
                        'bbox': bbox if bbox is not None else []
                    })
                    total_text.append(text_str)
                    
                except Exception as parse_error:
                    logger.warning(f"Error parsing OCR result at index {i}: {parse_error}. Result: {result}")
                    continue
            
            return {
                'full_text': ' '.join(total_text),
                'detailed_results': extracted_texts,
                'text_found': len(total_text) > 0
            }
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'full_text': '',
                'detailed_results': [],
                'text_found': False,
                'error': str(e)
            }
    
    def detect_objects(self, image_path: str) -> Dict[str, Any]:
        """
        Detect objects in the image using YOLO.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing detected objects and their details
            
        Learning Points:
        - YOLO detects 80 object classes (person, car, cat, dog, etc.)
        - Returns bounding boxes, class labels, and confidence scores
        - Confidence threshold filters out uncertain detections
        """
        try:
            # Run inference
            results = self.yolo_model(image_path, verbose=False)
            
            # Parse results
            detected_objects = []
            object_labels = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get class name and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    label = result.names[class_id]
                    
                    # Only include high-confidence detections
                    if confidence > 0.3:
                        detected_objects.append({
                            'label': label,
                            'confidence': confidence,
                            'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        })
                        object_labels.append(label)
            
            # Count occurrences of each object
            object_counts = {}
            for label in object_labels:
                object_counts[label] = object_counts.get(label, 0) + 1
            
            return {
                'objects': detected_objects,
                'object_labels': list(set(object_labels)),  # Unique labels
                'object_counts': object_counts,
                'total_objects': len(detected_objects)
            }
            
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return {
                'objects': [],
                'object_labels': [],
                'object_counts': {},
                'total_objects': 0,
                'error': str(e)
            }
    
    def generate_caption(self, image_path: str) -> Dict[str, str]:
        """
        Generate a natural language caption for the image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing the generated caption
            
        Learning Points:
        - BLIP understands scene context and relationships
        - Can generate detailed descriptions including actions and attributes
        """
        try:
            if self.blip_model is None:
                return {
                    'caption': 'Caption generation unavailable',
                    'success': False,
                    'error': 'BLIP model not loaded'
                }
            
            image = self.preprocess_image(image_path)
            
            # Prepare inputs
            inputs = self.blip_processor(image, return_tensors="pt").to(
                self.device, 
                torch.float16 if self.device == 'cuda' else torch.float32
            )
            
            # Generate caption with memory optimization
            with torch.no_grad():
                generated_ids = self.blip_model.generate(**inputs, max_length=50)
                caption = self.blip_processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )[0].strip()
            
            # Clear GPU cache
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            return {
                'caption': caption,
                'success': True
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"GPU out of memory for captioning. Skipping caption.")
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                return {
                    'caption': 'Image processed (caption skipped due to memory)',
                    'success': False,
                    'error': 'GPU out of memory'
                }
            else:
                logger.error(f"Caption generation error: {e}")
                return {
                    'caption': '',
                    'success': False,
                    'error': str(e)
                }
        except Exception as e:
            logger.error(f"Caption generation error: {e}")
            return {
                'caption': '',
                'success': False,
                'error': str(e)
            }
    
    def generate_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Generate CLIP embedding for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Numpy array containing the image embedding (512-dimensional vector)
            
        Learning Points:
        - CLIP embeddings are 512-dimensional vectors
        - Images with similar visual content have similar embeddings
        - These embeddings can be compared with text embeddings
        """
        max_retries = 2
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                image = self.preprocess_image(image_path)
                
                # Process image
                inputs = self.clip_processor(
                    images=image, 
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate embedding with memory optimization
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    # Normalize for cosine similarity
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Convert to numpy before clearing CUDA cache
                    embedding = image_features.cpu().numpy()[0]
                
                # Clear GPU cache if using CUDA
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                
                # Validate embedding
                if np.any(np.isnan(embedding)) or np.all(embedding == 0):
                    logger.warning("Invalid embedding detected, regenerating...")
                    retry_count += 1
                    continue
                
                return embedding
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"GPU out of memory. Clearing cache and retrying on CPU...")
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                    # Retry on CPU
                    try:
                        image = self.preprocess_image(image_path)
                        inputs = self.clip_processor(images=image, return_tensors="pt").to('cpu')
                        with torch.no_grad():
                            model_cpu = self.clip_model.to('cpu')
                            image_features = model_cpu.get_image_features(**inputs)
                            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                            embedding = image_features.numpy()[0]
                        # Move model back to original device
                        self.clip_model.to(self.device)
                        return embedding
                    except Exception as cpu_error:
                        logger.error(f"CPU fallback also failed: {cpu_error}")
                        return np.zeros(512)
                elif "torch.load" in str(e) or "safetensors" in str(e):
                    # Model loading issue, try reloading
                    logger.warning("Model loading issue, reinitializing CLIP...")
                    self._clip_model = None
                    self._clip_processor = None
                    retry_count += 1
                    continue
                else:
                    logger.error(f"Embedding generation error: {e}")
                    return np.zeros(512)
            except Exception as e:
                logger.error(f"Embedding generation error: {e}")
                retry_count += 1
                if retry_count >= max_retries:
                    return np.zeros(512)
        
        # If we exhausted retries
        return np.zeros(512)
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate CLIP embedding for text (used for search queries).
        
        Args:
            text: Text string to embed
            
        Returns:
            Numpy array containing the text embedding (512-dimensional vector)
            
        Learning Points:
        - Text and image embeddings live in the same space
        - Can compute similarity between any image and any text
        - This enables natural language image search
        """
        try:
            # Process text
            inputs = self.clip_processor(
                text=[text], 
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Generate embedding with memory optimization
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                # Normalize for cosine similarity
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Convert to numpy before clearing cache
                embedding = text_features.cpu().numpy()[0]
            
            # Clear GPU cache if using CUDA
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            return embedding
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"GPU out of memory for text embedding. Clearing cache and retrying on CPU...")
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                # Retry on CPU
                try:
                    inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to('cpu')
                    with torch.no_grad():
                        model_cpu = self.clip_model.to('cpu')
                        text_features = model_cpu.get_text_features(**inputs)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        embedding = text_features.numpy()[0]
                    # Move model back to original device
                    self.clip_model.to(self.device)
                    return embedding
                except Exception as cpu_error:
                    logger.error(f"CPU fallback also failed: {cpu_error}")
                    return np.zeros(512)
            else:
                logger.error(f"Text embedding generation error: {e}")
                return np.zeros(512)
        except Exception as e:
            logger.error(f"Text embedding generation error: {e}")
            return np.zeros(512)
    
    def process_image_complete(self, image_path: str) -> Dict[str, Any]:
        """
        Complete image processing pipeline: OCR + Object Detection + Caption + Embedding.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing all extracted information
            
        Learning Points:
        - This combines all AI models to create a rich representation
        - Multiple modalities provide better search results
        - Each type of information serves different search needs
        """
        logger.info(f"Processing image: {image_path}")
        
        # Extract all information
        ocr_result = self.extract_text_ocr(image_path)
        objects_result = self.detect_objects(image_path)
        caption_result = self.generate_caption(image_path)
        embedding = self.generate_image_embedding(image_path)
        
        # Combine metadata for rich search
        combined_text = ' '.join(filter(None, [
            ocr_result['full_text'],
            ' '.join(objects_result['object_labels']),
            caption_result.get('caption', '')
        ]))
        
        return {
            'image_path': image_path,
            'ocr': ocr_result,
            'objects': objects_result,
            'caption': caption_result,
            'embedding': embedding,
            'combined_text': combined_text,
            'metadata': {
                'has_text': ocr_result['text_found'],
                'object_count': objects_result['total_objects'],
                'has_caption': caption_result.get('success', False)
            }
        }


# Global instance (singleton pattern)
_processor_instance = None

def get_image_processor() -> ImageProcessor:
    """
    Get or create the global ImageProcessor instance.
    
    Learning Note:
    - Singleton pattern ensures models are loaded only once
    - Saves memory and initialization time
    """
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = ImageProcessor()
    return _processor_instance


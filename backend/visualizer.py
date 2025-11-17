"""
Vector Database Visualization Module
=====================================
This module provides visualization capabilities for the vector database,
allowing you to see how images are organized in the embedding space.

Learning Points:
- Dimensionality reduction techniques (t-SNE, UMAP, PCA)
- How to visualize high-dimensional vectors
- Understanding the embedding space
- Clustering patterns in image data
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
from pathlib import Path

from backend.vector_database import get_vector_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorVisualizer:
    """
    Visualize vector embeddings in 2D/3D space.
    
    Learning Notes:
    - Embeddings are 512-dimensional (CLIP) - impossible to visualize directly
    - Dimensionality reduction projects high-D vectors to 2D/3D
    - Preserves relative distances (similar images stay close)
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        self.vector_db = get_vector_database()
        logger.info("Vector Visualizer initialized")
    
    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str = 'tsne',
        n_components: int = 2,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Reduce embedding dimensions for visualization.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            method: 'tsne', 'pca', or 'umap'
            n_components: 2 for 2D, 3 for 3D
            random_state: Random seed for reproducibility
            
        Returns:
            Reduced embeddings of shape (n_samples, n_components)
            
        Learning Points:
        - t-SNE: Best for visualization, preserves local structure
        - PCA: Fast, linear, preserves global structure
        - UMAP: Balance of both, faster than t-SNE
        """
        logger.info(f"Reducing {embeddings.shape} embeddings using {method}...")
        
        if method == 'tsne':
            # t-SNE (t-distributed Stochastic Neighbor Embedding)
            # Great for visualization but slow for large datasets
            # Perplexity: balance between local and global structure
            reducer = TSNE(
                n_components=n_components,
                random_state=random_state,
                perplexity=min(30, len(embeddings) - 1),
                max_iter=1000  # Changed from n_iter to max_iter (newer scikit-learn)
            )
            
        elif method == 'pca':
            # PCA (Principal Component Analysis)
            # Fast, deterministic, but may not capture complex patterns
            reducer = PCA(
                n_components=n_components,
                random_state=random_state
            )
            
        else:
            # Default to PCA if unknown method
            logger.warning(f"Unknown method '{method}', using PCA")
            reducer = PCA(n_components=n_components)
        
        # Perform reduction
        reduced = reducer.fit_transform(embeddings)
        logger.info(f"✓ Reduced to {reduced.shape}")
        
        return reduced
    
    def create_2d_visualization(
        self,
        sample_size: Optional[int] = None,
        method: str = 'tsne',
        color_by: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Create an interactive 2D visualization of the vector database.
        
        Args:
            sample_size: Number of images to visualize (None = all)
            method: Dimensionality reduction method
            color_by: How to color points ('auto', 'objects', 'has_text')
            
        Returns:
            Dictionary with plotly figure and metadata
            
        Learning Points:
        - Interactive plots help understand data distribution
        - Clustering reveals similar image groups
        - Outliers may indicate unique or unusual images
        """
        try:
            # Get all embeddings from database
            total_count = self.vector_db.count()
            
            if total_count == 0:
                return {
                    'error': 'No images in database',
                    'figure': None
                }
            
            # Determine sample size
            if sample_size is None or sample_size > total_count:
                sample_size = total_count
            
            logger.info(f"Visualizing {sample_size} images...")
            
            # Get sample of data
            data = self.vector_db.collection.get(
                limit=sample_size,
                include=['embeddings', 'metadatas', 'documents']
            )
            
            # Extract embeddings and metadata
            embeddings = np.array(data['embeddings'])
            metadatas = data['metadatas']
            ids = data['ids']
            
            # Check for valid embeddings
            if np.any(np.isnan(embeddings)) or np.all(embeddings == 0):
                return {
                    'error': 'Invalid embeddings detected. Please re-upload your images.',
                    'figure': None
                }
            
            # Reduce dimensions to 2D
            try:
                coords_2d = self.reduce_dimensions(
                    embeddings, 
                    method=method, 
                    n_components=2
                )
            except Exception as reduce_error:
                logger.error(f"Dimension reduction error: {reduce_error}")
                return {
                    'error': f'Dimension reduction failed: {str(reduce_error)}. Try using PCA instead of t-SNE.',
                    'figure': None
                }
            
            # Prepare data for plotting
            plot_data = {
                'x': coords_2d[:, 0],
                'y': coords_2d[:, 1],
                'ids': ids,
                'hover_text': []
            }
            
            # Determine coloring
            if color_by == 'auto':
                # Auto-detect best coloring scheme
                has_objects = any(
                    json.loads(m.get('object_labels', '[]')) if isinstance(m.get('object_labels'), str) else []
                    for m in metadatas
                )
                color_by = 'objects' if has_objects else 'has_text'
            
            # Create hover text and colors
            colors = []
            for i, meta in enumerate(metadatas):
                # Parse metadata
                caption = meta.get('caption', 'No caption')
                has_text = meta.get('has_text', False)
                object_count = meta.get('object_count', 0)
                
                # Create hover text
                hover = f"ID: {ids[i][:20]}...<br>"
                hover += f"Caption: {caption}<br>"
                hover += f"Objects: {object_count}<br>"
                hover += f"Has Text: {has_text}"
                plot_data['hover_text'].append(hover)
                
                # Determine color
                if color_by == 'has_text':
                    colors.append('Has Text' if has_text else 'No Text')
                elif color_by == 'objects':
                    if object_count == 0:
                        colors.append('No Objects')
                    elif object_count <= 2:
                        colors.append('Few Objects (1-2)')
                    elif object_count <= 5:
                        colors.append('Some Objects (3-5)')
                    else:
                        colors.append('Many Objects (6+)')
                else:
                    colors.append('Image')
            
            plot_data['colors'] = colors
            
            # Create plotly figure
            fig = px.scatter(
                x=plot_data['x'],
                y=plot_data['y'],
                color=plot_data['colors'],
                hover_data={'hover_text': plot_data['hover_text']},
                title=f'Image Embedding Space Visualization ({method.upper()})',
                labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
                width=1000,
                height=800
            )
            
            fig.update_traces(
                marker=dict(size=8, opacity=0.7),
                hovertemplate='%{customdata[0]}<extra></extra>',
                customdata=[[text] for text in plot_data['hover_text']]
            )
            
            fig.update_layout(
                hovermode='closest',
                showlegend=True,
                legend_title=color_by.replace('_', ' ').title()
            )
            
            # Convert to JSON for API response
            fig_json = fig.to_json()
            
            return {
                'figure': json.loads(fig_json),
                'method': method,
                'sample_size': sample_size,
                'total_images': total_count,
                'color_by': color_by
            }
            
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return {
                'error': str(e),
                'figure': None
            }
    
    def create_3d_visualization(
        self,
        sample_size: Optional[int] = None,
        method: str = 'pca'
    ) -> Dict[str, Any]:
        """
        Create an interactive 3D visualization.
        
        Learning Points:
        - 3D adds one more dimension of information
        - Can reveal structure not visible in 2D
        - More immersive but harder to interpret
        """
        try:
            # Get data
            total_count = self.vector_db.count()
            
            if total_count == 0:
                return {'error': 'No images in database', 'figure': None}
            
            if sample_size is None or sample_size > total_count:
                sample_size = total_count
            
            data = self.vector_db.collection.get(
                limit=sample_size,
                include=['embeddings', 'metadatas']
            )
            
            embeddings = np.array(data['embeddings'])
            metadatas = data['metadatas']
            ids = data['ids']
            
            # Reduce to 3D
            coords_3d = self.reduce_dimensions(
                embeddings,
                method=method,
                n_components=3
            )
            
            # Prepare plot data
            colors = [m.get('object_count', 0) for m in metadatas]
            hover_texts = [
                f"ID: {ids[i][:20]}...<br>Caption: {m.get('caption', 'N/A')}"
                for i, m in enumerate(metadatas)
            ]
            
            # Create 3D scatter plot
            fig = go.Figure(data=[go.Scatter3d(
                x=coords_3d[:, 0],
                y=coords_3d[:, 1],
                z=coords_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=colors,
                    colorscale='Viridis',
                    showscale=True,
                    opacity=0.7,
                    colorbar=dict(title="Object Count")
                ),
                text=hover_texts,
                hoverinfo='text'
            )])
            
            fig.update_layout(
                title=f'3D Image Embedding Space ({method.upper()})',
                scene=dict(
                    xaxis_title='Dimension 1',
                    yaxis_title='Dimension 2',
                    zaxis_title='Dimension 3'
                ),
                width=1000,
                height=800
            )
            
            fig_json = fig.to_json()
            
            return {
                'figure': json.loads(fig_json),
                'method': method,
                'sample_size': sample_size,
                'total_images': total_count
            }
            
        except Exception as e:
            logger.error(f"3D visualization error: {e}")
            return {'error': str(e), 'figure': None}
    
    def analyze_clusters(
        self,
        n_clusters: int = 5,
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform clustering analysis on embeddings.
        
        Learning Points:
        - Clustering groups similar images automatically
        - K-means is a simple, effective clustering algorithm
        - Useful for discovering categories in unlabeled data
        """
        try:
            from sklearn.cluster import KMeans
            
            total_count = self.vector_db.count()
            if total_count == 0:
                return {'error': 'No images in database'}
            
            if sample_size is None or sample_size > total_count:
                sample_size = total_count
            
            # Get embeddings
            data = self.vector_db.collection.get(
                limit=sample_size,
                include=['embeddings', 'metadatas']
            )
            
            embeddings = np.array(data['embeddings'])
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Analyze each cluster
            clusters_info = []
            for i in range(n_clusters):
                cluster_indices = np.where(cluster_labels == i)[0]
                cluster_metadatas = [data['metadatas'][idx] for idx in cluster_indices]
                
                # Find common characteristics
                avg_objects = np.mean([m.get('object_count', 0) for m in cluster_metadatas])
                text_percentage = sum(m.get('has_text', False) for m in cluster_metadatas) / len(cluster_metadatas) * 100
                
                # Sample captions
                sample_captions = [m.get('caption', '') for m in cluster_metadatas[:3]]
                
                clusters_info.append({
                    'cluster_id': i,
                    'size': len(cluster_indices),
                    'avg_object_count': round(avg_objects, 2),
                    'text_percentage': round(text_percentage, 2),
                    'sample_captions': sample_captions
                })
            
            return {
                'n_clusters': n_clusters,
                'sample_size': sample_size,
                'clusters': clusters_info
            }
            
        except Exception as e:
            logger.error(f"Clustering error: {e}")
            return {'error': str(e)}


# Global instance
_visualizer_instance = None

def get_visualizer() -> VectorVisualizer:
    """Get or create the global visualizer instance."""
    global _visualizer_instance
    if _visualizer_instance is None:
        _visualizer_instance = VectorVisualizer()
    return _visualizer_instance


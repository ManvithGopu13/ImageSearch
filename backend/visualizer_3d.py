"""
Advanced 3D Vector Space Visualization
=======================================
Creates an interactive 3D visualization with actual images displayed in vector space.

Features:
- 3D scatter plot with image thumbnails
- Interactive zoom, pan, rotate
- Clustering visualization
- Connection lines between similar images
- Real image previews in 3D space
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
import logging
import json
from pathlib import Path
import base64
from PIL import Image
import io

from backend.vector_database import get_vector_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorVisualizer3D:
    """
    Advanced 3D visualizer with actual images shown in space.
    """
    
    def __init__(self):
        self.vector_db = get_vector_database()
        logger.info("3D Vector Visualizer initialized")
    
    def image_to_base64(self, image_path: str, size: int = 100) -> str:
        """Convert image to base64 for embedding in HTML."""
        try:
            img = Image.open(image_path)
            img.thumbnail((size, size), Image.Resampling.LANCZOS)
            
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            logger.error(f"Error converting image: {e}")
            return ""
    
    def create_network_map(
        self,
        sample_size: Optional[int] = 50,
        method: str = 'pca',
        n_clusters: int = 6,
        connection_threshold: float = 0.75
    ) -> Dict[str, Any]:
        """
        Create an interactive 2D network map (like NetworkX) with visible nodes and hover.
        """
        try:
            total_count = self.vector_db.count()
            
            if total_count == 0:
                return {'error': 'No images in database', 'figure': None}
            
            if sample_size is None or sample_size > total_count:
                sample_size = min(total_count, 50)
            
            logger.info(f"Creating network map for {sample_size} images...")
            
            # Get ALL data from database
            all_data = self.vector_db.get_all()
            
            if not all_data or len(all_data) == 0:
                return {'error': 'No data found in database', 'figure': None}
            
            # Sample if needed
            if len(all_data) > sample_size:
                import random
                all_data = random.sample(all_data, sample_size)
            
            # Extract embeddings and metadata
            embeddings = []
            metadatas = []
            
            for item in all_data:
                if item.get('embedding') is not None:
                    emb = item['embedding']
                    # Handle both list and numpy array
                    if isinstance(emb, list):
                        emb = np.array(emb)
                    embeddings.append(emb)
                    metadatas.append(item.get('metadata', {}))
            
            if len(embeddings) == 0:
                return {'error': 'No valid embeddings found', 'figure': None}
            
            embeddings = np.array(embeddings)
            logger.info(f"Processing {len(embeddings)} embeddings, shape: {embeddings.shape}")
            
            # Validate
            if np.any(np.isnan(embeddings)) or np.all(embeddings == 0):
                return {'error': 'Invalid embeddings detected', 'figure': None}
            
            # Use 2D for clearer network visualization (like NetworkX)
            if method == 'tsne':
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1), max_iter=500)
            else:
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2, random_state=42)
            
            coords_2d = reducer.fit_transform(embeddings)
            logger.info(f"Reduced to 2D: {coords_2d.shape}")
            
            # Clustering
            from sklearn.cluster import KMeans
            n_clusters = min(n_clusters, len(embeddings))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(embeddings)
            
            # Calculate similarity for connections
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(embeddings)
            
            # Create network edges
            edge_x, edge_y = [], []
            edge_count = 0
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    if similarity_matrix[i][j] >= connection_threshold:
                        edge_x.extend([coords_2d[i, 0], coords_2d[j, 0], None])
                        edge_y.extend([coords_2d[i, 1], coords_2d[j, 1], None])
                        edge_count += 1
            
            logger.info(f"Created {edge_count} connections")
            
            # Build figure traces
            traces = []
            
            # Add connection lines
            if len(edge_x) > 0:
                traces.append(go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(color='rgba(150, 150, 150, 0.3)', width=1),
                    hoverinfo='none',
                    showlegend=False,
                    name='Connections'
                ))
            
            # Color palette
            import plotly.express as px
            colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
            
            # Collect all node data
            all_x, all_y, all_colors, all_labels, all_hover, all_filenames = [], [], [], [], [], []
            
            for cluster_id in range(n_clusters):
                mask = clusters == cluster_id
                cluster_coords = coords_2d[mask]
                cluster_metas = [metadatas[i] for i, m in enumerate(mask) if m]
                
                color = colors[cluster_id % len(colors)]
                
                for idx, meta in enumerate(cluster_metas):
                    x_coord = cluster_coords[idx, 0]
                    y_coord = cluster_coords[idx, 1]
                    
                    file_path = meta.get('file_path', meta.get('image_path', ''))
                    caption = meta.get('caption', 'No caption')
                    objects = meta.get('object_count', 0)
                    ocr_text = meta.get('ocr_text', '')
                    
                    import os
                    filename = os.path.basename(file_path) if file_path else 'Unknown'
                    
                    # Short label
                    if caption and caption != 'No caption' and len(caption) > 5:
                        label = caption[:25]
                    else:
                        label = filename[:25]
                    
                    # Hover text
                    hover_text = f"<b>{label}</b><br>"
                    hover_text += f"Cluster {cluster_id + 1}<br>"
                    hover_text += f"Objects: {objects}<br>"
                    if ocr_text and len(ocr_text) > 0:
                        hover_text += f"Text: {ocr_text[:50]}...<br>"
                    hover_text += f"<i>File: {filename}</i>"
                    
                    all_x.append(x_coord)
                    all_y.append(y_coord)
                    all_colors.append(color)
                    all_labels.append(label)
                    all_hover.append(hover_text)
                    all_filenames.append(filename)
            
            # Add ALL nodes as one trace
            traces.append(go.Scatter(
                x=all_x,
                y=all_y,
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=all_colors,
                    line=dict(color='white', width=2),
                    opacity=0.9
                ),
                text=all_labels,
                textposition='top center',
                textfont=dict(size=9, color='white', family='Arial Black'),
                hovertext=all_hover,
                hoverinfo='text',
                customdata=all_filenames,
                name='Images',
                showlegend=False
            ))
            
            logger.info(f"Created {len(traces)} traces with {len(all_x)} nodes")
            
            # Create figure
            fig = go.Figure(data=traces)
            
            # 2D layout like NetworkX graphs
            fig.update_layout(
                title=dict(
                    text=f'🌐 Interactive Network Map - {len(embeddings)} Images',
                    x=0.5,
                    xanchor='center',
                    font=dict(size=20, color='white')
                ),
                xaxis=dict(
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False,
                    showline=False
                ),
                yaxis=dict(
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False,
                    showline=False
                ),
                plot_bgcolor='rgb(17, 17, 17)',
                paper_bgcolor='rgb(17, 17, 17)',
                font=dict(color='white'),
                hovermode='closest',
                showlegend=False,
                margin=dict(l=20, r=20, t=60, b=20)
            )
            
            logger.info("Figure created successfully")
            
            return {
                'figure': json.loads(fig.to_json()),
                'method': method,
                'sample_size': len(embeddings),
                'n_clusters': n_clusters,
                'connections': edge_count,
                'message': f'Network with {len(embeddings)} nodes and {edge_count} connections'
            }
            
        except Exception as e:
            logger.error(f"Network map error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': str(e), 'figure': None}
    
    def create_3d_visualization(
        self,
        sample_size: Optional[int] = None,
        method: str = 'pca',
        n_clusters: int = 5,
        show_connections: bool = True
    ) -> Dict[str, Any]:
        """
        Create advanced 3D visualization with images.
        
        Args:
            sample_size: Number of images to visualize
            method: 'pca' or 'tsne' for dimensionality reduction
            n_clusters: Number of clusters to identify
            show_connections: Whether to show connection lines
            
        Returns:
            Dictionary with plotly figure and metadata
        """
        try:
            total_count = self.vector_db.count()
            
            if total_count == 0:
                return {'error': 'No images in database', 'figure': None}
            
            if sample_size is None or sample_size > total_count:
                sample_size = min(total_count, 100)  # Limit for performance
            
            logger.info(f"Creating 3D visualization for {sample_size} images...")
            
            # Get data from database
            data = self.vector_db.collection.get(
                limit=sample_size,
                include=['embeddings', 'metadatas']
            )
            
            embeddings = np.array(data['embeddings'])
            metadatas = data['metadatas']
            ids = data['ids']
            
            # Validate embeddings
            if np.any(np.isnan(embeddings)) or np.all(embeddings == 0):
                return {
                    'error': 'Invalid embeddings. Please re-upload your images.',
                    'figure': None
                }
            
            # Reduce to 3D
            if method == 'tsne':
                reducer = TSNE(
                    n_components=3,
                    random_state=42,
                    perplexity=min(30, len(embeddings) - 1),
                    max_iter=500  # Faster for real-time
                )
            else:  # PCA is faster
                reducer = PCA(n_components=3, random_state=42)
            
            coords_3d = reducer.fit_transform(embeddings)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Prepare data for plotly
            x, y, z = coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]
            
            # Create hover text with image previews
            hover_texts = []
            image_paths = []
            
            for i, meta in enumerate(metadatas):
                caption = meta.get('caption', 'No caption')[:100]
                objects = meta.get('object_count', 0)
                has_text = meta.get('has_text', False)
                file_path = meta.get('file_path', '')
                
                hover_text = (
                    f"<b>Image {i+1}</b><br>"
                    f"Caption: {caption}<br>"
                    f"Objects: {objects}<br>"
                    f"Has Text: {'Yes' if has_text else 'No'}<br>"
                    f"Cluster: {cluster_labels[i]}"
                )
                hover_texts.append(hover_text)
                image_paths.append(file_path)
            
            # Color by cluster
            colors = cluster_labels
            
            # Create main scatter plot
            scatter = go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers+text',
                marker=dict(
                    size=10,
                    color=colors,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Cluster"),
                    line=dict(color='white', width=0.5),
                    opacity=0.8
                ),
                text=[f"Img {i+1}" for i in range(len(x))],
                textposition="top center",
                textfont=dict(size=8),
                hovertext=hover_texts,
                hoverinfo='text',
                name='Images'
            )
            
            fig_data = [scatter]
            
            # Add connection lines between close neighbors
            if show_connections:
                # Find k nearest neighbors for each point
                from sklearn.neighbors import NearestNeighbors
                
                n_neighbors = min(3, len(embeddings) - 1)
                nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords_3d)
                distances, indices = nbrs.kneighbors(coords_3d)
                
                # Create lines
                line_x, line_y, line_z = [], [], []
                
                for i in range(len(coords_3d)):
                    for j in indices[i][1:]:  # Skip self
                        if distances[i][np.where(indices[i] == j)[0][0]] < 2.0:  # Only close neighbors
                            line_x.extend([x[i], x[j], None])
                            line_y.extend([y[i], y[j], None])
                            line_z.extend([z[i], z[j], None])
                
                connections = go.Scatter3d(
                    x=line_x, y=line_y, z=line_z,
                    mode='lines',
                    line=dict(color='rgba(100, 100, 100, 0.2)', width=1),
                    hoverinfo='skip',
                    showlegend=False,
                    name='Connections'
                )
                fig_data.insert(0, connections)  # Add first so it's behind
            
            # Add cluster centers
            cluster_centers_3d = reducer.transform(kmeans.cluster_centers_)
            
            centers = go.Scatter3d(
                x=cluster_centers_3d[:, 0],
                y=cluster_centers_3d[:, 1],
                z=cluster_centers_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=20,
                    color='red',
                    symbol='diamond',
                    line=dict(color='white', width=2)
                ),
                name='Cluster Centers',
                hoverinfo='text',
                hovertext=[f'Cluster {i} Center' for i in range(len(cluster_centers_3d))]
            )
            fig_data.append(centers)
            
            # Create figure
            fig = go.Figure(data=fig_data)
            
            # Update layout for better 3D experience
            fig.update_layout(
                title=dict(
                    text=f'3D Vector Space Visualization ({method.upper()})<br>'
                         f'<sub>{sample_size} images, {n_clusters} clusters</sub>',
                    x=0.5,
                    xanchor='center'
                ),
                scene=dict(
                    xaxis=dict(
                        title='Dimension 1',
                        backgroundcolor="rgb(230, 230,230)",
                        gridcolor="white",
                        showbackground=True
                    ),
                    yaxis=dict(
                        title='Dimension 2',
                        backgroundcolor="rgb(230, 230,230)",
                        gridcolor="white",
                        showbackground=True
                    ),
                    zaxis=dict(
                        title='Dimension 3',
                        backgroundcolor="rgb(230, 230,230)",
                        gridcolor="white",
                        showbackground=True
                    ),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    ),
                    aspectmode='cube'
                ),
                width=1200,
                height=800,
                hovermode='closest',
                showlegend=True,
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(255, 255, 255, 0.8)'
                ),
                paper_bgcolor='rgb(243, 243, 243)',
                plot_bgcolor='rgb(243, 243, 243)'
            )
            
            # Add image thumbnails info
            image_info = {
                'image_paths': image_paths,
                'positions': coords_3d.tolist(),
                'clusters': cluster_labels.tolist()
            }
            
            return {
                'figure': json.loads(fig.to_json()),
                'method': method,
                'sample_size': sample_size,
                'n_clusters': n_clusters,
                'image_info': image_info,
                'message': f'Visualized {sample_size} images in 3D space'
            }
            
        except Exception as e:
            logger.error(f"3D visualization error: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'figure': None}


# Global instance
_visualizer_3d_instance = None

def get_visualizer_3d() -> VectorVisualizer3D:
    """Get or create the global 3D visualizer instance."""
    global _visualizer_3d_instance
    if _visualizer_3d_instance is None:
        _visualizer_3d_instance = VectorVisualizer3D()
    return _visualizer_3d_instance


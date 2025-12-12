import os
import numpy as np
import plotly.graph_objects as go

def visualize_pointclouds_plotly(pointclouds, save_path=None, marker_size=3, opacity=0.8, title="3D Point Cloud Visualization"):
    """
    Visualizes 3D point clouds interactively using Plotly and optionally saves as HTML.

    Args:
        pointclouds (dict or np.ndarray): {'Name': np.array(N,3)} or just np.array(N,3)
        save_path (str): Path to save .html file.
        marker_size (int): Size of points.
    """
    fig = go.Figure()

    # Handle single array input
    if isinstance(pointclouds, np.ndarray):
        if pointclouds.ndim != 2 or pointclouds.shape[1] != 3:
            raise ValueError("Point cloud must be (N, 3).")
        pointclouds = {"Point Cloud": pointclouds}

    for name, points in pointclouds.items():
        if not isinstance(points, np.ndarray) or points.shape[1] != 3:
             continue # Skip invalid data

        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(size=marker_size, opacity=opacity),
            name=name
        ))

    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        title=title,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if not save_path.lower().endswith('.html'):
            save_path += '.html'
        fig.write_html(save_path)
        print(f"Saved visualization to {save_path}")

    # fig.show() # Uncomment if running in notebook/local GUI
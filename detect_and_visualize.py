#!/usr/bin/env python3
"""
Easy-to-use 3D object detection and visualization pipeline.
Supports both .bin and .pcd point cloud formats.

Usage:
    python detect_and_visualize.py <path_to_point_cloud_file>
    python detect_and_visualize.py demo/data/kitti/000008.bin
    python detect_and_visualize.py demo/data/kitti/000008.pcd
    python detect_and_visualize.py demo/data/kitti/000008.bin --score-thr 0.5
    python detect_and_visualize.py demo/data/kitti/000008.pcd --score-thr 0.3
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from mmdet3d.apis import LidarDet3DInferencer

# ============================================================================
# CONFIGURATION - Edit these settings as needed
# ============================================================================
CONFIG = {
    # Model configuration file
    'model': 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py',
    
    # Model checkpoint file
    'checkpoint': 'hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth',
    
    # Device to use for inference
    'device': 'cuda:0',  # Use 'cpu' if no GPU available
    
    # Output directory for predictions
    'output_dir': 'detection_outputs',
    
    # Default confidence threshold
    'score_threshold': 0.5,
    
    # Downsample ratio for point cloud visualization (higher = faster but less detailed)
    'downsample_ratio': 1,
    
    # Output HTML filename (fixed name to avoid clutter)
    'output_html': 'detection_results.html',
}
# ============================================================================


def load_kitti_bin(bin_path):
    """Load KITTI binary point cloud file."""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3], points[:, 3]  # xyz, intensity


def prepare_point_cloud(pcd_path):
    """
    Prepare point cloud file for detection.
    Automatically converts PCD to BIN format if needed.
    
    Args:
        pcd_path (str): Path to point cloud file (.bin or .pcd)
    
    Returns:
        str: Path to BIN file ready for detection
    """
    file_ext = Path(pcd_path).suffix.lower()
    
    if file_ext == '.pcd':
        print(f"Detected PCD format, converting to BIN...")
        
        # Import converter (native implementation, no external deps)
        from demo.format_converter import pcd_to_bin

        # Convert PCD to BIN in the same directory
        bin_path = str(Path(pcd_path).with_suffix('.bin'))
        bin_path = pcd_to_bin(pcd_path, bin_path)
        print(f"✓ Conversion complete: {bin_path}\n")
        return bin_path
    elif file_ext == '.bin':
        return pcd_path
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Only .bin and .pcd are supported.")


def bbox_3d_corners(bbox):
    """Convert 3D bbox [x,y,z,l,w,h,rot] to 8 corners."""
    x, y, z, l, w, h, rot = bbox
    corners = np.array([
        [-l/2, -w/2, -h/2], [l/2, -w/2, -h/2],
        [l/2, w/2, -h/2], [-l/2, w/2, -h/2],
        [-l/2, -w/2, h/2], [l/2, -w/2, h/2],
        [l/2, w/2, h/2], [-l/2, w/2, h/2]
    ])
    # Rotate around Z axis
    rot_mat = np.array([
        [np.cos(rot), -np.sin(rot), 0],
        [np.sin(rot), np.cos(rot), 0],
        [0, 0, 1]
    ])
    corners = corners @ rot_mat.T
    corners += np.array([x, y, z])
    return corners


def create_bbox_edges(corners):
    """Create edges for drawing 3D bounding boxes."""
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical
    ]
    
    x_lines, y_lines, z_lines = [], [], []
    for edge in edges:
        for i in edge:
            x_lines.append(corners[i, 0])
            y_lines.append(corners[i, 1])
            z_lines.append(corners[i, 2])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)
    
    return x_lines, y_lines, z_lines


def run_detection(pcd_path, output_dir, score_threshold):
    """Run 3D object detection on point cloud file."""
    print(f"\n{'='*70}")
    print("Running 3D Object Detection...")
    print(f"{'='*70}")
    print(f"Input file: {pcd_path}")
    print(f"Model: {CONFIG['model']}")
    print(f"Checkpoint: {CONFIG['checkpoint']}")
    print(f"Device: {CONFIG['device']}")
    print(f"Score threshold: {score_threshold}")
    print(f"{'='*70}\n")
    
    # Convert to BIN format if needed
    bin_path = prepare_point_cloud(pcd_path)
    
    # Initialize inferencer
    inferencer = LidarDet3DInferencer(
        model=CONFIG['model'],
        weights=CONFIG['checkpoint'],
        device=CONFIG['device']
    )
    
    # Run inference
    result = inferencer(
        inputs=dict(points=bin_path),
        pred_score_thr=score_threshold,
        out_dir=output_dir,
        no_save_vis=True,
        print_result=False,
        show=False
    )
    
    # Get prediction file path (use original input filename)
    pcd_name = Path(pcd_path).stem
    pred_file = os.path.join(output_dir, 'preds', f'{pcd_name}.json')
    
    print(f"✓ Detection completed!")
    print(f"✓ Results saved to: {pred_file}\n")
    
    return pred_file, bin_path


def create_visualization(original_pcd_path, bin_path, pred_file, output_html):
    """Create interactive HTML visualization."""
    print(f"{'='*70}")
    print("Creating Visualization...")
    print(f"{'='*70}\n")
    
    # Load point cloud from BIN file
    points, intensity = load_kitti_bin(bin_path)
    
    # Downsample for performance
    points = points[::CONFIG['downsample_ratio']]
    intensity = intensity[::CONFIG['downsample_ratio']]
    
    # Load detection results
    with open(pred_file, 'r') as f:
        results = json.load(f)
    
    bboxes = results['bboxes_3d']
    scores = results['scores_3d']
    
    # Print summary
    print(f"Point cloud: {len(points):,} points (downsampled)")
    print(f"Detections: {len(bboxes)} objects found\n")
    
    if len(bboxes) > 0:
        print(f"{'ID':<4} {'Score':<8} {'X':<8} {'Y':<8} {'Z':<8} {'L':<7} {'W':<7} {'H':<7}")
        print(f"{'-'*70}")
        
        for i, (bbox, score) in enumerate(zip(bboxes, scores)):
            x, y, z, l, w, h, rot = bbox
            print(f"{i:<4} {score:<8.3f} {x:<8.2f} {y:<8.2f} {z:<8.2f} {l:<7.2f} {w:<7.2f} {h:<7.2f}")
    else:
        print("No objects detected above the confidence threshold.")
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add point cloud
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color=intensity,
            colorscale='Viridis',
            opacity=0.3
        ),
        name='Point Cloud',
        hoverinfo='skip'
    ))
    
    # Add bounding boxes
    for i, (bbox, score) in enumerate(zip(bboxes, scores)):
        corners = bbox_3d_corners(bbox)
        x_lines, y_lines, z_lines = create_bbox_edges(corners)
        
        # Color by confidence
        if score > 0.8:
            color = 'green'
            conf_level = 'High'
        elif score > 0.5:
            color = 'yellow'
            conf_level = 'Medium'
        else:
            color = 'red'
            conf_level = 'Low'
        
        fig.add_trace(go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode='lines',
            line=dict(color=color, width=4),
            name=f'Car {i} ({conf_level}: {score:.2f})',
            hovertext=f'<b>Car {i}</b><br>Score: {score:.3f}<br>Position: ({bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f})<br>Size: L={bbox[3]:.2f}m, W={bbox[4]:.2f}m, H={bbox[5]:.2f}m',
            hoverinfo='text'
        ))
    
    # Update layout (use original filename for display)
    pcd_name = Path(original_pcd_path).name
    fig.update_layout(
        title=f'3D Object Detection Results - {pcd_name}<br><sub>Green: High confidence (>0.8) | Yellow: Medium (>0.5) | Red: Low (≤0.5)</sub>',
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Z (meters)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=1400,
        height=900,
        showlegend=True,
        hovermode='closest'
    )
    
    # Save HTML
    fig.write_html(output_html)
    
    # Get hostname for remote access
    import socket
    hostname = socket.gethostname()
    
    print(f"\n{'='*70}")
    print(f"✓ Visualization saved to: {output_html}")
    print(f"{'='*70}\n")
    print("To view the visualization, run:")
    print(f"  ./serve.sh")
    print(f"\nThen open in your local browser:")
    print(f"  http://0.0.0.0:8000/{output_html}")
    print(f"  (or use the server's IP address if hostname doesn't work)\n")


def main():
    parser = argparse.ArgumentParser(
        description='Easy 3D object detection and visualization pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_and_visualize.py demo/data/kitti/000008.bin
  python detect_and_visualize.py demo/data/kitti/000008.pcd
  python detect_and_visualize.py demo/data/kitti/000008.bin --score-thr 0.5
  python detect_and_visualize.py demo/data/kitti/000003.pcd --score-thr 0.3
        """
    )
    
    parser.add_argument(
        'pcd_file',
        type=str,
        help='Path to point cloud file (.bin or .pcd format)'
    )
    
    parser.add_argument(
        '--score-thr',
        type=float,
        default=CONFIG['score_threshold'],
        help=f'Confidence score threshold (default: {CONFIG["score_threshold"]})'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=CONFIG['output_dir'],
        help=f'Directory to save detection results (default: {CONFIG["output_dir"]})'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.pcd_file):
        print(f"Error: Input file not found: {args.pcd_file}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use fixed output HTML filename
    output_html = CONFIG['output_html']
    
    # Run detection (returns pred_file and bin_path)
    pred_file, bin_path = run_detection(args.pcd_file, args.output_dir, args.score_thr)
    
    # Create visualization (pass original path for display, bin_path for loading)
    create_visualization(args.pcd_file, bin_path, pred_file, output_html)
    
    print("Pipeline completed successfully! 🎉\n")


if __name__ == '__main__':
    main()

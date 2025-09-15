import os

import numpy as np
from pypcd import pypcd


def pcd_to_bin(pcd_path, bin_path=None):
    """
    Convert PCD format point cloud to BIN format.
    
    Args:
        pcd_path (str): Path to input PCD file
        bin_path (str, optional): Path to output BIN file. If None, generates from input path.
    
    Returns:
        str: Path to the output BIN file
    """
    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"PCD file not found: {pcd_path}")
    
    # Generate output path if not provided
    if bin_path is None:
        bin_path = pcd_path.replace('.pcd', '.bin')
    
    # Load PCD data
    pcd_data = pypcd.PointCloud.from_path(pcd_path)
    
    # Extract point cloud data (x, y, z, intensity)
    points = np.zeros([pcd_data.pc_data.shape[0], 4], dtype=np.float32)
    points[:, 0] = pcd_data.pc_data['x'].copy()
    points[:, 1] = pcd_data.pc_data['y'].copy()
    points[:, 2] = pcd_data.pc_data['z'].copy()
    
    # Handle intensity field (might not exist in all PCD files)
    if 'intensity' in pcd_data.pc_data.dtype.names:
        points[:, 3] = pcd_data.pc_data['intensity'].copy().astype(np.float32)
    else:
        # Set default intensity if not available
        points[:, 3] = 0.0
    
    # Save as binary file
    with open(bin_path, 'wb') as f:
        f.write(points.tobytes())
    
    print(f"Converted PCD with {points.shape[0]} points to BIN format: {bin_path}")
    return bin_path


if __name__ == '__main__':
    # Example usage
    pcd_file = '/home/gwe/source/mmdetection3d/data/kitti/pcd_demo/0000000000.pcd'
    if os.path.exists(pcd_file):
        bin_file = pcd_to_bin(pcd_file, 'point_cloud_data.bin')
        print(f"Converted {pcd_file} to {bin_file}")
    else:
        print(f"Example PCD file not found: {pcd_file}")

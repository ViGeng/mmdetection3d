# 3D Object Detection Pipeline

Simple pipeline for detecting cars in point cloud data and visualizing results.

## Quick Start

```bash
# 1. Run detection
python detect_and_visualize.py demo/data/kitti/000008.bin

# 2. Start server
./serve.sh

# 3. Open in browser (replace gpu-01 with your server's hostname/IP)
http://gpu-01:8000/detection_results.html
```

## Usage

```bash
# Basic detection
python detect_and_visualize.py <path_to_bin_file>

# Custom confidence threshold
python detect_and_visualize.py demo/data/kitti/000008.bin --score-thr 0.5

# Different server port
./serve.sh 9000
```

## Configuration

Edit `CONFIG` in `detect_and_visualize.py`:
- `model` - Model config file
- `checkpoint` - Model weights
- `device` - cuda:0 or cpu
- `score_threshold` - Default confidence threshold (0.3)
- `downsample_ratio` - Point cloud downsampling (4)

## Output

- `detection_results.html` - Visualization (always overwritten, no clutter)
- `detection_outputs/preds/*.json` - Detection results per file

## Features

- Interactive 3D visualization (rotate, zoom, hover)
- Color-coded confidence: Green (>0.8), Yellow (>0.5), Red (≤0.5)
- Fixed output filename for easy access
- Separate server for flexibility

## Tips

💡 Keep server running, refresh browser after new detection  
💡 Server must be accessible from your local machine (use hostname or IP, not localhost)

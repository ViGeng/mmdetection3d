# 3D Object Detection Pipeline - Usage Guide

## Overview

`detect_and_visualize.py` is an easy-to-use script for running 3D object detection on point cloud data and creating interactive visualizations.

## Features

- **Multiple Format Support**: Accepts both `.bin` and `.pcd` point cloud formats
- **Automatic Conversion**: PCD files are automatically converted to BIN format for processing
- **Interactive Visualization**: Generates HTML visualization with Plotly
- **Confidence Filtering**: Filter detections by confidence threshold
- **Color-coded Results**: Objects colored by confidence level (Green/Yellow/Red)

## Usage

### Basic Usage

```bash
# Detect objects in a BIN file
python detect_and_visualize.py demo/data/kitti/000008.bin

# Detect objects in a PCD file (auto-converts to BIN)
python detect_and_visualize.py demo/data/kitti/000008.pcd
```

### With Confidence Threshold

```bash
# Only show high-confidence detections
python detect_and_visualize.py demo/data/kitti/000008.bin --score-thr 0.7

# Show more detections (lower threshold)
python detect_and_visualize.py demo/data/kitti/000008.pcd --score-thr 0.3
```

### Custom Output Directory

```bash
python detect_and_visualize.py demo/data/kitti/000008.bin --output-dir my_results
```

## File Format Support

### BIN Format (KITTI)
- Native format, used directly for detection
- Structure: Nx4 array (x, y, z, intensity)

### PCD Format
- Automatically converted to BIN before detection
- Converted file saved alongside original with `.bin` extension
- Original PCD file remains unchanged

## Output

1. **JSON Detection Results**: `detection_outputs/preds/<filename>.json`
2. **Interactive HTML Visualization**: `detection_results.html` (fixed name)

## Viewing Results

Start the web server:
```bash
./serve.sh
```

Then open in your browser:
```
http://0.0.0.0:8000/detection_results.html
```

## Confidence Level Colors

- 🟢 **Green**: High confidence (> 0.8)
- 🟡 **Yellow**: Medium confidence (> 0.5)
- 🔴 **Red**: Low confidence (≤ 0.5)

## Dependencies

- mmdet3d
- numpy
- plotly
- pypcd (for PCD file support)

## Implementation Details

The script uses:
- `demo/format_converter.py` for PCD to BIN conversion
- PointPillars model for 3D object detection
- Plotly for interactive 3D visualization

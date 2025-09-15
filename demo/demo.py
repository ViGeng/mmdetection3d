#!/usr/bin/env python3
import logging
import os
import tempfile
from argparse import ArgumentParser

from format_converter import pcd_to_bin
from mmengine.logging import print_log

from mmdet3d.apis import LidarDet3DInferencer

# ==================== CONFIGURATION SECTION ====================
# You can modify these default values instead of passing command line arguments
CONFIG = {
    # Input/Output paths
    'pcd': "/home/gwe/source/mmdetection3d/data/kitti/pcd_demo/0000000000.pcd",  # Path to PCD file (e.g., 'demo/data/kitti/sample.pcd')
    'model': 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py',  # Config file
    'weights': 'hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth',  # Checkpoint file
    'out_dir': 'outputs',  # Output directory
    
    # Detection parameters
    'device': 'cuda:0',  # Device for inference
    'pred_score_thr': 0.3,  # Detection score threshold
    
    # Visualization options
    'show': False,  # Show online visualization
    'wait_time': -1,  # Wait time for visualization
    'no_save_vis': False,  # Don't save visualization
    'no_save_pred': False,  # Don't save predictions
    'print_result': True,  # Print detection results
    
    # Conversion options
    'keep_bin': False,  # Keep converted BIN file
}


def update_config(**kwargs):
    """
    Helper function to update configuration values.
    
    Example usage:
        update_config(
            pcd='my_pointcloud.pcd',
            device='cpu',
            show=True
        )
    """
    for key, value in kwargs.items():
        if key in CONFIG:
            CONFIG[key] = value
        else:
            print_log(f"Warning: Unknown config key '{key}'", logger='current', level=logging.WARNING)


# ==================== USAGE EXAMPLES ====================
# Uncomment and modify one of these examples to set your configuration:

# Example 1: Basic setup for a specific PCD file
# update_config(
#     pcd='demo/data/kitti/000008.pcd',
#     show=True,
#     print_result=True
# )

# Example 2: CPU inference with custom output directory
# update_config(
#     pcd='my_data/pointcloud.pcd',
#     device='cpu',
#     out_dir='my_results',
#     keep_bin=True
# )

# Example 3: High-precision detection
# update_config(
#     pcd='demo/data/scan.pcd',
#     pred_score_thr=0.1,
#     show=True,
#     wait_time=2.0
# )

# ================================================================


def parse_args():
    """Parse command line arguments with config defaults."""
    parser = ArgumentParser(description='3D Object Detection Demo for PCD files')
    
    # Positional arguments (optional if config is set)
    parser.add_argument('pcd', nargs='?', default=CONFIG['pcd'], 
                       help='Point cloud file in PCD format')
    parser.add_argument('model', nargs='?', default=CONFIG['model'],
                       help='Config file')
    parser.add_argument('weights', nargs='?', default=CONFIG['weights'],
                       help='Checkpoint file')
    
    # Optional arguments with config defaults
    parser.add_argument('--device', default=CONFIG['device'], 
                       help='Device used for inference')
    parser.add_argument('--pred-score-thr', type=float, default=CONFIG['pred_score_thr'],
                       help='bbox score threshold')
    parser.add_argument('--out-dir', type=str, default=CONFIG['out_dir'],
                       help='Output directory of prediction and visualization results.')
    parser.add_argument('--show', action='store_true', default=CONFIG['show'],
                       help='Show online visualization results')
    parser.add_argument('--wait-time', type=float, default=CONFIG['wait_time'],
                       help='The interval of show (s). Demo will be blocked in showing '
                            'results, if wait_time is -1. Defaults to -1.')
    parser.add_argument('--no-save-vis', action='store_true', default=CONFIG['no_save_vis'],
                       help='Do not save detection visualization results')
    parser.add_argument('--no-save-pred', action='store_true', default=CONFIG['no_save_pred'],
                       help='Do not save detection prediction results')
    parser.add_argument('--print-result', action='store_true', default=CONFIG['print_result'],
                       help='Whether to print the results.')
    parser.add_argument('--keep-bin', action='store_true', default=CONFIG['keep_bin'],
                       help='Keep the converted BIN file after processing')
    
    return parser.parse_args()


def main():
    """Main function to run PCD file detection."""
    args = parse_args()
    
    # Validate required arguments
    if not args.pcd:
        print_log("Error: PCD file path is required. Set it in CONFIG or pass as argument.", 
                 logger='current', level=logging.ERROR)
        print_log("Example: python demo.py input.pcd [model] [weights]", logger='current')
        return
    
    if not args.model:
        print_log("Error: Model config is required. Set it in CONFIG or pass as argument.", 
                 logger='current', level=logging.ERROR)
        return
        
    if not args.weights:
        print_log("Error: Model weights are required. Set it in CONFIG or pass as argument.", 
                 logger='current', level=logging.ERROR)
        return
    
    # Print current configuration
    print_log("=== Detection Configuration ===", logger='current')
    print_log(f"PCD file: {args.pcd}", logger='current')
    print_log(f"Model config: {args.model}", logger='current')
    print_log(f"Model weights: {args.weights}", logger='current')
    print_log(f"Output directory: {args.out_dir}", logger='current')
    print_log(f"Device: {args.device}", logger='current')
    print_log(f"Score threshold: {args.pred_score_thr}", logger='current')
    print_log("==============================", logger='current')
    
    # Validate input file
    if not os.path.exists(args.pcd):
        print_log(f"PCD file not found: {args.pcd}", logger='current', level=logging.ERROR)
        return
    
    if not args.pcd.lower().endswith('.pcd'):
        print_log(f"Input file must be a PCD file: {args.pcd}", logger='current', level=logging.ERROR)
        return
    
    # Convert PCD to BIN format
    print_log(f"Converting PCD file to BIN format: {args.pcd}", logger='current')
    
    if args.keep_bin:
        # Create BIN file in the same directory as PCD
        bin_file = args.pcd.replace('.pcd', '.bin')
    else:
        # Create temporary BIN file
        temp_dir = tempfile.mkdtemp()
        bin_file = os.path.join(temp_dir, 'temp_pointcloud.bin')
    
    try:
        pcd_to_bin(args.pcd, bin_file)
        print_log(f"Successfully converted to: {bin_file}", logger='current')
    except Exception as e:
        print_log(f"Failed to convert PCD to BIN: {e}", logger='current', level=logging.ERROR)
        return
    
    # Prepare arguments for inference
    call_args = {
        'inputs': {'points': bin_file},
        'pred_score_thr': args.pred_score_thr,
        'out_dir': args.out_dir,
        'show': args.show,
        'wait_time': args.wait_time,
        'no_save_vis': args.no_save_vis,
        'no_save_pred': args.no_save_pred,
        'print_result': args.print_result
    }
    
    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''
    
    # Check display device
    if os.environ.get('DISPLAY') is None and call_args['show']:
        print_log(
            'Display device not found. `--show` is forced to False',
            logger='current',
            level=logging.WARNING)
        call_args['show'] = False
    
    # Initialize inferencer
    print_log(f"Initializing inferencer with model: {args.model}", logger='current')
    try:
        inferencer = LidarDet3DInferencer(
            model=args.model,
            weights=args.weights,
            device=args.device
        )
    except Exception as e:
        print_log(f"Failed to initialize inferencer: {e}", logger='current', level=logging.ERROR)
        return
    
    # Run inference
    print_log("Running 3D object detection...", logger='current')
    try:
        inferencer(**call_args)
        print_log("Detection completed successfully!", logger='current')
    except Exception as e:
        print_log(f"Detection failed: {e}", logger='current', level=logging.ERROR)
        return
    
    # Clean up temporary BIN file if not keeping it
    if not args.keep_bin and os.path.exists(bin_file):
        try:
            os.remove(bin_file)
            if bin_file.startswith(tempfile.gettempdir()):
                os.rmdir(os.path.dirname(bin_file))
        except:
            pass  # Ignore cleanup errors
    
    # Print results location
    if call_args['out_dir'] != '' and not (call_args['no_save_vis'] and call_args['no_save_pred']):
        print_log(f'Results have been saved to: {call_args["out_dir"]}', logger='current')


if __name__ == '__main__':
    main()
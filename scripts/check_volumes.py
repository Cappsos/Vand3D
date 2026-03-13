import numpy as np
import nibabel as nib
import argparse
import os
from pathlib import Path

def load_volume(volume_path):
    """Load volume from .nii.gz or .npy file"""
    if volume_path.endswith('.nii.gz') or volume_path.endswith('.nii'):
        img = nib.load(volume_path)
        volume = img.get_fdata()
    elif volume_path.endswith('.npy'):
        volume = np.load(volume_path)
    else:
        raise ValueError(f"Unsupported file format: {volume_path}")
    
    return volume

def analyze_volume(volume_path):
    """Analyze a single volume and return statistics"""
    try:
        volume = load_volume(volume_path)
        
        stats = {
            'path': volume_path,
            'shape': volume.shape,
            'min': float(np.min(volume)),
            'max': float(np.max(volume)),
            'mean': float(np.mean(volume)),
            'std': float(np.std(volume)),
            'median': float(np.median(volume)),
            'non_zero_count': int(np.count_nonzero(volume)),
            'total_voxels': int(volume.size),
            'dtype': str(volume.dtype)
        }
        
        return stats
    
    except Exception as e:
        print(f"Error processing {volume_path}: {e}")
        return None

def print_stats(stats):
    """Print volume statistics in a formatted way"""
    if stats is None:
        return
    
    print(f"\nVolume: {os.path.basename(stats['path'])}")
    print(f"  Shape: {stats['shape']}")
    print(f"  Data type: {stats['dtype']}")
    print(f"  Min: {stats['min']:.6f}")
    print(f"  Max: {stats['max']:.6f}")
    print(f"  Mean: {stats['mean']:.6f}")
    print(f"  Std: {stats['std']:.6f}")
    print(f"  Median: {stats['median']:.6f}")
    print(f"  Non-zero voxels: {stats['non_zero_count']:,} / {stats['total_voxels']:,}")
    print(f"  Non-zero ratio: {stats['non_zero_count']/stats['total_voxels']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Analyze volume statistics')
    parser.add_argument('volume_path', type=str, help='Path to volume file (.nii.gz, .nii, or .npy)')
    parser.add_argument('--recursive', '-r', action='store_true', 
                       help='Process all volumes in directory recursively')
    parser.add_argument('--pattern', '-p', type=str, default='*.nii.gz',
                       help='File pattern for recursive search (default: *.nii.gz)')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.volume_path):
        # Single file
        stats = analyze_volume(args.volume_path)
        print_stats(stats)
        
    elif os.path.isdir(args.volume_path) and args.recursive:
        # Directory with recursive search
        from glob import glob
        
        # Support multiple patterns
        patterns = args.pattern.split(',')
        all_files = []
        
        for pattern in patterns:
            pattern = pattern.strip()
            search_pattern = os.path.join(args.volume_path, '**', pattern)
            files = glob(search_pattern, recursive=True)
            all_files.extend(files)
        
        all_files = sorted(set(all_files))  # Remove duplicates and sort
        
        print(f"Found {len(all_files)} volume files")
        
        for volume_path in all_files:
            stats = analyze_volume(volume_path)
            print_stats(stats)
            
    elif os.path.isdir(args.volume_path):
        # Directory without recursive search
        from glob import glob
        
        patterns = args.pattern.split(',')
        all_files = []
        
        for pattern in patterns:
            pattern = pattern.strip()
            search_pattern = os.path.join(args.volume_path, pattern)
            files = glob(search_pattern)
            all_files.extend(files)
        
        all_files = sorted(set(all_files))
        
        print(f"Found {len(all_files)} volume files")
        
        for volume_path in all_files:
            stats = analyze_volume(volume_path)
            print_stats(stats)
    
    else:
        print(f"Error: {args.volume_path} is not a valid file or directory")

if __name__ == "__main__":
    main()
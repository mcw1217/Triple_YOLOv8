import os
from pathlib import Path

def rename_files(data_folder):
    for subset in ['train', 'val', 'test']:
        subset_path = Path(data_folder) / subset
        
        rgb_path = subset_path / 'RGB'
        depth_path = subset_path / 'D'
        labels_path = subset_path / 'labels'
        
        # Get sorted file lists
        rgb_files = sorted(rgb_path.glob('*'))
        depth_files = sorted(depth_path.glob('*'))
        labels_files = sorted(labels_path.glob('*'))
        
        assert len(rgb_files) == len(depth_files) == len(labels_files), "Mismatched number of files in folders"

        for i, (rgb_file, depth_file, labels_file) in enumerate(zip(rgb_files, depth_files, labels_files), start=1):
            new_name = f"{i:06d}"
            
            # Determine file extensions
            rgb_ext = rgb_file.suffix
            depth_ext = depth_file.suffix
            labels_ext = labels_file.suffix
            
            # Define new file paths
            new_rgb_file = rgb_path / f"{new_name}{rgb_ext}"
            new_depth_file = depth_path / f"{new_name}{depth_ext}"
            new_labels_file = labels_path / f"{new_name}{labels_ext}"
            
            # Rename files
            rgb_file.rename(new_rgb_file)
            depth_file.rename(new_depth_file)
            labels_file.rename(new_labels_file)
            
            print(f"Renamed: {rgb_file} -> {new_rgb_file}")
            print(f"Renamed: {depth_file} -> {new_depth_file}")
            print(f"Renamed: {labels_file} -> {new_labels_file}")

if __name__ == "__main__":
    data_folder = 'dataset/new_data'  # Adjust this path if necessary
    rename_files(data_folder)

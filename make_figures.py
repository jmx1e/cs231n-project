import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(filepath):
    """Load and return image as numpy array"""
    if os.path.exists(filepath):
        return np.array(Image.open(filepath))
    else:
        # Return placeholder if image doesn't exist
        return np.zeros((256, 256, 3), dtype=np.uint8)

def get_image_paths(base_dir, task, method, dataset, image_id):
    """Get paths for different reconstruction methods"""
    paths = {}
    
    # Ground truth and measurement from baselines
    baseline_dir = os.path.join(base_dir, "baselines")
    if task == "symmetric":
        task_prefix = "hdr-"
    elif task == "over":
        task_prefix = "hdr-highlight-"
    elif task == "under":
        task_prefix = "hdr-shadow-"
    
    baseline_folder = f"{task_prefix}baseline-{dataset}"
    paths['gt'] = os.path.join(baseline_dir, baseline_folder, "original", f"{image_id:05d}.png")
    paths['meas'] = os.path.join(baseline_dir, baseline_folder, "measurements", f"{image_id:05d}.png")
    paths['naive'] = os.path.join(baseline_dir, baseline_folder, "reconstructed", f"{image_id:05d}.png")
    
    # Method results from task-specific directories
    task_dir = os.path.join(base_dir, task)
    method_map = {
        'pixel-4k': f'pixel-{dataset}',
        'pixel-1k': f'pixel1-{dataset}',
        'ldm-1k': f'ldm-{dataset}',
        'sd1.5-1k': f'sd15-{dataset}'
    }
    
    for method_key, method_name in method_map.items():
        method_path = os.path.join(task_dir, method_name, "samples", f"{image_id:05d}_run0000.png")
        paths[method_key] = method_path
    
    return paths

def create_grid(images, save_path, figsize=(16, 14), is_small_grid=False):
    """Create and save image grid"""
    rows, cols = len(images), len(images[0])
    
    fig = plt.figure(figsize=figsize)
    fig.patch.set_alpha(0.0)
    
    # Create grid with tight spacing for both small and large grids
    gs = fig.add_gridspec(rows, cols, hspace=0.02, wspace=0.02)
    
    axes = [[fig.add_subplot(gs[i, j]) for j in range(cols)] for i in range(rows)]
    
    if rows == 1:
        axes = [axes[0]]
    if cols == 1:
        axes = [[row[0]] for row in axes]
    
    # Add column labels
    col_labels = ['Measurement', 'Baseline', 'SD1.5-1k', 'LDM-1k', 'Pixel-1k', 'Pixel-4k', 'Ground Truth']
    for j, label in enumerate(col_labels):
        axes[0][j].set_title(label, fontsize=12, pad=10)
    
    for i in range(len(images)):
        for j in range(cols):
            axes[i][j].imshow(images[i][j])
            axes[i][j].axis('off')
    
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1, transparent=True)
    plt.close()

def make_small_grids(base_dir):
    """Create 4x7 grids for each task with selected samples"""
    tasks = ['symmetric', 'over', 'under']
    methods = ['meas', 'naive', 'sd1.5-1k', 'ldm-1k', 'pixel-1k', 'pixel-4k', 'gt']
    
    # Different images for each task - now 2 samples each
    ffhq_samples = {
        'symmetric': [2, 3],
        'over': [0, 1], 
        'under': [4, 9]
    }
    imagenet_samples = {
        'symmetric': [2, 7],
        'over': [3, 8],
        'under': [1, 5]
    }
    
    # Create figures directory
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    for task in tasks:
        print(f"Creating small grid for {task}...")
        grid_images = []
        
        # FFHQ samples (2 rows)
        for img_id in ffhq_samples[task]:
            row_images = []
            for method in methods:
                paths = get_image_paths(base_dir, task, method, 'ffhq', img_id)
                img = load_image(paths[method])
                row_images.append(img)
            grid_images.append(row_images)
        
        # ImageNet samples (2 rows)
        for img_id in imagenet_samples[task]:
            row_images = []
            for method in methods:
                paths = get_image_paths(base_dir, task, method, 'imagenet', img_id)
                img = load_image(paths[method])
                row_images.append(img)
            grid_images.append(row_images)
        
        save_path = os.path.join(figures_dir, f"grid_small_{task}.png")
        create_grid(grid_images, save_path, figsize=(14, 8), is_small_grid=True)

def make_large_grids(base_dir):
    """Create 10x7 grids for each task and dataset combination"""
    tasks = ['symmetric', 'over', 'under']
    datasets = ['ffhq', 'imagenet']
    methods = ['meas', 'naive', 'sd1.5-1k', 'ldm-1k', 'pixel-1k', 'pixel-4k', 'gt']
    
    # Create figures directory
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    for task in tasks:
        for dataset in datasets:
            print(f"Creating large grid for {task} - {dataset}...")
            grid_images = []
            
            # All 10 images (0-9) as rows
            for img_id in range(10):
                row_images = []
                for method in methods:
                    paths = get_image_paths(base_dir, task, method, dataset, img_id)
                    img = load_image(paths[method])
                    row_images.append(img)
                grid_images.append(row_images)
            
            save_path = os.path.join(figures_dir, f"grid_large_{task}_{dataset}.png")
            create_grid(grid_images, save_path, figsize=(14, 20))

def main():
    base_dir = "results"
    
    # Check if results directory exists
    if not os.path.exists(base_dir):
        print(f"Results directory '{base_dir}' not found!")
        return
    
    print("Creating figure grids...")
    
    # Create small grids (6x7)
    make_small_grids(base_dir)
    
    # Create large grids (10x7) 
    make_large_grids(base_dir)
    
    print("All grids created successfully!")
    print("Small grids: figures/grid_small_symmetric.png, figures/grid_small_over.png, figures/grid_small_under.png")
    print("Large grids: figures/grid_large_[task]_[dataset].png (6 total)")

if __name__ == "__main__":
    main()

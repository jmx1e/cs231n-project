import os
import json
import matplotlib.pyplot as plt
import numpy as np
import re

def parse_nfe_from_folder(folder_name):
    """Extract NFE value from folder name like 'pixel-50', 'pixel-1k', etc."""
    match = re.search(r'pixel-(\d+)([k]?)', folder_name)
    if match:
        number = int(match.group(1))
        suffix = match.group(2)
        if suffix == 'k':
            return number * 1000
        return number
    return None

def read_metrics(metrics_path):
    """Read SSIM and LPIPS values from metrics.json"""
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        
        # Get SSIM mean values and compute average
        ssim_values = data.get('ssim', {}).get('mean', [])
        ssim_avg = np.mean(ssim_values) if ssim_values else None
        
        # Get LPIPS mean values and compute average
        lpips_values = data.get('lpips', {}).get('mean', [])
        lpips_avg = np.mean(lpips_values) if lpips_values else None
        
        return ssim_avg, lpips_avg
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None, None

def main():
    ablation_dir = 'results/ablation'
    
    if not os.path.exists(ablation_dir):
        print(f"Directory {ablation_dir} not found!")
        return
    
    # Collect data points
    data_points = []
    
    for folder_name in os.listdir(ablation_dir):
        if folder_name.startswith('pixel-'):
            nfe = parse_nfe_from_folder(folder_name)
            if nfe is not None:
                metrics_path = os.path.join(ablation_dir, folder_name, 'metrics.json')
                ssim_avg, lpips_avg = read_metrics(metrics_path)
                
                if ssim_avg is not None and lpips_avg is not None:
                    data_points.append((nfe, ssim_avg, lpips_avg))
                    print(f"NFE: {nfe}, SSIM: {ssim_avg:.4f}, LPIPS: {lpips_avg:.4f}")
    
    if not data_points:
        print("No valid data points found!")
        return
    
    # Sort by NFE
    data_points.sort(key=lambda x: x[0])
    
    # Extract data for plotting
    nfe_values = [point[0] for point in data_points]
    ssim_values = [point[1] for point in data_points]
    lpips_values = [point[2] for point in data_points]
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot SSIM on left y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('NFE (Neural Function Evaluations)', fontsize=18)
    ax1.set_ylabel('SSIM', color=color1, fontsize=18)
    line1 = ax1.plot(nfe_values, ssim_values, 'o-', color=color1, label='SSIM', linewidth=2, markersize=6)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=16)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for LPIPS
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('LPIPS', color=color2, fontsize=18)
    line2 = ax2.plot(nfe_values, lpips_values, 's-', color=color2, label='LPIPS', linewidth=2, markersize=6)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=16)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('ablation_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved as 'ablation_plot.png'")

if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import numpy as np

def create_clean_table(csv_file_path, save_path='professional_table.png'):
    # Read and process the CSV
    df = pd.read_csv(csv_file_path)
    
    # Extract data starting from row 2 (skip the header rows)
    data = df.iloc[2:].reset_index(drop=True)
    
    # Clean column names for display
    display_columns = ['Task', 'Method', 'PSNR', 'SSIM', 'LPIPS', 'PSNR', 'SSIM', 'LPIPS']
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for display
    table_data = []
    for _, row in data.iterrows():
        row_data = []
        for i, val in enumerate(row):
            if pd.isna(val):
                row_data.append('')
            elif i > 1:  # Format numeric columns
                try:
                    num_val = float(val)
                    if num_val < 1:
                        row_data.append(f"{num_val:.3f}")
                    else:
                        row_data.append(f"{num_val:.2f}")
                except:
                    row_data.append(str(val))
            else:
                row_data.append(str(val))
        table_data.append(row_data)
    
    # Create the table
    table = ax.table(cellText=table_data,
                    colLabels=display_columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.12, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Header styling
    for i in range(len(display_columns)):
        table[(0, i)].set_facecolor('#f0f0f0')
        table[(0, i)].set_text_props(weight='bold')
        table[(0, i)].set_edgecolor('#cccccc')
    
    # Row styling
    for i in range(1, len(table_data) + 1):
        for j in range(len(display_columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f9f9f9')
            else:
                table[(i, j)].set_facecolor('white')
            table[(i, j)].set_edgecolor('#cccccc')
            
            # Bold task names
            if j == 0 and table_data[i-1][j]:  # Task column, non-empty
                table[(i, j)].set_text_props(weight='bold')
    
    # Add dataset headers manually
    ax.text(0.225, 0.95, 'FFHQ', transform=ax.transAxes, ha='center', va='center',
            fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor='#e8e8e8', edgecolor='#cccccc'))
    
    ax.text(0.775, 0.95, 'ImageNet', transform=ax.transAxes, ha='center', va='center',
            fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor='#e8e8e8', edgecolor='#cccccc'))
    
    plt.title('Model Performance Metrics', fontsize=16, fontweight='bold', pad=40)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def create_html_table(csv_file_path, save_path='table.html'):
    # Read and process data
    df = pd.read_csv(csv_file_path)
    data = df.iloc[2:].reset_index(drop=True)
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }
            .table-container {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                text-align: center;
                color: #333;
                margin-bottom: 30px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 14px;
            }
            .dataset-header {
                background-color: #f0f0f0;
                font-weight: bold;
                text-align: center;
                padding: 12px;
                border: 1px solid #ddd;
                font-size: 16px;
            }
            .metric-header {
                background-color: #e8e8e8;
                font-weight: bold;
                text-align: center;
                padding: 10px;
                border: 1px solid #ddd;
            }
            .task-cell {
                font-weight: bold;
                background-color: #f9f9f9;
                padding: 8px 12px;
                border: 1px solid #ddd;
                text-align: center;
            }
            .method-cell {
                padding: 8px 12px;
                border: 1px solid #ddd;
                text-align: center;
                background-color: #f9f9f9;
            }
            .data-cell {
                padding: 8px 12px;
                border: 1px solid #ddd;
                text-align: center;
            }
            .row-even {
                background-color: #f9f9f9;
            }
            .row-odd {
                background-color: white;
            }
        </style>
    </head>
    <body>
        <div class="table-container">
            <h1>Model Performance Comparison</h1>
            <table>
                <thead>
                    <tr>
                        <th rowspan="2" class="dataset-header">Task</th>
                        <th rowspan="2" class="dataset-header">Method</th>
                        <th colspan="3" class="dataset-header">FFHQ</th>
                        <th colspan="3" class="dataset-header">ImageNet</th>
                    </tr>
                    <tr>
                        <th class="metric-header">PSNR ↑</th>
                        <th class="metric-header">SSIM ↑</th>
                        <th class="metric-header">LPIPS ↓</th>
                        <th class="metric-header">PSNR ↑</th>
                        <th class="metric-header">SSIM ↑</th>
                        <th class="metric-header">LPIPS ↓</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Add data rows
    current_task = None
    row_count = 0
    
    for _, row in data.iterrows():
        task = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ''
        method = str(row.iloc[1]) if pd.notna(row.iloc[1]) else ''
        
        # Determine if this is a new task
        if task and task != current_task:
            current_task = task
            task_display = task
        else:
            task_display = ''
        
        row_class = 'row-even' if row_count % 2 == 0 else 'row-odd'
        
        html_content += f'<tr class="{row_class}">\n'
        html_content += f'    <td class="task-cell">{task_display}</td>\n'
        html_content += f'    <td class="method-cell">{method}</td>\n'
        
        # Add metric values
        for i in range(2, len(row)):
            val = row.iloc[i]
            if pd.isna(val):
                formatted_val = ''
            else:
                try:
                    num_val = float(val)
                    if num_val < 1:
                        formatted_val = f"{num_val:.3f}"
                    else:
                        formatted_val = f"{num_val:.2f}"
                except:
                    formatted_val = str(val)
            
            html_content += f'    <td class="data-cell">{formatted_val}</td>\n'
        
        html_content += '</tr>\n'
        row_count += 1
    
    html_content += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(save_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML table saved to {save_path}")

# Usage example
if __name__ == "__main__":
    csv_file = 'metrics_table.csv'  # Replace with your file path
    
    # Create matplotlib table
    print("Creating matplotlib table...")
    create_clean_table(csv_file)
    
    # Create HTML table
    print("Creating HTML table...")
    create_html_table(csv_file)
    
    print("Done! Check the generated files.")

# import os
# import re
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import json
# from matplotlib.backends.backend_pdf import PdfPages
# from tabulate import tabulate

# # Try to import great_tables, fallback gracefully if not available
# try:
#     from great_tables import GT, md
#     GREAT_TABLES_AVAILABLE = True
# except ImportError:
#     GREAT_TABLES_AVAILABLE = False
#     print("Warning: great_tables not available. Install with: pip install great_tables")

# def parse_eval_file(filepath):
#     """Parse evaluation JSON file to extract PSNR, SSIM, LPIPS metrics"""
#     metrics = {}
    
#     # Try JSON file first
#     json_path = filepath.replace('eval.md', 'metrics.json')
    
#     if os.path.exists(json_path):
#         try:
#             with open(json_path, 'r') as f:
#                 data = json.load(f)
            
#             # Extract metrics and calculate mean from the 10 values
#             for metric in ['psnr', 'ssim', 'lpips']:
#                 if metric in data and 'mean' in data[metric]:
#                     values = data[metric]['mean']
#                     if isinstance(values, list) and len(values) >= 10:
#                         mean_val = np.mean(values)
#                         metrics[metric] = (mean_val, 0.0)  # No std, just set to 0
#                     else:
#                         print(f"Warning: {metric} data format unexpected in {json_path}")
#                         metrics[metric] = (0, 0)
#                 else:
#                     print(f"Warning: {metric} not found in {json_path}")
#                     metrics[metric] = (0, 0)
                    
#         except Exception as e:
#             print(f"Error reading JSON {json_path}: {e}")
#             metrics = {'psnr': (0, 0), 'ssim': (0, 0), 'lpips': (0, 0)}
#     else:
#         print(f"Warning: JSON file not found: {json_path}")
#         metrics = {'psnr': (0, 0), 'ssim': (0, 0), 'lpips': (0, 0)}
    
#     return metrics

# def get_eval_path(base_dir, task, method, dataset):
#     """Get path to evaluation file"""
#     # Method name mapping for folders
#     method_map = {
#         'baseline': 'baseline', 
#         'pixel-4k': 'pixel',
#         'pixel-1k': 'pixel1',
#         'ldm-1k': 'ldm',
#         'sd1.5-1k': 'sd15'
#     }
    
#     folder_name = method_map[method]
    
#     # Handle baseline case - it's in baselines directory
#     if method == 'baseline':
#         if task == 'symmetric':
#             task_prefix = 'hdr'
#         elif task == 'overexposed':
#             task_prefix = 'hdr-highlight'
#         elif task == 'underexposed':
#             task_prefix = 'hdr-shadow'
        
#         eval_path = os.path.join(base_dir, 'baselines', f'{task_prefix}-baseline-{dataset}', 'eval.md')
#     else:
#         # Other methods are in task-specific directories
#         # Map task names to actual directory names
#         task_dir_map = {
#             'symmetric': 'symmetric',
#             'overexposed': 'over', 
#             'underexposed': 'under'
#         }
#         actual_task_dir = task_dir_map[task]
#         eval_path = os.path.join(base_dir, actual_task_dir, f'{folder_name}-{dataset}', 'eval.md')
    
#     return eval_path

# def collect_all_metrics(base_dir):
#     """Collect all metrics into a structured dictionary"""
#     tasks = ['symmetric', 'overexposed', 'underexposed']
#     methods = ['baseline', 'pixel-4k', 'pixel-1k', 'ldm-1k', 'sd1.5-1k']
#     datasets = ['ffhq', 'imagenet']
    
#     data = {}
    
#     for task in tasks:
#         data[task] = {}
#         for method in methods:
#             data[task][method] = {}
#             for dataset in datasets:
#                 eval_path = get_eval_path(base_dir, task, method, dataset)
#                 print(f"Reading: {eval_path}")
#                 metrics_data = parse_eval_file(eval_path)
#                 data[task][method][dataset] = metrics_data
    
#     return data

# def format_metric(mean, std, metric_name):
#     """Format metric without std (just mean)"""
#     if metric_name == 'psnr':
#         return f"{mean:.2f}"
#     elif metric_name == 'ssim':
#         return f"{mean:.3f}"
#     elif metric_name == 'lpips':
#         return f"{mean:.3f}"
#     else:
#         return f"{mean:.3f}"

# def create_table(data):
#     """Create the formatted table"""
#     tasks = ['symmetric', 'overexposed', 'underexposed']
#     methods = ['baseline', 'pixel-4k', 'pixel-1k', 'ldm-1k', 'sd1.5-1k']
#     datasets = ['FFHQ', 'ImageNet']
#     metrics = ['PSNR', 'SSIM', 'LPIPS']
    
#     # Create multi-level column headers - Dataset as outer, Metric as inner
#     col_tuples = []
#     for dataset in datasets:
#         for metric in metrics:
#             col_tuples.append((dataset, metric))
    
#     columns = pd.MultiIndex.from_tuples(col_tuples, names=['Dataset', 'Metric'])
    
#     # Create multi-level row headers - Task as outer, Method as inner
#     row_tuples = []
#     for task in tasks:
#         for method in methods:
#             row_tuples.append((task.capitalize(), method))
    
#     index = pd.MultiIndex.from_tuples(row_tuples, names=['Task', 'Method'])
    
#     # Fill the table
#     table_data = []
#     for task in tasks:
#         for method in methods:
#             row = []
#             for dataset_name in ['ffhq', 'imagenet']:
#                 for metric_name in ['psnr', 'ssim', 'lpips']:
#                     mean, std = data[task][method][dataset_name][metric_name]
#                     formatted = format_metric(mean, std, metric_name)
#                     row.append(formatted)
#             table_data.append(row)
    
#     df = pd.DataFrame(table_data, index=index, columns=columns)
#     return df

# def create_pdf_table(df, output_path):
#     """Create and save table as PDF"""
#     fig, ax = plt.subplots(figsize=(16, 10))
#     ax.axis('off')
    
#     # Prepare the data matrix
#     tasks = ['Symmetric', 'Overexposed', 'Underexposed']
#     methods = ['baseline', 'pixel-4k', 'pixel-1k', 'ldm-1k', 'sd1.5-1k']
#     datasets = ['FFHQ', 'ImageNet']
#     metrics = ['PSNR', 'SSIM', 'LPIPS']
    
#     # Create the full data matrix including labels
#     rows = []
    
#     # Header row 1: Dataset names
#     header1 = ['', ''] + ['FFHQ'] * 3 + ['ImageNet'] * 3
#     rows.append(header1)
    
#     # Header row 2: Metric names  
#     header2 = ['Task', 'Method'] + metrics * 2
#     rows.append(header2)
    
#     # Data rows
#     for i, task in enumerate(tasks):
#         for j, method in enumerate(methods):
#             row_data = []
#             if j == 0:  # First method of each task
#                 row_data.append(task)
#             else:
#                 row_data.append('')  # Empty for merged appearance
#             row_data.append(method)
            
#             # Add metric values
#             dataset_map = {'ffhq': 0, 'imagenet': 1}  # Map to indices
#             for dataset_name in ['ffhq', 'imagenet']:
#                 for metric_name in ['psnr', 'ssim', 'lpips']:
#                     idx = (i * 5) + j  # Calculate row index in dataframe
#                     col_idx = dataset_map[dataset_name] * 3 + metrics.index(metric_name.upper())
#                     value = df.iloc[idx, col_idx]
#                     row_data.append(value)
            
#             rows.append(row_data)
    
#     # Create table using matplotlib's table
#     table = ax.table(cellText=rows[2:],  # Data rows only
#                     colLabels=header2,    # Column headers
#                     cellLoc='center',
#                     loc='center',
#                     bbox=[0, 0, 1, 1])
    
#     # Style the table
#     table.auto_set_font_size(False)
#     table.set_fontsize(11)
#     table.scale(1, 2)
    
#     # Add dataset header row manually
#     for i, text in enumerate(header1[2:], 2):  # Skip first two empty cells
#         if text == 'FFHQ':
#             # Merge FFHQ cells
#             for j in range(3):
#                 if i + j < len(header1):
#                     table.add_cell(0, i + j, width=1/len(header2), height=1/len(rows), 
#                                  text='FFHQ' if j == 1 else '', loc='center')
#         elif text == 'ImageNet' and i == 5:  # Only add once
#             for j in range(3):
#                 if i + j < len(header1):
#                     table.add_cell(0, i + j, width=1/len(header2), height=1/len(rows),
#                                  text='ImageNet' if j == 1 else '', loc='center')
    
#     # Style headers
#     num_cols = len(header2)
#     num_rows = len(rows) - 2  # Exclude header rows
    
#     # Make headers bold
#     for i in range(num_cols):
#         table[(0, i)].set_text_props(weight='bold')
#         table[(0, i)].set_facecolor('#f0f0f0')
    
#     # Add lines by setting cell edges
#     # Horizontal lines
#     for i in range(num_rows + 1):
#         for j in range(num_cols):
#             cell = table[(i, j)]
            
#             # Line between headers and data
#             if i == 0:
#                 cell.set_linewidth(2)
#                 cell.set_edgecolor('black')
            
#             # Lines between task groups (every 5 rows after header)
#             elif i in [6, 11]:  # After 5th and 10th data rows
#                 cell.set_linewidth(2)
#                 cell.set_edgecolor('black')
#             else:
#                 cell.set_linewidth(0.5)
#                 cell.set_edgecolor('gray')
    
#     # Vertical lines
#     for i in range(num_rows + 1):
#         for j in range(num_cols):
#             cell = table[(i, j)]
            
#             # Line between task/method columns and data
#             if j == 1:
#                 cell.set_linewidth(2)
#                 cell.set_edgecolor('black')
            
#             # Line between FFHQ and ImageNet
#             elif j == 4:  # After LPIPS of FFHQ
#                 cell.set_linewidth(2) 
#                 cell.set_edgecolor('black')
#             else:
#                 cell.set_linewidth(0.5)
#                 cell.set_edgecolor('gray')
    
#     # Make task and method columns bold
#     for i in range(1, num_rows + 1):
#         table[(i, 0)].set_text_props(weight='bold')  # Task
#         table[(i, 1)].set_text_props(weight='bold')  # Method
    
#     plt.title('Evaluation Metrics Table', fontsize=16, fontweight='bold', pad=20)
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     print(f"Table saved as PDF: {output_path}")

# def highlight_best_values(df, metrics=['PSNR', 'SSIM', 'LPIPS']):
#     """Highlight best values in each metric column (higher is better for PSNR/SSIM, lower for LPIPS)"""
#     df_styled = df.copy()
    
#     for dataset in ['FFHQ', 'ImageNet']:
#         for metric in metrics:
#             col = (dataset, metric)
#             if col in df.columns:
#                 # Convert string values to float for comparison
#                 values = df[col].apply(lambda x: float(x) if isinstance(x, str) else x)
                
#                 if metric == 'LPIPS':
#                     # Lower is better for LPIPS
#                     best_idx = values.idxmin()
#                 else:
#                     # Higher is better for PSNR and SSIM
#                     best_idx = values.idxmax()
                
#                 # Mark best value with ** for LaTeX bold
#                 df_styled.loc[best_idx, col] = f"**{df.loc[best_idx, col]}**"
    
#     return df_styled

# def create_latex_table(df, output_path):
#     """Create publication-ready LaTeX table using pandas"""
#     # Highlight best values
#     df_highlighted = highlight_best_values(df)
    
#     # Configure LaTeX output
#     latex_str = df_highlighted.to_latex(
#         float_format="%.3f",
#         escape=False,  # Allow LaTeX formatting like **bold**
#         multirow=True,
#         multicolumn=True,
#         column_format='ll|ccc|ccc',  # Task, Method | FFHQ metrics | ImageNet metrics
#         caption="Evaluation metrics for HDR reconstruction methods across different tasks and datasets. Values are means calculated from 10 samples. Best values in each metric are highlighted in bold.",
#         label="tab:evaluation_metrics",
#         position='htbp'
#     )
    
#     # Post-process LaTeX to use booktabs and improve formatting
#     latex_str = latex_str.replace('\\begin{tabular}', '\\begin{tabular}')
#     latex_str = latex_str.replace('\\toprule', '\\toprule')
#     latex_str = latex_str.replace('\\midrule', '\\midrule') 
#     latex_str = latex_str.replace('\\bottomrule', '\\bottomrule')
    
#     # Add booktabs package requirement comment
#     latex_header = """% Add to document preamble:
# % \\usepackage{booktabs}
# % \\usepackage{multirow}

# """
    
#     latex_str = latex_header + latex_str
    
#     # Save LaTeX table
#     with open(output_path, 'w') as f:
#         f.write(latex_str)
    
#     print(f"LaTeX table saved: {output_path}")
#     return latex_str

# def create_great_table(df, output_path):
#     """Create modern table using great_tables package"""
#     if not GREAT_TABLES_AVAILABLE:
#         print("Skipping great_tables output - package not available")
#         return
    
#     try:
#         # Flatten the dataframe for great_tables
#         df_flat = df.copy()
#         df_flat.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
#         df_flat = df_flat.reset_index()
        
#         # Create GT table
#         gt_table = (
#             GT(df_flat)
#             .tab_header(
#                 title="HDR Reconstruction Evaluation Metrics",
#                 subtitle="Performance comparison across tasks, methods, and datasets"
#             )
#             .tab_spanner(
#                 label="FFHQ Dataset",
#                 columns=["FFHQ_PSNR", "FFHQ_SSIM", "FFHQ_LPIPS"]
#             )
#             .tab_spanner(
#                 label="ImageNet Dataset", 
#                 columns=["ImageNet_PSNR", "ImageNet_SSIM", "ImageNet_LPIPS"]
#             )
#             .cols_label(
#                 Task="Task",
#                 Method="Method",
#                 FFHQ_PSNR="PSNR",
#                 FFHQ_SSIM="SSIM", 
#                 FFHQ_LPIPS="LPIPS",
#                 ImageNet_PSNR="PSNR",
#                 ImageNet_SSIM="SSIM",
#                 ImageNet_LPIPS="LPIPS"
#             )
#             .fmt_number(
#                 columns=["FFHQ_PSNR", "ImageNet_PSNR"],
#                 decimals=2
#             )
#             .fmt_number(
#                 columns=["FFHQ_SSIM", "FFHQ_LPIPS", "ImageNet_SSIM", "ImageNet_LPIPS"],
#                 decimals=3
#             )
#             .tab_options(
#                 table_font_size="12px",
#                 heading_title_font_size="16px",
#                 column_labels_font_weight="bold"
#             )
#         )
        
#         # Save as HTML
#         html_path = output_path.replace('.txt', '.html')
#         gt_table.save(html_path)
#         print(f"Great Tables HTML saved: {html_path}")
        
#     except Exception as e:
#         print(f"Error creating great_tables output: {e}")

# def create_tabulate_table(df, output_path):
#     """Create clean text table using tabulate"""
#     # Flatten the multi-index for tabulate
#     df_flat = df.copy()
    
#     # Create clean column names
#     new_columns = []
#     for col in df.columns:
#         new_columns.append(f"{col[0]} {col[1]}")
#     df_flat.columns = new_columns
    
#     # Reset index to get Task and Method as columns
#     df_flat = df_flat.reset_index()
    
#     # Create tabulate table with different formats
#     formats = ['grid', 'fancy_grid', 'pipe', 'latex']
    
#     for fmt in formats:
#         table_str = tabulate(
#             df_flat,
#             headers='keys',
#             tablefmt=fmt,
#             floatfmt='.3f',
#             showindex=False
#         )
        
#         format_path = output_path.replace('.txt', f'_tabulate_{fmt}.txt')
#         with open(format_path, 'w') as f:
#             f.write(f"HDR Reconstruction Evaluation Metrics\n")
#             f.write("=" * 50 + "\n\n")
#             f.write(table_str)
        
#         print(f"Tabulate table ({fmt}) saved: {format_path}")

# def create_improved_pdf_table(df, output_path):
#     """Create improved PDF table with better styling"""
#     fig, ax = plt.subplots(figsize=(20, 12))
#     ax.axis('off')
    
#     # Prepare clean data for display
#     df_display = df.reset_index()
    
#     # Create table data with proper headers
#     table_data = []
    
#     # Add data rows
#     for idx, row in df_display.iterrows():
#         row_data = [row['Task'], row['Method']]
#         for col in df.columns:
#             row_data.append(row[col])
#         table_data.append(row_data)
    
#     # Create column labels
#     col_labels = ['Task', 'Method']
#     for dataset in ['FFHQ', 'ImageNet']:
#         for metric in ['PSNR', 'SSIM', 'LPIPS']:
#             col_labels.append(f'{dataset}\n{metric}')
    
#     # Create the table
#     table = ax.table(
#         cellText=table_data,
#         colLabels=col_labels,
#         cellLoc='center',
#         loc='center',
#         bbox=[0, 0, 1, 1]
#     )
    
#     # Improved styling
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     table.scale(1, 2.5)
    
#     # Style headers
#     num_cols = len(col_labels)
#     num_rows = len(table_data)
    
#     # Header styling
#     for j in range(num_cols):
#         cell = table[(0, j)]
#         cell.set_text_props(weight='bold', color='white')
#         cell.set_facecolor('#2E86AB')
#         cell.set_height(0.08)
    
#     # Data cell styling with alternating colors
#     for i in range(1, num_rows + 1):
#         for j in range(num_cols):
#             cell = table[(i, j)]
            
#             # Alternate row colors
#             if i % 2 == 0:
#                 cell.set_facecolor('#F8F9FA')
#             else:
#                 cell.set_facecolor('white')
            
#             # Bold formatting for task and method columns
#             if j in [0, 1]:
#                 cell.set_text_props(weight='bold')
            
#             # Highlight best values (you could implement this based on your logic)
#             cell.set_linewidth(0.5)
#             cell.set_edgecolor('#CCCCCC')
    
#     # Add title
#     plt.suptitle('HDR Reconstruction Evaluation Metrics', 
#                 fontsize=16, fontweight='bold', y=0.95)
    
#     # Add subtitle
#     plt.figtext(0.5, 0.92, 
#                'Performance comparison across tasks, methods, and datasets',
#                ha='center', fontsize=12, style='italic')
    
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', 
#                 facecolor='white', edgecolor='none')
#     plt.close()
    
#     print(f"Improved PDF table saved: {output_path}")

# def save_table(df, output_path):
#     """Save table in multiple formats with improved styling"""
#     # Create output directory
#     output_dir = os.path.dirname(output_path)
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Save as CSV
#     csv_path = output_path.replace('.txt', '.csv')
#     df.to_csv(csv_path)
#     print(f"Table saved as CSV: {csv_path}")
    
#     # Save as LaTeX (publication-ready)
#     latex_path = output_path.replace('.txt', '.tex')
#     create_latex_table(df, latex_path)
    
#     # Save using great_tables (if available)
#     create_great_table(df, output_path)
    
#     # Save using tabulate (multiple formats)
#     create_tabulate_table(df, output_path)
    
#     # Save improved PDF
#     pdf_path = output_path.replace('.txt', '_improved.pdf')
#     create_improved_pdf_table(df, pdf_path)
    
#     # Keep original PDF for comparison
#     pdf_path_original = output_path.replace('.txt', '_original.pdf')
#     create_pdf_table(df, pdf_path_original)
    
#     # Save as formatted text (original)
#     with open(output_path, 'w') as f:
#         f.write("HDR Reconstruction Evaluation Metrics\n")
#         f.write("=" * 80 + "\n\n")
#         f.write("Format: mean (calculated from 10 samples)\n")
#         f.write("Tasks: symmetric, overexposed, underexposed\n")
#         f.write("Methods: baseline, pixel-4k, pixel-1k, ldm-1k, sd1.5-1k\n")
#         f.write("Datasets: FFHQ, ImageNet\n")
#         f.write("Metrics: PSNR (higher is better), SSIM (higher is better), LPIPS (lower is better)\n\n")
#         f.write(str(df))
    
#     print(f"Table saved as text: {output_path}")
    
#     # Print summary
#     print("\nTable files generated:")
#     print(f"  - CSV: {csv_path}")
#     print(f"  - LaTeX: {latex_path}")
#     print(f"  - PDF (improved): {pdf_path}")
#     print(f"  - PDF (original): {pdf_path_original}")
#     print(f"  - Text: {output_path}")
#     if GREAT_TABLES_AVAILABLE:
#         print(f"  - HTML: {output_path.replace('.txt', '.html')}")
#     print(f"  - Tabulate formats: {output_dir}/*_tabulate_*.txt")
    
#     # Also print to console
#     print("\nEvaluation Metrics Table:")
#     print("=" * 80)
#     print(df)

# def main():
#     base_dir = "results"
    
#     # Check if results directory exists
#     if not os.path.exists(base_dir):
#         print(f"Results directory '{base_dir}' not found!")
#         return
    
#     print("Collecting metrics from evaluation files...")
#     data = collect_all_metrics(base_dir)
    
#     print("\nCreating table...")
#     df = create_table(data)
    
#     # Create output directory
#     os.makedirs("tables", exist_ok=True)
#     output_path = "tables/metrics_table.txt"
    
#     save_table(df, output_path)
    
#     print(f"\nTable creation completed!")

# if __name__ == "__main__":
#     main()

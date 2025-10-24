import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def compress_and_convert_to_pdf(input_path, output_path, quality=85, dpi=150, resize_factor=1.0):
    """
    Compress PNG image and convert to PDF
    
    Args:
        input_path: Path to input PNG file
        output_path: Path to output PDF file
        quality: JPEG quality for compression (1-100)
        dpi: DPI for PDF output
        resize_factor: Factor to resize image (0.5 = half size, 1.0 = original)
    """
    try:
        # Load the image
        img = Image.open(input_path)
        
        # Resize image if needed for additional compression
        if resize_factor != 1.0:
            new_size = (int(img.width * resize_factor), int(img.height * resize_factor))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert RGBA to RGB if necessary (for JPEG compression)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        
        # Create figure with matplotlib for PDF output
        fig, ax = plt.subplots(figsize=(img.width/dpi, img.height/dpi), dpi=dpi)
        ax.imshow(img)
        ax.axis('off')
        
        # Remove all margins and padding
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.patch.set_facecolor('white')
        
        # Save as PDF with compression
        plt.savefig(output_path, format='pdf', dpi=dpi, bbox_inches='tight', 
                   pad_inches=0, facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Converted: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")

def compress_figures(figures_dir="figures", quality=85, dpi=150, resize_factor=1.0):
    """
    Compress all PNG files in figures directory and convert to PDF
    
    Args:
        figures_dir: Directory containing PNG files
        quality: JPEG quality for compression
        dpi: DPI for PDF output
        resize_factor: Factor to resize image
    """
    if not os.path.exists(figures_dir):
        print(f"Figures directory '{figures_dir}' not found!")
        return
    
    # Create compressed directory inside figures directory
    compressed_dir = os.path.join(figures_dir, "compressed")
    os.makedirs(compressed_dir, exist_ok=True)
    
    # Find all PNG files
    png_files = glob.glob(os.path.join(figures_dir, "*.png"))
    
    if not png_files:
        print(f"No PNG files found in {figures_dir}")
        return
    
    print(f"Found {len(png_files)} PNG files to process...")
    
    for png_file in png_files:
        # Generate output filename (replace .png with .pdf)
        base_name = os.path.splitext(os.path.basename(png_file))[0]
        pdf_file = os.path.join(compressed_dir, f"{base_name}.pdf")
        
        # Convert and compress
        compress_and_convert_to_pdf(png_file, pdf_file, quality=quality, dpi=dpi, resize_factor=resize_factor)
    
    print(f"\nCompression complete! {len(png_files)} files processed.")
    print(f"PDF files saved in: {compressed_dir}")

def main():
    """Main function with different quality options"""
    figures_dir = "figures"
    
    print("PNG to PDF Compression Tool")
    print("=" * 30)
    
    # Check if figures directory exists
    if not os.path.exists(figures_dir):
        print(f"Figures directory '{figures_dir}' not found!")
        return
    
    # Option to choose compression level
    print("\nChoose compression level:")
    print("1. High quality (DPI=300, larger file)")
    print("2. Medium quality (DPI=150, balanced)")
    print("3. Low quality (DPI=100, smaller file)")
    print("4. Ultra low quality (DPI=75, smallest readable)")
    print("5. Maximum compression (DPI=50, may be hard to read)")
    
    choice = input("\nEnter choice (1-5) or press Enter for medium quality: ").strip()
    
    if choice == "1":
        dpi, quality, resize_factor = 300, 95, 1.0
        print("Using high quality settings...")
    elif choice == "3":
        dpi, quality, resize_factor = 100, 80, 1.0
        print("Using low quality settings...")
    elif choice == "4":
        dpi, quality, resize_factor = 75, 70, 1.0
        print("Using ultra low quality settings...")
    elif choice == "5":
        dpi, quality, resize_factor = 50, 60, 0.8
        print("Using maximum compression (DPI=50, 80% size, 60% quality)...")
    else:
        dpi, quality, resize_factor = 150, 85, 1.0
        print("Using medium quality settings...")
    
    # Process files
    compress_figures(figures_dir, quality=quality, dpi=dpi, resize_factor=resize_factor)

if __name__ == "__main__":
    main()

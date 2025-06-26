import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

def save_png(npz_path, output_dir):
    data = np.load(npz_path)
    images = data['arr_0'].astype(np.uint8)  # [N, H, W, C]
    os.makedirs(output_dir, exist_ok=True)

    for i in range(min(100, len(images))):
        img = Image.fromarray(images[i])
        img.save(os.path.join(output_dir, f"sample_{i:04d}.png"))

npz_path = r"C:\Users\mobil\Desktop\25summer\GenPalm\Diff-Palm\DiffModels\output\test-large\samples_200x128x128x3.npz"
output_dir = 'generated_images'

def show_images(npz_path):
    data = np.load(npz_path)
    images = data['arr_0']  # shape: [N, H, W, C]

    # Normalize to [0, 1] for display
    images = images.astype(np.uint8)

    # Plot a grid of N images
    N = 16  # number of images to show
    cols = 4
    rows = N // cols

    plt.figure(figsize=(12, 6))
    for i in range(N):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# save_png(npz_path, output_dir)
show_images(npz_path)
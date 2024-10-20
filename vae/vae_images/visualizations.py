from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from moviepy.editor import ImageSequenceClip
import random

def create_disentanglement_videos(vae, dataloader, device, output_dir, range_value=5, num_steps=100, fps=30):
    vae.eval()
    with torch.no_grad():
        # Get a random batch from the dataloader
        random_batch = next(iter(dataloader))

        # Select a random image from this batch
        random_index = random.randint(0, random_batch[0].size(0) - 1)
        sample_image = random_batch[0][random_index].unsqueeze(0).to(device)

        # Encode the image
        z,decoded,mu,logvar = vae(sample_image)

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(output_dir, f"{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        # Get the original image and its unmodified reconstruction
        original_img = sample_image.squeeze().cpu().numpy()
        unmodified_reconstruction = vae.decoder(mu).squeeze().cpu().numpy()

        # Calculate step size
        step_size = (2 * range_value) / (num_steps - 1)

        # Create frames
        frames = []
        for dim in range(z.shape[1]):
            print(f"Creating frames for dimension {dim}")

            for i in range(num_steps):
                # Calculate the value to add to the original latent dimension
                add_value = -range_value + i * step_size

                # Copy the latent vector and modify the specific dimension
                z_modified = mu.clone()
                z_modified[0, dim] += add_value

                # Decode the modified latent vector
                reconstructed = vae.decoder(z_modified).squeeze().cpu().numpy()

                # Create a figure with 3 subplots
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f"Dimension {dim}, Added: {add_value:.2f}", fontsize=16)

                # Original image
                axs[0].imshow(original_img, cmap='gray')
                axs[0].set_title("Original Image")
                axs[0].axis('off')

                # Unmodified reconstruction
                axs[1].imshow(unmodified_reconstruction, cmap='gray')
                axs[1].set_title("Unmodified Reconstruction")
                axs[1].axis('off')

                # Modified reconstruction
                axs[2].imshow(reconstructed, cmap='gray')
                axs[2].set_title("Modified Reconstruction")
                axs[2].axis('off')

                plt.tight_layout()

                # Save the figure
                frame_path = os.path.join(output_dir, f"frame_{dim:02d}_{i:03d}.png")
                plt.savefig(frame_path)
                frames.append(frame_path)
                plt.close(fig)

        # Create video from frames
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(os.path.join(output_dir, "combined_disentanglement.mp4"))

        # Clean up frame images
        for frame in frames:
            os.remove(frame)

    print("Combined disentanglement video created successfully!")

# Usage example:
# create_combined_disentanglement_video(vae_model, train_loader, device, "disentanglement_videos", range_value=5, num_steps=100)
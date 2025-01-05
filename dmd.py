import numpy as np
from scipy.linalg import svd, lstsq
import cv2
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
import pandas as pd

class DMDModel:
    def __init__(self, n_components=9):
        """
        Discrete Modal Decomposition model
        Args:
            n_components: Number of SVD components to keep (default=9 to match PTM/HSH)
        """
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.H = None
        self.W = None
        self.C = None
        self.light_weights = None

    def _normalize_light_dirs(self, azimuths, elevations):
        """Convert light directions to normalized 3D vectors"""
        # Convert to radians
        azimuths = np.deg2rad(azimuths)
        elevations = np.deg2rad(elevations)
        
        # Convert to Cartesian coordinates (x, y, z)
        x = np.cos(elevations) * np.cos(azimuths)
        y = np.cos(elevations) * np.sin(azimuths)
        z = np.sin(elevations)
        
        # Stack into array of 3D vectors
        light_dirs = np.stack([x, y, z], axis=1)
        
        return light_dirs

    def model_fit(self, azimuths, elevations, target_images):
        """
        Fit DMD model to the captured data.
        """
        N, H, W, C = target_images.shape
        self.H, self.W, self.C = H, W, C
        
        # Normalize target images to [0,1]
        target_images = target_images.astype(np.float32) / 255.0
        
        # Normalize light directions
        light_dirs = self._normalize_light_dirs(azimuths, elevations)  # (N, 3)
        
        # Fit each color channel separately
        self.components = np.zeros((self.n_components, H, W, C))
        self.mean = np.zeros((H, W, C))
        self.light_weights = np.zeros((3, self.n_components, C))
        
        for c in range(C):
            # Get data for current channel
            X = target_images[..., c].reshape(N, -1)  # (N, H*W)
            
            # Compute mean and center data
            channel_mean = np.mean(X, axis=0)
            X_centered = X - channel_mean
            
            # Compute SVD
            U, S, Vt = svd(X_centered, full_matrices=False)
            
            # Keep top n_components
            components = Vt[:self.n_components] * S[:self.n_components, np.newaxis]
            
            # Compute weights relating light directions to components
            # Using least squares to find relationship between light directions and component activations
            weights, *_ = lstsq(light_dirs, U[:, :self.n_components])
            
            # Store results for this channel
            self.components[:, :, :, c] = components.reshape(self.n_components, H, W)
            self.mean[:, :, c] = channel_mean.reshape(H, W)
            self.light_weights[:, :, c] = weights
            
        print(f"Fitted components range: min={self.components.min():.4f}, max={self.components.max():.4f}")
        
        return self.components, self.mean, self.light_weights

    def relight(self, azimuths, elevations, save_paths, target_images=None):
        """
        Relight using the fitted DMD model and save images with enhanced comparison plots.
        """
        if self.components is None or self.mean is None or self.light_weights is None:
            raise ValueError("Model must be fitted first")
        
        if len(save_paths) != len(azimuths):
            raise ValueError("Number of save paths must match number of light directions")
        
        # Create comparison directory
        comparison_dir = os.path.join(os.path.dirname(save_paths[0]), "comparison")
        os.makedirs(comparison_dir, exist_ok=True)
            
        # Convert new light directions to normalized vectors
        light_dirs = self._normalize_light_dirs(azimuths, elevations)  # (M, 3)
        M = len(azimuths)
        
        # Initialize output array
        relit = np.zeros((M, self.H, self.W, self.C))
        
        # Relight each color channel
        for c in range(self.C):
            # Compute activations for new light directions
            activations = np.dot(light_dirs, self.light_weights[:, :, c])  # (M, n_components)
            
            # Reconstruct images
            channel_recon = np.dot(activations, self.components[:, :, :, c].reshape(self.n_components, -1))
            channel_recon = channel_recon + self.mean[:, :, c].reshape(-1)
            
            relit[:, :, :, c] = channel_recon.reshape(M, self.H, self.W)
        
        relit = np.clip(relit, 0, 1)
        
        # Initialize arrays to store metrics
        mse_values = []
        psnr_values = []
        ssim_values = []
        
        # Save images and create detailed comparisons
        for i, path in enumerate(save_paths):
            # Convert to uint8
            relit_img = (relit[i] * 255).astype(np.uint8)
            cv2.imwrite(path, relit_img)
            
            if target_images is not None:
                target_img = target_images[i].astype(np.uint8)
                
                # Compute error metrics
                mse = np.mean((target_img - relit_img) ** 2)
                psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
                
                # Convert to grayscale for SSIM
                target_gray = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)
                relit_gray = cv2.cvtColor(relit_img, cv2.COLOR_RGB2GRAY)
                ssim = structural_similarity(target_gray, relit_gray)
                
                # Store metrics
                mse_values.append(mse)
                psnr_values.append(psnr)
                ssim_values.append(ssim)
                
                # Compute difference image
                diff_img = cv2.absdiff(target_img, relit_img)
                diff_heatmap = cv2.applyColorMap(diff_img, cv2.COLORMAP_JET)
                
                # Create enhanced comparison plot
                plt.figure(figsize=(15, 10))
                
                # Original images
                plt.subplot(2, 3, 1)
                plt.imshow(target_img)
                plt.title('Target Image')
                plt.axis('off')
                
                plt.subplot(2, 3, 2)
                plt.imshow(relit_img)
                plt.title('Reconstructed Image')
                plt.axis('off')
                
                # Difference heatmap
                plt.subplot(2, 3, 3)
                plt.imshow(cv2.cvtColor(diff_heatmap, cv2.COLOR_BGR2RGB))
                plt.title('Difference Heatmap')
                plt.colorbar(label='Error Magnitude')
                plt.axis('off')
                
                # Error distribution histogram
                plt.subplot(2, 3, 4)
                plt.hist(diff_img.ravel(), bins=50, color='blue', alpha=0.7)
                plt.title('Error Distribution')
                plt.xlabel('Error Magnitude')
                plt.ylabel('Pixel Count')
                
                # Per-channel error
                plt.subplot(2, 3, 5)
                channel_errors = [np.mean((target_img[:,:,c] - relit_img[:,:,c])**2) for c in range(3)]
                plt.bar(['Red', 'Green', 'Blue'], channel_errors)
                plt.title('Per-channel MSE')
                plt.ylabel('Mean Squared Error')
                
                # Add text with metrics
                plt.subplot(2, 3, 6)
                plt.text(0.1, 0.8, f'MSE: {mse:.2f}', fontsize=12)
                plt.text(0.1, 0.6, f'PSNR: {psnr:.2f} dB', fontsize=12)
                plt.text(0.1, 0.4, f'SSIM: {ssim:.4f}', fontsize=12)
                plt.text(0.1, 0.2, f'Mean Error: {np.mean(diff_img):.2f}', fontsize=12)
                plt.axis('off')
                plt.title('Quality Metrics')
                
                # Overall title with light direction info
                plt.suptitle(f'Comparison Analysis (Az: {azimuths[i]:.1f}°, El: {elevations[i]:.1f}°)', fontsize=14)
                
                # Adjust layout and save
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                comparison_path = os.path.join(comparison_dir, f'comparison_{i:03d}.png')
                plt.savefig(comparison_path, bbox_inches='tight', dpi=150)
                plt.close()
        
        # Create summary plot of metrics across all images
        plt.figure(figsize=(15, 5))

        # MSE plot
        plt.subplot(1, 3, 1)
        plt.plot(mse_values, label='MSE')
        mean_mse = np.mean(mse_values)
        plt.axhline(y=mean_mse, color='r', linestyle='--', label=f'Mean MSE: {mean_mse:.2f}')
        plt.title('MSE across Images')
        plt.xlabel('Image Index')
        plt.ylabel('MSE')
        plt.grid(True)
        plt.legend()

        # PSNR plot
        plt.subplot(1, 3, 2)
        plt.plot(psnr_values, label='PSNR')
        mean_psnr = np.mean(psnr_values)
        plt.axhline(y=mean_psnr, color='r', linestyle='--', label=f'Mean PSNR: {mean_psnr:.2f}')
        plt.title('PSNR across Images')
        plt.xlabel('Image Index')
        plt.ylabel('PSNR (dB)')
        plt.grid(True)
        plt.legend()

        # SSIM plot
        plt.subplot(1, 3, 3)
        plt.plot(ssim_values, label='SSIM')
        mean_ssim = np.mean(ssim_values)
        plt.axhline(y=mean_ssim, color='r', linestyle='--', label=f'Mean SSIM: {mean_ssim:.4f}')
        plt.title('SSIM across Images')
        plt.xlabel('Image Index')
        plt.ylabel('SSIM')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        summary_path = os.path.join(comparison_dir, 'metrics_summary.png')
        plt.savefig(summary_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'Image_Index': range(len(save_paths)),
            'Azimuth': azimuths,
            'Elevation': elevations,
            'MSE': mse_values,
            'PSNR': psnr_values,
            'SSIM': ssim_values
        })
        metrics_df.to_csv(os.path.join(comparison_dir, 'metrics.csv'), index=False)
        
        return relit
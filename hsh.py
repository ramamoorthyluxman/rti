import numpy as np
from scipy.linalg import lstsq
import cv2
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
import pandas as pd

class HSHModel:
    def __init__(self, order=2):
        """
        Hemispherical Harmonics model
        Args:
            order: Order of harmonics (default=2 gives 9 basis functions)
        """
        self.order = order
        self.coefficients = None
        self.H = None
        self.W = None
        self.C = None
        self.num_basis = (order + 1) ** 2

    def _compute_hsh_basis(self, lps_cartesian):
        """
        Compute HSH basis functions for given light positions in Cartesian coordinates.
        Args:
            lps_cartesian: numpy array of shape (N, 3) containing x, y, z coordinates
                          of light positions
        Returns:
            basis: numpy array of shape (N, num_basis) containing HSH basis functions
        """
        # Normalize the vectors if they aren't already
        norms = np.sqrt(np.sum(lps_cartesian**2, axis=1))
        x = lps_cartesian[:, 0] / norms
        y = lps_cartesian[:, 1] / norms
        z = lps_cartesian[:, 2] / norms
        
        # Convert to spherical coordinates
        theta = np.arccos(z)  # colatitude [0, pi/2]
        phi = np.arctan2(y, x)  # azimuth [0, 2pi]
        # Ensure phi is in [0, 2pi]
        phi = np.where(phi < 0, phi + 2*np.pi, phi)
        
        basis = []
        
        # Order 0 (1 function)
        basis.append(np.ones_like(theta) / np.sqrt(2*np.pi))
        
        if self.order >= 1:
            # Order 1 (3 functions)
            basis.extend([
                np.sqrt(6)/2 * np.sin(theta) * np.cos(phi),
                np.sqrt(3)/2 * np.cos(theta),
                np.sqrt(6)/2 * np.sin(theta) * np.sin(phi)
            ])
        
        if self.order >= 2:
            # Order 2 (5 functions)
            basis.extend([
                np.sqrt(30)/4 * np.sin(theta)**2 * np.cos(2*phi),
                np.sqrt(30)/2 * np.sin(theta) * np.cos(theta) * np.cos(phi),
                np.sqrt(5)/4 * (3*np.cos(theta)**2 - 1),
                np.sqrt(30)/2 * np.sin(theta) * np.cos(theta) * np.sin(phi),
                np.sqrt(30)/4 * np.sin(theta)**2 * np.sin(2*phi)
            ])
            
        return np.stack(basis, axis=1)

    def model_fit(self, lps_cartesian, target_images):
        """
        Fit HSH model to the captured data using Cartesian light positions.
        Args:
            lps_cartesian: numpy array of shape (N, 3) containing x, y, z coordinates
                          of light positions
            target_images: numpy array of shape (N, H, W, C) containing target images
        """
        N, H, W, C = target_images.shape
        self.H, self.W, self.C = H, W, C
        
        # Normalize target images to [0,1]
        target_images = target_images.astype(np.float32) / 255.0
        
        # Compute HSH basis
        basis = self._compute_hsh_basis(lps_cartesian)  # (N, num_basis)
        
        # Solve for each color channel separately
        coefficients = np.zeros((self.num_basis, H, W, C))
        
        for c in range(C):
            # Reshape target images for current channel
            Y = target_images[..., c].reshape(N, -1)  # (N, H*W)
            
            # Solve least squares system
            channel_coeffs = lstsq(basis, Y)[0]  # (num_basis, H*W)
            coefficients[..., c] = channel_coeffs.reshape(self.num_basis, H, W)
        
        self.coefficients = coefficients
        print(f"Fitted coefficients range: min={coefficients.min():.4f}, max={coefficients.max():.4f}")
        
        return self.coefficients

    def relight(self, lps_cartesian, save_paths, target_images=None):
        """
        Relight using the fitted HSH model and save images with enhanced comparison plots.
        Args:
            lps_cartesian: numpy array of shape (M, 3) containing x, y, z coordinates
                          of light positions for relighting
            save_paths: list of M paths where to save the relit images
            target_images: optional numpy array of shape (M, H, W, C) containing 
                          ground truth images for comparison
        """
        if self.coefficients is None:
            raise ValueError("Model must be fitted first")
        
        if len(save_paths) != len(lps_cartesian):
            raise ValueError("Number of save paths must match number of light positions")
        
        # Create comparison directory
        comparison_dir = os.path.join(os.path.dirname(save_paths[0]), "comparison")
        os.makedirs(comparison_dir, exist_ok=True)
            
        # Compute basis for new light positions
        basis = self._compute_hsh_basis(lps_cartesian)
        M = len(lps_cartesian)
        
        # Initialize output array
        relit = np.zeros((M, self.H, self.W, self.C))
        
        # Relight each color channel separately
        for c in range(self.C):
            coeffs_c = self.coefficients[..., c].reshape(self.num_basis, -1)
            relit_c = np.dot(basis, coeffs_c)
            relit[..., c] = relit_c.reshape(M, self.H, self.W)
        
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
                
                # Overall title with light position info
                plt.suptitle(f'Comparison Analysis (x: {lps_cartesian[i,0]:.2f}, y: {lps_cartesian[i,1]:.2f}, z: {lps_cartesian[i,2]:.2f})', 
                           fontsize=14)
                
                # Adjust layout and save
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                comparison_path = os.path.join(comparison_dir, f'comparison_{i:03d}.png')
                plt.savefig(comparison_path, bbox_inches='tight', dpi=150)
                plt.close()
        
        if target_images is not None:
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
                'X': lps_cartesian[:, 0],
                'Y': lps_cartesian[:, 1],
                'Z': lps_cartesian[:, 2],
                'MSE': mse_values,
                'PSNR': psnr_values,
                'SSIM': ssim_values
            })
            metrics_df.to_csv(os.path.join(comparison_dir, 'metrics.csv'), index=False)
        
        return relit
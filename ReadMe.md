# Reflectance Transformation Imaging (RTI) Model Implementations

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Implementation Details](#implementation-details)
4. [Model Architectures](#model-architectures)
5. [Usage Guide](#usage-guide)
6. [Analysis Tools](#analysis-tools)
7. [Performance Optimization](#performance-optimization)
8. [Technical Considerations](#technical-considerations)
9. [Contributing](#contributing)
10. [References](#references)

## Introduction

Reflectance Transformation Imaging (RTI) is an advanced computational photography technique that captures surface shape and color information to enable interactive relighting of objects. This repository provides robust implementations of three primary RTI algorithms, each with its own mathematical foundation and optimization approach.

### Key Features
- High-performance Python implementations of PTM, HSH, and DMD models
- Comprehensive analysis and visualization tools
- Flexible data loading and preprocessing pipeline
- Extensive error metrics and quality assessment
- Optimized numerical computations
- Cross-platform compatibility

## Mathematical Foundations

### Polynomial Texture Mapping (PTM)
PTM represents surface reflectance using a biquadratic polynomial:

```
L(u,v,lx,ly) = a0(u,v)lx² + a1(u,v)ly² + a2(u,v)lxly + a3(u,v)lx + a4(u,v)ly + a5(u,v)
```

where:
- (u,v) are pixel coordinates
- (lx,ly) are normalized light coordinates
- ai(u,v) are the polynomial coefficients

### Hemispherical Harmonics (HSH)
HSH adapts spherical harmonics to the hemispherical domain using the basis functions:

Order 0:
```
H₀₀(θ,φ) = 1/√(2π)
```

Order 1:
```
H₁₀(θ,φ) = √(6)/2 * sin(θ)cos(φ)
H₁₁(θ,φ) = √(3)/2 * cos(θ)
H₁₂(θ,φ) = √(6)/2 * sin(θ)sin(φ)
```

Order 2:
```
H₂₀(θ,φ) = √(30)/4 * sin²(θ)cos(2φ)
H₂₁(θ,φ) = √(30)/2 * sin(θ)cos(θ)cos(φ)
H₂₂(θ,φ) = √(5)/4 * (3cos²(θ) - 1)
H₂₃(θ,φ) = √(30)/2 * sin(θ)cos(θ)sin(φ)
H₂₄(θ,φ) = √(30)/4 * sin²(θ)sin(2φ)
```

### Discrete Modal Decomposition (DMD)
DMD uses SVD to decompose the reflectance function:

```
R = USVᵀ
```
where:
- R is the reflectance matrix
- U contains the lighting basis functions
- S contains singular values
- V contains the spatial basis functions

## Implementation Details

### PTM Model Implementation
```python
class PTMModel:
    def __init__(self):
        """
        Initializes PTM model with:
        - 6 basis functions
        - Least squares solver
        - Per-channel coefficient computation
        """
```

Key features:
- Efficient matrix operations using NumPy
- Parallel processing of color channels
- Memory-optimized coefficient storage
- Robust light position normalization

### HSH Model Implementation
```python
class HSHModel:
    def __init__(self, order=2):
        """
        Initializes HSH model with:
        - Configurable order (default=2 giving 9 basis functions)
        - Spherical coordinate conversion
        - Automatic basis function generation
        """
```

Optimizations:
- Vectorized basis function computation
- Cached intermediate results
- Efficient coordinate transformations
- Dynamic basis function generation

### DMD Model Implementation
```python
class DMDModel:
    def __init__(self, n_components=9):
        """
        Initializes DMD model with:
        - Adaptive component selection
        - SVD-based decomposition
        - Automatic rank determination
        """
```

Advanced features:
- Truncated SVD for large datasets
- Automatic rank estimation
- Progressive component computation
- Memory-efficient matrix operations

## Model Architectures

### Data Flow
```
Raw Images → Preprocessing → Model Fitting → Coefficient Generation → Relighting
    ↓            ↓              ↓                    ↓                  ↓
Validation  → Normalization → Optimization → Error Analysis → Quality Assessment
```

### Processing Pipeline
1. **Data Loading**
   - Image normalization (0-1 range)
   - Light position processing
   - Memory mapping for large datasets

2. **Model Fitting**
   - Basis function computation
   - Coefficient optimization
   - Error minimization

3. **Relighting**
   - Coefficient interpolation
   - Image reconstruction
   - Quality metrics computation

## Technical Considerations

### Memory Management
- Efficient matrix operations using strided arrays
- Memory mapping for large datasets
- Garbage collection optimization
- Coefficient compression techniques

### Numerical Stability
- Condition number monitoring
- SVD stability checks
- Regularization parameters
- Error propagation control

### Performance Optimization
```python
# Example of optimized basis computation
def _compute_ptm_basis(self, lps_cartesian):
    lx = lps_cartesian[:, 0]
    ly = lps_cartesian[:, 1]
    return np.stack([
        np.ones_like(lx),
        lx,
        ly,
        lx**2,
        ly**2,
        lx*ly
    ], axis=1)
```

### Error Metrics
1. **Mean Squared Error (MSE)**
   ```python
   mse = np.mean((target_img - relit_img) ** 2)
   ```

2. **Peak Signal-to-Noise Ratio (PSNR)**
   ```python
   psnr = 20 * np.log10(255.0 / np.sqrt(mse))
   ```

3. **Structural Similarity Index (SSIM)**
   ```python
   ssim = structural_similarity(target_gray, relit_gray)
   ```

## Analysis Tools

### Visualization Components
1. **Error Heatmaps**
   - Per-pixel error visualization
   - Color-coded difference maps
   - Regional error analysis

2. **Quality Metrics**
   - MSE distribution plots
   - PSNR trend analysis
   - SSIM visualization

3. **Comparison Tools**
   - Side-by-side visualization
   - Error magnitude plots
   - Channel-wise analysis

### Output Analysis
```python
class AnalysisTools:
    def generate_error_heatmap(self, target, relit):
        diff = cv2.absdiff(target, relit)
        return cv2.applyColorMap(diff, cv2.COLORMAP_JET)

    def compute_metrics(self, target, relit):
        return {
            'mse': np.mean((target - relit) ** 2),
            'psnr': self._compute_psnr(target, relit),
            'ssim': self._compute_ssim(target, relit)
        }
```

## Usage Examples

### Basic Usage
```python
# Initialize model
ptm = PTMModel()

# Fit model
coeffs = ptm.model_fit(
    lps_cartesian=lps_cartesian,
    target_images=target_images
)

# Relight
relit_images = ptm.relight(
    lps_cartesian=lps_cartesian,
    target_images=target_images,
    save_paths=save_paths
)
```

### Advanced Configuration
```python
# HSH with higher order
hsh = HSHModel(order=3)

# DMD with custom components
dmd = DMDModel(n_components=12)

# Custom fitting parameters
coeffs = model.model_fit(
    lps_cartesian=lps_cartesian,
    target_images=target_images,
    regularization=0.01,
    max_iterations=1000
)
```

## Performance Comparison

| Metric | PTM | HSH | DMD |
|--------|-----|-----|-----|
| Basis Functions | 6 | 9 (order=2) | 9 (default) |
| Memory (GB)* | 0.5-1 | 1-2 | 2-4 |
| Training Time** | Fast | Medium | Slow |
| Accuracy*** | Good | Better | Best |
| Specular Handling | Limited | Good | Excellent |

\* For typical 4K resolution dataset
\** On standard hardware (CPU only)
\*** Based on average PSNR/SSIM metrics

## Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit pull request with:
   - Description of changes
   - Test results
   - Performance impact
   - Documentation updates

### Code Style
- Follow PEP 8 guidelines
- Document all functions
- Include type hints
- Add unit tests

## References

1. Malzbender, T., Gelb, D., & Wolters, H. (2001). Polynomial texture maps. In Proceedings of SIGGRAPH 2001 (pp. 519-528).

2. Gautron, P., Křivánek, J., Pattanaik, S. N., & Bouatouch, K. (2004). A novel hemispherical basis for accurate and efficient rendering. In Rendering Techniques 2004 (pp. 321-330).

3. Wang, O., Gunawardane, P., Scher, S., & Davis, J. (2009). Material classification using BRDF slices. In 2009 IEEE Conference on Computer Vision and Pattern Recognition (pp. 2805-2811).

4. Pan, R., & Skala, V. (2015). A two-level approach to implicit surface modeling with compactly supported radial basis functions. Engineering with Computers, 31(1), 187-198.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
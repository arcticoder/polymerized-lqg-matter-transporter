# Video/Animation Pipeline Documentation

## Overview

The video/animation pipeline generates high-quality visualizations of Newton-Raphson solver simulations for the enhanced stargate transporter. The system samples the simulation at 1-second intervals, renders frames showing energy density evolution, and assembles them into MP4 videos.

## Mathematical Framework

### Frame Sampling
- **Sampling interval**: Δt = 1 second
- **Frame index**: j = 0, 1, 2, ..., N-1
- **Time at frame j**: t_j = j × 1s
- **Total frames**: N = ⌊T_sim/1s⌋ + 1

### Energy Density Calculation
At each frame j, compute the energy density on a 2D slice:
```
E_ij^(j) = T_00(ρ_i, z_j; t_j)
```
where:
- `ρ_i` is the radial coordinate
- `z_j` is the axial coordinate
- `t_j` is the time at frame j
- `T_00` is the time-time component of the stress-energy tensor

### Field Components Visualized
1. **Energy Density**: T₀₀ - Dominant energy component
2. **Metric Components**: g₀₀, g₁₁ - Spacetime curvature
3. **Solver Information**: Convergence, iterations, residuals

## Pipeline Components

### 1. SimulationVideoRenderer Class
- **Purpose**: Core rendering engine for simulation frames
- **Key Methods**:
  - `render_simulation_frames()`: Generate PNG frames at 1fps
  - `assemble_video()`: Use FFmpeg to create MP4 video
- **Features**:
  - Custom colormaps for energy density visualization
  - Real-time convergence information display
  - Automatic scaling and frame optimization

### 2. YouTubeUploader Class (Optional)
- **Purpose**: Automated YouTube upload and playlist management
- **Authentication**: OAuth 2.0 flow with Google APIs
- **Features**:
  - Automatic playlist creation per repository
  - Unlisted video uploads with scientific metadata
  - Error handling and retry mechanisms

### 3. Simple Demo Script
- **File**: `scripts/simple_video_demo.py`
- **Purpose**: Standalone video generation without external dependencies
- **Features**:
  - Self-contained frame rendering
  - Basic FFmpeg integration
  - Local video output only

## Usage Instructions

### Quick Start (Recommended)
```bash
# Navigate to project directory
cd c:\Users\echo_\Code\asciimath\polymerized-lqg-matter-transporter

# Run simple video demo
python scripts/simple_video_demo.py
```

### Full Pipeline with YouTube Upload
```bash
# Install dependencies
pip install -r requirements_video.txt

# Setup environment
cp .env.example .env
# Edit .env with your Google OAuth credentials

# Run full pipeline
python scripts/render_and_upload.py
```

### Custom Configuration
```python
# Modify simulation parameters
T_sim = 30.0  # Simulation duration (seconds)

# Adjust solver settings
solver_config = {
    'max_iterations': 50,
    'tolerance': 1e-8,
    'acceleration': 'anderson'
}

# Configure video quality
render_config = {
    'fps': 1,
    'dpi': 150,
    'quality': 'high'
}
```

## Output Specifications

### Frame Format
- **Resolution**: 120-150 DPI (configurable)
- **Format**: PNG with transparent background support
- **Naming**: `frame_XXXX.png` (zero-padded 4 digits)
- **Content**: 2×2 subplot layout showing:
  1. Energy density T₀₀ with payload boundary
  2. Metric component g₀₀
  3. Metric component g₁₁  
  4. Solver convergence information and statistics

### Video Format
- **Codec**: H.264 (libx264)
- **Frame Rate**: 1 fps (1-second real-time per frame)
- **Resolution**: Based on input frame DPI
- **Format**: MP4 with fast-start for streaming
- **Quality**: CRF 18-28 (configurable)

## Performance Optimization

### Solver Settings for Video Generation
```python
# Balanced settings for speed vs quality
solver_config = {
    'max_iterations': 25,     # Reduced for faster solving
    'tolerance': 1e-6,        # Relaxed but adequate
    'acceleration': 'anderson' # Fastest convergence
}

# Optimized grid resolution
field_config = {
    'nx': 24, 'ny': 24, 'nz': 12,  # Balanced resolution
    'field_components': 3           # Essential components only
}
```

### Rendering Optimizations
- **Grid reduction**: Lower resolution for faster computation
- **Component selection**: Visualize only essential field components
- **Memory management**: Automatic cleanup of temporary frames
- **Progress monitoring**: Real-time feedback during rendering

## System Requirements

### Core Dependencies
- **Python**: ≥3.8
- **NumPy**: ≥1.21.0
- **Matplotlib**: ≥3.5.0
- **JAX**: ≥0.4.0 (for solver acceleration)
- **SciPy**: ≥1.7.0

### Video Processing
- **FFmpeg**: Must be installed and in system PATH
  - Windows: Download from https://ffmpeg.org/download.html
  - Linux: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`

### YouTube Upload (Optional)
- **google-auth**: ≥2.0.0
- **google-auth-oauthlib**: ≥0.5.0
- **google-api-python-client**: ≥2.0.0
- **OAuth credentials**: Google Cloud Console setup required

## Troubleshooting

### Common Issues

#### FFmpeg Not Found
```
Error: FFmpeg not found
Solution: Install FFmpeg and add to system PATH
```

#### Low Frame Success Rate
```
Issue: <50% frames rendered successfully
Causes: 
  - Solver convergence failures
  - Memory limitations
  - Grid resolution too high
Solutions:
  - Reduce max_iterations
  - Increase tolerance
  - Lower grid resolution
```

#### YouTube Authentication Errors
```
Issue: OAuth flow fails
Solutions:
  - Verify client_secrets.json path
  - Check Google Cloud Console API enablement
  - Ensure YouTube Data API v3 is enabled
```

### Performance Tuning

#### For Faster Rendering
```python
# Speed-optimized settings
field_config = {
    'nx': 16, 'ny': 16, 'nz': 8,
    'field_components': 2
}

solver_config = {
    'max_iterations': 15,
    'tolerance': 1e-5
}

render_config = {
    'dpi': 100,
    'quality': 'low'
}
```

#### For Higher Quality
```python
# Quality-optimized settings  
field_config = {
    'nx': 48, 'ny': 48, 'nz': 24,
    'field_components': 6
}

solver_config = {
    'max_iterations': 100,
    'tolerance': 1e-10
}

render_config = {
    'dpi': 200,
    'quality': 'high'
}
```

## Example Outputs

### Typical Video Metrics
- **Duration**: 15-30 seconds (15-30 frames at 1fps)
- **File Size**: 5-50 MB (depends on quality and resolution)
- **Render Time**: 2-10 seconds per frame
- **Total Generation Time**: 1-5 minutes for 20-frame video

### Frame Content Description
Each frame shows:
1. **Energy Density**: Plasma colormap showing T₀₀ evolution
2. **Metric Components**: RdBu colormap for spacetime curvature
3. **Convergence Info**: Newton-Raphson solver performance
4. **Physical Overlay**: Payload boundary and coordinate system

## Integration with YouTube

### Playlist Organization
- **Naming**: Repository name (e.g., "polymerized-lqg-matter-transporter")
- **Privacy**: Unlisted by default
- **Categories**: Science & Technology
- **Tags**: physics, simulation, general relativity, loop quantum gravity

### Video Metadata
- **Title**: "Enhanced Stargate Transporter Simulation - [timestamp]"
- **Description**: Detailed simulation parameters and results
- **Thumbnails**: Automatically generated from first frame

### Channel Integration
Videos are automatically uploaded to: https://www.youtube.com/channel/UCzvJDXYHv7MZW5CwswDXXPw

## Future Enhancements

### Planned Features
1. **Interactive Controls**: Mouse-over information display
2. **3D Visualization**: Volume rendering of full 3D fields
3. **Real-time Streaming**: Live simulation broadcasting
4. **Comparative Analysis**: Multi-configuration side-by-side videos
5. **Advanced Analytics**: Performance metrics overlay

### Research Applications
- **Educational Content**: Teaching general relativity concepts
- **Research Documentation**: Preserving simulation results
- **Collaboration**: Sharing findings with research community
- **Validation**: Visual verification of numerical methods

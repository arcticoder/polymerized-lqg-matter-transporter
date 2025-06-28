"""
Simplified Video Generation for Newton-Raphson Solver

This script creates a basic video from simulation frames without requiring
external dependencies for YouTube upload.

Usage:
    python scripts/simple_video_demo.py

Mathematical Framework:
    - Sample at Δt = 1s: t_j = j·1s
    - Energy density: E_ij = T_00(ρ_i, z_j; t_j)
    - Render N = ⌊T_sim/1s⌋ + 1 frames

Author: Enhanced Implementation  
Created: June 27, 2025
"""

import os
import sys
import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from core.enhanced_stargate_transporter import EnhancedStargateTransporter, EnhancedTransporterConfig
from solvers.newton_raphson_solver import NewtonRaphsonIterativeSolver

def stress_energy_function(X, Y, Z, t, config):
    """
    Time-dependent stress-energy tensor.
    
    Args:
        X, Y, Z: Coordinate meshgrids
        t: Time parameter
        config: Field configuration
        
    Returns:
        Stress-energy tensor components
    """
    r_cyl = np.sqrt(X**2 + Y**2)
    
    # Time-evolving exotic matter source
    base_source = np.exp(-(r_cyl - 1.5)**2 / 0.8) * np.exp(-(Z**2) / 30.0)
    
    # Complex time dynamics
    oscillation = np.sin(t * 2 * np.pi / 8.0)  # 8-second period
    growth = 1 + 0.3 * np.tanh(t / 4.0)        # Gradual growth
    pulse = np.exp(-((t - 10.0)**2) / 16.0)    # Pulse at t=10s
    
    time_factor = growth * (1 + 0.2 * oscillation + 0.4 * pulse)
    source = base_source * time_factor * 1e-5
    
    # Expand to field components using NumPy
    n_components = config['field_components']
    source_expanded = np.zeros(X.shape + (n_components,))
    
    for i in range(n_components):
        component_factor = 1.0 + 0.1 * (i % 3) / 3.0
        source_expanded[:, :, :, i] = source * component_factor
        
    return source_expanded

def generate_simulation_video(T_sim=20.0, output_dir=None):
    """
    Generate complete simulation video.
    
    Args:
        T_sim: Total simulation time (seconds)
        output_dir: Output directory (temp if None)
        
    Returns:
        Video generation results
    """
    start_time = time.time()
    
    print("🎬 NEWTON-RAPHSON SIMULATION VIDEO GENERATOR")
    print("=" * 60)
    print(f"Simulation duration: {T_sim:.0f} seconds")
    print(f"Frame sampling: Δt = 1.0 s")
    
    # Setup output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="newton_raphson_video_")
    else:
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"Output directory: {output_dir}")
    
    # Initialize enhanced transporter
    config = EnhancedTransporterConfig(
        R_payload=2.0,
        R_neck=0.08,
        L_corridor=50.0,
        corridor_mode="sinusoidal",
        v_conveyor_max=1e6
    )
    transporter = EnhancedStargateTransporter(config)
    
    # Initialize Newton-Raphson solver with optimized settings
    solver_config = {
        'max_iterations': 25,     # Balanced for speed vs accuracy
        'tolerance': 1e-6,        # Reasonable tolerance
        'initial_damping': 0.8,
        'line_search': True,
        'jacobian_method': 'forward',
        'acceleration': 'anderson'
    }
    
    field_config = {
        'nx': 24, 'ny': 24, 'nz': 12,  # Optimized resolution
        'field_components': 3,          # Reduced components for speed
        'boundary_conditions': 'asymptotically_flat'
    }
    
    solver = NewtonRaphsonIterativeSolver(transporter, solver_config, field_config)
    
    # Generate frame times
    frame_times = np.arange(0, T_sim + 1, 1.0)  # 1-second intervals
    n_frames = len(frame_times)
    
    print(f"\n🖼️ RENDERING {n_frames} FRAMES")
    print("-" * 40)
    
    # Setup visualization grid
    grid = solver.grid
    z_slice = grid['nz'] // 2  # Middle slice
    X_2d = grid['X'][:, :, z_slice]
    Y_2d = grid['Y'][:, :, z_slice]
    
    # Storage for global color scaling
    energy_values = []
    
    # Render frames
    successful_frames = 0
    render_start = time.time()
    
    for j, t in enumerate(frame_times):
        try:
            print(f"  Frame {j+1:2d}/{n_frames}: t = {t:4.1f}s ", end="")
            frame_start = time.time()
            
            # Solve field equation
            result = solver.solve_nonlinear_field(
                lambda X, Y, Z, t: stress_energy_function(X, Y, Z, t, field_config), 
                t
            )
            
            # Extract 2D slice
            solution = result['solution']
            field_2d = solution[:, :, z_slice, :]
            
            # Compute energy density approximation
            energy_density = np.sum(field_2d**2, axis=2)
            energy_values.append(energy_density)
            
            # Create visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Enhanced Stargate Transporter - Newton-Raphson Solver\nt = {t:.1f} s', 
                        fontsize=14, fontweight='bold')
            
            # Energy density plot
            extent = [-config.R_payload*3, config.R_payload*3, 
                     -config.R_payload*3, config.R_payload*3]
            
            im1 = ax1.imshow(energy_density.T, origin='lower', extent=extent,
                           cmap='plasma', aspect='equal')
            ax1.set_title('Energy Density T₀₀')
            ax1.set_xlabel('x (m)')
            ax1.set_ylabel('y (m)')
            plt.colorbar(im1, ax=ax1, label='J/m³', shrink=0.8)
            
            # Add payload boundary
            circle = plt.Circle((0, 0), config.R_payload, 
                              fill=False, color='white', linewidth=2, linestyle='--')
            ax1.add_patch(circle)
            
            # Field component visualizations
            for idx, (ax, comp_idx, title) in enumerate([(ax2, 0, 'g₀₀'), (ax3, 1, 'g₁₁')]):
                if comp_idx < field_2d.shape[2]:
                    im = ax.imshow(field_2d[:, :, comp_idx].T, origin='lower', 
                                 extent=extent, cmap='RdBu_r', aspect='equal')
                    ax.set_title(f'Metric Component {title}')
                    ax.set_xlabel('x (m)')
                    ax.set_ylabel('y (m)')
                    plt.colorbar(im, ax=ax, shrink=0.8)
            
            # Solver information
            ax4.text(0.05, 0.85, "Newton-Raphson Results:", fontweight='bold', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.05, 0.75, f"Converged: {'✅' if result['converged'] else '❌'}", 
                    transform=ax4.transAxes)
            ax4.text(0.05, 0.65, f"Iterations: {result['iterations']}", 
                    transform=ax4.transAxes)
            ax4.text(0.05, 0.55, f"Residual: {result['final_residual']:.2e}", 
                    transform=ax4.transAxes)
            ax4.text(0.05, 0.45, f"Solve time: {result['timing']['total_time']:.2f}s", 
                    transform=ax4.transAxes)
            
            # Field statistics
            ax4.text(0.05, 0.30, "Field Statistics:", fontweight='bold', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.05, 0.20, f"Energy max: {np.max(energy_density):.2e}", 
                    transform=ax4.transAxes)
            ax4.text(0.05, 0.10, f"Field norm: {np.linalg.norm(field_2d):.2e}", 
                    transform=ax4.transAxes)
            ax4.text(0.05, 0.00, f"Grid: {field_config['nx']}×{field_config['ny']}×{field_config['nz']}", 
                    transform=ax4.transAxes)
            
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            
            plt.tight_layout()
            
            # Save frame
            frame_path = os.path.join(output_dir, f"frame_{j:04d}.png")
            plt.savefig(frame_path, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close()
            
            frame_time = time.time() - frame_start
            print(f"✅ ({frame_time:.2f}s)")
            successful_frames += 1
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    render_time = time.time() - render_start
    
    print(f"\n📊 FRAME RENDERING COMPLETE")
    print("-" * 40)
    print(f"Successful frames: {successful_frames}/{n_frames}")
    print(f"Success rate: {successful_frames/n_frames:.1%}")
    print(f"Total render time: {render_time:.1f}s")
    print(f"Average per frame: {render_time/n_frames:.2f}s")
    
    if successful_frames == 0:
        print("❌ No frames rendered successfully")
        return None
    
    # Assemble video with FFmpeg
    print(f"\n🎥 ASSEMBLING VIDEO")
    print("-" * 40)
    
    video_path = os.path.join(output_dir, "newton_raphson_simulation.mp4")
    
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', '1',  # 1 fps
        '-i', os.path.join(output_dir, 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',  # Good quality
        '-movflags', '+faststart',
        video_path
    ]
    
    try:
        print("Running FFmpeg...")
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        
        # Check video file
        if os.path.exists(video_path):
            video_size = os.path.getsize(video_path) / (1024*1024)  # MB
            print(f"✅ Video created successfully!")
            print(f"Video path: {video_path}")
            print(f"Video size: {video_size:.1f} MB")
            print(f"Duration: {successful_frames} seconds (1 fps)")
            
            return {
                'success': True,
                'video_path': video_path,
                'frames_rendered': successful_frames,
                'total_time': time.time() - start_time,
                'video_size_mb': video_size
            }
        else:
            print("❌ Video file not created")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error: {e}")
        print("Make sure FFmpeg is installed and in PATH")
        return None
    except FileNotFoundError:
        print("❌ FFmpeg not found")
        print("Please install FFmpeg:")
        print("  Windows: https://ffmpeg.org/download.html")
        print("  Linux: sudo apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        return None

def main():
    """Main video generation demo."""
    
    print("🚀 Starting Newton-Raphson simulation video generation...")
    
    # Generate video
    result = generate_simulation_video(T_sim=15.0)  # 15-second simulation
    
    if result and result['success']:
        print(f"\n🎉 SUCCESS!")
        print(f"Video generated in {result['total_time']:.1f}s")
        print(f"Output: {result['video_path']}")
        
        # Optionally open video
        open_video = input("\n🎬 Open video file? (y/N): ").lower().strip() == 'y'
        if open_video:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(result['video_path'])}")
    else:
        print("❌ Video generation failed")

if __name__ == "__main__":
    main()

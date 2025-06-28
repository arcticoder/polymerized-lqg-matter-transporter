"""
Basic Video Demo for Enhanced Stargate Transporter

This creates a simple animation showing the enhanced transporter field evolution
without the complex Newton-Raphson solver to avoid JAX compatibility issues.

Usage:
    python scripts/basic_video_demo.py

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

def generate_basic_video(T_sim=20.0, output_dir=None):
    """
    Generate basic video showing enhanced transporter field evolution.
    
    Args:
        T_sim: Total simulation time (seconds)
        output_dir: Output directory (temp if None)
        
    Returns:
        Video generation results
    """
    start_time = time.time()
    
    print("🎬 ENHANCED TRANSPORTER BASIC VIDEO GENERATOR")
    print("=" * 55)
    print(f"Simulation duration: {T_sim:.0f} seconds")
    print(f"Frame sampling: Δt = 1.0 s")
    
    # Setup output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="transporter_video_")
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
    
    # Generate frame times
    frame_times = np.arange(0, T_sim + 1, 1.0)  # 1-second intervals
    n_frames = len(frame_times)
    
    print(f"\n🖼️ RENDERING {n_frames} FRAMES")
    print("-" * 40)
    
    # Setup visualization grid
    nr, nz = 100, 200  # Resolution for visualization
    r_max = config.R_payload * 3
    z_range = config.L_corridor
    
    r = np.linspace(0, r_max, nr)
    z = np.linspace(-z_range/2, z_range/2, nz)
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    # Render frames
    successful_frames = 0
    render_start = time.time()
    
    for j, t in enumerate(frame_times):
        try:
            print(f"  Frame {j+1:2d}/{n_frames}: t = {t:4.1f}s ", end="")
            frame_start = time.time()
            
            # Compute enhanced field configuration
            field_config = transporter.compute_complete_field_configuration(t)
            
            # Extract field components
            conveyor_velocity = field_config.get('conveyor_velocity', 0)
            field_strength = field_config.get('field_strength', 1.0)
            
            # Compute stress-energy density on the grid
            stress_energy = np.zeros_like(R)
            for i in range(nr):
                for k in range(nz):
                    stress_energy[i, k] = transporter.stress_energy_density(R[i, k], 0.0, Z[i, k])
            
            # Add time-dependent evolution
            time_modulation = 1 + 0.3 * np.sin(t * 2 * np.pi / 10.0)  # 10-second period
            stress_energy_evolved = stress_energy * time_modulation
            
            # Compute energy density (simplified)
            energy_density = np.abs(stress_energy_evolved)
            
            # Create visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Enhanced Stargate Transporter - Field Evolution\nt = {t:.1f} s', 
                        fontsize=16, fontweight='bold')
            
            # Stress-energy density (cylindrical coordinates)
            im1 = ax1.imshow(energy_density, origin='lower', 
                           extent=[z[0], z[-1], r[0], r[-1]],
                           cmap='plasma', aspect='auto')
            ax1.set_title('Energy Density T₀₀')
            ax1.set_xlabel('z (m)')
            ax1.set_ylabel('ρ (m)')
            plt.colorbar(im1, ax=ax1, label='|T₀₀| (J/m³)', shrink=0.8)
            
            # Add corridor boundaries
            ax1.axhline(y=config.R_payload, color='white', linestyle='--', linewidth=2, alpha=0.8)
            ax1.axhline(y=config.R_neck, color='cyan', linestyle='--', linewidth=2, alpha=0.8)
            ax1.text(0.02, 0.95, 'Payload Region', transform=ax1.transAxes, 
                    color='white', fontweight='bold')
            
            # Cross-section at z=0
            z_center_idx = nz // 2
            energy_cross_section = energy_density[:, z_center_idx]
            
            ax2.plot(r, energy_cross_section, 'b-', linewidth=2, label='Energy Density')
            ax2.axvline(x=config.R_payload, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Payload Boundary')
            ax2.axvline(x=config.R_neck, color='cyan', linestyle='--', linewidth=2, alpha=0.7, label='Neck Radius')
            ax2.set_xlabel('Radius ρ (m)')
            ax2.set_ylabel('Energy Density (J/m³)')
            ax2.set_title('Radial Profile at z=0')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Field strength evolution
            ax3.clear()
            time_history = frame_times[:j+1]
            if len(time_history) > 1:
                # Simple field strength evolution
                field_evolution = [1 + 0.3 * np.sin(t_hist * 2 * np.pi / 10.0) for t_hist in time_history]
                ax3.plot(time_history, field_evolution, 'g-', linewidth=2, marker='o', markersize=4)
                ax3.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7)
                ax3.set_xlabel('Time (s)')
                ax3.set_ylabel('Field Strength')
                ax3.set_title('Field Evolution Over Time')
                ax3.grid(True, alpha=0.3)
                ax3.set_xlim(0, T_sim)
                
            # System information
            ax4.text(0.05, 0.90, "Enhanced Transporter Status:", fontweight='bold', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.05, 0.80, f"Time: {t:.1f} s", transform=ax4.transAxes)
            ax4.text(0.05, 0.70, f"Conveyor velocity: {conveyor_velocity:.2e} m/s", transform=ax4.transAxes)
            ax4.text(0.05, 0.60, f"Field strength: {field_strength:.3f}", transform=ax4.transAxes)
            ax4.text(0.05, 0.50, f"Max energy density: {np.max(energy_density):.2e} J/m³", transform=ax4.transAxes)
            
            # Configuration info
            ax4.text(0.05, 0.35, "Configuration:", fontweight='bold', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.05, 0.25, f"Payload radius: {config.R_payload:.1f} m", transform=ax4.transAxes)
            ax4.text(0.05, 0.15, f"Neck radius: {config.R_neck:.2f} m", transform=ax4.transAxes)
            ax4.text(0.05, 0.05, f"Corridor length: {config.L_corridor:.0f} m", transform=ax4.transAxes)
            
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
    
    video_path = os.path.join(output_dir, "enhanced_transporter_demo.mp4")
    
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-r', '1',  # 1 fps (simpler format that works)
        '-i', os.path.join(output_dir, 'frame_%04d.png'),
        '-vcodec', 'libx264',
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
    """Main basic video generation demo."""
    
    print("🚀 Starting enhanced transporter video generation...")
    
    # Generate video
    result = generate_basic_video(T_sim=15.0)  # 15-second simulation
    
    if result and result['success']:
        print(f"\n🎉 SUCCESS!")
        print(f"Video generated in {result['total_time']:.1f}s")
        print(f"Output: {result['video_path']}")
        print(f"\n📺 Your video is ready for upload to:")
        print(f"   https://www.youtube.com/channel/UCzvJDXYHv7MZW5CwswDXXPw")
        
        # Optionally open video
        open_video = input("\n🎬 Open video file? (y/N): ").lower().strip() == 'y'
        if open_video:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(result['video_path'])}")
    else:
        print("❌ Video generation failed")

if __name__ == "__main__":
    main()

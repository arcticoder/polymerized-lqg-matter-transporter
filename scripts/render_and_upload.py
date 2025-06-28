"""
Video/Animation Pipeline for Newton-Raphson Solver Simulations

This module provides automated video generation and YouTube upload capabilities
for visualizing the enhanced stargate transporter simulation results.

Mathematical Framework:
    - Sample simulation at Δt = 1s intervals
    - Frame j corresponds to t_j = j·1s  
    - Energy density: E_ij^(j) = T_00(ρ_i, z_j; t_j)
    - Generate N = ⌊T_sim/1s⌋ + 1 frames

Author: Enhanced Implementation
Created: June 27, 2025
"""

import os
import sys
import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import time
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

# YouTube API imports
try:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
    YOUTUBE_AVAILABLE = True
except ImportError:
    print("⚠️ YouTube API libraries not available. Install with: pip install google-auth-oauthlib google-api-python-client")
    YOUTUBE_AVAILABLE = False

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from core.enhanced_stargate_transporter import EnhancedStargateTransporter, EnhancedTransporterConfig
from solvers.newton_raphson_solver import NewtonRaphsonIterativeSolver

class SimulationVideoRenderer:
    """
    Automated video generation from Newton-Raphson solver simulations.
    
    Samples simulation at 1fps, renders energy density evolution,
    and provides YouTube upload integration.
    """
    
    def __init__(self, 
                 transporter: EnhancedStargateTransporter,
                 solver: NewtonRaphsonIterativeSolver,
                 render_config: Optional[Dict] = None):
        """
        Initialize video renderer.
        
        Args:
            transporter: Enhanced stargate transporter instance
            solver: Newton-Raphson solver instance  
            render_config: Rendering configuration parameters
        """
        self.transporter = transporter
        self.solver = solver
        
        # Default rendering configuration
        if render_config is None:
            render_config = {
                'fps': 1,                    # Frames per second (1fps = 1s intervals)
                'dpi': 150,                  # Image resolution
                'figsize': (10, 8),          # Figure size (width, height)
                'colormap': 'plasma',        # Colormap for energy density
                'frame_format': 'png',       # Frame image format
                'video_codec': 'libx264',    # Video codec
                'video_format': 'mp4',       # Video format
                'pixel_format': 'yuv420p',   # Pixel format for compatibility
                'quality': 'high',           # Video quality: 'low', 'medium', 'high'
                'bitrate': '2M'              # Video bitrate
            }
        self.render_config = render_config
        
        # Setup custom colormap for energy density
        self._setup_colormaps()
        
        print(f"SimulationVideoRenderer initialized:")
        print(f"  Sampling rate: {render_config['fps']} fps ({1/render_config['fps']:.1f}s intervals)")
        print(f"  Resolution: {render_config['dpi']} dpi")
        print(f"  Figure size: {render_config['figsize']}")
        print(f"  Video codec: {render_config['video_codec']}")
        
    def _setup_colormaps(self):
        """Setup custom colormaps for different field visualizations."""
        
        # Energy density colormap (dark to bright plasma)
        colors_energy = ['#000033', '#000055', '#330055', '#660055', 
                        '#990055', '#CC0055', '#FF3366', '#FF6699', 
                        '#FFCCDD', '#FFFFFF']
        self.energy_cmap = LinearSegmentedColormap.from_list('energy_density', colors_energy)
        
        # Field component colormap (diverging)
        colors_field = ['#0000FF', '#3366FF', '#66CCFF', '#FFFFFF', 
                       '#FFCC66', '#FF6633', '#FF0000']
        self.field_cmap = LinearSegmentedColormap.from_list('field_components', colors_field)
        
        print(f"  ✅ Custom colormaps configured")
        
    def stress_energy_function(self, X, Y, Z, t):
        """
        Time-dependent stress-energy tensor for visualization.
        
        Args:
            X, Y, Z: Coordinate meshgrids
            t: Time parameter
            
        Returns:
            Stress-energy tensor components
        """
        r_cyl = np.sqrt(X**2 + Y**2)
        
        # Evolving energy source with time dependence
        base_source = np.exp(-(r_cyl - 1.5)**2 / 0.8) * np.exp(-(Z**2) / 30.0)
        
        # Complex time evolution
        oscillation = np.sin(t * 2 * np.pi / 10.0)  # 10-second period
        growth = 1 + 0.3 * np.tanh(t / 5.0)  # Gradual growth
        pulse = np.exp(-((t - 15.0)**2) / 25.0)  # Pulse at t=15s
        
        time_factor = growth * (1 + 0.2 * oscillation + 0.5 * pulse)
        source = base_source * time_factor * 1e-5
        
        # Expand to field components
        n_components = self.solver.field_config['field_components']
        source_expanded = np.zeros(X.shape + (n_components,))
        
        for i in range(n_components):
            component_factor = 1.0 + 0.1 * (i % 3) / 3.0
            source_expanded = source_expanded.at[:, :, :, i].set(source * component_factor)
            
        return source_expanded
        
    def render_simulation_frames(self, T_sim: float, 
                                output_dir: Optional[str] = None) -> Dict:
        """
        Render simulation frames at 1-second intervals.
        
        Args:
            T_sim: Total simulation time (seconds)
            output_dir: Output directory for frames (temp dir if None)
            
        Returns:
            Rendering results and metadata
        """
        start_time = time.time()
        
        print(f"\n🎬 RENDERING SIMULATION FRAMES")
        print("-" * 50)
        print(f"Simulation time: {T_sim:.1f} seconds")
        
        # Calculate frame times
        dt = 1.0 / self.render_config['fps']  # Time interval (1s for 1fps)
        frame_times = np.arange(0, T_sim + dt, dt)
        n_frames = len(frame_times)
        
        print(f"Frame interval: {dt:.1f} seconds")
        print(f"Total frames: {n_frames}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="simulation_frames_")
        else:
            os.makedirs(output_dir, exist_ok=True)
            
        print(f"Output directory: {output_dir}")
        
        # Setup grid for visualization (2D slice at z=0)
        grid = self.solver.grid
        nx, ny = grid['nx'], grid['ny']
        z_slice = nx // 2  # Middle slice in z-direction
        
        X_2d = grid['X'][:, :, z_slice]
        Y_2d = grid['Y'][:, :, z_slice]
        
        # Storage for frame data
        frame_metadata = []
        energy_range_global = [float('inf'), float('-inf')]
        
        print(f"Starting frame rendering...")
        render_start = time.time()
        
        for j, t in enumerate(frame_times):
            frame_start = time.time()
            
            try:
                # Solve field equation at time t
                result = self.solver.solve_nonlinear_field(self.stress_energy_function, t)
                solution = result['solution']
                
                # Extract 2D slice for visualization
                field_2d = solution[:, :, z_slice, :]  # Shape: (nx, ny, n_components)
                
                # Compute energy density T_00 approximation
                energy_density = np.sum(field_2d**2, axis=2)  # Sum over field components
                
                # Update global range for consistent colorbar
                energy_range_global[0] = min(energy_range_global[0], np.min(energy_density))
                energy_range_global[1] = max(energy_range_global[1], np.max(energy_density))
                
                # Create visualization
                fig, axes = plt.subplots(2, 2, figsize=self.render_config['figsize'])
                fig.suptitle(f'Enhanced Stargate Transporter Simulation\nt = {t:.1f} s', 
                           fontsize=16, fontweight='bold')
                
                # Energy density plot
                ax1 = axes[0, 0]
                im1 = ax1.imshow(energy_density.T, origin='lower', 
                               extent=[-self.transporter.config.R_payload*3, 
                                      self.transporter.config.R_payload*3,
                                      -self.transporter.config.R_payload*3,
                                      self.transporter.config.R_payload*3],
                               cmap=self.energy_cmap, aspect='equal')
                ax1.set_title('Energy Density T₀₀')
                ax1.set_xlabel('x (m)')
                ax1.set_ylabel('y (m)')
                plt.colorbar(im1, ax=ax1, label='Energy Density (J/m³)')
                
                # Add payload region indicator
                circle = plt.Circle((0, 0), self.transporter.config.R_payload, 
                                  fill=False, color='white', linewidth=2, linestyle='--')
                ax1.add_patch(circle)
                
                # Field component 0
                ax2 = axes[0, 1]
                im2 = ax2.imshow(field_2d[:, :, 0].T, origin='lower',
                               extent=[-self.transporter.config.R_payload*3, 
                                      self.transporter.config.R_payload*3,
                                      -self.transporter.config.R_payload*3,
                                      self.transporter.config.R_payload*3],
                               cmap=self.field_cmap, aspect='equal')
                ax2.set_title('Metric Component g₀₀')
                ax2.set_xlabel('x (m)')
                ax2.set_ylabel('y (m)')
                plt.colorbar(im2, ax=ax2, label='Field Amplitude')
                
                # Field component 1  
                ax3 = axes[1, 0]
                im3 = ax3.imshow(field_2d[:, :, 1].T, origin='lower',
                               extent=[-self.transporter.config.R_payload*3, 
                                      self.transporter.config.R_payload*3,
                                      -self.transporter.config.R_payload*3,
                                      self.transporter.config.R_payload*3],
                               cmap=self.field_cmap, aspect='equal')
                ax3.set_title('Metric Component g₁₁')
                ax3.set_xlabel('x (m)')
                ax3.set_ylabel('y (m)')
                plt.colorbar(im3, ax=ax3, label='Field Amplitude')
                
                # Convergence information
                ax4 = axes[1, 1]
                ax4.text(0.1, 0.8, f"Newton-Raphson Results:", fontweight='bold', transform=ax4.transAxes)
                ax4.text(0.1, 0.7, f"Converged: {'✅' if result['converged'] else '❌'}", transform=ax4.transAxes)
                ax4.text(0.1, 0.6, f"Iterations: {result['iterations']}", transform=ax4.transAxes)
                ax4.text(0.1, 0.5, f"Final residual: {result['final_residual']:.2e}", transform=ax4.transAxes)
                ax4.text(0.1, 0.4, f"Solve time: {result['timing']['total_time']:.3f}s", transform=ax4.transAxes)
                
                # Field statistics
                ax4.text(0.1, 0.25, f"Field Statistics:", fontweight='bold', transform=ax4.transAxes)
                ax4.text(0.1, 0.15, f"Energy range: [{np.min(energy_density):.2e}, {np.max(energy_density):.2e}]", transform=ax4.transAxes)
                ax4.text(0.1, 0.05, f"Field norm: {np.linalg.norm(field_2d):.2e}", transform=ax4.transAxes)
                
                ax4.set_xlim(0, 1)
                ax4.set_ylim(0, 1)
                ax4.axis('off')
                
                plt.tight_layout()
                
                # Save frame
                frame_filename = f"frame_{j:04d}.{self.render_config['frame_format']}"
                frame_path = os.path.join(output_dir, frame_filename)
                plt.savefig(frame_path, dpi=self.render_config['dpi'], 
                           bbox_inches='tight', facecolor='white')
                plt.close()
                
                # Store metadata
                frame_data = {
                    'frame_number': j,
                    'time': t,
                    'filename': frame_filename,
                    'path': frame_path,
                    'converged': result['converged'],
                    'iterations': result['iterations'],
                    'residual': result['final_residual'],
                    'solve_time': result['timing']['total_time'],
                    'energy_range': [float(np.min(energy_density)), float(np.max(energy_density))],
                    'field_norm': float(np.linalg.norm(field_2d))
                }
                frame_metadata.append(frame_data)
                
                frame_time = time.time() - frame_start
                
                if j % 5 == 0 or j == n_frames - 1:
                    print(f"  Frame {j+1:3d}/{n_frames}: t={t:5.1f}s, "
                          f"converged={'✅' if result['converged'] else '❌'}, "
                          f"time={frame_time:.2f}s")
                    
            except Exception as e:
                print(f"  ❌ Error rendering frame {j}: {e}")
                continue
        
        render_time = time.time() - render_start
        total_time = time.time() - start_time
        
        print(f"\n📊 FRAME RENDERING COMPLETE")
        print("-" * 50)
        print(f"Frames rendered: {len(frame_metadata)}/{n_frames}")
        print(f"Success rate: {len(frame_metadata)/n_frames:.1%}")
        print(f"Render time: {render_time:.1f}s")
        print(f"Average time per frame: {render_time/n_frames:.2f}s")
        print(f"Total time: {total_time:.1f}s")
        print(f"Output directory: {output_dir}")
        
        return {
            'output_dir': output_dir,
            'frame_metadata': frame_metadata,
            'n_frames': len(frame_metadata),
            'success_rate': len(frame_metadata) / n_frames,
            'timing': {
                'render_time': render_time,
                'total_time': total_time,
                'time_per_frame': render_time / n_frames if n_frames > 0 else 0
            },
            'energy_range_global': energy_range_global,
            'simulation_parameters': {
                'T_sim': T_sim,
                'dt': dt,
                'n_frames_target': n_frames
            }
        }
    
    def assemble_video(self, frame_dir: str, output_path: Optional[str] = None) -> Dict:
        """
        Assemble video from rendered frames using FFmpeg.
        
        Args:
            frame_dir: Directory containing frame images
            output_path: Output video path (auto-generated if None)
            
        Returns:
            Video assembly results
        """
        print(f"\n🎥 ASSEMBLING VIDEO WITH FFMPEG")
        print("-" * 50)
        
        if output_path is None:
            output_path = os.path.join(frame_dir, f"simulation_{int(time.time())}.{self.render_config['video_format']}")
            
        print(f"Input frames: {frame_dir}")
        print(f"Output video: {output_path}")
        print(f"Target FPS: {self.render_config['fps']}")
        
        # FFmpeg command configuration
        quality_settings = {
            'low': ['-crf', '28'],
            'medium': ['-crf', '23'], 
            'high': ['-crf', '18']
        }
        
        crf_settings = quality_settings.get(self.render_config['quality'], quality_settings['medium'])
        
        ffmpeg_cmd = [
            'ffmpeg', '-y',  # Overwrite output file
            '-framerate', str(self.render_config['fps']),
            '-i', os.path.join(frame_dir, f"frame_%04d.{self.render_config['frame_format']}"),
            '-c:v', self.render_config['video_codec'],
            '-pix_fmt', self.render_config['pixel_format'],
            *crf_settings,
            '-movflags', '+faststart',  # Enable fast streaming
            output_path
        ]
        
        start_time = time.time()
        
        try:
            print(f"Running FFmpeg command...")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            
            assembly_time = time.time() - start_time
            
            # Get video file size
            video_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            
            print(f"✅ Video assembly successful!")
            print(f"Assembly time: {assembly_time:.1f}s")
            print(f"Video size: {video_size / (1024*1024):.1f} MB")
            print(f"Video path: {output_path}")
            
            return {
                'success': True,
                'output_path': output_path,
                'assembly_time': assembly_time,
                'video_size_mb': video_size / (1024*1024),
                'ffmpeg_command': ' '.join(ffmpeg_cmd)
            }
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error: {e}")
            print(f"FFmpeg stdout: {e.stdout}")
            print(f"FFmpeg stderr: {e.stderr}")
            
            return {
                'success': False,
                'error': str(e),
                'ffmpeg_stdout': e.stdout,
                'ffmpeg_stderr': e.stderr
            }
        except FileNotFoundError:
            print(f"❌ FFmpeg not found. Please install FFmpeg.")
            return {
                'success': False,
                'error': 'FFmpeg not found. Please install FFmpeg.'
            }

class YouTubeUploader:
    """
    YouTube video upload and playlist management.
    """
    
    def __init__(self, client_secrets_path: str):
        """
        Initialize YouTube uploader.
        
        Args:
            client_secrets_path: Path to Google OAuth client secrets JSON
        """
        self.client_secrets_path = client_secrets_path
        self.youtube = None
        self.authenticated = False
        
        if not YOUTUBE_AVAILABLE:
            print("⚠️ YouTube upload functionality not available")
            return
            
        print(f"YouTubeUploader initialized:")
        print(f"  Client secrets: {client_secrets_path}")
        
    def authenticate(self) -> bool:
        """
        Authenticate with YouTube API using OAuth flow.
        
        Returns:
            True if authentication successful
        """
        if not YOUTUBE_AVAILABLE:
            print("❌ YouTube API libraries not available")
            return False
            
        try:
            print(f"\n🔐 YOUTUBE AUTHENTICATION")
            print("-" * 50)
            
            scopes = [
                "https://www.googleapis.com/auth/youtube.upload",
                "https://www.googleapis.com/auth/youtube"
            ]
            
            flow = InstalledAppFlow.from_client_secrets_file(
                self.client_secrets_path, scopes=scopes
            )
            
            print("Opening browser for OAuth authentication...")
            credentials = flow.run_console()
            
            self.youtube = build("youtube", "v3", credentials=credentials)
            self.authenticated = True
            
            print("✅ YouTube authentication successful!")
            return True
            
        except Exception as e:
            print(f"❌ YouTube authentication failed: {e}")
            self.authenticated = False
            return False
    
    def get_or_create_playlist(self, repo_name: str) -> Optional[str]:
        """
        Get existing playlist or create new one for repository.
        
        Args:
            repo_name: Repository name for playlist
            
        Returns:
            Playlist ID if successful, None otherwise
        """
        if not self.authenticated:
            print("❌ Not authenticated with YouTube")
            return None
            
        try:
            print(f"\n📝 PLAYLIST MANAGEMENT")
            print("-" * 50)
            print(f"Repository: {repo_name}")
            
            # Search for existing playlist
            playlists = self.youtube.playlists().list(
                part="id,snippet",
                mine=True,
                maxResults=50
            ).execute()
            
            existing_playlist = None
            for playlist in playlists.get("items", []):
                if playlist["snippet"]["title"] == repo_name:
                    existing_playlist = playlist
                    break
                    
            if existing_playlist:
                playlist_id = existing_playlist["id"]
                print(f"✅ Found existing playlist: {playlist_id}")
                return playlist_id
            else:
                # Create new playlist
                print(f"Creating new playlist for {repo_name}...")
                
                playlist_body = {
                    "snippet": {
                        "title": repo_name,
                        "description": f"Simulation videos from {repo_name} repository",
                        "defaultLanguage": "en"
                    },
                    "status": {
                        "privacyStatus": "unlisted"  # Unlisted by default
                    }
                }
                
                playlist = self.youtube.playlists().insert(
                    part="snippet,status",
                    body=playlist_body
                ).execute()
                
                playlist_id = playlist["id"]
                print(f"✅ Created new playlist: {playlist_id}")
                return playlist_id
                
        except HttpError as e:
            print(f"❌ YouTube API error: {e}")
            return None
        except Exception as e:
            print(f"❌ Playlist error: {e}")
            return None
    
    def upload_video(self, video_path: str, playlist_id: str, 
                    title: Optional[str] = None, 
                    description: Optional[str] = None) -> Optional[str]:
        """
        Upload video to YouTube and add to playlist.
        
        Args:
            video_path: Path to video file
            playlist_id: Target playlist ID
            title: Video title (auto-generated if None)
            description: Video description
            
        Returns:
            Video ID if successful, None otherwise
        """
        if not self.authenticated:
            print("❌ Not authenticated with YouTube")
            return None
            
        try:
            print(f"\n📤 YOUTUBE VIDEO UPLOAD")
            print("-" * 50)
            print(f"Video path: {video_path}")
            print(f"Playlist ID: {playlist_id}")
            
            # Auto-generate title if not provided
            if title is None:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                title = f"Enhanced Stargate Transporter Simulation - {timestamp}"
                
            if description is None:
                description = """
Simulation of enhanced stargate transporter using Newton-Raphson solver
for nonlinear Einstein field equations.

Generated by polymerized-lqg-matter-transporter
Mathematical framework: Loop Quantum Gravity + General Relativity
Solver: Newton-Raphson iterative method with Anderson acceleration
                """.strip()
                
            print(f"Title: {title}")
            
            # Prepare video metadata
            video_body = {
                "snippet": {
                    "title": title,
                    "description": description,
                    "tags": [
                        "physics", "simulation", "general relativity",
                        "loop quantum gravity", "newton raphson",
                        "stargate", "exotic matter", "spacetime"
                    ],
                    "categoryId": "28"  # Science & Technology
                },
                "status": {
                    "privacyStatus": "unlisted",  # Unlisted by default
                    "selfDeclaredMadeForKids": False
                }
            }
            
            # Create media upload
            media = MediaFileUpload(
                video_path,
                chunksize=-1,  # Upload in single request
                resumable=True,
                mimetype="video/mp4"
            )
            
            print("Uploading video...")
            upload_start = time.time()
            
            # Upload video
            request = self.youtube.videos().insert(
                part="snippet,status",
                body=video_body,
                media_body=media
            )
            
            response = request.execute()
            video_id = response["id"]
            
            upload_time = time.time() - upload_start
            print(f"✅ Video uploaded successfully!")
            print(f"Video ID: {video_id}")
            print(f"Upload time: {upload_time:.1f}s")
            
            # Add to playlist
            print("Adding to playlist...")
            
            playlist_item_body = {
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {
                        "kind": "youtube#video",
                        "videoId": video_id
                    }
                }
            }
            
            self.youtube.playlistItems().insert(
                part="snippet",
                body=playlist_item_body
            ).execute()
            
            print(f"✅ Added to playlist: {playlist_id}")
            print(f"Video URL: https://www.youtube.com/watch?v={video_id}")
            
            return video_id
            
        except HttpError as e:
            print(f"❌ YouTube API error: {e}")
            return None
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return None

def main():
    """
    Complete video generation and upload pipeline demonstration.
    """
    print("="*70)
    print("VIDEO/ANIMATION PIPELINE FOR NEWTON-RAPHSON SOLVER")
    print("="*70)
    
    # Initialize enhanced transporter
    config = EnhancedTransporterConfig(
        R_payload=2.0,
        R_neck=0.08, 
        L_corridor=50.0,
        corridor_mode="sinusoidal",
        v_conveyor_max=1e6
    )
    transporter = EnhancedStargateTransporter(config)
    
    # Initialize Newton-Raphson solver
    solver_config = {
        'max_iterations': 30,  # Reduced for video generation
        'tolerance': 1e-6,     # Relaxed tolerance for speed
        'initial_damping': 0.8,
        'line_search': True,
        'jacobian_method': 'forward',
        'acceleration': 'anderson'
    }
    
    field_config = {
        'nx': 32, 'ny': 32, 'nz': 16,  # Reduced resolution for speed
        'field_components': 4,          # Fewer components for demo
        'boundary_conditions': 'asymptotically_flat'
    }
    
    solver = NewtonRaphsonIterativeSolver(transporter, solver_config, field_config)
    
    # Initialize video renderer
    render_config = {
        'fps': 1,
        'dpi': 120,               # Reduced for faster rendering
        'figsize': (12, 10),
        'colormap': 'plasma',
        'video_codec': 'libx264',
        'quality': 'medium'
    }
    
    renderer = SimulationVideoRenderer(transporter, solver, render_config)
    
    # Render simulation frames
    T_sim = 20.0  # 20-second simulation
    print(f"\nStarting {T_sim:.0f}-second simulation rendering...")
    
    frame_results = renderer.render_simulation_frames(T_sim)
    
    if frame_results['success_rate'] < 0.5:
        print(f"❌ Low frame success rate: {frame_results['success_rate']:.1%}")
        return
        
    # Assemble video
    video_results = renderer.assemble_video(frame_results['output_dir'])
    
    if not video_results['success']:
        print(f"❌ Video assembly failed")
        return
        
    print(f"\n🎉 Video generation complete!")
    print(f"Video path: {video_results['output_path']}")
    print(f"Video size: {video_results['video_size_mb']:.1f} MB")
    
    # Optional YouTube upload (requires authentication)
    upload_to_youtube = input("\n🔗 Upload to YouTube? (y/N): ").lower().strip() == 'y'
    
    if upload_to_youtube and YOUTUBE_AVAILABLE:
        client_secrets_path = r"C:\Users\echo_\Code\asciimath\client_secret_868692484023-92sa85eaoiui1vpo2gcmskrl7fpq0vnm.apps.googleusercontent.com.json"
        
        if os.path.exists(client_secrets_path):
            uploader = YouTubeUploader(client_secrets_path)
            
            if uploader.authenticate():
                repo_name = "polymerized-lqg-matter-transporter"
                playlist_id = uploader.get_or_create_playlist(repo_name)
                
                if playlist_id:
                    video_id = uploader.upload_video(
                        video_results['output_path'],
                        playlist_id,
                        title=f"Newton-Raphson Solver Simulation - {time.strftime('%Y-%m-%d %H:%M:%S')}",
                        description=f"""
Enhanced Stargate Transporter Simulation using Newton-Raphson solver.

Simulation Parameters:
- Duration: {T_sim:.0f} seconds
- Grid resolution: {field_config['nx']}×{field_config['ny']}×{field_config['nz']}
- Field components: {field_config['field_components']}
- Solver iterations: {solver_config['max_iterations']}
- Frame rate: {render_config['fps']} fps

Results:
- Frames rendered: {frame_results['n_frames']}
- Success rate: {frame_results['success_rate']:.1%}
- Video size: {video_results['video_size_mb']:.1f} MB

Generated by polymerized-lqg-matter-transporter
                        """.strip()
                    )
                    
                    if video_id:
                        print(f"\n🎥 Upload successful!")
                        print(f"Video URL: https://www.youtube.com/watch?v={video_id}")
                    else:
                        print(f"❌ Upload failed")
                else:
                    print(f"❌ Could not access playlist")
            else:
                print(f"❌ YouTube authentication failed")
        else:
            print(f"❌ Client secrets file not found: {client_secrets_path}")
    
    return renderer

if __name__ == "__main__":
    video_renderer = main()

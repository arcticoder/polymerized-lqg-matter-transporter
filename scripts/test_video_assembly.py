"""
Manual Video Assembly Test

This script manually assembles a video from the generated frames
to troubleshoot the FFmpeg command issue.

Author: Enhanced Implementation
Created: June 27, 2025
"""

import os
import subprocess
import tempfile
import glob

def find_latest_frames_directory():
    """Find the most recent transporter video directory."""
    
    temp_dir = tempfile.gettempdir()
    pattern = os.path.join(temp_dir, "transporter_video_*")
    
    # Find all matching directories
    directories = glob.glob(pattern)
    
    if not directories:
        print("❌ No transporter video directories found")
        return None
        
    # Sort by modification time (most recent first)
    latest_dir = max(directories, key=os.path.getmtime)
    
    print(f"📁 Found latest directory: {latest_dir}")
    return latest_dir

def test_ffmpeg_assembly():
    """Test FFmpeg video assembly with the generated frames."""
    
    # Find the frames directory
    frames_dir = find_latest_frames_directory()
    if not frames_dir:
        return False
        
    # Check for PNG files
    png_files = glob.glob(os.path.join(frames_dir, "frame_*.png"))
    
    if not png_files:
        print("❌ No PNG frame files found")
        return False
        
    print(f"✅ Found {len(png_files)} PNG frames")
    
    # List first few frames
    for i, frame in enumerate(sorted(png_files)[:5]):
        print(f"  Frame {i}: {os.path.basename(frame)}")
    
    # Output video path
    video_path = os.path.join(frames_dir, "enhanced_transporter_manual.mp4")
    
    # Try different FFmpeg command variations
    commands_to_try = [
        # Original command
        [
            'ffmpeg', '-y',
            '-framerate', '1',
            '-i', os.path.join(frames_dir, 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            '-movflags', '+faststart',
            video_path
        ],
        
        # Simplified command
        [
            'ffmpeg', '-y',
            '-framerate', '1',
            '-i', os.path.join(frames_dir, 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            video_path
        ],
        
        # Even simpler
        [
            'ffmpeg', '-y',
            '-r', '1',
            '-i', os.path.join(frames_dir, 'frame_%04d.png'),
            '-vcodec', 'libx264',
            video_path
        ]
    ]
    
    for i, cmd in enumerate(commands_to_try):
        print(f"\n🔧 Trying FFmpeg command {i+1}/{len(commands_to_try)}:")
        print(f"   {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"✅ Success! Video created at: {video_path}")
                
                if os.path.exists(video_path):
                    video_size = os.path.getsize(video_path) / (1024*1024)
                    print(f"   Video size: {video_size:.1f} MB")
                    return True
                else:
                    print(f"❌ Return code 0 but no video file found")
                    
            else:
                print(f"❌ FFmpeg failed with return code: {result.returncode}")
                if result.stderr:
                    # Show only first few lines of error
                    error_lines = result.stderr.split('\n')[:5]
                    for line in error_lines:
                        if line.strip():
                            print(f"   Error: {line}")
                            
        except subprocess.TimeoutExpired:
            print(f"❌ FFmpeg timed out after 30 seconds")
        except Exception as e:
            print(f"❌ Exception: {e}")
            
    return False

def main():
    """Main manual video assembly test."""
    
    print("🔧 MANUAL VIDEO ASSEMBLY TEST")
    print("=" * 40)
    
    success = test_ffmpeg_assembly()
    
    if success:
        print(f"\n🎉 Video assembly successful!")
        print(f"Your enhanced stargate transporter video is ready!")
    else:
        print(f"\n❌ Video assembly failed")
        print(f"The frames were generated successfully, but FFmpeg assembly needs debugging")

if __name__ == "__main__":
    main()

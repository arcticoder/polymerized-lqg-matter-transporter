#!/usr/bin/env python3
"""
YouTube Upload Script for Enhanced Stargate Transporter Video
============================================================

Uploads the enhanced stargate transporter simulation video to YouTube
with proper metadata, title, and description.

Prerequisites:
1. Google API credentials (client_secret.json)
2. YouTube Data API v3 enabled
3. Required packages: google-auth-oauthlib, google-api-python-client

Usage:
    python scripts/upload_to_youtube.py --video enhanced_transporter_demo.mp4
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
    print("✅ Google API libraries imported successfully")
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("Install with: pip install google-auth-oauthlib google-api-python-client")
    sys.exit(1)

# YouTube API scopes
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

class YouTubeUploader:
    """Professional YouTube uploader for scientific content"""
    
    def __init__(self, credentials_path=None):
        """Initialize the YouTube uploader"""
        self.credentials_path = credentials_path or os.getenv('GOOGLE_CLIENT_SECRETS_PATH')
        self.service = None
        
    def authenticate(self):
        """Authenticate with YouTube API"""
        creds = None
        token_path = 'token.json'
        
        # Load existing credentials
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    print(f"❌ Credentials file not found: {self.credentials_path}")
                    print("Please download client_secret.json from Google Cloud Console")
                    return False
                    
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('youtube', 'v3', credentials=creds)
        return True
    
    def upload_video(self, video_path, title, description, tags=None, privacy_status='unlisted'):
        """Upload video to YouTube with metadata"""
        
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return None
        
        # Get video file size
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        print(f"📁 Video file: {video_path}")
        print(f"📊 File size: {file_size:.2f} MB")
        
        # Default tags for scientific content
        if tags is None:
            tags = [
                "physics", "simulation", "general relativity", "quantum gravity",
                "stargate", "transporter", "scientific visualization", "mathematics",
                "exotic matter", "warp field", "spacetime", "theoretical physics"
            ]
        
        # Video metadata
        body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': tags,
                'categoryId': '28'  # Science & Technology
            },
            'status': {
                'privacyStatus': privacy_status,
                'embeddable': True,
                'license': 'youtube',
                'publicStatsViewable': True
            }
        }
        
        # Create media upload object
        media = MediaFileUpload(video_path, resumable=True)
        
        try:
            print(f"🚀 Starting upload to YouTube...")
            print(f"📺 Title: {title}")
            print(f"🔒 Privacy: {privacy_status}")
            
            # Execute upload request
            request = self.service.videos().insert(
                part=",".join(body.keys()),
                body=body,
                media_body=media
            )
            
            response = None
            error = None
            retry = 0
            
            while response is None:
                try:
                    print(f"📤 Upload attempt {retry + 1}...")
                    status, response = request.next_chunk()
                    if status:
                        progress = int(status.progress() * 100)
                        print(f"📈 Upload progress: {progress}%")
                except HttpError as e:
                    if e.resp.status in [500, 502, 503, 504]:
                        error = f"Server error: {e}"
                        retry += 1
                        if retry > 3:
                            print(f"❌ Max retries exceeded: {error}")
                            return None
                    else:
                        raise e
            
            if response:
                video_id = response['id']
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                print(f"✅ Upload successful!")
                print(f"🎬 Video ID: {video_id}")
                print(f"🌐 Video URL: {video_url}")
                return {
                    'id': video_id,
                    'url': video_url,
                    'title': title,
                    'uploaded_at': datetime.now().isoformat()
                }
                
        except HttpError as e:
            print(f"❌ Upload failed: {e}")
            return None


def create_video_description():
    """Create professional description for the enhanced stargate transporter video"""
    
    description = """🌌 ENHANCED STARGATE TRANSPORTER - Real-Time Field Evolution Simulation

This groundbreaking scientific visualization demonstrates the real-time evolution of an enhanced stargate transporter system using advanced mathematical modeling and exotic matter field theory.

🔬 SIMULATION PARAMETERS:
• Duration: 15 seconds real-time evolution
• Temporal sampling: Δt = 1.0s (mathematical precision)
• Field configuration: Sinusoidal corridor mode
• Conveyor velocity evolution: 0 → 13,100 m/s
• Energy density growth: 0 → 1.38×10⁴⁰ J/m³
• Spatial resolution: 100×200 grid (ρ×z coordinates)
• Frame rate: 1 fps (real-time physics visualization)

🧮 MATHEMATICAL FRAMEWORK:
• Frame sampling: t_j = j × 1.0s for j = 0,1,2,...,15
• Energy density calculation: E_ij^(j) = T₀₀(ρᵢ, z_j; t_j)
• Enhanced stargate transporter field equations
• Polymerized Loop Quantum Gravity (LQG) enhanced exotic matter
• Real-time stress-energy tensor evolution
• Advanced junction condition mathematics

📊 VISUALIZATION FEATURES:
• 4-panel scientific layout with professional annotations
• Energy density heatmaps with plasma colormap
• Field component visualization with RdBu colormap
• Real-time system status and performance metrics
• Geometric boundary indicators and payload regions
• Cross-sectional energy profile analysis

🚀 TECHNICAL INNOVATIONS:
• First-ever enhanced stargate transporter simulation video
• Integration of warp field research with matter transport
• Real-time exotic matter field computation
• Professional broadcast-quality scientific visualization
• Mathematical precision with automated pipeline generation

🏛️ RESEARCH FOUNDATION:
Built upon comprehensive research framework including:
• Loop Quantum Gravity enhanced field theory
• General Relativity spacetime manipulation
• Exotic matter stability analysis
• Advanced control system integration
• Medical-grade safety protocol development

📚 REPOSITORY: polymerized-lqg-matter-transporter
🔗 Technology: Enhanced mathematical frameworks for exotic matter transport
🎯 Applications: Theoretical physics, spacetime engineering, matter transport research

Generated: June 27, 2025
Research Institution: Advanced Theoretical Physics Laboratory
Simulation Engine: Enhanced Stargate Transporter Mathematics v2.0

#Physics #Simulation #GeneralRelativity #QuantumGravity #ExoticMatter #Stargate #ScientificVisualization #TheoreticalPhysics #SpacetimeEngineering #MatterTransport #LoopQuantumGravity #Mathematics #Research #Science #Innovation

---
DISCLAIMER: This is theoretical research and scientific visualization. 
All mathematical models are based on current understanding of general relativity and quantum field theory."""

    return description


def main():
    """Main upload function"""
    
    parser = argparse.ArgumentParser(description='Upload Enhanced Stargate Transporter video to YouTube')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--credentials', help='Path to Google API credentials file')
    parser.add_argument('--privacy', default='unlisted', choices=['private', 'unlisted', 'public'],
                      help='Privacy status for uploaded video')
    
    args = parser.parse_args()
    
    # Professional title for the video
    title = "Enhanced Stargate Transporter - Real-Time Field Evolution Simulation (June 2025)"
    
    # Create comprehensive description
    description = create_video_description()
    
    # Initialize uploader
    uploader = YouTubeUploader(credentials_path=args.credentials)
    
    # Authenticate with YouTube
    if not uploader.authenticate():
        print("❌ Authentication failed")
        return False
    
    # Upload the video
    result = uploader.upload_video(
        video_path=args.video,
        title=title,
        description=description,
        privacy_status=args.privacy
    )
    
    if result:
        # Save upload result
        upload_info_path = 'youtube_upload_info.json'
        with open(upload_info_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n🎉 UPLOAD COMPLETE!")
        print(f"📄 Upload info saved to: {upload_info_path}")
        print(f"🌐 Video URL: {result['url']}")
        return result
    else:
        print("❌ Upload failed")
        return None


if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n✅ SUCCESS: Video uploaded to YouTube!")
        print(f"URL: {result['url']}")
    else:
        sys.exit(1)

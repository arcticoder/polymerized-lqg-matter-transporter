#!/usr/bin/env python3
"""
Simple YouTube Upload Script
"""
import os
import sys

# Set credentials path
os.environ['GOOGLE_CLIENT_SECRETS_PATH'] = r'C:\Users\echo_\Code\asciimath\client_secret_868692484023-92sa85eaoiui1vpo2gcmskrl7fpq0vnm.apps.googleusercontent.com.json'

try:
    from scripts.upload_to_youtube import YouTubeUploader, create_video_description
    import json
    
    print("✅ Modules imported successfully")
    
    # Initialize uploader
    uploader = YouTubeUploader()
    print("🚀 Initializing YouTube uploader...")
    
    # Authenticate
    print("🔐 Starting authentication...")
    if uploader.authenticate():
        print("✅ Authentication successful!")
        
        # Prepare video details
        title = "Enhanced Stargate Transporter - Real-Time Field Evolution Simulation (June 2025)"
        description = create_video_description()
        video_path = "enhanced_transporter_demo.mp4"
        
        print(f"📁 Video file: {video_path}")
        print(f"📺 Title: {title}")
        print("🚀 Starting upload...")
        
        # Upload video as PUBLIC
        result = uploader.upload_video(
            video_path=video_path,
            title=title,
            description=description,
            privacy_status='public'
        )
        
        if result:
            print(f"\n🎉 SUCCESS!")
            print(f"🌐 Video URL: {result['url']}")
            print(f"🆔 Video ID: {result['id']}")
            
            # Save result
            with open('youtube_upload_result.json', 'w') as f:
                json.dump(result, f, indent=2)
            print("📄 Upload details saved to youtube_upload_result.json")
            
        else:
            print("❌ Upload failed")
            sys.exit(1)
            
    else:
        print("❌ Authentication failed")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

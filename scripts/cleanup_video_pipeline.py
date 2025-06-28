#!/usr/bin/env python3
"""
Remove all video/YouTube‐related scripts and docs,
now that we've dropped the animation pipeline.
"""

import os
import sys

# adjust if your project root is elsewhere
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

to_remove = [
    # video generators & uploaders
    'scripts/render_and_upload.py',
    'scripts/basic_video_demo.py',
    'scripts/test_video_assembly.py',
    'scripts/upload_to_youtube.py',
    'scripts/simple_video_demo.py',
    'scripts/add_youtube_url.py',
    'upload_video_now.py',
    # video output
    'enhanced_transporter_demo.mp4',
    'youtube_upload_result.json',
    # docs and pipeline descriptions
    'VIDEO_PIPELINE.md',
    'YOUTUBE_UPLOAD_INSTRUCTIONS.md',
    'VIDEO_IMPLEMENTATION_COMPLETE.md',
    'VIDEO_SUCCESS_SUMMARY.md',
    'TASK_COMPLETION_SUMMARY.md',
    'requirements_video.txt',
]

removed_count = 0
for rel in to_remove:
    path = os.path.join(PROJECT_ROOT, rel)
    if os.path.exists(path):
        os.remove(path)
        print(f"Removed {rel}")
        removed_count += 1
    else:
        print(f"Not found (already removed?): {rel}")

print(f"✅ Video/YouTube pipeline has been cleaned up. Removed {removed_count} files.")

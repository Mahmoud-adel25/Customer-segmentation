#!/usr/bin/env python3
"""
Customer Segmentation App Launcher
=================================

Launch script for the Customer Segmentation Streamlit application.
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit application."""
    
    # Change to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    # Path to the main Streamlit app
    app_path = os.path.join("streamlit_app", "main.py")
    
    # Check if the app file exists
    if not os.path.exists(app_path):
        print(f"❌ Error: Streamlit app not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Launching Customer Segmentation App...")
    print(f"📂 Project directory: {project_dir}")
    print(f"🎯 App path: {app_path}")
    print("-" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

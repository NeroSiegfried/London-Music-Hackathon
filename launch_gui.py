#!/usr/bin/env python3
"""
ABRSM GUI Launcher

Simple launcher script for the ABRSM AI Music Feedback System GUI.
This ensures proper environment setup and dependency checks.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'tkinter',
        'matplotlib', 
        'numpy',
        'librosa'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def main():
    """Launch the GUI with proper setup"""
    print("🎵 ABRSM AI Music Feedback System - GUI Launcher")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Warning: Not in virtual environment")
        print("   Consider activating venv for best experience")
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("\n💡 To install missing dependencies:")
        print("   pip install -r requirements.txt")
        
        # Special note for tkinter on Linux
        if 'tkinter' in missing:
            print("\n📋 For tkinter on Ubuntu/Debian:")
            print("   sudo apt-get install python3-tk")
        
        return 1
    
    print("✅ All dependencies available")
    
    # Check if analysis modules are available
    try:
        from enhanced_main import MusicAnalyzer
        print("✅ ABRSM analysis modules loaded")
    except ImportError as e:
        print(f"❌ Failed to load analysis modules: {e}")
        print("   Make sure you're in the correct directory")
        return 1
    
    # Launch GUI
    print("\n🚀 Launching GUI interface...")
    print("   Close the GUI window to return to terminal")
    print("-" * 50)
    
    try:
        print("🚀 Launching Enhanced ABRSM Analysis GUI...")
        import tkinter as tk
        from enhanced_gui_interface import EnhancedABRSMGUI
        
        root = tk.Tk()
        app = EnhancedABRSMGUI(root)
        root.mainloop()
        
        print("\n👋 GUI closed successfully")
        return 0
    except Exception as e:
        print(f"\n❌ GUI error: {e}")
        print("\nFor troubleshooting:")
        print("1. Ensure all dependencies are installed")
        print("2. Check that you have a display available (X11/Wayland)")
        print("3. Try running: python enhanced_gui_interface.py directly")
        return 1

if __name__ == "__main__":
    sys.exit(main())

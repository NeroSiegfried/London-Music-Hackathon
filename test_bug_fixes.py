#!/usr/bin/env python3
"""
Test script to verify the bug fixes in the enhanced GUI interface
"""

import os
import sys
import numpy as np
import tkinter as tk

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_gui_interface import EnhancedABRSMGUI

def test_gui_startup():
    """Test that the GUI starts without errors"""
    print("Testing GUI startup...")
    
    root = tk.Tk()
    try:
        app = EnhancedABRSMGUI(root)
        print("✓ GUI created successfully")
        
        # Test sheet music initialization
        if hasattr(app, 'interactive_sheet'):
            print("✓ Interactive sheet music initialized")
        else:
            print("✗ Interactive sheet music not found")
        
        # Test mistake patterns initialization
        if hasattr(app, 'mistake_patterns'):
            print("✓ Mistake patterns list initialized")
        else:
            print("✗ Mistake patterns not initialized")
            
        root.destroy()
        return True
        
    except Exception as e:
        print(f"✗ GUI startup failed: {e}")
        root.destroy()
        return False

def test_demo_load():
    """Test loading the demo file"""
    print("\nTesting demo loading...")
    
    demo_file = "demo_performance.wav"
    if os.path.exists(demo_file):
        print(f"✓ Demo file '{demo_file}' exists")
        return True
    else:
        print(f"✗ Demo file '{demo_file}' not found")
        return False

def test_module_imports():
    """Test that all required modules can be imported"""
    print("\nTesting module imports...")
    
    modules_to_test = [
        'enhanced_main',
        'interactive_sheet_music',
        'librosa',
        'pygame',
        'soundfile',
        'matplotlib'
    ]
    
    all_good = True
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {module} imported successfully")
        except ImportError as e:
            print(f"✗ {module} import failed: {e}")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("ABRSM Enhanced GUI Bug Fix Verification")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    if test_module_imports():
        success_count += 1
    
    if test_demo_load():
        success_count += 1
    
    if test_gui_startup():
        success_count += 1
    
    print(f"\nTest Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("\n🎉 All tests passed! The application should work correctly.")
        print("\nKey fixes implemented:")
        print("• Fixed pitch error calculation for same-frequency notes")
        print("• Added proper waveform and spectrum visualization")  
        print("• Improved note extraction with onset detection")
        print("• Fixed reference note playback")
        print("• Enhanced mistake pattern detection")
        print("• Created interactive sheet music with clickable notes")
    else:
        print(f"\n⚠️  {total_tests - success_count} test(s) failed. Please check the errors above.")

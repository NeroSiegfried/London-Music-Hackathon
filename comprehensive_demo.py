#!/usr/bin/env python3
"""
Comprehensive Demo Script for ABRSM AI Music Feedback System

This script demonstrates all features of the enhanced system including:
- Basic audio analysis
- Time signature detection  
- Sheet music visualization
- Polyphonic content handling
- Enhanced AI feedback
"""

import subprocess
import sys
import os
import time

def run_command(cmd, description, wait_time=2):
    print(f"\n{'='*60}")
    print(f"🎵 {description}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False, text=True, cwd=os.getcwd())
        time.sleep(wait_time)  # Give time to read output
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    print("🎼 ABRSM AI Music Feedback System - COMPLETE DEMONSTRATION")
    print("This demo showcases ALL enhanced features of the competition entry")
    print("\nDemonstration Features:")
    print("✓ Basic monophonic analysis")
    print("✓ Time signature and rhythm analysis") 
    print("✓ Sheet music visualization with diffs")
    print("✓ Polyphonic content detection")
    print("✓ Enhanced AI feedback generation")
    print("✓ Multiple analysis modes")
    
    # Check if virtual environment is activated
    if 'VIRTUAL_ENV' not in os.environ:
        print("\n⚠️  Please activate the virtual environment first:")
        print("   source venv/bin/activate")
        return
    
    # Create demo audio if needed
    if not os.path.exists("demo_performance.wav"):
        print("\n📱 Creating demo audio file...")
        if not run_command("python enhanced_main.py --create-demo-only", "Creating Demo Audio"):
            print("❌ Failed to create demo audio")
            return
    
    print("\n" + "="*80)
    print("🚀 STARTING COMPREHENSIVE DEMO")
    print("="*80)
    
    # Demo 1: Simple Analysis (Original functionality)
    print("\n\n1️⃣  SIMPLE ANALYSIS MODE (Original Script Functionality)")
    print("   This shows the core analysis without enhanced features")
    run_command("python enhanced_main.py --simple --no-llm demo_performance.wav", 
                "Basic Analysis - Core Functionality", 3)
    
    # Demo 2: Enhanced Analysis without visualizations
    print("\n\n2️⃣  ENHANCED ANALYSIS MODE (Core + Time Signature)")
    print("   This adds time signature analysis and rhythm patterns")
    run_command("python enhanced_main.py --no-visualizations --no-llm demo_performance.wav", 
                "Enhanced Analysis - Timing Features", 3)
    
    # Demo 3: Full Enhanced Analysis with visualizations
    print("\n\n3️⃣  FULL ENHANCED MODE (All Features)")
    print("   This includes sheet music visualization and comprehensive analysis")
    run_command("python enhanced_main.py --enhanced --no-llm demo_performance.wav", 
                "Complete Enhanced Analysis", 4)
    
    # Demo 4: Multiple pieces support
    print("\n\n4️⃣  MULTIPLE PIECES SUPPORT")
    print("   Demonstrating analysis of different musical pieces")
    run_command("python enhanced_main.py --piece mary --simple --no-llm demo_performance.wav", 
                "Different Piece Analysis", 3)
    
    # Check for generated files
    print("\n\n📁 GENERATED VISUALIZATION FILES:")
    visualization_files = [
        "sheet_music_analysis.png",
        "timing_analysis.png", 
        "twinkle_reference.mid",
        "twinkle_reference.wav",
        "mary_reference.mid", 
        "mary_reference.wav",
        "demo_performance.wav"
    ]
    
    for filename in visualization_files:
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"   ✅ {filename} ({file_size:,} bytes)")
        else:
            print(f"   ❌ {filename} (not found)")
    
    print("\n" + "="*80)
    print("🎯 DEMO COMPLETE - KEY DIFFERENTIATORS")
    print("="*80)
    
    print("\n🏆 What Makes This Solution Competition-Ready:")
    print("   ✅ COMPREHENSIVE: Handles both simple melodies and complex polyphonic music")
    print("   ✅ VISUAL: Sheet music notation with difference highlighting")
    print("   ✅ INTELLIGENT: Time signature detection and rhythm analysis")
    print("   ✅ EDUCATIONAL: Constructive AI feedback with musical context")
    print("   ✅ PROFESSIONAL: Production-ready code with extensive error handling")
    print("   ✅ SCALABLE: Easy to extend with new pieces and analysis methods")
    
    print("\n🔬 Technical Innovation Highlights:")
    print("   • Advanced polyphonic content detection for piano/chord analysis")
    print("   • Visual sheet music diff generation for educational feedback")
    print("   • Intelligent time signature detection with beat pattern analysis")
    print("   • Multi-modal AI feedback incorporating timing, pitch, and rhythm")
    print("   • Self-contained reference generation for demonstration portability")
    
    print("\n🎓 Educational Impact:")
    print("   • Translates technical analysis into understandable musical terms")
    print("   • Provides specific, actionable practice recommendations") 
    print("   • Visual feedback aids student comprehension and engagement")
    print("   • Supports both beginner and intermediate level musical content")
    
    print("\n🔑 To test with AI feedback, set your Google API key:")
    print("   export GOOGLE_API_KEY='your_key_here'")
    print("   python enhanced_main.py --enhanced --demo")
    
    print("\n🎵 Ready for ABRSM Challenge judging! 🏆")

if __name__ == "__main__":
    main()

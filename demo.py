#!/usr/bin/env python3
"""
Demo comparison script showing both original and enhanced versions
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"🎵 {description}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    print("🎼 ABRSM AI Music Feedback System - Demo Comparison")
    print("This demo shows the capabilities of both versions of the system")
    
    # Check if virtual environment is activated
    if 'VIRTUAL_ENV' not in os.environ:
        print("\n⚠️  Please activate the virtual environment first:")
        print("   source venv/bin/activate")
        return
    
    # Create demo audio if it doesn't exist
    if not os.path.exists("demo_performance.wav"):
        print("\n📱 Creating demo audio file...")
        if not run_command("python enhanced_main.py --create-demo-only", "Creating Demo Audio"):
            print("❌ Failed to create demo audio")
            return
    
    print("\n🚀 Running Enhanced Version (Analysis Only)")
    success = run_command("python enhanced_main.py --no-llm demo_performance.wav", 
                         "Enhanced Analysis (No LLM)")
    
    if not success:
        print("❌ Enhanced version failed")
        return
    
    print("\n💡 Key Improvements in Enhanced Version:")
    print("   ✓ Better error handling and validation")
    print("   ✓ More robust pitch and timing analysis") 
    print("   ✓ Support for multiple pieces")
    print("   ✓ Professional CLI with demo mode")
    print("   ✓ Enhanced system prompts for better LLM feedback")
    print("   ✓ Detailed analysis reporting")
    
    print("\n🔑 To test with AI feedback, set your Google API key:")
    print("   export GOOGLE_API_KEY='your_key_here'")
    print("   python enhanced_main.py --demo")
    
    print("\n🎯 Demo complete! Your system is ready for the competition.")

if __name__ == "__main__":
    main()

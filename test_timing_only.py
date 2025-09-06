#!/usr/bin/env python3
"""
Test just the timing calculation to see if it's fixed
"""

import sys
import json
from enhanced_main import MusicAnalyzer

def test_timing_only():
    print("🕐 Testing Timing Calculation")
    print("=" * 30)
    
    # Test with the demo file
    analyzer = MusicAnalyzer('twinkle')
    
    try:
        # Analyze the demo performance
        f0, times, onsets, enhanced_analysis = analyzer.analyze_with_enhancements(
            'demo_performance.wav',
            generate_visualizations=False,
            detect_polyphony=False,
            analyze_timing=False
        )
        
        if enhanced_analysis:
            note_details = enhanced_analysis["standard_analysis"]["note_details"]
            
            print("📊 Timing Analysis (should be per-note intervals now):")
            for i, detail in enumerate(note_details):
                timing_dev = detail.get('timing_deviation_ms', 'N/A')
                print(f"   Note {i+1}: {timing_dev}ms")
            
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_timing_only()

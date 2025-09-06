#!/usr/bin/env python3
"""
Test GUI analysis pipeline
"""

import os
import sys

# Test the modules can be imported
try:
    from enhanced_main import MusicAnalyzer, PIECES, get_feedback_from_llm
    from sheet_music_visualizer import SheetMusicVisualizer
    from time_signature_analyzer import TimeSignatureAnalyzer
    from polyphonic_analyzer import PolyphonicAnalyzer
    MODULES_AVAILABLE = True
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Could not import modules: {e}")
    sys.exit(1)

# Test analysis pipeline
def test_analysis():
    print("\n=== Testing Analysis Pipeline ===")
    
    # Check if demo file exists
    demo_file = "demo_performance.wav"
    if not os.path.exists(demo_file):
        print(f"✗ Demo file {demo_file} not found")
        return False
        
    print(f"✓ Demo file {demo_file} found")
    
    # Test analyzer creation
    try:
        analyzer = MusicAnalyzer(piece_key='twinkle')
        print("✓ Analyzer created successfully")
    except Exception as e:
        print(f"✗ Failed to create analyzer: {e}")
        return False
    
    # Test analysis
    try:
        print("Running analysis...")
        f0, times, onsets, enhanced_analysis = analyzer.analyze_with_enhancements(
            demo_file,
            generate_visualizations=False,
            detect_polyphony=True,
            analyze_timing=True
        )
        
        if enhanced_analysis is None:
            print("✗ Analysis returned None")
            return False
            
        print("✓ Analysis completed successfully")
        
        # Check structure
        print(f"✓ Analysis type: {type(enhanced_analysis)}")
        print(f"✓ Top-level keys: {list(enhanced_analysis.keys())}")
        
        if 'standard_analysis' in enhanced_analysis:
            sa = enhanced_analysis['standard_analysis']
            print(f"✓ Standard analysis keys: {list(sa.keys())}")
            
            if 'note_details' in sa:
                print(f"✓ Found {len(sa['note_details'])} note details")
            
            if 'overall_assessment' in sa:
                oa = sa['overall_assessment']
                print(f"✓ Overall assessment: {oa}")
                
        if 'enhanced_features' in enhanced_analysis:
            ef = enhanced_analysis['enhanced_features']
            print(f"✓ Enhanced features: {list(ef.keys())}")
            
        return True
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing ABRSM GUI Analysis Pipeline")
    print("=" * 40)
    
    success = test_analysis()
    
    if success:
        print("\n✓ All tests passed! The analysis pipeline is working correctly.")
        print("The GUI should work properly with demo analysis.")
    else:
        print("\n✗ Tests failed! There are issues with the analysis pipeline.")

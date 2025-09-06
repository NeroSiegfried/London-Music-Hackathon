#!/usr/bin/env python3
"""
Test the improved note matching algorithm
"""

import sys
import json
from enhanced_main import MusicAnalyzer, PIECES

def test_note_matching():
    print("🎵 Testing Improved Note Matching Algorithm")
    print("=" * 50)
    
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
            analysis_data = enhanced_analysis["standard_analysis"]
            
            print(f"🎼 Piece: {analysis_data['piece_title']}")
            print(f"📊 Expected notes: {analysis_data['analysis_metadata']['expected_notes']}")
            print(f"🎵 Detected onsets: {analysis_data['analysis_metadata']['detected_onsets']}")
            
            # Check alignment data
            alignment = analysis_data.get("alignment", [])
            performance_notes = analysis_data.get("performance_notes", [])
            reference_notes = analysis_data.get("reference_notes", [])
            
            print(f"\n🔗 Alignment Results:")
            print(f"   Reference notes: {len(reference_notes)}")
            print(f"   Performance notes: {len(performance_notes)}")
            print(f"   Alignment pairs: {len(alignment)}")
            
            print(f"\n📝 Note-by-Note Analysis:")
            for i, (ref_idx, perf_idx) in enumerate(alignment):
                if ref_idx is not None and perf_idx is not None:
                    ref_note = reference_notes[ref_idx]
                    perf_note = performance_notes[perf_idx]
                    print(f"   Match {i+1}: Ref[{ref_idx+1}] {ref_note['note_name']} ↔ Perf[{perf_idx+1}] {perf_note['note_name']}")
                elif ref_idx is None:
                    perf_note = performance_notes[perf_idx]
                    print(f"   Extra {i+1}: Perf[{perf_idx+1}] {perf_note['note_name']} (EXTRA)")
                elif perf_idx is None:
                    ref_note = reference_notes[ref_idx]
                    print(f"   Missing {i+1}: Ref[{ref_idx+1}] {ref_note['note_name']} (MISSED)")
            
            # Overall assessment
            overall = analysis_data.get("overall_assessment", {})
            print(f"\n📈 Overall Assessment:")
            print(f"   Completion Rate: {overall.get('completion_rate', 'N/A')}%")
            print(f"   Missed Notes: {overall.get('missed_notes', 0)}")
            print(f"   Extra Notes: {overall.get('extra_notes', 0)}")
            print(f"   Pitch Accuracy: {overall.get('pitch_accuracy', 'N/A')}%")
            print(f"   Timing Accuracy: {overall.get('timing_accuracy', 'N/A')}%")
            
            # Individual note details
            note_details = analysis_data.get("note_details", [])
            print(f"\n🎯 Individual Note Details:")
            for i, detail in enumerate(note_details):
                note_type = detail.get('note_type', 'unknown')
                expected_pitch = detail.get('expected_pitch', 'N/A')
                actual_pitch = detail.get('actual_pitch', 'N/A')
                timing_dev = detail.get('timing_deviation_ms', 'N/A')
                pitch_dev = detail.get('pitch_deviation_cents', 'N/A')
                
                print(f"   Note {i+1}: {expected_pitch} → {actual_pitch} "
                      f"({note_type}, timing: {timing_dev}ms, pitch: {pitch_dev}¢)")
            
            print(f"\n✅ Analysis completed successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_note_matching()

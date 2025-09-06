#!/usr/bin/env python3
"""
Check the alignment issue
"""

from enhanced_main import MusicAnalyzer

def check_alignment():
    analyzer = MusicAnalyzer('twinkle')
    f0, times, onsets, enhanced_analysis = analyzer.analyze_with_enhancements(
        'demo_performance.wav', generate_visualizations=False, 
        detect_polyphony=False, analyze_timing=False
    )
    
    if enhanced_analysis:
        analysis_data = enhanced_analysis['standard_analysis']
        alignment = analysis_data.get('alignment', [])
        ref_notes = analysis_data.get('reference_notes', [])
        perf_notes = analysis_data.get('performance_notes', [])
        
        print(f"Reference notes: {len(ref_notes)}")
        print(f"Performance notes: {len(perf_notes)}")
        print(f"Alignment pairs: {len(alignment)}")
        
        print("\nAlignment details:")
        used_perf = set()
        used_ref = set()
        
        for i, (ref_idx, perf_idx) in enumerate(alignment):
            if ref_idx is not None and perf_idx is not None:
                if ref_idx in used_ref:
                    print(f"⚠️  REF {ref_idx} used multiple times!")
                if perf_idx in used_perf:
                    print(f"⚠️  PERF {perf_idx} used multiple times!")
                used_ref.add(ref_idx)
                used_perf.add(perf_idx)
                
                ref_note = ref_notes[ref_idx]['note_name']
                perf_note = perf_notes[perf_idx]['note_name']
                print(f"  {i}: Ref[{ref_idx}] {ref_note} ↔ Perf[{perf_idx}] {perf_note}")
            elif ref_idx is None:
                print(f"  {i}: EXTRA Perf[{perf_idx}] {perf_notes[perf_idx]['note_name']}")
            elif perf_idx is None:
                print(f"  {i}: MISSED Ref[{ref_idx}] {ref_notes[ref_idx]['note_name']}")

if __name__ == "__main__":
    check_alignment()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Music Performance Feedback Generator using AI

This is an improved version of the ABRSM challenge proof-of-concept with:
- Better error handling and validation
- Support for multiple pieces/songs
- Enhanced algorithms
- Improved user interface
- Demo mode for easy testing
"""

import os # file handling
import sys # system operations
import json # report generation
import argparse # command-line argument parsing
import numpy as np # creating the graphs
import librosa # audio analysis / note detection
import mido # MIDI file handling MIDI to generate templates (expected vs generated)
import requests
from scipy.io import wavfile # reformatting to wav files
from pathlib import Path # file handling 
import traceback # error handling

# Import our new modules
try:
    from sheet_music_visualizer import create_visual_analysis
    from time_signature_analyzer import TimeSignatureAnalyzer, create_timing_visualization
    from polyphonic_analyzer import PolyphonicAnalyzer
    from audio_digitizer import AudioDigitizer
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    print("⚠️  Enhanced visualization features not available. Install matplotlib for full functionality.")

# --- Configuration ---
PIECES = {
    "twinkle": {
        "title": "Twinkle, Twinkle, Little Star",
        "melody": [
            {'pitch': 60, 'duration': 0.25}, {'pitch': 60, 'duration': 0.25},  # Twinkle twinkle
            {'pitch': 67, 'duration': 0.25}, {'pitch': 67, 'duration': 0.25},  # little star
            {'pitch': 69, 'duration': 0.25}, {'pitch': 69, 'duration': 0.25},  # How I wonder
            {'pitch': 67, 'duration': 0.5},                                    # what you are
            {'pitch': 65, 'duration': 0.25}, {'pitch': 65, 'duration': 0.25},  # Up above the
            {'pitch': 64, 'duration': 0.25}, {'pitch': 64, 'duration': 0.25},  # world so high
            {'pitch': 62, 'duration': 0.25}, {'pitch': 62, 'duration': 0.25},  # Like a diamond
            {'pitch': 60, 'duration': 0.5},                                    # in the sky
        ]
    },
    "mary": {
        "title": "Mary Had a Little Lamb",
        "melody": [
            {'pitch': 64, 'duration': 0.25}, {'pitch': 62, 'duration': 0.25}, {'pitch': 60, 'duration': 0.25}, {'pitch': 62, 'duration': 0.25},  # Mary had a little
            {'pitch': 64, 'duration': 0.25}, {'pitch': 64, 'duration': 0.25}, {'pitch': 64, 'duration': 0.5},                                    # lamb
            {'pitch': 62, 'duration': 0.25}, {'pitch': 62, 'duration': 0.25}, {'pitch': 62, 'duration': 0.5},                                    # little lamb
            {'pitch': 64, 'duration': 0.25}, {'pitch': 67, 'duration': 0.25}, {'pitch': 67, 'duration': 0.5},                                    # little lamb
        ]
    }
}

TEMPO_BPM = 100
SAMPLE_RATE = 22050

class MusicAnalyzer:
    def __init__(self, piece_key="twinkle", tempo=TEMPO_BPM, sample_rate=SAMPLE_RATE):
        self.piece_key = piece_key
        self.piece_info = PIECES[piece_key]
        self.tempo = tempo
        self.sample_rate = sample_rate
        self.reference_prefix = f"{piece_key}_reference"
        
        # Initialize audio digitizer for independent analysis
        if ENHANCED_FEATURES_AVAILABLE:
            self.digitizer = AudioDigitizer(sample_rate=sample_rate)
        
    def create_reference_data(self):
        """Enhanced reference generation with better error handling"""
        midi_path = f"{self.reference_prefix}.mid"
        wav_path = f"{self.reference_prefix}.wav"

        if os.path.exists(midi_path) and os.path.exists(wav_path):
            print(f"✓ Reference files for '{self.piece_info['title']}' already exist.")
            return True

        try:
            print(f"🎵 Generating reference files for '{self.piece_info['title']}'...")
            
            # MIDI Generation
            mid = mido.MidiFile()
            track = mido.MidiTrack()
            mid.tracks.append(track)

            ticks_per_beat = mid.ticks_per_beat
            seconds_per_beat = 60.0 / self.tempo

            for note in self.piece_info['melody']:
                duration_ticks = int(mido.second2tick(note['duration'] * 4 * seconds_per_beat, ticks_per_beat, self.tempo))
                track.append(mido.Message('note_on', note=note['pitch'], velocity=64, time=0))
                track.append(mido.Message('note_off', note=note['pitch'], velocity=64, time=duration_ticks))
            mid.save(midi_path)

            # WAV Generation with improved synthesis
            total_duration_seconds = sum(n['duration'] for n in self.piece_info['melody']) * 4 * seconds_per_beat
            wav_data = np.zeros(int(total_duration_seconds * self.sample_rate))
            current_time = 0.0

            for note in self.piece_info['melody']:
                frequency = librosa.midi_to_hz(note['pitch'])
                duration_samples = int(note['duration'] * 4 * seconds_per_beat * self.sample_rate)
                
                t = np.linspace(0., duration_samples / self.sample_rate, duration_samples, endpoint=False)
                # Add harmonics for richer sound
                note_wave = 0.5 * np.sin(2 * np.pi * frequency * t)
                note_wave += 0.1 * np.sin(2 * np.pi * frequency * 2 * t)  # octave
                note_wave += 0.05 * np.sin(2 * np.pi * frequency * 3 * t)  # fifth

                # Add envelope for natural attack/decay
                envelope = np.exp(-3 * t / (duration_samples / self.sample_rate))
                note_wave *= envelope

                start_sample = int(current_time * self.sample_rate)
                end_sample = start_sample + len(note_wave)
                if end_sample <= len(wav_data):
                    wav_data[start_sample:end_sample] += note_wave

                current_time += note['duration'] * 4 * seconds_per_beat

            # Normalize and save
            if np.max(np.abs(wav_data)) > 0:
                wav_data = wav_data / np.max(np.abs(wav_data)) * 0.9
            wavfile.write(wav_path, self.sample_rate, (wav_data * 32767).astype(np.int16))
            print(f"✓ Reference files '{midi_path}' and '{wav_path}' created.")
            return True
            
        except Exception as e:
            print(f"❌ Error creating reference files: {e}")
            return False

    def create_demo_audio(self):
        """Create demo audio - reference melody with one note MISSING (13 notes instead of 14)"""
        demo_path = "demo_performance.wav"
        
        if os.path.exists(demo_path):
            print(f"✓ Demo audio file already exists: {demo_path}")
            return True
        
        try:
            print("🎵 Creating demo performance audio (missing one note)...")
            
            # Reference melody has 14 notes
            reference_melody = self.piece_info['melody']
            print(f"Reference melody has {len(reference_melody)} notes")
            
            # Create demo with MISSING FINAL NOTE (so demo has 13 notes)
            demo_melody = reference_melody[:-1]  # Remove last note
            print(f"Demo performance will have {len(demo_melody)} notes (missing final note)")
            
            seconds_per_beat = 60.0 / self.tempo
            total_duration = sum(n['duration'] for n in demo_melody) * 4 * seconds_per_beat
            
            wav_data = np.zeros(int(total_duration * self.sample_rate))
            current_time = 0.0
            
            for i, note in enumerate(demo_melody):
                frequency = librosa.midi_to_hz(note['pitch'])
                duration_samples = int(note['duration'] * 4 * seconds_per_beat * self.sample_rate)
                
                t = np.linspace(0., duration_samples / self.sample_rate, duration_samples, endpoint=False)
                # Vary the sound slightly for demo
                note_wave = 0.6 * np.sin(2 * np.pi * frequency * t)
                note_wave += 0.08 * np.sin(2 * np.pi * frequency * 2 * t)
                
                # Add envelope
                envelope = np.exp(-2 * t / (duration_samples / self.sample_rate))
                note_wave *= envelope
                
                start_sample = int(current_time * self.sample_rate)
                end_sample = start_sample + len(note_wave)
                if end_sample <= len(wav_data):
                    wav_data[start_sample:end_sample] += note_wave
                
                current_time += note['duration'] * 4 * seconds_per_beat
            
            # Normalize and save
            if np.max(np.abs(wav_data)) > 0:
                wav_data = wav_data / np.max(np.abs(wav_data)) * 0.8
            wavfile.write(demo_path, self.sample_rate, (wav_data * 32767).astype(np.int16))
            print(f"✓ Demo audio created: {demo_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating demo audio: {e}")
            return False

    def analyze_performance_with_digitizer(self, audio_path):
        """Use new digitizer for independent audio analysis"""
        if not ENHANCED_FEATURES_AVAILABLE or not hasattr(self, 'digitizer'):
            print("❌ Audio digitizer not available")
            return None
        
        try:
            # Use digitizer for independent analysis
            reference_melody = self.piece_info['melody']
            digitized_notes = self.digitizer.digitize_performance(
                audio_path, 
                reference_melody, 
                tempo_bpm=self.tempo
            )
            
            # Convert to the format expected by the rest of the system
            report = {
                "metadata": {
                    "piece": self.piece_info['title'],
                    "total_notes": len(reference_melody),
                    "detected_notes": len([n for n in digitized_notes if n['is_present']]),
                    "missed_notes": len([n for n in digitized_notes if not n['is_present']]),
                    "analysis_method": "Independent Audio Digitizer"
                },
                "note_details": []
            }
            
            # Convert digitized notes to report format
            for note_data in digitized_notes:
                if note_data['is_present']:
                    # Calculate timing and pitch deviations
                    timing_dev_ms = round((note_data['actual_time'] - note_data['expected_time']) * 1000)
                    
                    if note_data['actual_pitch_midi'] is not None:
                        pitch_dev_cents = round((note_data['actual_pitch_midi'] - 
                                                note_data.get('expected_pitch_midi', 
                                                librosa.note_to_midi(note_data['expected_pitch']))) * 100)
                    else:
                        pitch_dev_cents = "MISSED"
                    
                    report["note_details"].append({
                        "note_index": note_data['note_index'] + 1,
                        "expected_pitch": note_data['expected_pitch'],
                        "expected_time": round(note_data['expected_time'], 2),
                        "actual_pitch": note_data.get('actual_pitch', "MISSED"),
                        "actual_time": round(note_data['actual_time'], 2) if note_data['actual_time'] else None,
                        "timing_deviation_ms": timing_dev_ms,
                        "pitch_deviation_cents": pitch_dev_cents,
                        "confidence": note_data.get('confidence', 0.0),
                        "energy_level": note_data.get('energy_level', 0.0)
                    })
                else:
                    # Missing note
                    report["note_details"].append({
                        "note_index": note_data['note_index'] + 1,
                        "expected_pitch": note_data['expected_pitch'],
                        "expected_time": round(note_data['expected_time'], 2),
                        "actual_pitch": "MISSED",
                        "actual_time": None,
                        "timing_deviation_ms": "MISSED",
                        "pitch_deviation_cents": "MISSED",
                        "confidence": 0.0,
                        "energy_level": 0.0
                    })
            
            return report
            
        except Exception as e:
            print(f"❌ Error in digitizer analysis: {e}")
            traceback.print_exc()
            return None

    def analyze_performance_audio(self, audio_path):
        """Enhanced audio analysis with better error handling"""
        print(f"🎧 Analyzing performance file: {audio_path}...")
        
        if not os.path.exists(audio_path):
            print(f"❌ Audio file not found: {audio_path}")
            return None, None, None
            
        try:
            # Load audio with error handling
            y, sr_original = librosa.load(audio_path, sr=self.sample_rate)
            
            if len(y) == 0:
                print("❌ Audio file appears to be empty")
                return None, None, None
                
            # Enhanced pitch extraction
            hop_length = 512
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, 
                fmin=librosa.note_to_hz('A0'), 
                fmax=librosa.note_to_hz('C8'),
                hop_length=hop_length
            )
            
            # Get timestamps for pitch values
            times = librosa.times_like(f0, sr=self.sample_rate, hop_length=hop_length)

            # Enhanced onset detection with improved parameters for first note detection
            onset_frames = librosa.onset.onset_detect(
                y=y, 
                sr=self.sample_rate, 
                units='frames',
                hop_length=hop_length,
                backtrack=True,
                pre_max=10,     # Reduced for better first note detection
                post_max=10,    # Reduced for better first note detection
                pre_avg=50,     # Reduced for better sensitivity
                post_avg=50,    # Reduced for better sensitivity
                delta=0.05,     # Lower threshold for weak first notes
                wait=5          # Reduced wait time
            )
            onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate, hop_length=hop_length)
            
            print(f"✓ Found {len(onset_times)} note onsets")
            print(f"✓ Extracted {np.sum(~np.isnan(f0))} pitch measurements")
            
            return onset_times, f0, times
            
        except Exception as e:
            print(f"❌ Error analyzing audio: {e}")
            traceback.print_exc()
            return None, None, None

    def compare_performances(self, audio_path, use_digitizer=True):
        """Enhanced performance comparison with multiple analysis methods"""
        print(f"\n🎯 Comparing performance against '{self.piece_info['title']}'...")
        
        if use_digitizer and ENHANCED_FEATURES_AVAILABLE:
            # Use new digitizer approach
            return self.analyze_performance_with_digitizer(audio_path)
        
        # Fallback to original method if digitizer not available
        onset_times, f0, times = self.analyze_performance_audio(audio_path)
        
        if onset_times is None:
            return None
            
        reference_melody = self.piece_info['melody']
        report = {
            "metadata": {
                "piece": self.piece_info['title'],
                "total_notes": len(reference_melody),
                "detected_notes": 0,
                "missed_notes": 0,
                "analysis_method": "Traditional Onset Detection"
            },
            "note_details": []
        }
        
        # Calculate expected timings for reference notes
        seconds_per_beat = 60.0 / self.tempo
        expected_times = []
        current_time = 0.0
        
        for note in reference_melody:
            expected_times.append(current_time)
            current_time += note['duration'] * 4 * seconds_per_beat
        
        # Track which onsets have been used
        used_onsets = set()
        missed_notes = 0
        
        # Match each reference note to detected onsets
        for i, (ref_note, ref_onset) in enumerate(zip(reference_melody, expected_times)):
            ref_pitch_midi = ref_note['pitch']
            
            # Find available onsets within a reasonable time window
            time_window = 1.0  # seconds
            available_onsets = []
            available_indices = []
            
            for j, perf_onset in enumerate(onset_times):
                if j not in used_onsets and abs(perf_onset - ref_onset) <= time_window:
                    available_onsets.append(perf_onset)
                    available_indices.append(j)
            
            available_onsets = np.array(available_onsets)
            
            if len(available_onsets) == 0:
                missed_notes += 1
                report["note_details"].append({
                    "note_index": i + 1,
                    "expected_pitch": librosa.midi_to_note(ref_pitch_midi),
                    "expected_time": round(ref_onset, 2),
                    "timing_deviation_ms": "MISSED",
                    "pitch_deviation_cents": "MISSED"
                })
                continue
            else:
                # Normal case - find closest available onset within window
                available_time_diffs = np.abs(available_onsets - ref_onset)
                closest_available_idx = np.argmin(available_time_diffs)
                perf_onset = available_onsets[closest_available_idx]
                
                # Mark this onset as used
                original_idx = available_indices[closest_available_idx]
                used_onsets.add(original_idx)
            
            # Calculate timing deviation
            timing_deviation_ms = round((perf_onset - ref_onset) * 1000)
            
            # Extract pitch at this onset
            onset_frame = int(perf_onset * self.sample_rate / 512)
            
            if onset_frame < len(f0):
                pitch_window_start = max(0, onset_frame - 5)
                pitch_window_end = min(len(f0), onset_frame + 15)
                
                pitch_segment = f0[pitch_window_start:pitch_window_end]
                valid_pitches = pitch_segment[~np.isnan(pitch_segment)]
                
                if len(valid_pitches) > 0:
                    detected_pitch_hz = np.median(valid_pitches)
                    detected_pitch_midi = librosa.hz_to_midi(detected_pitch_hz)
                    detected_pitch_name = librosa.midi_to_note(detected_pitch_midi)
                    pitch_deviation_cents = round((detected_pitch_midi - ref_pitch_midi) * 100)
                else:
                    detected_pitch_name = "UNCLEAR"
                    pitch_deviation_cents = "UNCLEAR"
            else:
                detected_pitch_name = "UNCLEAR"
                pitch_deviation_cents = "UNCLEAR"
            
            report["note_details"].append({
                "note_index": i + 1,
                "expected_pitch": librosa.midi_to_note(ref_pitch_midi),
                "expected_time": round(ref_onset, 2),
                "actual_pitch": detected_pitch_name,
                "actual_time": round(perf_onset, 2),
                "timing_deviation_ms": timing_deviation_ms,
                "pitch_deviation_cents": pitch_deviation_cents
            })
        
        report["metadata"]["detected_notes"] = len(reference_melody) - missed_notes
        report["metadata"]["missed_notes"] = missed_notes
        

        return report

    def analyze_with_enhancements(self, audio_path, generate_visualizations=True, 
                                 detect_polyphony=True, analyze_timing=True):
        """
        Enhanced analysis with sheet music, timing, and polyphonic capabilities
        """
        # Standard monophonic analysis
        f0, times, onsets = self.analyze_performance_audio(audio_path)
        if f0 is None:
            return None, None, None, None
        # Generate standard report
        standard_report = self.compare_performances(audio_path, use_digitizer=False)
        enhanced_analysis = {
            "standard_analysis": standard_report if isinstance(standard_report, dict) else json.loads(standard_report),
            "enhanced_features": {}
        }
        # Time signature analysis
        if analyze_timing and ENHANCED_FEATURES_AVAILABLE:
            print("⏱️  Analyzing time signature and rhythm patterns...")
            timing_analyzer = TimeSignatureAnalyzer()
            timing_analysis = timing_analyzer.analyze_time_signature_impact(
                self.piece_info['melody'], 
                standard_report if isinstance(standard_report, dict) else json.loads(standard_report), 
                self.tempo
            )
            enhanced_analysis["enhanced_features"]["timing_analysis"] = timing_analysis
            if generate_visualizations:
                timing_viz_path = create_timing_visualization(timing_analysis)
                enhanced_analysis["enhanced_features"]["timing_visualization"] = timing_viz_path
        # Check for polyphonic content
        if detect_polyphony and len(onsets) > len(self.piece_info['melody']) * 1.5:
            print("🎹 Potential polyphonic content detected, running advanced analysis...")
            if ENHANCED_FEATURES_AVAILABLE:
                poly_analyzer = PolyphonicAnalyzer()
                poly_result = poly_analyzer.analyze_polyphonic_performance(audio_path)
                if poly_result:
                    enhanced_analysis["enhanced_features"]["polyphonic_analysis"] = poly_result
        # Sheet music visualization
        if generate_visualizations and ENHANCED_FEATURES_AVAILABLE:
            print("🎼 Generating sheet music visualization...")
            try:
                # Determine time signature from analysis or use default
                time_sig = (4, 4)
                if "timing_analysis" in enhanced_analysis["enhanced_features"]:
                    time_sig = enhanced_analysis["enhanced_features"]["timing_analysis"].get("detected_time_signature", (4, 4))
                sheet_music_path = create_visual_analysis(
                    self.piece_info['melody'], 
                    standard_report, 
                    time_sig
                )
                enhanced_analysis["enhanced_features"]["sheet_music_visualization"] = sheet_music_path
            except Exception as e:
                print(f"⚠️  Sheet music visualization failed: {e}")
        return f0, times, onsets, enhanced_analysis

def main():
    """Enhanced main function with better CLI and error handling"""
    parser = argparse.ArgumentParser(description="Enhanced ABRSM Music Performance Analyzer")
    parser.add_argument("--piece", choices=list(PIECES.keys()), default="twinkle",
                        help="Choose which piece to analyze")
    parser.add_argument("--performance", type=str, 
                        help="Path to performance audio file (if not provided, demo mode will be used)")
    parser.add_argument("--generate-refs", action="store_true",
                        help="Generate reference files and exit")
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode with generated performance")
    parser.add_argument("--tempo", type=int, default=TEMPO_BPM,
                        help=f"Tempo in BPM (default: {TEMPO_BPM})")
    parser.add_argument("--no-visualizations", action="store_true",
                        help="Skip generating visualization files")
    parser.add_argument("--output", type=str, default="analysis_report.json",
                        help="Output file for analysis results")
    parser.add_argument("--use-digitizer", action="store_true", default=True,
                        help="Use new audio digitizer for analysis")
    
    args = parser.parse_args()
    
    print("🎼 Enhanced ABRSM Music Performance Analyzer")
    print(f"📝 Analyzing piece: {PIECES[args.piece]['title']}")
    
    # Initialize analyzer
    analyzer = MusicAnalyzer(piece_key=args.piece, tempo=args.tempo)
    
    # Generate reference data
    if not analyzer.create_reference_data():
        print("❌ Failed to create reference data")
        sys.exit(1)
    
    if args.generate_refs:
        print("✓ Reference files generated successfully")
        return
    
    # Handle demo mode
    if args.demo or not args.performance:
        print("\n🎭 Demo Mode: Creating demonstration performance...")
        if not analyzer.create_demo_audio():
            print("❌ Failed to create demo audio")
            sys.exit(1)
        performance_path = "demo_performance.wav"
    else:
        performance_path = args.performance
    
    # Analyze performance
    report = analyzer.compare_performances(performance_path, use_digitizer=args.use_digitizer)
    
    if report is None:
        print("❌ Analysis failed")
        sys.exit(1)
    
    # Display results
    print(f"\n📊 Analysis Results for '{report['metadata']['piece']}':")
    print(f"   Total Notes: {report['metadata']['total_notes']}")
    print(f"   Detected: {report['metadata']['detected_notes']}")
    print(f"   Missed: {report['metadata']['missed_notes']}")
    print(f"   Method: {report['metadata']['analysis_method']}")
    
    # Save detailed report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Detailed report saved to: {args.output}")
    
    # Generate visualizations if available and not disabled
    if ENHANCED_FEATURES_AVAILABLE and not args.no_visualizations:
        try:
            # Sheet music visualization
            vis_path = create_visual_analysis(performance_path, f"{analyzer.reference_prefix}.wav", report)
            if vis_path:
                print(f"✓ Sheet music visualization: {vis_path}")
            
            # Timing analysis
            timing_path = create_timing_visualization(report, f"timing_analysis_{args.piece}.png")
            if timing_path:
                print(f"✓ Timing analysis: {timing_path}")
                
        except Exception as e:
            print(f"⚠️  Visualization generation failed: {e}")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()

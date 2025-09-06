#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Music Performance Feedback Generator using AI

This is an improved version of the ABRSM challenge proof-of-concept with:
- Better error handling and validation
- Support for multiple pieces/songs
- Enhanced            elif len(available_onsets) == 0:
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
                used_onsets.add(original_idx)thms
- Improved user interface
- Demo mode for easy testing
"""

import os
import sys
import json
import argparse
import numpy as np
import librosa
import mido
import requests
from scipy.io import wavfile
from pathlib import Path
import traceback

# Import our new modules
try:
    from sheet_music_visualizer import create_visual_analysis
    from time_signature_analyzer import TimeSignatureAnalyzer, create_timing_visualization
    from polyphonic_analyzer import PolyphonicAnalyzer
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
            {'pitch': 67, 'duration': 0.5},                                  # what you are
            {'pitch': 65, 'duration': 0.25}, {'pitch': 65, 'duration': 0.25},  # Up above the
            {'pitch': 64, 'duration': 0.25}, {'pitch': 64, 'duration': 0.25},  # world so high
            {'pitch': 62, 'duration': 0.25}, {'pitch': 62, 'duration': 0.25},  # Like a diamond
            {'pitch': 60, 'duration': 0.5},                                  # in the sky
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
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C6'),
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
            
            return f0, times, onset_times
            
        except Exception as e:
            print(f"❌ Error analyzing audio: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return None, None, None

    def compare_and_generate_report(self, performance_f0, performance_times, performance_onsets):
        """Enhanced comparison using optimal sequence alignment"""
        print("📊 Comparing performance to reference...")
        
        report = {
            "piece_title": self.piece_info['title'],
            "analysis_metadata": {
                "expected_notes": len(self.piece_info['melody']),
                "detected_onsets": len(performance_onsets),
                "tempo_bpm": self.tempo
            },
            "overall_assessment": {},
            "note_details": []
        }

        # Calculate reference note timings
        seconds_per_beat = 60.0 / self.tempo
        reference_notes = []
        current_time = 0.0
        for i, note in enumerate(self.piece_info['melody']):
            reference_notes.append({
                'index': i,
                'onset': current_time,
                'pitch_midi': note['pitch'],
                'duration': note['duration'] * 4 * seconds_per_beat,
                'note_name': librosa.midi_to_note(note['pitch'])
            })
            current_time += note['duration'] * 4 * seconds_per_beat

        # Extract performance notes with pitch analysis
        performance_notes = []
        for i, onset in enumerate(performance_onsets):
            # Estimate duration until next onset or end
            if i < len(performance_onsets) - 1:
                duration = performance_onsets[i + 1] - onset
            else:
                duration = performance_times[-1] - onset
            
            # Extract pitch for this note
            start_idx = np.argmin(np.abs(performance_times - onset))
            end_idx = np.argmin(np.abs(performance_times - (onset + min(duration * 0.8, 0.5))))
            
            note_f0 = performance_f0[start_idx:end_idx]
            note_f0 = note_f0[~np.isnan(note_f0)]
            
            if len(note_f0) > 0:
                # Use median frequency for stability
                median_f0 = np.median(note_f0)
                pitch_midi = librosa.hz_to_midi(median_f0)
                note_name = librosa.midi_to_note(pitch_midi)
            else:
                pitch_midi = 60  # Default to C4 if no pitch detected
                note_name = "C4"
            
            performance_notes.append({
                'index': i,
                'onset': onset,
                'pitch_midi': pitch_midi,
                'duration': duration,
                'note_name': note_name
            })

        # Optimal sequence alignment using dynamic programming
        alignment = self._optimal_note_alignment(reference_notes, performance_notes)
        
        # Process alignment results
        pitch_deviations = []
        timing_deviations = []
        missed_notes = 0
        extra_notes = 0
        
        for align_pair in alignment:
            ref_idx, perf_idx = align_pair
            
            if ref_idx is None:
                # Extra note in performance
                extra_notes += 1
                perf_note = performance_notes[perf_idx]
                report["note_details"].append({
                    "note_index": f"EXTRA-{perf_idx + 1}",
                    "expected_pitch": "N/A",
                    "actual_pitch": perf_note['note_name'],
                    "expected_time": "N/A",
                    "actual_time": round(perf_note['onset'], 2),
                    "timing_deviation_ms": "EXTRA",
                    "pitch_deviation_cents": "EXTRA",
                    "note_type": "extra"
                })
            elif perf_idx is None:
                # Missing note from reference
                missed_notes += 1
                ref_note = reference_notes[ref_idx]
                report["note_details"].append({
                    "note_index": ref_idx + 1,
                    "expected_pitch": ref_note['note_name'],
                    "actual_pitch": "MISSED",
                    "expected_time": round(ref_note['onset'], 2),
                    "actual_time": "MISSED",
                    "timing_deviation_ms": "MISSED",
                    "pitch_deviation_cents": "MISSED",
                    "note_type": "missed"
                })
            else:
                # Matched notes
                ref_note = reference_notes[ref_idx]
                perf_note = performance_notes[perf_idx]
                
                # Calculate timing deviation relative to interval, not absolute time
                if ref_idx > 0:
                    # Get previous reference note timing
                    prev_ref_note = reference_notes[ref_idx - 1]
                    expected_interval = ref_note['onset'] - prev_ref_note['onset']
                    
                    # Find corresponding previous performance note
                    prev_perf_idx = None
                    for prev_ref, prev_perf in alignment:
                        if prev_ref == ref_idx - 1:
                            prev_perf_idx = prev_perf
                            break
                    
                    if prev_perf_idx is not None and prev_perf_idx < len(performance_notes):
                        prev_perf_note = performance_notes[prev_perf_idx]
                        actual_interval = perf_note['onset'] - prev_perf_note['onset']
                        timing_dev = (actual_interval - expected_interval) * 1000  # in ms
                    else:
                        # Fallback to absolute timing for first note or missing previous
                        timing_dev = (perf_note['onset'] - ref_note['onset']) * 1000
                else:
                    # First note - use absolute timing from start
                    timing_dev = (perf_note['onset'] - ref_note['onset']) * 1000
                
                timing_deviations.append(abs(timing_dev))
                
                # Calculate pitch deviation
                pitch_dev = (perf_note['pitch_midi'] - ref_note['pitch_midi']) * 100  # in cents
                
                # Handle small deviations as correct
                if abs(pitch_dev) < 5:
                    pitch_dev = 0
                
                pitch_deviations.append(abs(pitch_dev))
                
                report["note_details"].append({
                    "note_index": ref_idx + 1,
                    "expected_pitch": ref_note['note_name'],
                    "actual_pitch": perf_note['note_name'],
                    "expected_time": round(ref_note['onset'], 2),
                    "actual_time": round(perf_note['onset'], 2),
                    "timing_deviation_ms": round(timing_dev, 1),
                    "pitch_deviation_cents": round(pitch_dev, 1),
                    "note_type": "matched"
                })
        
        # Calculate overall statistics
        total_notes = len(reference_notes)
        matched_notes = sum(1 for ref_idx, perf_idx in alignment if ref_idx is not None and perf_idx is not None)
        
        report["overall_assessment"]["completion_rate"] = round((matched_notes / total_notes) * 100, 1) if total_notes > 0 else 0
        report["overall_assessment"]["missed_notes"] = missed_notes
        report["overall_assessment"]["extra_notes"] = extra_notes
        
        if pitch_deviations:
            report["overall_assessment"]["avg_pitch_error_cents"] = round(np.mean(pitch_deviations))
            report["overall_assessment"]["pitch_accuracy"] = round(100 - min(100, np.mean(pitch_deviations) / 5), 1)
            
        if timing_deviations:
            report["overall_assessment"]["avg_timing_error_ms"] = round(np.mean(timing_deviations))
            report["overall_assessment"]["timing_accuracy"] = round(100 - min(100, np.mean(timing_deviations) / 20), 1)

        # Store alignment for sheet music visualization
        report["alignment"] = alignment
        report["reference_notes"] = reference_notes
        report["performance_notes"] = performance_notes

        return json.dumps(report, indent=2)

    def _optimal_note_alignment(self, reference_notes, performance_notes):
        """
        Optimal sequence alignment using dynamic programming
        Maximizes alignment score while allowing skips in either sequence
        """
        ref_len = len(reference_notes)
        perf_len = len(performance_notes)
        
        # DP table: dp[i][j] = best score for aligning ref[0:i] with perf[0:j]
        dp = np.full((ref_len + 1, perf_len + 1), -float('inf'))
        dp[0][0] = 0
        
        # Backtrack table to reconstruct alignment
        backtrack = {}
        
        # Fill DP table
        for i in range(ref_len + 1):
            for j in range(perf_len + 1):
                if dp[i][j] == -float('inf'):
                    continue
                
                # Option 1: Skip reference note (missed note)
                if i < ref_len:
                    miss_cost = -2  # Penalty for missing a reference note
                    if dp[i + 1][j] < dp[i][j] + miss_cost:
                        dp[i + 1][j] = dp[i][j] + miss_cost
                        backtrack[(i + 1, j)] = (i, j, 'miss_ref')
                
                # Option 2: Skip performance note (extra note)
                if j < perf_len:
                    extra_cost = -1  # Penalty for extra performance note
                    if dp[i][j + 1] < dp[i][j] + extra_cost:
                        dp[i][j + 1] = dp[i][j] + extra_cost
                        backtrack[(i, j + 1)] = (i, j, 'miss_perf')
                
                # Option 3: Match notes
                if i < ref_len and j < perf_len:
                    match_score = self._calculate_match_score(reference_notes[i], performance_notes[j])
                    if dp[i + 1][j + 1] < dp[i][j] + match_score:
                        dp[i + 1][j + 1] = dp[i][j] + match_score
                        backtrack[(i + 1, j + 1)] = (i, j, 'match')
        
        # Reconstruct alignment
        alignment = []
        i, j = ref_len, perf_len
        
        while i > 0 or j > 0:
            if (i, j) not in backtrack:
                break
                
            prev_i, prev_j, action = backtrack[(i, j)]
            
            if action == 'match':
                alignment.append((prev_i, prev_j))
            elif action == 'miss_ref':
                alignment.append((prev_i, None))  # Reference note missed
            elif action == 'miss_perf':
                alignment.append((None, prev_j))  # Extra performance note
            
            i, j = prev_i, prev_j
        
        alignment.reverse()
        return alignment
    
    def _calculate_match_score(self, ref_note, perf_note):
        """Calculate matching score between reference and performance notes"""
        # Time similarity (closer in time = higher score)
        time_diff = abs(ref_note['onset'] - perf_note['onset'])
        if time_diff < 0.2:  # Very close timing
            time_score = 5
        elif time_diff < 0.5:  # Good timing
            time_score = 3
        elif time_diff < 1.0:  # Acceptable timing
            time_score = 1
        else:  # Poor timing
            time_score = -1
        
        # Pitch similarity (closer in pitch = higher score)
        pitch_diff = abs(ref_note['pitch_midi'] - perf_note['pitch_midi'])
        if pitch_diff < 0.1:  # Same note
            pitch_score = 10
        elif pitch_diff < 0.5:  # Very close
            pitch_score = 8
        elif pitch_diff < 1:  # Close (within semitone)
            pitch_score = 5
        elif pitch_diff < 2:  # Within a tone
            pitch_score = 2
        elif pitch_diff < 6:  # Within tritone
            pitch_score = -1
        else:  # Far apart
            pitch_score = -3
        
        return time_score + pitch_score

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
        standard_report = self.compare_and_generate_report(f0, times, onsets)
        
        enhanced_analysis = {
            "standard_analysis": json.loads(standard_report),
            "enhanced_features": {}
        }
        
        # Time signature analysis
        if analyze_timing and ENHANCED_FEATURES_AVAILABLE:
            print("⏱️  Analyzing time signature and rhythm patterns...")
            timing_analyzer = TimeSignatureAnalyzer()
            timing_analysis = timing_analyzer.analyze_time_signature_impact(
                self.piece_info['melody'], 
                json.loads(standard_report), 
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

def get_feedback_from_llm(report_json, api_key, enhanced_analysis=None):
    """Enhanced LLM feedback with timing and visual analysis context"""
    print("🤖 Generating personalized feedback...")

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

    # Enhanced system prompt with new capabilities
    system_prompt = """You are a warm, encouraging, and professional music teacher working for ABRSM. 
Your role is to provide constructive feedback to music students based on comprehensive technical analysis of their performance.

You now have access to enhanced analysis including:
- Time signature and rhythm pattern analysis
- Visual sheet music comparisons  
- Potential polyphonic content detection
- Detailed timing compensation patterns

Your feedback must follow these guidelines:
1. TONE: Always be positive, encouraging, and supportive. You're helping a student improve.
2. STRUCTURE: Start with genuine praise, provide 1-2 specific areas for improvement, end with encouragement.
3. LANGUAGE: Use simple, musical terms. Avoid technical jargon.
4. SPECIFICITY: Reference specific musical elements like timing, rhythm patterns, and note accuracy.
5. ACTIONABLE ADVICE: Give practical tips for improvement.
6. LENGTH: Keep it concise but meaningful - about 4-5 sentences.

If timing analysis is available, comment on rhythm and tempo consistency.
If polyphonic content is detected, acknowledge the complexity and provide appropriate guidance.
Always encourage continued practice and improvement!"""
    
    # Prepare enhanced context
    context_parts = [f"Standard Performance Analysis:\n{report_json}"]
    
    if enhanced_analysis and "enhanced_features" in enhanced_analysis:
        features = enhanced_analysis["enhanced_features"]
        
        if "timing_analysis" in features:
            timing_info = features["timing_analysis"]
            context_parts.append(f"\nTiming Analysis: {json.dumps(timing_info, indent=2)}")
        
        if "polyphonic_analysis" in features:
            poly_info = features["polyphonic_analysis"] 
            context_parts.append(f"\nPolyphonic Content Detected - Complexity: {poly_info.get('complexity_score', {})}")
    
    user_query = "Please provide encouraging, specific feedback based on this comprehensive analysis:\n\n" + "\n".join(context_parts)

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 600
        }
    }

    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, 
                               json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        candidate = result.get("candidates", [{}])[0]
        content = candidate.get("content", {}).get("parts", [{}])[0]
        feedback = content.get("text", "Sorry, I couldn't generate feedback at this time.")
        
        return feedback
        
    except requests.exceptions.RequestException as e:
        return f"❌ Error connecting to AI service: {e}"
    except Exception as e:
        return f"❌ Error generating feedback: {e}"

def create_demo_audio():
    """Create a demo audio file for testing"""
    demo_path = "demo_performance.wav"
    if os.path.exists(demo_path):
        return demo_path
        
    print("🎤 Creating demo audio file...")
    
    # Create a slightly imperfect version of Twinkle Twinkle
    melody = PIECES["twinkle"]["melody"]
    tempo = 95  # Slightly slower
    sr = 22050
    seconds_per_beat = 60.0 / tempo
    
    total_duration = sum(n['duration'] for n in melody) * 4 * seconds_per_beat
    wav_data = np.zeros(int(total_duration * sr))
    current_time = 0.0

    for i, note in enumerate(melody):
        # Add some pitch variation (slightly flat on some notes)
        pitch_variation = -0.3 if i in [2, 5, 8] else 0.1 if i in [1, 6] else 0
        frequency = librosa.midi_to_hz(note['pitch'] + pitch_variation)
        
        # Add some timing variation
        timing_variation = 0.05 if i in [3, 7] else -0.03 if i in [1, 4] else 0
        duration_samples = int((note['duration'] * 4 * seconds_per_beat + timing_variation) * sr)
        
        t = np.linspace(0., duration_samples / sr, duration_samples, endpoint=False)
        note_wave = 0.4 * np.sin(2 * np.pi * frequency * t)
        
        # Add some noise to simulate real recording
        note_wave += np.random.normal(0, 0.02, len(note_wave))
        
        # Envelope
        envelope = np.exp(-2 * t / (duration_samples / sr))
        note_wave *= envelope

        start_sample = int(current_time * sr)
        end_sample = start_sample + len(note_wave)
        if end_sample <= len(wav_data):
            wav_data[start_sample:end_sample] += note_wave

        current_time += note['duration'] * 4 * seconds_per_beat + timing_variation

    # Normalize and save
    if np.max(np.abs(wav_data)) > 0:
        wav_data = wav_data / np.max(np.abs(wav_data)) * 0.8
    wavfile.write(demo_path, sr, (wav_data * 32767).astype(np.int16))
    print(f"✓ Demo audio created: {demo_path}")
    return demo_path

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced AI Music Performance Feedback Generator (ABRSM Challenge)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_main.py my_performance.wav
  python enhanced_main.py --piece mary my_performance.wav  
  python enhanced_main.py --demo
  python enhanced_main.py --create-demo-only
        """
    )
    
    parser.add_argument("audio_file", nargs='?', help="Path to your performance audio file")
    parser.add_argument("--api-key", help="Google AI Studio API key (or set GOOGLE_API_KEY env var)")
    parser.add_argument("--piece", choices=list(PIECES.keys()), default="twinkle", 
                       help="Which piece to analyze against")
    parser.add_argument("--demo", action="store_true", 
                       help="Run with automatically generated demo audio")
    parser.add_argument("--create-demo-only", action="store_true",
                       help="Just create demo audio file and exit")
    parser.add_argument("--no-llm", action="store_true",
                       help="Skip LLM feedback generation (just show analysis)")
    parser.add_argument("--no-visualizations", action="store_true",
                       help="Skip generating sheet music and timing visualizations")
    parser.add_argument("--enhanced", action="store_true",
                       help="Enable all enhanced features (timing analysis, visualizations, polyphonic detection)")
    parser.add_argument("--simple", action="store_true",
                       help="Simple analysis only (equivalent to original script)")
    
    args = parser.parse_args()

    # Handle demo creation
    if args.create_demo_only:
        create_demo_audio()
        return

    # Handle demo mode
    if args.demo:
        audio_file = create_demo_audio()
    elif args.audio_file:
        audio_file = args.audio_file
    else:
        parser.print_help()
        return

    # API key handling
    if not args.no_llm:
        api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️  No API key provided. Running in analysis-only mode.")
            print("   Set GOOGLE_API_KEY environment variable or use --api-key for full feedback.")
            args.no_llm = True

    print(f"\n🎼 ABRSM AI Music Feedback System")
    print(f"   Analyzing: {PIECES[args.piece]['title']}")
    print(f"   Audio: {audio_file}")
    if args.enhanced or (not args.simple and not args.no_visualizations):
        print(f"   Mode: Enhanced Analysis with Visualizations")
    else:
        print(f"   Mode: Standard Analysis")
    print("=" * 50)

    try:
        # Initialize analyzer
        analyzer = MusicAnalyzer(piece_key=args.piece)
        
        # Create reference
        if not analyzer.create_reference_data():
            print("❌ Failed to create reference data")
            return

        # Choose analysis type
        if args.simple:
            # Original simple analysis
            f0, times, onsets = analyzer.analyze_performance_audio(audio_file)
            if f0 is None:
                return
            report = analyzer.compare_and_generate_report(f0, times, onsets)
            enhanced_analysis = None
            
            print("\n📋 ANALYSIS REPORT")
            print("=" * 50)
            print(report)
            
        else:
            # Enhanced analysis
            generate_viz = not args.no_visualizations and ENHANCED_FEATURES_AVAILABLE
            f0, times, onsets, enhanced_analysis = analyzer.analyze_with_enhancements(
                audio_file, 
                generate_visualizations=generate_viz,
                detect_polyphony=True,
                analyze_timing=True
            )
            
            if enhanced_analysis is None:
                return
                
            print("\n📋 STANDARD ANALYSIS REPORT")
            print("=" * 50)
            print(json.dumps(enhanced_analysis["standard_analysis"], indent=2))
            
            # Display enhanced features
            if enhanced_analysis.get("enhanced_features"):
                print("\n🔬 ENHANCED ANALYSIS FEATURES")
                print("=" * 50)
                
                features = enhanced_analysis["enhanced_features"]
                
                if "timing_analysis" in features:
                    timing = features["timing_analysis"]
                    print(f"\n⏱️  Time Signature Analysis:")
                    print(f"   • Detected: {timing.get('detected_time_signature', 'Unknown')}")
                    print(f"   • Tempo: {timing.get('tempo_bpm', 'Unknown')} BPM")
                    print(f"   • Beat Consistency: {timing.get('beat_analysis', {}).get('beat_consistency', 'N/A')}%")
                    print(f"   • Difficulty: {timing.get('signature_difficulty', 'Unknown')}")
                    
                    if "timing_visualization" in features:
                        print(f"   • Timing chart saved: {features['timing_visualization']}")
                
                if "polyphonic_analysis" in features:
                    poly = features["polyphonic_analysis"]
                    complexity = poly.get("complexity_score", {})
                    print(f"\n🎹 Polyphonic Content Detected:")
                    print(f"   • Complexity: {complexity.get('score', 'Unknown')}/100")
                    print(f"   • Difficulty: {complexity.get('difficulty', 'Unknown')}")
                    print(f"   • Avg Simultaneous Notes: {complexity.get('avg_simultaneous_notes', 'Unknown')}")
                    print(f"   • Chord Changes: {complexity.get('chord_changes', 'Unknown')}")
                
                if "sheet_music_visualization" in features:
                    print(f"\n🎼 Sheet Music Analysis:")
                    print(f"   • Visual comparison saved: {features['sheet_music_visualization']}")

        # Generate feedback if API key available
        if not args.no_llm:
            if args.simple:
                feedback = get_feedback_from_llm(report, api_key)
            else:
                feedback = get_feedback_from_llm(
                    json.dumps(enhanced_analysis["standard_analysis"], indent=2), 
                    api_key, 
                    enhanced_analysis
                )
            print(f"\n🎯 YOUR PERSONALIZED FEEDBACK")
            print("=" * 50)
            print(feedback)
        
        print(f"\n✨ Analysis complete! Keep practicing! 🎵")
        
        # Summary of generated files
        if not args.simple and not args.no_visualizations and ENHANCED_FEATURES_AVAILABLE:
            print(f"\n📁 Generated Files:")
            if enhanced_analysis and enhanced_analysis.get("enhanced_features"):
                for key, path in enhanced_analysis["enhanced_features"].items():
                    if key.endswith("_visualization") and isinstance(path, str):
                        print(f"   • {path}")
        
    except KeyboardInterrupt:
        print("\n\n👋 Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()

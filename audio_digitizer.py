#!/usr/bin/env python3
"""
Audio Digitization Module

This module properly digitizes audio input by:
1. Segmenting audio based on expected note timings
2. Analyzing each segment for actual note content
3. Determining presence/absence and pitch for each expected note
"""

import numpy as np
import librosa
import scipy.signal
from typing import List, Dict, Tuple, Optional

class AudioDigitizer:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def digitize_performance(self, audio_path: str, reference_melody: List[Dict], tempo_bpm: float = 100) -> List[Dict]:
        """
        Digitize audio performance by independently analyzing audio content
        
        Args:
            audio_path: Path to audio file
            reference_melody: List of reference notes (used only for comparison, not segmentation)
            tempo_bpm: Tempo in beats per minute (for timing reference)
            
        Returns:
            List of digitized notes with presence/absence and pitch info
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Step 1: Detect pitch changes independently from audio
        detected_notes = self._detect_notes_from_audio(y, sr)
        
        print(f"Audio analysis found {len(detected_notes)} notes")
        
        # Step 2: Match detected notes to reference melody
        aligned_results = self._align_detected_to_reference(detected_notes, reference_melody)
        
        return aligned_results
    
    def _detect_notes_from_audio(self, y: np.ndarray, sr: int) -> List[Dict]:
        """
        Detect notes from audio by finding pitch changes and energy levels
        """
        hop_length = 512
        
        # Extract pitch using PYIN
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'),
            sr=sr,
            hop_length=hop_length
        )
        
        # Get time stamps
        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        
        # Calculate energy envelope
        rms_energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Find onset points (pitch and energy changes)
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, units='frames', hop_length=hop_length,
            backtrack=True, pre_max=2, post_max=2, pre_avg=2, post_avg=3,
            delta=0.05, wait=5
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
        
        # Add onset at time 0 if not detected (common issue with first note)
        if len(onset_times) == 0 or onset_times[0] > 0.1:
            onset_times = np.concatenate([[0.0], onset_times])
            print(f"Added onset at time 0.0 for first note")
        
        print(f"Raw onset detection found {len(onset_times)} potential note starts")
        
        # Detect note segments based on pitch stability and energy
        detected_notes = []
        
        # Energy threshold for note detection
        energy_threshold = np.max(rms_energy) * 0.01
        
        for i, onset_time in enumerate(onset_times):
            # Determine note end time
            if i < len(onset_times) - 1:
                end_time = onset_times[i + 1]
            else:
                end_time = len(y) / sr
            
            # Extract segment
            start_sample = int(onset_time * sr)
            end_sample = int(end_time * sr)
            segment = y[start_sample:end_sample]
            
            if len(segment) < hop_length:
                continue
                
            # Calculate segment energy
            segment_energy = np.sqrt(np.mean(segment ** 2))
            
            # Check if this is a real note (has sufficient energy)
            if segment_energy < energy_threshold:
                continue
                
            # Extract pitch from this segment
            start_frame = int(onset_time * sr / hop_length)
            end_frame = int(end_time * sr / hop_length)
            
            if start_frame < len(f0) and end_frame <= len(f0):
                segment_f0 = f0[start_frame:end_frame]
                segment_voiced = voiced_flag[start_frame:end_frame]
                
                # Get valid pitches
                valid_f0 = segment_f0[segment_voiced & ~np.isnan(segment_f0)]
                
                if len(valid_f0) > 0:
                    # Use median frequency for stability
                    median_f0 = np.median(valid_f0)
                    pitch_midi = librosa.hz_to_midi(median_f0)
                    note_name = librosa.midi_to_note(pitch_midi)
                    
                    # Calculate pitch stability
                    pitch_std = np.std(librosa.hz_to_midi(valid_f0)) if len(valid_f0) > 1 else 0
                    confidence = max(0, min(1, 1.0 - pitch_std / 2))
                    
                    detected_notes.append({
                        'onset': float(onset_time),
                        'duration': float(end_time - onset_time),
                        'pitch_midi': float(pitch_midi),
                        'note_name': note_name,
                        'energy': float(segment_energy),
                        'confidence': float(confidence),
                        'pitch_stability': float(pitch_std)
                    })
        
        print(f"After filtering: {len(detected_notes)} stable notes detected")
        return detected_notes
    
    def _align_detected_to_reference(self, detected_notes: List[Dict], reference_melody: List[Dict]) -> List[Dict]:
        """
        Align detected notes to reference melody to identify which reference notes are present/missing
        """
        aligned_results = []
        
        # For each reference note, find the best matching detected note
        for ref_idx, ref_note in enumerate(reference_melody):
            ref_pitch = ref_note['pitch']
            ref_pitch_name = librosa.midi_to_note(ref_pitch)
            
            # Calculate expected timing for this reference note
            expected_time = ref_note.get('onset', ref_idx * 0.6)  # Fallback timing
            
            # Find closest detected note in time and pitch
            best_match = None
            best_score = float('inf')
            
            for det_note in detected_notes:
                # Calculate time difference
                time_diff = abs(det_note['onset'] - expected_time)
                
                # Calculate pitch difference  
                pitch_diff = abs(det_note['pitch_midi'] - ref_pitch)
                
                # Combined score (prioritize time proximity)
                score = time_diff + pitch_diff * 0.1
                
                # Only consider reasonable matches (within 2 seconds and 12 semitones)
                if time_diff < 2.0 and pitch_diff < 12 and score < best_score:
                    best_score = score
                    best_match = det_note
            
            # Create result entry
            if best_match:
                aligned_results.append({
                    'note_index': ref_idx,
                    'expected_pitch': ref_pitch_name,
                    'expected_time': float(expected_time),
                    'expected_duration': float(ref_note.get('duration', 0.6)),
                    'is_present': True,
                    'actual_pitch': best_match['note_name'],
                    'actual_pitch_midi': float(best_match['pitch_midi']),
                    'actual_time': float(best_match['onset']),
                    'confidence': float(best_match['confidence']),
                    'energy_level': float(best_match['energy']),
                    'pitch_stability': float(best_match['pitch_stability']),
                    'note_type': 'matched'
                })
                
                # Remove matched note so it can't be matched again
                detected_notes.remove(best_match)
            else:
                aligned_results.append({
                    'note_index': ref_idx,
                    'expected_pitch': ref_pitch_name,
                    'expected_time': float(expected_time),
                    'expected_duration': float(ref_note.get('duration', 0.6)),
                    'is_present': False,
                    'actual_pitch': None,
                    'actual_time': None,
                    'confidence': 0.0,
                    'energy_level': 0.0,
                    'pitch_stability': 0.0,
                    'note_type': 'missed'
                })
        
        return aligned_results
    
    def compare_to_reference(self, digitized_notes: List[Dict]) -> Dict:
        """
        Compare digitized notes to reference and generate analysis report
        """
        report = {
            'piece_title': 'Performance Analysis',
            'analysis_metadata': {
                'total_notes': len(digitized_notes),
                'analysis_method': 'digitization'
            },
            'overall_assessment': {
                'completion_rate': 0,
                'missed_notes': 0,
                'pitch_accuracy': 0,
                'timing_accuracy': 0
            },
            'note_details': [],
            'alignment': [],
            'reference_notes': [],
            'performance_notes': []
        }
        
        matched_notes = 0
        missed_notes = 0
        pitch_deviations = []
        timing_deviations = []
        
        for note in digitized_notes:
            note_index = note['note_index']
            expected_pitch = note['expected_pitch']
            
            if note['is_present']:
                matched_notes += 1
                actual_pitch = note['actual_pitch']
                actual_time = note['actual_time']
                
                # Calculate pitch deviation
                expected_midi = librosa.note_to_hz(expected_pitch)
                actual_midi = note.get('actual_pitch_midi', librosa.note_to_hz(actual_pitch))
                pitch_dev = (actual_midi - librosa.note_to_hz(expected_pitch)) / librosa.note_to_hz(expected_pitch) * 1200  # in cents
                
                # Calculate timing deviation
                timing_dev = (actual_time - note['expected_time']) * 1000  # in ms
                
                pitch_deviations.append(abs(pitch_dev))
                timing_deviations.append(abs(timing_dev))
                
                # Add to note details
                report['note_details'].append({
                    'note_index': note_index + 1,
                    'expected_pitch': expected_pitch,
                    'actual_pitch': actual_pitch,
                    'expected_time': float(note['expected_time']),
                    'actual_time': float(actual_time),
                    'timing_deviation_ms': round(float(timing_dev), 1),
                    'pitch_deviation_cents': round(float(pitch_dev), 1),
                    'note_type': 'matched',
                    'confidence': float(note['confidence']),
                    'energy_level': float(note['energy_level'])
                })
                
                # Add to alignment (for compatibility)
                report['alignment'].append((note_index, note_index))
                
                # Add to performance notes
                report['performance_notes'].append({
                    'note_name': actual_pitch,
                    'pitch_midi': float(actual_midi),
                    'onset': float(actual_time),
                    'confidence': float(note['confidence'])
                })
                
            else:
                missed_notes += 1
                report['note_details'].append({
                    'note_index': note_index + 1,
                    'expected_pitch': expected_pitch,
                    'actual_pitch': 'MISSED',
                    'expected_time': float(note['expected_time']),
                    'actual_time': 'MISSED',
                    'timing_deviation_ms': 'MISSED',
                    'pitch_deviation_cents': 'MISSED',
                    'note_type': 'missed',
                    'confidence': 0.0,
                    'energy_level': float(note['energy_level'])
                })
                
                # Add to alignment
                report['alignment'].append((note_index, None))
            
            # Add to reference notes (for compatibility)
            report['reference_notes'].append({
                'note_name': expected_pitch,
                'pitch_midi': float(librosa.note_to_hz(expected_pitch)),
                'onset': float(note['expected_time']),
                'duration': float(note['expected_duration'])
            })
        
        # Calculate overall statistics
        total_notes = len(digitized_notes)
        report['overall_assessment']['completion_rate'] = round((matched_notes / total_notes) * 100, 1) if total_notes > 0 else 0
        report['overall_assessment']['missed_notes'] = missed_notes
        
        if pitch_deviations:
            report['overall_assessment']['pitch_accuracy'] = round(100 - min(100, np.mean(pitch_deviations) / 5), 1)
        
        if timing_deviations:
            report['overall_assessment']['timing_accuracy'] = round(100 - min(100, np.mean(timing_deviations) / 20), 1)
        
        return report

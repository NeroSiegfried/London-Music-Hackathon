#!/usr/bin/env python3
"""
Polyphonic Music Analysis Module

This module extends the ABRSM system to handle polyphonic music
(multiple simultaneous notes) such as piano performances with chords,
bass lines, and melody lines.
"""

import numpy as np
import librosa
from scipy.signal import find_peaks
from collections import defaultdict
import json

class PolyphonicAnalyzer:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.min_note_duration = 0.1  # Minimum note duration in seconds
        self.max_simultaneous_notes = 6  # Maximum expected simultaneous notes
        
    def analyze_polyphonic_performance(self, audio_path, reference_piece=None):
        """
        Analyze a polyphonic audio performance
        
        Args:
            audio_path: Path to audio file
            reference_piece: Optional reference for comparison
            
        Returns:
            Dictionary with polyphonic analysis results
        """
        
        print(f"🎹 Analyzing polyphonic performance: {audio_path}")
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract multiple pitch tracks
            pitch_tracks = self._extract_multiple_pitches(y, sr)
            
            # Detect chord progressions
            chord_progression = self._detect_chord_progression(pitch_tracks)
            
            # Separate melody and harmony
            melody_line, harmony_lines = self._separate_melody_harmony(pitch_tracks)
            
            # Analyze voice leading
            voice_leading = self._analyze_voice_leading(harmony_lines)
            
            # Onset detection for polyphonic content
            onset_analysis = self._polyphonic_onset_detection(y, sr)
            
            return {
                "analysis_type": "polyphonic",
                "pitch_tracks": pitch_tracks,
                "chord_progression": chord_progression,
                "melody_line": melody_line,
                "harmony_lines": harmony_lines,
                "voice_leading": voice_leading,
                "onset_analysis": onset_analysis,
                "complexity_score": self._calculate_complexity_score(pitch_tracks, chord_progression)
            }
            
        except Exception as e:
            print(f"❌ Error in polyphonic analysis: {e}")
            return None

    def _extract_multiple_pitches(self, y, sr):
        """
        Extract multiple simultaneous pitch tracks using harmonic source separation
        """
        
        # Use harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Extract fundamental frequency and harmonics
        pitches = []
        magnitudes = []
        
        # Use multiple methods for robust pitch detection
        
        # Method 1: STFT-based multi-pitch detection
        stft = librosa.stft(y_harmonic, hop_length=512)
        magnitude = np.abs(stft)
        
        # Find peaks in frequency domain for each time frame
        freq_bins = librosa.fft_frequencies(sr=sr)
        times = librosa.frames_to_time(range(stft.shape[1]), sr=sr, hop_length=512)
        
        pitch_tracks = []
        
        for t_idx in range(stft.shape[1]):
            frame_magnitude = magnitude[:, t_idx]
            
            # Find peaks in this frame
            peaks, properties = find_peaks(frame_magnitude, 
                                         height=np.max(frame_magnitude) * 0.1,
                                         distance=20)  # Minimum distance between peaks
            
            # Convert peak indices to frequencies
            frame_freqs = freq_bins[peaks]
            frame_mags = frame_magnitude[peaks]
            
            # Filter to musical range and sort by magnitude
            musical_range = (80, 2000)  # Typical piano range in Hz
            valid_indices = (frame_freqs >= musical_range[0]) & (frame_freqs <= musical_range[1])
            
            if np.any(valid_indices):
                valid_freqs = frame_freqs[valid_indices]
                valid_mags = frame_mags[valid_indices]
                
                # Sort by magnitude and take top N
                sorted_indices = np.argsort(valid_mags)[::-1]
                top_freqs = valid_freqs[sorted_indices[:self.max_simultaneous_notes]]
                top_mags = valid_mags[sorted_indices[:self.max_simultaneous_notes]]
                
                pitch_tracks.append({
                    'time': times[t_idx],
                    'frequencies': top_freqs.tolist(),
                    'magnitudes': top_mags.tolist(),
                    'pitches_midi': [librosa.hz_to_midi(f) for f in top_freqs if f > 0]
                })
        
        # Method 2: Add traditional single-pitch detection for comparison
        f0, voiced_flag, voiced_probs = librosa.pyin(y_harmonic, 
                                                   fmin=librosa.note_to_hz('C2'), 
                                                   fmax=librosa.note_to_hz('C7'))
        
        # Combine results
        single_pitch_track = {
            'method': 'pyin_single_pitch',
            'times': librosa.times_like(f0, sr=sr),
            'frequencies': f0,
            'voiced_probabilities': voiced_probs
        }
        
        return {
            'multi_pitch_frames': pitch_tracks,
            'single_pitch_reference': single_pitch_track,
            'analysis_params': {
                'hop_length': 512,
                'max_simultaneous_notes': self.max_simultaneous_notes,
                'frequency_range': musical_range
            }
        }

    def _detect_chord_progression(self, pitch_tracks):
        """
        Detect chord progressions from pitch tracks
        """
        
        if not pitch_tracks.get('multi_pitch_frames'):
            return {"chords": [], "progression": "Unable to detect"}
        
        chords = []
        frames = pitch_tracks['multi_pitch_frames']
        
        # Group frames into chord segments
        chord_segments = []
        current_chord_notes = set()
        segment_start_time = 0
        
        for i, frame in enumerate(frames):
            frame_notes = set()
            
            # Convert frequencies to note names
            for freq in frame.get('frequencies', []):
                if freq > 0:
                    try:
                        midi_note = librosa.hz_to_midi(freq)
                        note_name = librosa.midi_to_note(midi_note, unicode=False)
                        # Normalize to just note name (remove octave for chord detection)
                        note_base = note_name[:-1] if note_name[-1].isdigit() else note_name
                        frame_notes.add(note_base)
                    except:
                        continue
            
            # Check if this represents a new chord
            if len(frame_notes) >= 2 and frame_notes != current_chord_notes:
                # Save previous chord if it exists
                if current_chord_notes and i > 0:
                    chord_segments.append({
                        'notes': list(current_chord_notes),
                        'start_time': segment_start_time,
                        'end_time': frame['time'],
                        'duration': frame['time'] - segment_start_time
                    })
                
                current_chord_notes = frame_notes
                segment_start_time = frame['time']
        
        # Add final chord
        if current_chord_notes and frames:
            chord_segments.append({
                'notes': list(current_chord_notes),
                'start_time': segment_start_time,
                'end_time': frames[-1]['time'],
                'duration': frames[-1]['time'] - segment_start_time
            })
        
        # Identify chord types
        identified_chords = []
        for segment in chord_segments:
            chord_name = self._identify_chord_type(segment['notes'])
            identified_chords.append({
                'chord': chord_name,
                'notes': segment['notes'],
                'start_time': round(segment['start_time'], 2),
                'duration': round(segment['duration'], 2)
            })
        
        return {
            'chords': identified_chords,
            'progression': ' - '.join([chord['chord'] for chord in identified_chords]),
            'total_chords': len(identified_chords)
        }

    def _identify_chord_type(self, notes):
        """
        Identify chord type from a set of note names
        """
        if len(notes) < 2:
            return "Single Note"
        elif len(notes) == 2:
            return f"{sorted(notes)[0]} interval"
        
        # Simple chord identification for common triads
        notes_sorted = sorted(notes)
        
        # Common chord patterns (simplified)
        chord_patterns = {
            frozenset(['C', 'E', 'G']): 'C Major',
            frozenset(['C', 'E♭', 'G']): 'C Minor',
            frozenset(['F', 'A', 'C']): 'F Major',
            frozenset(['G', 'B', 'D']): 'G Major',
            frozenset(['A', 'C', 'E']): 'A Minor',
            frozenset(['D', 'F♯', 'A']): 'D Major',
            frozenset(['E', 'G♯', 'B']): 'E Major',
        }
        
        notes_set = frozenset(notes)
        chord_name = chord_patterns.get(notes_set)
        
        if chord_name:
            return chord_name
        else:
            # Generic description
            root = sorted(notes)[0]
            if len(notes) == 3:
                return f"{root} triad"
            else:
                return f"{root} chord ({len(notes)} notes)"

    def _separate_melody_harmony(self, pitch_tracks):
        """
        Attempt to separate melody line from harmony
        """
        
        frames = pitch_tracks.get('multi_pitch_frames', [])
        if not frames:
            return None, []
        
        melody_line = []
        harmony_lines = [[] for _ in range(self.max_simultaneous_notes - 1)]
        
        for frame in frames:
            frequencies = frame.get('frequencies', [])
            magnitudes = frame.get('magnitudes', [])
            
            if not frequencies:
                continue
            
            # Sort by magnitude (loudest is likely melody)
            sorted_indices = np.argsort(magnitudes)[::-1]
            
            # Melody is typically the highest or loudest note
            if len(frequencies) > 0:
                melody_freq = frequencies[sorted_indices[0]]  # Loudest note
                melody_line.append({
                    'time': frame['time'],
                    'frequency': melody_freq,
                    'magnitude': magnitudes[sorted_indices[0]],
                    'midi_note': librosa.hz_to_midi(melody_freq) if melody_freq > 0 else None
                })
                
                # Remaining notes are harmony
                for i, freq_idx in enumerate(sorted_indices[1:]):
                    if i < len(harmony_lines) and freq_idx < len(frequencies):
                        harmony_lines[i].append({
                            'time': frame['time'],
                            'frequency': frequencies[freq_idx],
                            'magnitude': magnitudes[freq_idx]
                        })
        
        return melody_line, harmony_lines

    def _analyze_voice_leading(self, harmony_lines):
        """
        Analyze voice leading in harmonic parts
        """
        
        voice_analysis = []
        
        for voice_idx, voice in enumerate(harmony_lines):
            if len(voice) < 2:
                continue
            
            intervals = []
            smoothness_score = 0
            
            for i in range(1, len(voice)):
                if voice[i-1]['frequency'] > 0 and voice[i]['frequency'] > 0:
                    # Calculate interval in semitones
                    prev_midi = librosa.hz_to_midi(voice[i-1]['frequency'])
                    curr_midi = librosa.hz_to_midi(voice[i]['frequency'])
                    interval = abs(curr_midi - prev_midi)
                    intervals.append(interval)
                    
                    # Smooth voice leading prefers small intervals
                    if interval <= 2:  # Step or same note
                        smoothness_score += 2
                    elif interval <= 4:  # Small leap
                        smoothness_score += 1
                    # Large leaps get 0 points
            
            if intervals:
                avg_interval = np.mean(intervals)
                smoothness_percentage = (smoothness_score / len(intervals)) * 50  # Scale to percentage
                
                voice_analysis.append({
                    'voice_number': voice_idx + 1,
                    'average_interval': round(avg_interval, 2),
                    'smoothness_score': round(smoothness_percentage, 1),
                    'total_notes': len(voice),
                    'analysis': 'smooth' if smoothness_percentage > 70 else 'moderate' if smoothness_percentage > 40 else 'jumpy'
                })
        
        return voice_analysis

    def _polyphonic_onset_detection(self, y, sr):
        """
        Enhanced onset detection for polyphonic content
        """
        
        # Use multiple onset detection methods
        onset_methods = ['energy', 'spectral_centroid', 'spectral_rolloff']
        all_onsets = {}
        
        for method in onset_methods:
            if method == 'energy':
                onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
            elif method == 'spectral_centroid':
                onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', 
                                                  feature=librosa.feature.spectral_centroid)
            elif method == 'spectral_rolloff':
                onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time',
                                                  feature=librosa.feature.spectral_rolloff)
            
            all_onsets[method] = onsets.tolist()
        
        # Combine onset detections
        combined_onsets = set()
        for onsets in all_onsets.values():
            combined_onsets.update(onsets)
        
        return {
            'individual_methods': all_onsets,
            'combined_onsets': sorted(list(combined_onsets)),
            'total_onsets': len(combined_onsets)
        }

    def _calculate_complexity_score(self, pitch_tracks, chord_progression):
        """
        Calculate a complexity score for the polyphonic piece
        """
        
        frames = pitch_tracks.get('multi_pitch_frames', [])
        chords = chord_progression.get('chords', [])
        
        # Factors contributing to complexity
        avg_simultaneous_notes = 0
        if frames:
            note_counts = [len(frame.get('frequencies', [])) for frame in frames]
            avg_simultaneous_notes = np.mean(note_counts)
        
        chord_changes = len(chords)
        max_notes = max([len(frame.get('frequencies', [])) for frame in frames]) if frames else 0
        
        # Calculate score (0-100)
        complexity = min(100, (avg_simultaneous_notes * 15) + (chord_changes * 5) + (max_notes * 10))
        
        difficulty_levels = {
            (0, 20): "Beginner - Simple melody",
            (20, 40): "Elementary - Basic harmony", 
            (40, 60): "Intermediate - Multi-voice texture",
            (60, 80): "Advanced - Complex harmony",
            (80, 100): "Expert - Highly polyphonic"
        }
        
        difficulty = "Advanced"
        for (min_score, max_score), level in difficulty_levels.items():
            if min_score <= complexity < max_score:
                difficulty = level
                break
        
        return {
            'score': round(complexity, 1),
            'difficulty': difficulty,
            'avg_simultaneous_notes': round(avg_simultaneous_notes, 1),
            'chord_changes': chord_changes,
            'max_simultaneous_notes': max_notes
        }

    def compare_polyphonic_performance(self, analysis_result, reference_analysis=None):
        """
        Compare polyphonic performance against reference (if available)
        """
        
        if not reference_analysis:
            return {
                "comparison_type": "standalone_analysis",
                "melody_accuracy": "Not available - no reference",
                "harmony_accuracy": "Not available - no reference", 
                "recommendations": self._generate_polyphonic_recommendations(analysis_result)
            }
        
        # Compare melody lines
        melody_comparison = self._compare_melody_lines(
            analysis_result.get('melody_line', []),
            reference_analysis.get('melody_line', [])
        )
        
        # Compare chord progressions
        chord_comparison = self._compare_chord_progressions(
            analysis_result.get('chord_progression', {}),
            reference_analysis.get('chord_progression', {})
        )
        
        return {
            "comparison_type": "reference_comparison",
            "melody_accuracy": melody_comparison,
            "harmony_accuracy": chord_comparison,
            "recommendations": self._generate_polyphonic_recommendations(analysis_result, reference_analysis)
        }

    def _generate_polyphonic_recommendations(self, analysis_result, reference_analysis=None):
        """
        Generate recommendations for polyphonic performance improvement
        """
        
        recommendations = []
        complexity = analysis_result.get('complexity_score', {})
        voice_leading = analysis_result.get('voice_leading', [])
        
        # Complexity-based recommendations
        if complexity.get('score', 0) > 60:
            recommendations.append("This is a complex polyphonic piece. Focus on one voice at a time before combining them.")
        
        # Voice leading recommendations
        for voice in voice_leading:
            if voice.get('analysis') == 'jumpy':
                recommendations.append(f"Voice {voice['voice_number']} has large intervals. Practice smooth connections between notes.")
        
        # Melody vs harmony balance
        avg_notes = complexity.get('avg_simultaneous_notes', 0)
        if avg_notes > 4:
            recommendations.append("With many simultaneous notes, ensure the melody line remains clear and prominent.")
        elif avg_notes < 2:
            recommendations.append("Consider adding more harmonic content to create fuller texture.")
        
        # General polyphonic recommendations
        recommendations.extend([
            "Practice hands separately before combining them.",
            "Use a metronome to maintain steady rhythm across all voices.",
            "Listen for balance between melody and accompaniment."
        ])
        
        return recommendations

def enhance_main_script_for_polyphony():
    """
    Return code snippet to integrate polyphonic analysis into main script
    """
    
    integration_code = '''
# Add this to enhanced_main.py imports
from polyphonic_analyzer import PolyphonicAnalyzer

# Add this method to MusicAnalyzer class
def analyze_polyphonic_audio(self, audio_path):
    """Analyze polyphonic audio content"""
    poly_analyzer = PolyphonicAnalyzer()
    return poly_analyzer.analyze_polyphonic_performance(audio_path)

# Add this to main() function after monophonic analysis
# Check if the piece might be polyphonic
if len(performance_onsets) > len(self.piece_info['melody']) * 1.5:
    print("🎹 Detected potential polyphonic content, running advanced analysis...")
    poly_result = analyzer.analyze_polyphonic_audio(args.audio_file)
    if poly_result:
        print("\\n📊 POLYPHONIC ANALYSIS")
        print("=" * 50)
        print(json.dumps(poly_result, indent=2))
'''
    
    return integration_code

if __name__ == "__main__":
    # Test the polyphonic analyzer
    print("Polyphonic Analysis Module - Test Mode")
    print("This module can analyze:")
    print("✓ Multiple simultaneous pitches")
    print("✓ Chord progressions") 
    print("✓ Voice leading")
    print("✓ Melody vs harmony separation")
    print("✓ Complexity scoring")
    
    print("\n🔗 Integration code for main script:")
    print(enhance_main_script_for_polyphony())

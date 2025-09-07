#!/usr/bin/env python3
"""
Core Music Analysis Module

This is the main analysis engine that coordinates all music analysis components.
It provides a unified interface for analyzing musical performances and comparing 
them against reference pieces.
"""

import numpy as np
import librosa
import json
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Import component modules
from audio_digitizer import AudioDigitizer
from musicxml_parser import extract_melody_from_musicxml, get_musicxml_tempo
from polyphonic_analyzer import PolyphonicAnalyzer

# Define standard test pieces
PIECES = {
    "twinkle": {
        "title": "Twinkle Twinkle Little Star",
        "melody": [
            {"pitch": 60, "duration": 0.5, "velocity": 80},  # C4
            {"pitch": 60, "duration": 0.5, "velocity": 80},  # C4
            {"pitch": 67, "duration": 0.5, "velocity": 80},  # G4
            {"pitch": 67, "duration": 0.5, "velocity": 80},  # G4
            {"pitch": 69, "duration": 0.5, "velocity": 80},  # A4
            {"pitch": 69, "duration": 0.5, "velocity": 80},  # A4
            {"pitch": 67, "duration": 1.0, "velocity": 80},  # G4
            {"pitch": 65, "duration": 0.5, "velocity": 80},  # F4
            {"pitch": 65, "duration": 0.5, "velocity": 80},  # F4
            {"pitch": 64, "duration": 0.5, "velocity": 80},  # E4
            {"pitch": 64, "duration": 0.5, "velocity": 80},  # E4
            {"pitch": 62, "duration": 0.5, "velocity": 80},  # D4
            {"pitch": 62, "duration": 0.5, "velocity": 80},  # D4
            {"pitch": 60, "duration": 1.0, "velocity": 80},  # C4
        ]
    },
    "mary": {
        "title": "Mary Had a Little Lamb",
        "melody": [
            {"pitch": 64, "duration": 0.5, "velocity": 80},  # E4
            {"pitch": 62, "duration": 0.5, "velocity": 80},  # D4
            {"pitch": 60, "duration": 0.5, "velocity": 80},  # C4
            {"pitch": 62, "duration": 0.5, "velocity": 80},  # D4
            {"pitch": 64, "duration": 0.5, "velocity": 80},  # E4
            {"pitch": 64, "duration": 0.5, "velocity": 80},  # E4
            {"pitch": 64, "duration": 1.0, "velocity": 80},  # E4
            {"pitch": 62, "duration": 0.5, "velocity": 80},  # D4
            {"pitch": 62, "duration": 0.5, "velocity": 80},  # D4
            {"pitch": 62, "duration": 1.0, "velocity": 80},  # D4
            {"pitch": 64, "duration": 0.5, "velocity": 80},  # E4
            {"pitch": 67, "duration": 0.5, "velocity": 80},  # G4
            {"pitch": 67, "duration": 1.0, "velocity": 80},  # G4
        ]
    }
}

class MusicAnalyzer:
    """
    Core music analysis engine that coordinates all analysis components
    """
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.audio_digitizer = AudioDigitizer(sample_rate)
        self.polyphonic_analyzer = PolyphonicAnalyzer(sample_rate)
        
    def analyze_performance(self, audio_path: str, reference_piece: str = "twinkle", 
                          reference_musicxml: Optional[str] = None) -> Dict:
        """
        Main analysis function that analyzes a musical performance
        
        Args:
            audio_path: Path to the audio file to analyze
            reference_piece: Key of reference piece from PIECES dict
            reference_musicxml: Optional path to MusicXML reference file
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        print(f"🎵 Starting analysis of {audio_path}")
        print(f"📊 Reference: {reference_piece}")
        
        # Get reference melody
        reference_melody = self._get_reference_melody(reference_piece, reference_musicxml)
        if not reference_melody:
            return {"error": "Could not load reference melody"}
        
        print(f"📝 Reference melody has {len(reference_melody)} notes")
        
        # Detect if audio is polyphonic or monophonic
        is_polyphonic = self._detect_polyphonic_content(audio_path)
        
        if is_polyphonic:
            print("🎹 Detected polyphonic content - using advanced analysis")
            polyphonic_analysis = self.polyphonic_analyzer.analyze_polyphonic_performance(
                audio_path, reference_melody
            )
            if polyphonic_analysis:
                # Combine with standard analysis
                standard_analysis = self._perform_standard_analysis(audio_path, reference_melody)
                return {
                    "analysis_type": "combined",
                    "polyphonic_analysis": polyphonic_analysis,
                    "standard_analysis": standard_analysis,
                    "reference_melody": reference_melody,
                    "audio_file": audio_path
                }
        
        # Standard monophonic analysis
        print("🎵 Using standard monophonic analysis")
        standard_analysis = self._perform_standard_analysis(audio_path, reference_melody)
        
        return {
            "analysis_type": "standard",
            "standard_analysis": standard_analysis,
            "reference_melody": reference_melody,
            "audio_file": audio_path
        }
    
    def _get_reference_melody(self, reference_piece: str, musicxml_path: Optional[str] = None) -> List[Dict]:
        """Get reference melody from various sources"""
        
        # Try MusicXML first if provided
        if musicxml_path and os.path.exists(musicxml_path):
            print(f"📄 Loading melody from MusicXML: {musicxml_path}")
            melody = extract_melody_from_musicxml(musicxml_path)
            if melody:
                return melody
        
        # Try to find MusicXML file based on piece name
        musicxml_candidates = [
            f"midi/{reference_piece}.mxl",
            f"midi/{reference_piece}_reference.mxl",
            f"{reference_piece}.mxl",
            f"test_5measures.mxl"  # Default test file
        ]
        
        for candidate in musicxml_candidates:
            if os.path.exists(candidate):
                print(f"📄 Found MusicXML file: {candidate}")
                melody = extract_melody_from_musicxml(candidate)
                if melody:
                    return melody
        
        # Fallback to built-in pieces
        if reference_piece in PIECES:
            print(f"📚 Using built-in melody for {reference_piece}")
            return PIECES[reference_piece]["melody"]
        
        # Ultimate fallback
        print("⚠️ Using default twinkle melody as fallback")
        return PIECES["twinkle"]["melody"]
    
    def _detect_polyphonic_content(self, audio_path: str) -> bool:
        """Detect if audio contains polyphonic (multiple simultaneous notes) content"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Use harmonic-percussive separation
            y_harmonic, _ = librosa.effects.hpss(y)
            
            # Analyze spectral content
            stft = librosa.stft(y_harmonic)
            magnitude = np.abs(stft)
            
            # Count strong frequency components per frame
            threshold = np.percentile(magnitude.flatten(), 85)
            strong_components_per_frame = np.sum(magnitude > threshold, axis=0)
            
            # If consistently more than 3-4 strong components, likely polyphonic
            avg_components = np.mean(strong_components_per_frame)
            is_polyphonic = avg_components > 4
            
            print(f"📊 Polyphonic detection: {avg_components:.1f} avg components, {'Yes' if is_polyphonic else 'No'}")
            return is_polyphonic
            
        except Exception as e:
            print(f"⚠️ Polyphonic detection failed: {e}")
            return False
    
    def _perform_standard_analysis(self, audio_path: str, reference_melody: List[Dict]) -> Dict:
        """Perform standard monophonic analysis"""
        
        # Use audio digitizer for main analysis
        digitized_notes = self.audio_digitizer.digitize_performance(
            audio_path, reference_melody
        )
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(digitized_notes, reference_melody)
        
        # Detailed note analysis
        note_details = self._analyze_note_details(digitized_notes, reference_melody)
        
        # Generate feedback
        feedback = self._generate_feedback(metrics, note_details)
        
        return {
            "digitized_notes": digitized_notes,
            "performance_notes": digitized_notes,  # Alias for compatibility
            "metrics": metrics,
            "note_details": note_details,
            "feedback": feedback,
            "timestamp": self._get_timestamp()
        }
    
    def _calculate_performance_metrics(self, performance_notes: List[Dict], 
                                     reference_melody: List[Dict]) -> Dict:
        """Calculate overall performance metrics"""
        
        total_notes = len(reference_melody)
        detected_notes = len([n for n in performance_notes if n.get('detected', False)])
        missed_notes = total_notes - detected_notes
        
        # Calculate accuracy categories
        excellent = len([n for n in performance_notes if n.get('accuracy') == 'excellent'])
        good = len([n for n in performance_notes if n.get('accuracy') == 'good'])
        poor = len([n for n in performance_notes if n.get('accuracy') == 'poor'])
        
        # Overall score calculation
        accuracy_score = (excellent * 100 + good * 80 + poor * 60) / max(total_notes * 100, 1)
        timing_score = 100 - min(50, np.mean([abs(n.get('timing_error', 0)) for n in performance_notes if 'timing_error' in n]) / 10)
        pitch_score = 100 - min(50, np.mean([abs(n.get('pitch_error', 0)) for n in performance_notes if 'pitch_error' in n]) / 20)
        
        overall_score = (accuracy_score + timing_score + pitch_score) / 3
        
        return {
            "total_notes": total_notes,
            "detected_notes": detected_notes,
            "missed_notes": missed_notes,
            "note_accuracy": (detected_notes / max(total_notes, 1)) * 100,
            "excellent_notes": excellent,
            "good_notes": good,
            "poor_notes": poor,
            "accuracy_score": accuracy_score,
            "timing_score": timing_score,
            "pitch_score": pitch_score,
            "overall_score": overall_score
        }
    
    def _analyze_note_details(self, performance_notes: List[Dict], 
                            reference_melody: List[Dict]) -> List[Dict]:
        """Analyze individual note details"""
        
        note_details = []
        
        for i, (perf_note, ref_note) in enumerate(zip(performance_notes, reference_melody)):
            
            if perf_note.get('detected', False):
                # Calculate deviations
                pitch_error = perf_note.get('pitch', 0) - ref_note.get('pitch', 0)
                pitch_error_cents = pitch_error * 100  # Convert to cents (approximate)
                
                timing_error = (perf_note.get('onset', 0) - ref_note.get('start_time', i * 0.5)) * 1000  # ms
                
                # Determine accuracy level
                if abs(pitch_error_cents) < 25 and abs(timing_error) < 50:
                    accuracy = "excellent"
                elif abs(pitch_error_cents) < 50 and abs(timing_error) < 100:
                    accuracy = "good"
                elif abs(pitch_error_cents) < 100 and abs(timing_error) < 200:
                    accuracy = "poor"
                else:
                    accuracy = "needs_work"
                
                note_details.append({
                    "note_number": i + 1,
                    "expected_pitch": ref_note.get('pitch', 0),
                    "detected_pitch": perf_note.get('pitch', 0),
                    "pitch_deviation_cents": pitch_error_cents,
                    "timing_deviation_ms": timing_error,
                    "expected_time": ref_note.get('start_time', i * 0.5),
                    "detected_time": perf_note.get('onset', 0),
                    "duration": perf_note.get('duration', 0),
                    "accuracy": accuracy,
                    "confidence": perf_note.get('confidence', 0.5)
                })
            else:
                # Missed note
                note_details.append({
                    "note_number": i + 1,
                    "expected_pitch": ref_note.get('pitch', 0),
                    "detected_pitch": None,
                    "pitch_deviation_cents": "MISSED",
                    "timing_deviation_ms": "MISSED",
                    "expected_time": ref_note.get('start_time', i * 0.5),
                    "detected_time": None,
                    "duration": 0,
                    "accuracy": "missed",
                    "confidence": 0.0
                })
        
        return note_details
    
    def _generate_feedback(self, metrics: Dict, note_details: List[Dict]) -> str:
        """Generate text feedback based on analysis"""
        
        feedback_lines = []
        
        # Overall performance
        score = metrics.get('overall_score', 0)
        if score >= 90:
            feedback_lines.append("🌟 Excellent performance!")
        elif score >= 80:
            feedback_lines.append("👍 Good performance with room for improvement.")
        elif score >= 70:
            feedback_lines.append("📈 Fair performance, focus on accuracy.")
        else:
            feedback_lines.append("📚 Keep practicing - you're on the right track!")
        
        feedback_lines.append(f"Overall Score: {score:.1f}/100")
        feedback_lines.append("")
        
        # Specific areas
        if metrics.get('missed_notes', 0) > 0:
            feedback_lines.append(f"⚠️ {metrics['missed_notes']} notes were missed")
        
        if metrics.get('timing_score', 100) < 80:
            feedback_lines.append("⏱️ Focus on timing accuracy - try practicing with a metronome")
        
        if metrics.get('pitch_score', 100) < 80:
            feedback_lines.append("🎵 Work on pitch accuracy - practice scales and use a tuner")
        
        # Note-specific feedback
        problematic_notes = [n for n in note_details if n.get('accuracy') in ['poor', 'needs_work']]
        if problematic_notes:
            feedback_lines.append("")
            feedback_lines.append("📝 Focus on these notes:")
            for note in problematic_notes[:3]:  # Show first 3
                feedback_lines.append(f"   Note {note['note_number']}: {note.get('accuracy', 'unknown')} accuracy")
        
        return "\n".join(feedback_lines)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for analysis"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def export_analysis(self, analysis_result: Dict, output_path: str) -> bool:
        """Export analysis results to JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(analysis_result, f, indent=2, default=str)
            print(f"📄 Analysis exported to {output_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to export analysis: {e}")
            return False

def main():
    """Test the music analyzer"""
    analyzer = MusicAnalyzer()
    
    # Test with available audio file
    test_files = ["test_5measures.wav", "audio/test_5measures.wav", "demo_performance.wav"]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"🧪 Testing with {test_file}")
            result = analyzer.analyze_performance(test_file, "twinkle")
            
            if "error" not in result:
                print("✅ Analysis completed successfully!")
                
                # Print basic metrics
                if "standard_analysis" in result:
                    metrics = result["standard_analysis"]["metrics"]
                    print(f"📊 Overall Score: {metrics['overall_score']:.1f}/100")
                    print(f"🎵 Note Accuracy: {metrics['note_accuracy']:.1f}%")
                    print(f"⏱️ Timing Score: {metrics['timing_score']:.1f}/100")
                    print(f"🎯 Pitch Score: {metrics['pitch_score']:.1f}/100")
                
                # Export result
                analyzer.export_analysis(result, f"analysis_output_{test_file.replace('.wav', '.json')}")
            else:
                print(f"❌ Analysis failed: {result['error']}")
            
            break
    else:
        print("⚠️ No test audio files found")

if __name__ == "__main__":
    main()

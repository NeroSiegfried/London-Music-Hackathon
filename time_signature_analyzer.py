#!/usr/bin/env python3
"""
Time Signature Analysis Module

This module analyzes rhythmic patterns and time signatures to provide
better understanding of timing relationships in musical performances.
"""

import numpy as np
import librosa
from collections import Counter
import matplotlib.pyplot as plt

class TimeSignatureAnalyzer:
    def __init__(self):
        self.common_signatures = [
            (4, 4),  # Common time
            (3, 4),  # Waltz time
            (2, 4),  # March time
            (6, 8),  # Compound duple
            (9, 8),  # Compound triple
            (12, 8), # Compound quadruple
        ]

    def analyze_time_signature_impact(self, reference_melody, performance_data, detected_tempo):
        """
        Analyze how time signature affects performance accuracy
        
        Args:
            reference_melody: List of reference notes with durations
            performance_data: Performance analysis data
            detected_tempo: Detected tempo in BPM
            
        Returns:
            Dictionary with time signature analysis
        """
        
        # Calculate the natural time signature of the piece
        detected_signature = self._detect_time_signature(reference_melody)
        
        # Analyze beat patterns
        beat_analysis = self._analyze_beat_patterns(reference_melody, performance_data, detected_signature)
        
        # Analyze timing compensation
        compensation_analysis = self._analyze_timing_compensation(performance_data, detected_signature)
        
        # Generate recommendations
        recommendations = self._generate_timing_recommendations(beat_analysis, compensation_analysis)
        
        return {
            "detected_time_signature": detected_signature,
            "tempo_bpm": detected_tempo,
            "beat_analysis": beat_analysis,
            "timing_compensation": compensation_analysis,
            "recommendations": recommendations,
            "signature_difficulty": self._assess_signature_difficulty(detected_signature)
        }

    def _detect_time_signature(self, melody):
        """Detect the most likely time signature from melody structure"""
        
        # Calculate total duration in quarter notes
        total_duration = sum(note['duration'] * 4 for note in melody)
        
        # Look for patterns in note durations
        durations = [note['duration'] * 4 for note in melody]  # Convert to quarter notes
        
        # Common patterns for different time signatures
        if total_duration % 4 == 0 and len([d for d in durations if d == 0.5]) > 0:
            return (4, 4)  # 4/4 time - common time
        elif total_duration % 3 == 0:
            return (3, 4)  # 3/4 time - waltz
        elif total_duration % 2 == 0 and max(durations) <= 1:
            return (2, 4)  # 2/4 time - march
        else:
            return (4, 4)  # Default to 4/4

    def _analyze_beat_patterns(self, reference_melody, performance_data, time_signature):
        """Analyze how well the performer follows beat patterns"""
        
        beats_per_measure = time_signature[0]
        note_value = time_signature[1]
        
        # Calculate expected beat positions
        beat_positions = []
        current_beat = 0
        
        for note in reference_melody:
            beat_positions.append(current_beat)
            current_beat += note['duration'] * 4  # Convert to quarter note beats
        
        # Analyze performance timing relative to beats
        strong_beats = []  # Beats 1, 3 in 4/4 or beat 1 in 3/4
        weak_beats = []   # Beats 2, 4 in 4/4 or beats 2, 3 in 3/4
        
        performance_notes = performance_data.get('note_details', [])
        
        for i, (ref_beat, perf_note) in enumerate(zip(beat_positions, performance_notes)):
            if 'timing_deviation_ms' in perf_note and isinstance(perf_note['timing_deviation_ms'], (int, float)):
                beat_in_measure = (ref_beat % beats_per_measure)
                timing_error = abs(perf_note['timing_deviation_ms'])
                
                if time_signature == (4, 4):
                    if beat_in_measure in [0, 2]:  # Strong beats (1, 3)
                        strong_beats.append(timing_error)
                    else:  # Weak beats (2, 4)
                        weak_beats.append(timing_error)
                elif time_signature == (3, 4):
                    if beat_in_measure == 0:  # Strong beat (1)
                        strong_beats.append(timing_error)
                    else:  # Weak beats (2, 3)
                        weak_beats.append(timing_error)
        
        return {
            "strong_beat_accuracy": np.mean(strong_beats) if strong_beats else 0,
            "weak_beat_accuracy": np.mean(weak_beats) if weak_beats else 0,
            "beat_consistency": self._calculate_beat_consistency(performance_notes),
            "rhythmic_pattern_adherence": self._analyze_rhythmic_patterns(reference_melody, performance_notes)
        }

    def _analyze_timing_compensation(self, performance_data, time_signature):
        """
        Analyze if the performer compensates timing errors to maintain overall rhythm
        """
        performance_notes = performance_data.get('note_details', [])
        timing_deviations = []
        
        for note in performance_notes:
            if 'timing_deviation_ms' in note and isinstance(note['timing_deviation_ms'], (int, float)):
                timing_deviations.append(note['timing_deviation_ms'])
        
        if len(timing_deviations) < 3:
            return {"compensation_detected": False}
        
        # Look for compensation patterns
        compensations = []
        for i in range(1, len(timing_deviations)):
            prev_dev = timing_deviations[i-1]
            curr_dev = timing_deviations[i]
            
            # Check if current note compensates for previous note's timing
            if (prev_dev > 50 and curr_dev < -20) or (prev_dev < -50 and curr_dev > 20):
                compensations.append(abs(prev_dev + curr_dev))
        
        compensation_ratio = len(compensations) / max(len(timing_deviations) - 1, 1)
        
        # Analyze overall drift
        cumulative_drift = np.cumsum(timing_deviations)
        max_drift = np.max(np.abs(cumulative_drift))
        
        return {
            "compensation_detected": compensation_ratio > 0.3,
            "compensation_ratio": compensation_ratio,
            "average_compensation": np.mean(compensations) if compensations else 0,
            "overall_drift_ms": max_drift,
            "drift_pattern": "accelerating" if cumulative_drift[-1] < -100 else "slowing" if cumulative_drift[-1] > 100 else "stable"
        }

    def _calculate_beat_consistency(self, performance_notes):
        """Calculate how consistent the beat timing is"""
        timing_deviations = []
        
        for note in performance_notes:
            if 'timing_deviation_ms' in note and isinstance(note['timing_deviation_ms'], (int, float)):
                timing_deviations.append(note['timing_deviation_ms'])
        
        if len(timing_deviations) < 2:
            return 100  # Perfect if too few notes to judge
        
        # Calculate standard deviation of timing errors
        std_dev = np.std(timing_deviations)
        
        # Convert to consistency score (0-100)
        consistency = max(0, 100 - (std_dev / 5))  # 5ms std = 99% consistency
        return min(100, consistency)

    def _analyze_rhythmic_patterns(self, reference_melody, performance_notes):
        """Analyze adherence to rhythmic patterns"""
        
        # Group notes by duration patterns
        ref_pattern = []
        perf_pattern = []
        
        for i, ref_note in enumerate(reference_melody):
            ref_duration = ref_note['duration']
            ref_pattern.append(ref_duration)
            
            if i < len(performance_notes):
                perf_note = performance_notes[i]
                # Estimate performed duration from timing (simplified)
                if i < len(performance_notes) - 1 and 'detected_time' in perf_note and 'detected_time' in performance_notes[i+1]:
                    perf_duration = performance_notes[i+1]['detected_time'] - perf_note['detected_time']
                    perf_pattern.append(perf_duration)
        
        # Calculate pattern similarity
        if len(perf_pattern) >= len(ref_pattern) * 0.7:  # At least 70% of notes detected
            pattern_similarity = self._calculate_pattern_similarity(ref_pattern, perf_pattern[:len(ref_pattern)])
        else:
            pattern_similarity = 50  # Moderate score if many notes missed
        
        return {
            "pattern_similarity": pattern_similarity,
            "rhythmic_accuracy": pattern_similarity  # Simplified for now
        }

    def _calculate_pattern_similarity(self, ref_pattern, perf_pattern):
        """Calculate similarity between rhythmic patterns"""
        if len(ref_pattern) != len(perf_pattern):
            return 50  # Default moderate score
        
        differences = []
        for ref, perf in zip(ref_pattern, perf_pattern):
            if isinstance(perf, (int, float)) and isinstance(ref, (int, float)):
                diff = abs(ref - perf) / max(ref, 0.1)  # Relative difference
                differences.append(diff)
        
        if not differences:
            return 50
        
        avg_diff = np.mean(differences)
        similarity = max(0, 100 - (avg_diff * 100))
        return min(100, similarity)

    def _assess_signature_difficulty(self, time_signature):
        """Assess the difficulty level of the time signature"""
        difficulty_levels = {
            (4, 4): "Beginner - Most common time signature",
            (2, 4): "Beginner - Simple march time", 
            (3, 4): "Intermediate - Waltz time, requires triplet feel",
            (6, 8): "Advanced - Compound time, needs subdivision awareness",
            (5, 4): "Advanced - Asymmetrical, challenging to maintain",
            (7, 8): "Expert - Complex asymmetrical meter"
        }
        
        return difficulty_levels.get(time_signature, "Advanced - Complex time signature")

    def _generate_timing_recommendations(self, beat_analysis, compensation_analysis):
        """Generate specific recommendations based on timing analysis"""
        recommendations = []
        
        # Strong vs weak beat analysis
        strong_accuracy = beat_analysis.get("strong_beat_accuracy", 0)
        weak_accuracy = beat_analysis.get("weak_beat_accuracy", 0)
        
        if strong_accuracy > weak_accuracy + 50:
            recommendations.append("Focus on weak beats (2 & 4 in 4/4). Try practicing with a metronome emphasizing all beats equally.")
        elif weak_accuracy > strong_accuracy + 50:
            recommendations.append("Work on strong beat placement (1 & 3 in 4/4). These beats should feel more anchored and precise.")
        
        # Consistency analysis
        consistency = beat_analysis.get("beat_consistency", 100)
        if consistency < 70:
            recommendations.append("Practice with a metronome to improve timing consistency. Start slowly and gradually increase tempo.")
        elif consistency < 85:
            recommendations.append("Good timing overall! Fine-tune with slow practice to achieve even more consistent rhythm.")
        
        # Compensation analysis
        if compensation_analysis.get("compensation_detected", False):
            drift = compensation_analysis.get("drift_pattern", "stable")
            if drift == "accelerating":
                recommendations.append("You tend to speed up through the piece. Practice with a metronome and focus on maintaining steady tempo.")
            elif drift == "slowing":
                recommendations.append("You tend to slow down through the piece. Work on maintaining energy and forward momentum.")
            else:
                recommendations.append("Good timing compensation! You naturally adjust to stay in time.")
        
        # Rhythmic patterns
        pattern_accuracy = beat_analysis.get("rhythmic_pattern_adherence", {}).get("pattern_similarity", 100)
        if pattern_accuracy < 70:
            recommendations.append("Work on note duration accuracy. Practice counting beats out loud while playing.")
        
        return recommendations

def create_timing_visualization(timing_analysis, output_path="visualizations/timing_analysis.png"):
    """Create a visualization of timing analysis results"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Time Signature and Rhythm Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Strong vs Weak Beat Accuracy
    beat_data = timing_analysis.get("beat_analysis", {})
    strong_acc = beat_data.get("strong_beat_accuracy", 0)
    weak_acc = beat_data.get("weak_beat_accuracy", 0)
    
    ax1.bar(['Strong Beats\n(1, 3)', 'Weak Beats\n(2, 4)'], [strong_acc, weak_acc], 
            color=['darkblue', 'lightblue'], alpha=0.7)
    ax1.set_ylabel('Average Timing Error (ms)')
    ax1.set_title('Beat Type Accuracy')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time Signature Info
    signature = timing_analysis.get("detected_time_signature", (4, 4))
    tempo = timing_analysis.get("tempo_bpm", 100)
    difficulty = timing_analysis.get("signature_difficulty", "")
    
    ax2.text(0.5, 0.7, f"Time Signature: {signature[0]}/{signature[1]}", 
             ha='center', va='center', fontsize=14, fontweight='bold')
    ax2.text(0.5, 0.5, f"Tempo: {tempo} BPM", 
             ha='center', va='center', fontsize=12)
    ax2.text(0.5, 0.3, f"Difficulty: {difficulty.split(' - ')[0]}", 
             ha='center', va='center', fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Piece Information')
    
    # Plot 3: Timing Compensation
    comp_data = timing_analysis.get("timing_compensation", {})
    comp_ratio = comp_data.get("compensation_ratio", 0) * 100
    drift = comp_data.get("overall_drift_ms", 0)
    
    ax3.bar(['Compensation\nRate (%)', 'Overall\nDrift (ms)'], 
            [comp_ratio, abs(drift)], color=['green', 'orange'], alpha=0.7)
    ax3.set_title('Timing Compensation Analysis')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Beat Consistency
    consistency = beat_data.get("beat_consistency", 100)
    pattern_acc = beat_data.get("rhythmic_pattern_adherence", {}).get("pattern_similarity", 100)
    
    ax4.bar(['Beat\nConsistency', 'Pattern\nAccuracy'], 
            [consistency, pattern_acc], color=['purple', 'teal'], alpha=0.7)
    ax4.set_ylabel('Score (%)')
    ax4.set_title('Rhythmic Accuracy')
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Timing analysis visualization saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Test the timing analyzer
    test_melody = [
        {'pitch': 60, 'duration': 0.25}, {'pitch': 60, 'duration': 0.25},
        {'pitch': 67, 'duration': 0.25}, {'pitch': 67, 'duration': 0.25},
        {'pitch': 69, 'duration': 0.25}, {'pitch': 69, 'duration': 0.25},
        {'pitch': 67, 'duration': 0.5},
    ]
    
    test_performance = {
        'note_details': [
            {'timing_deviation_ms': 50}, {'timing_deviation_ms': -30},
            {'timing_deviation_ms': 120}, {'timing_deviation_ms': -80},
            {'timing_deviation_ms': 40}, {'timing_deviation_ms': -60},
            {'timing_deviation_ms': 20}
        ]
    }
    
    analyzer = TimeSignatureAnalyzer()
    results = analyzer.analyze_time_signature_impact(test_melody, test_performance, 100)
    
    print("Time Signature Analysis Results:")
    for key, value in results.items():
        print(f"{key}: {value}")
    
    create_timing_visualization(results)
    print("Test timing visualization created!")

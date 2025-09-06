#!/usr/bin/env python3
"""
Advanced Mistake Detection and Performance Diff Analysis

This module provides sophisticated analysis of musical performances to detect:
- Systematic mistakes and error patterns
- Retry attempts and corrections
- Section-by-section performance differences
- Musical interpretation variations
"""

import numpy as np
import librosa
import json
from scipy import signal
from sklearn.cluster import DBSCAN
from dtaidistance import dtw
import matplotlib.pyplot as plt

class PerformanceDiffAnalyzer:
    def __init__(self, reference_data, performance_data):
        self.reference_data = reference_data
        self.performance_data = performance_data
        self.mistake_patterns = []
        self.retry_sections = []
        self.performance_sections = []
        
    def analyze_comprehensive_diff(self):
        """
        Perform comprehensive difference analysis between reference and performance
        """
        analysis_result = {
            "mistake_patterns": self.detect_mistake_patterns(),
            "retry_analysis": self.detect_retry_attempts(),
            "section_analysis": self.analyze_performance_sections(),
            "musical_interpretation": self.analyze_musical_interpretation(),
            "recommendations": self.generate_improvement_recommendations()
        }
        
        return analysis_result
    
    def detect_mistake_patterns(self):
        """
        Detect systematic mistake patterns in the performance
        """
        patterns = []
        note_details = self.performance_data.get("note_details", [])
        
        if not note_details:
            return patterns
        
        # 1. Pitch Pattern Analysis
        pitch_patterns = self._analyze_pitch_patterns(note_details)
        patterns.extend(pitch_patterns)
        
        # 2. Timing Pattern Analysis
        timing_patterns = self._analyze_timing_patterns(note_details)
        patterns.extend(timing_patterns)
        
        # 3. Sequential Error Analysis
        sequential_patterns = self._analyze_sequential_errors(note_details)
        patterns.extend(sequential_patterns)
        
        # 4. Harmonic/Melodic Pattern Analysis
        harmonic_patterns = self._analyze_harmonic_patterns(note_details)
        patterns.extend(harmonic_patterns)
        
        return patterns
    
    def _analyze_pitch_patterns(self, note_details):
        """Analyze pitch-related mistake patterns"""
        patterns = []
        pitch_errors = []
        error_positions = []
        
        for i, note in enumerate(note_details):
            pitch_dev = note.get('pitch_deviation_cents', 0)
            if isinstance(pitch_dev, (int, float)):
                pitch_errors.append(pitch_dev)
                error_positions.append(i)
        
        if len(pitch_errors) < 3:
            return patterns
        
        # Systematic sharp/flat tendencies
        sharp_errors = [e for e in pitch_errors if e > 20]
        flat_errors = [e for e in pitch_errors if e < -20]
        
        if len(sharp_errors) > len(pitch_errors) * 0.6:
            patterns.append({
                "type": "systematic_sharp",
                "severity": "high" if np.mean(sharp_errors) > 50 else "medium",
                "description": f"Consistent sharp tendency (+{np.mean(sharp_errors):.1f}¢ average)",
                "affected_notes": [i+1 for i, e in enumerate(pitch_errors) if e > 20],
                "recommendation": "Focus on intonation exercises, check instrument tuning"
            })
        
        if len(flat_errors) > len(pitch_errors) * 0.6:
            patterns.append({
                "type": "systematic_flat",
                "severity": "high" if abs(np.mean(flat_errors)) > 50 else "medium",
                "description": f"Consistent flat tendency ({np.mean(flat_errors):.1f}¢ average)",
                "affected_notes": [i+1 for i, e in enumerate(pitch_errors) if e < -20],
                "recommendation": "Work on breath support and embouchure/bow pressure"
            })
        
        # Interval-specific errors
        interval_errors = self._analyze_interval_errors(note_details)
        patterns.extend(interval_errors)
        
        # Pitch instability
        unstable_notes = []
        for i, note in enumerate(note_details):
            if abs(note.get('pitch_deviation_cents', 0)) > 30:
                unstable_notes.append(i+1)
        
        if len(unstable_notes) > len(note_details) * 0.3:
            patterns.append({
                "type": "pitch_instability",
                "severity": "medium",
                "description": f"Pitch instability in {len(unstable_notes)} notes",
                "affected_notes": unstable_notes,
                "recommendation": "Practice long tones and pitch matching exercises"
            })
        
        return patterns
    
    def _analyze_timing_patterns(self, note_details):
        """Analyze timing-related mistake patterns"""
        patterns = []
        timing_errors = []
        error_positions = []
        
        for i, note in enumerate(note_details):
            timing_dev = note.get('timing_deviation_ms', 0)
            if isinstance(timing_dev, (int, float)):
                timing_errors.append(timing_dev)
                error_positions.append(i)
        
        if len(timing_errors) < 3:
            return patterns
        
        # Systematic rushing/dragging
        early_errors = [e for e in timing_errors if e < -50]
        late_errors = [e for e in timing_errors if e > 50]
        
        if len(early_errors) > len(timing_errors) * 0.6:
            patterns.append({
                "type": "systematic_rushing",
                "severity": "high" if abs(np.mean(early_errors)) > 100 else "medium",
                "description": f"Consistent rushing ({np.mean(early_errors):.1f}ms average)",
                "affected_notes": [i+1 for i, e in enumerate(timing_errors) if e < -50],
                "recommendation": "Practice with metronome, focus on subdivisions"
            })
        
        if len(late_errors) > len(timing_errors) * 0.6:
            patterns.append({
                "type": "systematic_dragging",
                "severity": "high" if np.mean(late_errors) > 100 else "medium",
                "description": f"Consistent dragging (+{np.mean(late_errors):.1f}ms average)",
                "affected_notes": [i+1 for i, e in enumerate(timing_errors) if e > 50],
                "recommendation": "Work on forward motion and phrase direction"
            })
        
        # Tempo instability
        tempo_changes = self._analyze_tempo_instability(note_details)
        if tempo_changes:
            patterns.extend(tempo_changes)
        
        # Rhythmic pattern errors
        rhythmic_errors = self._analyze_rhythmic_patterns(note_details)
        patterns.extend(rhythmic_errors)
        
        return patterns
    
    def _analyze_interval_errors(self, note_details):
        """Analyze errors in specific musical intervals"""
        patterns = []
        
        if len(note_details) < 2:
            return patterns
        
        interval_errors = {}
        
        for i in range(1, len(note_details)):
            prev_note = note_details[i-1]
            curr_note = note_details[i]
            
            prev_pitch = prev_note.get('expected_pitch', '')
            curr_pitch = curr_note.get('expected_pitch', '')
            
            if prev_pitch and curr_pitch:
                try:
                    prev_midi = librosa.note_to_midi(prev_pitch)
                    curr_midi = librosa.note_to_midi(curr_pitch)
                    interval = curr_midi - prev_midi
                    
                    curr_error = curr_note.get('pitch_deviation_cents', 0)
                    if isinstance(curr_error, (int, float)) and abs(curr_error) > 30:
                        if interval not in interval_errors:
                            interval_errors[interval] = []
                        interval_errors[interval].append((i+1, curr_error))
                except:
                    continue
        
        # Find problematic intervals
        for interval, errors in interval_errors.items():
            if len(errors) >= 2:  # Multiple errors on same interval type
                avg_error = np.mean([e[1] for e in errors])
                interval_name = self._interval_to_name(interval)
                
                patterns.append({
                    "type": "interval_specific_error",
                    "severity": "medium",
                    "description": f"Consistent errors on {interval_name} intervals ({avg_error:.1f}¢)",
                    "affected_notes": [e[0] for e in errors],
                    "recommendation": f"Practice {interval_name} intervals separately"
                })
        
        return patterns
    
    def _analyze_sequential_errors(self, note_details):
        """Analyze errors that occur in sequence"""
        patterns = []
        
        error_sequences = []
        current_sequence = []
        
        for i, note in enumerate(note_details):
            pitch_error = note.get('pitch_deviation_cents', 0)
            timing_error = note.get('timing_deviation_ms', 0)
            
            has_error = False
            if isinstance(pitch_error, (int, float)) and abs(pitch_error) > 30:
                has_error = True
            if isinstance(timing_error, (int, float)) and abs(timing_error) > 100:
                has_error = True
            
            if has_error:
                current_sequence.append(i+1)
            else:
                if len(current_sequence) >= 3:  # Sequence of 3+ errors
                    error_sequences.append(current_sequence.copy())
                current_sequence = []
        
        # Check final sequence
        if len(current_sequence) >= 3:
            error_sequences.append(current_sequence)
        
        for sequence in error_sequences:
            patterns.append({
                "type": "error_sequence",
                "severity": "high" if len(sequence) > 5 else "medium",
                "description": f"Sequential errors in notes {sequence[0]}-{sequence[-1]}",
                "affected_notes": sequence,
                "recommendation": "Break down this passage and practice slowly"
            })
        
        return patterns
    
    def detect_retry_attempts(self):
        """
        Detect sections where the performer likely stopped and restarted
        """
        retry_analysis = {
            "retry_sections": [],
            "correction_attempts": [],
            "performance_flow": "continuous"
        }
        
        note_details = self.performance_data.get("note_details", [])
        if len(note_details) < 3:
            return retry_analysis
        
        # Look for timing discontinuities that suggest restarts
        timing_jumps = []
        
        for i in range(1, len(note_details)):
            curr_time = note_details[i].get('detected_time', 0)
            prev_time = note_details[i-1].get('detected_time', 0)
            expected_time = note_details[i].get('expected_time', 0)
            prev_expected = note_details[i-1].get('expected_time', 0)
            
            if (isinstance(curr_time, (int, float)) and isinstance(prev_time, (int, float)) and
                isinstance(expected_time, (int, float)) and isinstance(prev_expected, (int, float))):
                
                actual_gap = curr_time - prev_time
                expected_gap = expected_time - prev_expected
                
                # Look for unusually long pauses or backward jumps
                if actual_gap > expected_gap * 3 and actual_gap > 2.0:  # Long pause
                    timing_jumps.append({
                        "type": "long_pause",
                        "location": f"After note {i}",
                        "duration": actual_gap,
                        "expected_duration": expected_gap,
                        "likely_cause": "Hesitation or preparation for difficult section"
                    })
                elif curr_time < prev_time - 0.5:  # Backward jump
                    timing_jumps.append({
                        "type": "restart",
                        "location": f"Note {i+1}",
                        "backward_jump": prev_time - curr_time,
                        "likely_cause": "Performer restarted from earlier point"
                    })
        
        retry_analysis["retry_sections"] = timing_jumps
        
        # Analyze performance flow
        if len(timing_jumps) == 0:
            retry_analysis["performance_flow"] = "continuous"
        elif len(timing_jumps) <= 2:
            retry_analysis["performance_flow"] = "mostly_continuous"
        else:
            retry_analysis["performance_flow"] = "fragmented"
        
        # Look for correction patterns
        corrections = self._detect_corrections(note_details)
        retry_analysis["correction_attempts"] = corrections
        
        return retry_analysis
    
    def _detect_corrections(self, note_details):
        """Detect potential corrections within the performance"""
        corrections = []
        
        for i in range(2, len(note_details)):
            # Look for patterns where accuracy suddenly improves after errors
            prev_prev_note = note_details[i-2]
            prev_note = note_details[i-1]
            curr_note = note_details[i]
            
            # Check if there was an error followed by improvement
            prev_error = self._calculate_note_error_score(prev_note)
            curr_accuracy = self._calculate_note_accuracy_score(curr_note)
            
            if prev_error > 0.7 and curr_accuracy > 0.8:  # Error followed by good performance
                corrections.append({
                    "location": f"Note {i+1}",
                    "type": "immediate_correction",
                    "description": "Accuracy improved immediately after error"
                })
        
        return corrections
    
    def analyze_performance_sections(self):
        """
        Analyze performance by musical sections/phrases
        """
        section_analysis = {
            "section_breakdown": [],
            "difficulty_assessment": {},
            "consistency_analysis": {}
        }
        
        note_details = self.performance_data.get("note_details", [])
        if len(note_details) < 4:
            return section_analysis
        
        # Divide performance into sections (every 4-8 notes)
        section_size = min(8, max(4, len(note_details) // 4))
        sections = []
        
        for i in range(0, len(note_details), section_size):
            section_notes = note_details[i:i+section_size]
            section_analysis_data = self._analyze_section(section_notes, i+1, i+len(section_notes))
            sections.append(section_analysis_data)
        
        section_analysis["section_breakdown"] = sections
        
        # Overall difficulty assessment
        section_analysis["difficulty_assessment"] = self._assess_section_difficulties(sections)
        
        # Consistency analysis
        section_analysis["consistency_analysis"] = self._analyze_section_consistency(sections)
        
        return section_analysis
    
    def _analyze_section(self, section_notes, start_note, end_note):
        """Analyze a specific section of the performance"""
        section_data = {
            "range": f"Notes {start_note}-{end_note}",
            "note_count": len(section_notes),
            "pitch_accuracy": 0,
            "timing_accuracy": 0,
            "overall_score": 0,
            "main_issues": [],
            "strengths": []
        }
        
        pitch_errors = []
        timing_errors = []
        
        for note in section_notes:
            pitch_dev = note.get('pitch_deviation_cents', 0)
            timing_dev = note.get('timing_deviation_ms', 0)
            
            if isinstance(pitch_dev, (int, float)):
                pitch_errors.append(abs(pitch_dev))
            if isinstance(timing_dev, (int, float)):
                timing_errors.append(abs(timing_dev))
        
        # Calculate section metrics
        if pitch_errors:
            avg_pitch_error = np.mean(pitch_errors)
            section_data["pitch_accuracy"] = max(0, 100 - avg_pitch_error / 2)
            
            if avg_pitch_error < 20:
                section_data["strengths"].append("Excellent pitch accuracy")
            elif avg_pitch_error > 50:
                section_data["main_issues"].append("Pitch inaccuracy")
        
        if timing_errors:
            avg_timing_error = np.mean(timing_errors)
            section_data["timing_accuracy"] = max(0, 100 - avg_timing_error / 10)
            
            if avg_timing_error < 50:
                section_data["strengths"].append("Good timing precision")
            elif avg_timing_error > 100:
                section_data["main_issues"].append("Timing issues")
        
        # Overall score
        section_data["overall_score"] = (section_data["pitch_accuracy"] + section_data["timing_accuracy"]) / 2
        
        return section_data
    
    def analyze_musical_interpretation(self):
        """
        Analyze musical interpretation aspects
        """
        interpretation_analysis = {
            "tempo_variations": {},
            "dynamic_expression": {},
            "phrasing_analysis": {},
            "musical_character": {}
        }
        
        note_details = self.performance_data.get("note_details", [])
        
        # Tempo variation analysis
        interpretation_analysis["tempo_variations"] = self._analyze_tempo_variations(note_details)
        
        # Phrasing analysis
        interpretation_analysis["phrasing_analysis"] = self._analyze_phrasing(note_details)
        
        # Musical character assessment
        interpretation_analysis["musical_character"] = self._assess_musical_character(note_details)
        
        return interpretation_analysis
    
    def _analyze_tempo_variations(self, note_details):
        """Analyze tempo variations for musical expression"""
        tempo_analysis = {
            "tempo_stability": 0,
            "expressive_variations": [],
            "problematic_variations": []
        }
        
        if len(note_details) < 3:
            return tempo_analysis
        
        # Calculate local tempo for each note
        local_tempos = []
        for i in range(1, len(note_details)):
            curr_time = note_details[i].get('detected_time', 0)
            prev_time = note_details[i-1].get('detected_time', 0)
            
            if isinstance(curr_time, (int, float)) and isinstance(prev_time, (int, float)):
                note_duration = curr_time - prev_time
                if note_duration > 0:
                    local_tempo = 60.0 / (note_duration * 4)  # Approximate BPM
                    local_tempos.append(local_tempo)
        
        if local_tempos:
            tempo_std = np.std(local_tempos)
            tempo_mean = np.mean(local_tempos)
            
            tempo_analysis["tempo_stability"] = max(0, 100 - (tempo_std / tempo_mean) * 100)
            
            # Look for intentional vs problematic variations
            for i, tempo in enumerate(local_tempos):
                deviation = abs(tempo - tempo_mean) / tempo_mean
                if deviation > 0.2:  # 20% deviation
                    if deviation < 0.4:  # Moderate deviation might be expressive
                        tempo_analysis["expressive_variations"].append({
                            "location": f"Note {i+2}",
                            "tempo": tempo,
                            "deviation": f"{deviation*100:.1f}%"
                        })
                    else:  # Large deviation likely problematic
                        tempo_analysis["problematic_variations"].append({
                            "location": f"Note {i+2}",
                            "tempo": tempo,
                            "deviation": f"{deviation*100:.1f}%"
                        })
        
        return tempo_analysis
    
    def generate_improvement_recommendations(self):
        """
        Generate specific improvement recommendations based on analysis
        """
        recommendations = {
            "priority_areas": [],
            "practice_exercises": [],
            "technical_focus": [],
            "musical_development": []
        }
        
        # Analyze mistake patterns to generate recommendations
        for pattern in self.mistake_patterns:
            if pattern.get("severity") == "high":
                recommendations["priority_areas"].append({
                    "area": pattern["type"],
                    "description": pattern["description"],
                    "action": pattern.get("recommendation", "Focus practice on this area")
                })
        
        # Technical recommendations
        note_details = self.performance_data.get("note_details", [])
        pitch_errors = [abs(n.get('pitch_deviation_cents', 0)) for n in note_details 
                       if isinstance(n.get('pitch_deviation_cents'), (int, float))]
        timing_errors = [abs(n.get('timing_deviation_ms', 0)) for n in note_details 
                        if isinstance(n.get('timing_deviation_ms'), (int, float))]
        
        if pitch_errors and np.mean(pitch_errors) > 30:
            recommendations["technical_focus"].append({
                "area": "Pitch Accuracy",
                "exercises": ["Long tone exercises", "Interval training", "Tuning exercises"]
            })
        
        if timing_errors and np.mean(timing_errors) > 75:
            recommendations["technical_focus"].append({
                "area": "Rhythm Precision",
                "exercises": ["Metronome practice", "Subdivision exercises", "Clapping rhythms"]
            })
        
        return recommendations
    
    # Utility methods
    def _calculate_note_error_score(self, note):
        """Calculate overall error score for a note (0-1, higher = more error)"""
        pitch_error = note.get('pitch_deviation_cents', 0)
        timing_error = note.get('timing_deviation_ms', 0)
        
        pitch_score = 0
        timing_score = 0
        
        if isinstance(pitch_error, (int, float)):
            pitch_score = min(1.0, abs(pitch_error) / 100)  # Normalize to 0-1
        
        if isinstance(timing_error, (int, float)):
            timing_score = min(1.0, abs(timing_error) / 200)  # Normalize to 0-1
        
        return (pitch_score + timing_score) / 2
    
    def _calculate_note_accuracy_score(self, note):
        """Calculate accuracy score for a note (0-1, higher = more accurate)"""
        return 1.0 - self._calculate_note_error_score(note)
    
    def _interval_to_name(self, semitones):
        """Convert semitone interval to musical name"""
        interval_names = {
            0: "Unison", 1: "Minor 2nd", 2: "Major 2nd", 3: "Minor 3rd",
            4: "Major 3rd", 5: "Perfect 4th", 6: "Tritone", 7: "Perfect 5th",
            8: "Minor 6th", 9: "Major 6th", 10: "Minor 7th", 11: "Major 7th",
            12: "Octave"
        }
        return interval_names.get(abs(semitones), f"{abs(semitones)} semitones")
    
    def _analyze_tempo_instability(self, note_details):
        """Analyze tempo instability patterns"""
        patterns = []
        # Implementation for tempo instability analysis
        return patterns
    
    def _analyze_rhythmic_patterns(self, note_details):
        """Analyze rhythmic pattern errors"""
        patterns = []
        # Implementation for rhythmic pattern analysis
        return patterns
    
    def _analyze_harmonic_patterns(self, note_details):
        """Analyze harmonic/melodic pattern errors"""
        patterns = []
        # Implementation for harmonic pattern analysis
        return patterns
    
    def _analyze_phrasing(self, note_details):
        """Analyze musical phrasing"""
        phrasing_analysis = {}
        # Implementation for phrasing analysis
        return phrasing_analysis
    
    def _assess_musical_character(self, note_details):
        """Assess musical character and expression"""
        character_assessment = {}
        # Implementation for musical character assessment
        return character_assessment
    
    def _assess_section_difficulties(self, sections):
        """Assess difficulty levels of different sections"""
        difficulty_assessment = {}
        # Implementation for section difficulty assessment
        return difficulty_assessment
    
    def _analyze_section_consistency(self, sections):
        """Analyze consistency across sections"""
        consistency_analysis = {}
        # Implementation for section consistency analysis
        return consistency_analysis

def create_performance_diff_report(reference_data, performance_data, output_path=None):
    """
    Create a comprehensive performance difference report
    """
    analyzer = PerformanceDiffAnalyzer(reference_data, performance_data)
    analysis = analyzer.analyze_comprehensive_diff()
    
    # Generate detailed report
    report = {
        "timestamp": np.datetime64('now').isoformat(),
        "performance_summary": {
            "total_notes": len(performance_data.get("note_details", [])),
            "major_issues_detected": len([p for p in analysis["mistake_patterns"] if p.get("severity") == "high"]),
            "retry_attempts": len(analysis["retry_analysis"]["retry_sections"]),
            "overall_consistency": analysis["retry_analysis"]["performance_flow"]
        },
        "detailed_analysis": analysis,
        "improvement_plan": analysis["recommendations"]
    }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    return report

if __name__ == "__main__":
    # Example usage
    print("Performance Diff Analyzer - Testing Module")
    
    # This would normally be called with real performance data
    sample_performance = {
        "note_details": [
            {"note_index": 1, "pitch_deviation_cents": 25, "timing_deviation_ms": -30, "detected_time": 0.5},
            {"note_index": 2, "pitch_deviation_cents": -15, "timing_deviation_ms": 80, "detected_time": 0.8},
            {"note_index": 3, "pitch_deviation_cents": 45, "timing_deviation_ms": -60, "detected_time": 1.1},
        ]
    }
    
    analyzer = PerformanceDiffAnalyzer({}, sample_performance)
    patterns = analyzer.detect_mistake_patterns()
    
    print(f"Detected {len(patterns)} mistake patterns")
    for pattern in patterns:
        print(f"- {pattern['type']}: {pattern['description']}")

#!/usr/bin/env python3
"""
Sheet Music Visualization and Diff Display Module

This module adds visual sheet music representation and difference highlighting
to the ABRSM AI Music Feedback System.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import librosa

class SheetMusicVisualizer:
    def __init__(self):
        # Staff line positions (treble clef)
        self.staff_lines = [0, 0.5, 1, 1.5, 2]  # E, G, B, D, F
        self.line_to_note = {
            # Lines
            0: 'E4',    # Bottom line
            0.5: 'F4',  # Below first space
            1: 'G4',    # Second line
            1.5: 'A4',  # Second space
            2: 'B4',    # Third line
            2.5: 'C5',  # Third space
            3: 'D5',    # Fourth line
            3.5: 'E5',  # Fourth space
            4: 'F5',    # Top line
        }
        
        # Reverse mapping
        self.note_to_line = {v: k for k, v in self.line_to_note.items()}
        
        # Add more notes for extended range
        extended_notes = {
            'C4': -0.5, 'D4': 0.25, 'E4': 0, 'F4': 0.5, 'G4': 1, 'A4': 1.5, 
            'B4': 2, 'C5': 2.5, 'D5': 3, 'E5': 3.5, 'F5': 4, 'G5': 4.5
        }
        self.note_to_line.update(extended_notes)

    def midi_to_staff_position(self, midi_note):
        """Convert MIDI note number to staff line position"""
        note_name = librosa.midi_to_note(midi_note)
        return self.note_to_line.get(note_name, 1)  # Default to G4 if not found

    def create_sheet_music_visualization(self, reference_melody, performance_data, 
                                       output_path="sheet_music_diff.png", 
                                       time_signature=(4, 4)):
        """
        Create a sheet music visualization showing reference vs performance with diffs
        
        Args:
            reference_melody: List of reference notes with pitch and duration
            performance_data: Analysis data from performance
            output_path: Where to save the visualization
            time_signature: Tuple of (beats_per_measure, note_value)
        """
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('Sheet Music Analysis: Reference vs Performance', fontsize=16, fontweight='bold')
        
        # Draw reference sheet music
        self._draw_staff_and_notes(ax1, reference_melody, "Reference (Expected)", 
                                 time_signature, is_reference=True)
        
        # Draw performance sheet music
        self._draw_staff_and_notes(ax2, performance_data.get('note_details', []), 
                                 "Performance (Detected)", time_signature, is_reference=False)
        
        # Draw difference visualization
        self._draw_difference_analysis(ax3, reference_melody, performance_data.get('note_details', []),
                                     time_signature)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Sheet music visualization saved to: {output_path}")
        return output_path

    def _draw_staff_and_notes(self, ax, notes_data, title, time_signature, is_reference=True):
        """Draw staff lines and notes"""
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Draw staff lines
        staff_width = 12
        for line_pos in self.staff_lines:
            ax.plot([0, staff_width], [line_pos, line_pos], 'k-', linewidth=1)
        
        # Draw time signature
        ax.text(-0.8, 1.8, str(time_signature[0]), fontsize=20, ha='center', va='center', fontweight='bold')
        ax.text(-0.8, 0.8, str(time_signature[1]), fontsize=20, ha='center', va='center', fontweight='bold')
        
        # Draw treble clef (simplified)
        ax.text(-0.5, 1, '𝄞', fontsize=30, ha='center', va='center')
        
        # Calculate measures and beat positions
        beats_per_measure = time_signature[0]
        current_beat = 0
        current_measure = 0
        x_position = 0.5
        
        # Draw measure lines
        for measure in range(int(staff_width / 3) + 1):
            measure_x = measure * 3
            if measure_x <= staff_width:
                ax.plot([measure_x, measure_x], [-0.5, 4.5], 'k-', linewidth=2)
        
        # Draw notes
        for i, note_data in enumerate(notes_data):
            if is_reference:
                # Reference data format
                midi_note = note_data['pitch']
                duration = note_data['duration']
                note_name = librosa.midi_to_note(midi_note)
            else:
                # Performance data format
                if 'detected_pitch' in note_data and note_data['detected_pitch'] != 'MISSED':
                    note_name = note_data['detected_pitch']
                    midi_note = librosa.note_to_midi(note_name)
                    duration = 0.25  # Default duration for detected notes
                else:
                    # Skip missed notes
                    continue
            
            # Calculate staff position
            staff_pos = self.midi_to_staff_position(midi_note)
            
            # Draw note head
            note_color = 'black' if is_reference else 'blue'
            circle = plt.Circle((x_position, staff_pos), 0.08, color=note_color, zorder=10)
            ax.add_patch(circle)
            
            # Add note name below staff
            ax.text(x_position, -0.8, note_name, ha='center', va='center', 
                   fontsize=10, color=note_color)
            
            # Draw stem (simplified)
            if staff_pos < 2:  # Stem up
                ax.plot([x_position + 0.08, x_position + 0.08], 
                       [staff_pos, staff_pos + 1.5], color=note_color, linewidth=2)
            else:  # Stem down
                ax.plot([x_position - 0.08, x_position - 0.08], 
                       [staff_pos, staff_pos - 1.5], color=note_color, linewidth=2)
            
            # Move to next position
            x_position += 0.8
            if x_position > staff_width:
                break
        
        # Set axis properties
        ax.set_xlim(-1, staff_width + 0.5)
        ax.set_ylim(-1.5, 5)
        ax.set_aspect('equal')
        ax.axis('off')

    def _draw_difference_analysis(self, ax, reference_melody, performance_notes, time_signature):
        """Draw a visual representation of timing and pitch differences"""
        ax.set_title("Difference Analysis", fontsize=14, fontweight='bold', pad=20)
        
        # Create a timeline visualization
        max_notes = max(len(reference_melody), len(performance_notes))
        
        # Draw timeline
        timeline_y = 2
        ax.plot([0, max_notes + 1], [timeline_y, timeline_y], 'k-', linewidth=2)
        
        # Analyze differences
        for i in range(min(len(reference_melody), len(performance_notes))):
            ref_note = reference_melody[i]
            perf_note = performance_notes[i] if i < len(performance_notes) else None
            
            x_pos = i + 0.5
            
            if perf_note and 'timing_deviation_ms' in perf_note and perf_note['timing_deviation_ms'] != 'MISSED':
                # Timing difference
                timing_dev = perf_note['timing_deviation_ms']
                if isinstance(timing_dev, (int, float)):
                    timing_color = 'red' if abs(timing_dev) > 100 else 'orange' if abs(timing_dev) > 50 else 'green'
                    timing_height = min(abs(timing_dev) / 200, 1)  # Normalize to 0-1
                    
                    # Draw timing bar
                    rect = Rectangle((x_pos - 0.1, timeline_y + 0.1), 0.2, timing_height, 
                                   facecolor=timing_color, alpha=0.7, label='Timing' if i == 0 else "")
                    ax.add_patch(rect)
                
                # Pitch difference
                pitch_dev = perf_note.get('pitch_deviation_cents', 0)
                if isinstance(pitch_dev, (int, float)):
                    pitch_color = 'red' if abs(pitch_dev) > 50 else 'orange' if abs(pitch_dev) > 20 else 'green'
                    pitch_height = min(abs(pitch_dev) / 100, 1)  # Normalize to 0-1
                    
                    # Draw pitch bar
                    rect = Rectangle((x_pos + 0.1, timeline_y + 0.1), 0.2, pitch_height, 
                                   facecolor=pitch_color, alpha=0.7, label='Pitch' if i == 0 else "")
                    ax.add_patch(rect)
            else:
                # Missed note
                rect = Rectangle((x_pos - 0.15, timeline_y + 0.1), 0.3, 0.5, 
                               facecolor='darkred', alpha=0.8, label='Missed' if i == 0 else "")
                ax.add_patch(rect)
            
            # Add note number
            ax.text(x_pos, timeline_y - 0.3, str(i + 1), ha='center', va='center', fontsize=10)
        
        # Add legend with colored circles
        ax.text(0, 3.5, "Timing/Pitch Accuracy:", fontsize=12, fontweight='bold')
        
        # Good accuracy - green circle
        circle1 = plt.Circle((0.2, 3.2), 0.05, color='green', clip_on=False)
        ax.add_patch(circle1)
        ax.text(0.4, 3.2, "Good (±50ms/±20¢)", fontsize=10, va='center')
        
        # Fair accuracy - orange circle
        circle2 = plt.Circle((0.2, 2.9), 0.05, color='orange', clip_on=False)
        ax.add_patch(circle2)
        ax.text(0.4, 2.9, "Fair (±100ms/±50¢)", fontsize=10, va='center')
        
        # Poor accuracy - red circle
        circle3 = plt.Circle((0.2, 2.6), 0.05, color='red', clip_on=False)
        ax.add_patch(circle3)
        ax.text(0.4, 2.6, "Needs Work (>100ms/>50¢)", fontsize=10, va='center')
        
        ax.set_xlim(-0.5, max_notes + 0.5)
        ax.set_ylim(0, 4)
        ax.axis('off')

# Integration function for the main script
def create_visual_analysis(reference_melody, performance_report, time_signature=(4, 4)):
    """
    Create visual sheet music analysis
    
    Args:
        reference_melody: List of reference notes
        performance_report: JSON analysis report
        time_signature: Tuple of (beats_per_measure, note_value)
    
    Returns:
        Path to generated visualization
    """
    import json
    
    visualizer = SheetMusicVisualizer()
    
    # Parse performance report if it's a string
    if isinstance(performance_report, str):
        performance_data = json.loads(performance_report)
    else:
        performance_data = performance_report
    
    output_path = "sheet_music_analysis.png"
    return visualizer.create_sheet_music_visualization(
        reference_melody, performance_data, output_path, time_signature
    )

if __name__ == "__main__":
    # Test the visualization
    test_melody = [
        {'pitch': 60, 'duration': 0.25}, {'pitch': 60, 'duration': 0.25},
        {'pitch': 67, 'duration': 0.25}, {'pitch': 67, 'duration': 0.25},
    ]
    
    test_performance = {
        'note_details': [
            {'detected_pitch': 'C4', 'timing_deviation_ms': 50, 'pitch_deviation_cents': 10},
            {'detected_pitch': 'C4', 'timing_deviation_ms': -30, 'pitch_deviation_cents': -5},
            {'detected_pitch': 'G4', 'timing_deviation_ms': 120, 'pitch_deviation_cents': 40},
            {'detected_pitch': 'G4', 'timing_deviation_ms': 80, 'pitch_deviation_cents': -20},
        ]
    }
    
    create_visual_analysis(test_melody, test_performance)
    print("Test visualization created!")

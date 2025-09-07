#!/usr/bin/env python3
"""
Enhanced Sheet Music Visualizer using music21
Properly generates sheet music with performance differences marked
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

try:
    from music21 import stream, note, pitch, duration, meter, tempo, key, bar, clef, metadata
    from music21.midi import translate
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("music21 not available - using fallback visualization")

class Music21SheetVisualizer:
    def __init__(self):
        self.staff_height = 4
        self.line_spacing = 0.5
        
    def create_sheet_music_from_melody(self, melody_data, performance_data=None, output_path="visualizations/sheet_music_analysis.png"):
        """Create sheet music using music21 with performance differences highlighted"""
        if not MUSIC21_AVAILABLE:
            return self._create_fallback_visualization(melody_data, performance_data, output_path)
        
        try:
            # Create a new music21 stream
            score = stream.Stream()
            
            # Add metadata
            score.append(metadata.Metadata())
            score.metadata.title = 'Performance Analysis'
            score.metadata.composer = 'ABRSM AI Analysis'
            
            # Add time signature (4/4 default)
            score.append(meter.TimeSignature('4/4'))
            
            # Add key signature (C major default)
            score.append(key.KeySignature(0))
            
            # Add tempo
            score.append(tempo.TempoIndication(number=120))
            
            # Add treble clef
            score.append(clef.TrebleClef())
            
            # Convert melody to music21 notes
            for i, note_data in enumerate(melody_data):
                # Create note
                n = note.Note()
                n.pitch.midi = note_data['pitch']
                
                # Set duration (convert from our format to music21 format)
                dur_quarters = note_data.get('duration', 0.25) * 4  # Convert to quarter notes
                n.duration = duration.Duration(quarterLength=dur_quarters)
                
                # Color coding based on performance
                if performance_data and 'note_details' in performance_data:
                    note_details = performance_data['note_details']
                    if i < len(note_details):
                        detail = note_details[i]
                        if detail.get('timing_deviation_ms') == 'MISSED':
                            n.style.color = 'red'
                        elif abs(detail.get('timing_deviation_ms', 0)) > 100:
                            n.style.color = 'orange'
                        elif abs(detail.get('pitch_deviation_cents', 0)) > 50:
                            n.style.color = 'blue'
                        else:
                            n.style.color = 'green'
                
                score.append(n)
            
            # Generate the sheet music image
            if hasattr(score, 'write'):
                # Try to write as PNG (requires additional dependencies)
                try:
                    score.write('musicxml.png', fp=output_path)
                    print(f"✓ Sheet music created with music21: {output_path}")
                    return output_path
                except:
                    # Fallback to MIDI then convert
                    midi_path = output_path.replace('.png', '.mid')
                    score.write('midi', fp=midi_path)
                    print(f"✓ MIDI created, converting to sheet music visualization...")
                    return self._create_visualization_from_midi(midi_path, performance_data, output_path)
            else:
                return self._create_fallback_visualization(melody_data, performance_data, output_path)
                
        except Exception as e:
            print(f"Error with music21 sheet generation: {e}")
            return self._create_fallback_visualization(melody_data, performance_data, output_path)
    
    def _create_visualization_from_midi(self, midi_path, performance_data, output_path):
        """Create visualization from MIDI file"""
        try:
            # Load MIDI with music21
            score = translate.midiFilePathToStream(midi_path)
            
            # Create matplotlib visualization
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Draw staff lines
            staff_y = 2
            for i in range(5):
                ax.axhline(y=staff_y + i * 0.5, color='black', linewidth=1)
            
            # Draw notes
            x_pos = 1
            note_width = 0.8
            
            notes = score.flat.notes
            for i, n in enumerate(notes):
                if hasattr(n, 'pitch'):
                    # Calculate y position based on pitch
                    midi_num = n.pitch.midi
                    # C4 (middle C) = 60, corresponds to staff_y + 1 (below staff)
                    y_pos = staff_y + (midi_num - 60) * 0.25
                    
                    # Color based on performance data
                    color = 'black'
                    if performance_data and 'note_details' in performance_data:
                        note_details = performance_data['note_details']
                        if i < len(note_details):
                            detail = note_details[i]
                            if detail.get('timing_deviation_ms') == 'MISSED':
                                color = 'red'
                            elif abs(detail.get('timing_deviation_ms', 0)) > 100:
                                color = 'orange'
                            elif abs(detail.get('pitch_deviation_cents', 0)) > 50:
                                color = 'blue'
                            else:
                                color = 'green'
                    
                    # Draw note
                    circle = plt.Circle((x_pos, y_pos), 0.15, color=color, fill=True)
                    ax.add_patch(circle)
                    
                    # Add note name
                    ax.text(x_pos, y_pos - 0.8, n.pitch.name, ha='center', va='top', fontsize=8)
                    
                    x_pos += note_width
            
            # Set up the plot
            ax.set_xlim(0, x_pos + 1)
            ax.set_ylim(staff_y - 1, staff_y + 5)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Add title
            title = "Sheet Music Analysis"
            if performance_data and 'metadata' in performance_data:
                title = f"Sheet Music Analysis: {performance_data['metadata'].get('piece', 'Unknown')}"
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            # Add legend
            legend_elements = [
                mpatches.Patch(color='green', label='Correct'),
                mpatches.Patch(color='blue', label='Pitch Error'),
                mpatches.Patch(color='orange', label='Timing Error'),
                mpatches.Patch(color='red', label='Missed Note')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Sheet music visualization created: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error creating visualization from MIDI: {e}")
            return self._create_fallback_visualization([], performance_data, output_path)
    
    def _create_fallback_visualization(self, melody_data, performance_data, output_path):
        """Fallback visualization without music21"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Draw staff
        staff_y = 2
        for i in range(5):
            ax.axhline(y=staff_y + i * 0.5, color='black', linewidth=1)
        
        # Draw clef symbol (approximation)
        ax.text(0.2, staff_y + 1, '𝄞', fontsize=40, va='center')
        
        # Draw notes if melody data available
        if melody_data:
            x_pos = 1
            note_width = 0.6
            
            for i, note_data in enumerate(melody_data):
                # Calculate y position based on MIDI pitch
                midi_pitch = note_data['pitch']
                # Map MIDI to staff position (C4=60 at staff_y+1)
                y_pos = staff_y + (midi_pitch - 60) * 0.25
                
                # Color based on performance
                color = 'black'
                if performance_data and 'note_details' in performance_data:
                    note_details = performance_data['note_details']
                    if i < len(note_details):
                        detail = note_details[i]
                        if detail.get('timing_deviation_ms') == 'MISSED':
                            color = 'red'
                        elif isinstance(detail.get('timing_deviation_ms'), (int, float)) and abs(detail.get('timing_deviation_ms', 0)) > 100:
                            color = 'orange'
                        elif isinstance(detail.get('pitch_deviation_cents'), (int, float)) and abs(detail.get('pitch_deviation_cents', 0)) > 50:
                            color = 'blue'
                        else:
                            color = 'green'
                
                # Draw note head
                circle = plt.Circle((x_pos, y_pos), 0.12, color=color, fill=True)
                ax.add_patch(circle)
                
                # Draw stem
                stem_height = 1.5 if y_pos < staff_y + 2 else -1.5
                ax.plot([x_pos + 0.12, x_pos + 0.12], [y_pos, y_pos + stem_height], color=color, linewidth=2)
                
                # Add ledger lines if needed
                if y_pos < staff_y:
                    for ledger_y in np.arange(staff_y - 0.5, y_pos - 0.25, -0.5):
                        ax.plot([x_pos - 0.2, x_pos + 0.2], [ledger_y, ledger_y], color='black', linewidth=1)
                elif y_pos > staff_y + 2:
                    for ledger_y in np.arange(staff_y + 2.5, y_pos + 0.25, 0.5):
                        ax.plot([x_pos - 0.2, x_pos + 0.2], [ledger_y, ledger_y], color='black', linewidth=1)
                
                x_pos += note_width
        
        # Setup plot
        ax.set_xlim(0, max(10, x_pos + 1))
        ax.set_ylim(staff_y - 2, staff_y + 5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title
        title = "Sheet Music Analysis (Fallback Mode)"
        if performance_data and 'metadata' in performance_data:
            title = f"Sheet Music Analysis: {performance_data['metadata'].get('piece', 'Unknown')}"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='green', label='Correct'),
            mpatches.Patch(color='blue', label='Pitch Error'), 
            mpatches.Patch(color='orange', label='Timing Error'),
            mpatches.Patch(color='red', label='Missed Note')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Fallback sheet music created: {output_path}")
        return output_path

# Main function for compatibility
def create_visual_analysis(reference_melody, performance_data, time_signature=(4, 4), output_path="visualizations/sheet_music_analysis.png"):
    """Create sheet music visualization with performance analysis"""
    visualizer = Music21SheetVisualizer()
    return visualizer.create_sheet_music_from_melody(reference_melody, performance_data, output_path)

if __name__ == "__main__":
    # Test with sample data
    test_melody = [
        {'pitch': 60, 'duration': 0.25},  # C4
        {'pitch': 62, 'duration': 0.25},  # D4  
        {'pitch': 64, 'duration': 0.25},  # E4
        {'pitch': 65, 'duration': 0.25},  # F4
    ]
    
    test_performance = {
        'metadata': {'piece': 'Test Piece'},
        'note_details': [
            {'timing_deviation_ms': 0, 'pitch_deviation_cents': 0},
            {'timing_deviation_ms': 50, 'pitch_deviation_cents': 25},
            {'timing_deviation_ms': 'MISSED', 'pitch_deviation_cents': 'MISSED'},
            {'timing_deviation_ms': 200, 'pitch_deviation_cents': 150},
        ]
    }
    
    create_visual_analysis(test_melody, test_performance)

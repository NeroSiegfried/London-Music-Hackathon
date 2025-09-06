#!/usr/bin/env python3
"""
Interactive Sheet Music Visualizer for ABRSM AI Music Feedback
Complete implementation with proper note matching, measure divisions, and note selection
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import librosa

class InteractiveSheetMusic:
    def __init__(self, parent_widget, on_note_click=None):
        self.parent = parent_widget
        self.on_note_click = on_note_click or self._default_note_click
        
        # Create the visualization
        self.fig = Figure(figsize=(16, 10), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, parent_widget)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Connect click events
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        
        # Storage for clickable note areas
        self.note_click_areas = []
        self.current_piece = None
        self.current_analysis = None
        self.view_mode = "traditional"  # "traditional" or "grid"
        
    def update_sheet_music(self, piece_info, analysis=None, view_mode="traditional"):
        """Update the sheet music display with piece and analysis data"""
        self.current_piece = piece_info
        self.current_analysis = analysis
        self.view_mode = view_mode
        
        print(f"🎼 Updating sheet music with analysis: {analysis is not None}")
        if analysis:
            print(f"   Analysis keys: {list(analysis.keys())}")
            if 'standard_analysis' in analysis:
                std_analysis = analysis['standard_analysis']
                print(f"   Standard analysis keys: {list(std_analysis.keys())}")
                note_details = std_analysis.get('note_details', [])
                print(f"   Found {len(note_details)} note details")
        
        self.fig.clear()
        
        if view_mode == "traditional":
            self._create_traditional_notation()
        else:
            self._create_grid_notation()
        
        self.canvas.draw()
    
    def _create_traditional_notation(self):
        """Create traditional sheet music notation"""
        # Create three subplots: Reference, Performance, Comparison
        ax_ref = self.fig.add_subplot(3, 1, 1)
        ax_perf = self.fig.add_subplot(3, 1, 2)
        ax_comp = self.fig.add_subplot(3, 1, 3)
        
        self._draw_traditional_reference_staff(ax_ref, "Reference Score")
        self._draw_traditional_performance_staff(ax_perf, "Your Performance")
        self._draw_comparison_view(ax_comp)
        
        self.fig.tight_layout()
    
    def _create_grid_notation(self):
        """Create FL Studio/grid-style notation"""
        ax_ref = self.fig.add_subplot(3, 1, 1)
        ax_perf = self.fig.add_subplot(3, 1, 2)
        ax_comp = self.fig.add_subplot(3, 1, 3)
        
        self._draw_grid_view(ax_ref, "Reference Score", reference=True)
        self._draw_grid_view(ax_perf, "Your Performance", reference=False)
        self._draw_grid_comparison(ax_comp)
        
        self.fig.tight_layout()
    
    def _draw_traditional_reference_staff(self, ax, title):
        """Draw traditional music staff with reference notes in proper 4/4 measures"""
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 10)
        
        # Draw staff lines
        staff_lines = [3, 4, 5, 6, 7]
        for line in staff_lines:
            ax.axhline(y=line, color='black', linewidth=1.5, alpha=0.8)
        
        # Draw treble clef and time signature
        ax.text(0.5, 5, 'G', fontsize=32, ha='center', va='center', fontweight='bold')
        ax.text(1.2, 6, '4', fontsize=16, ha='center', va='center', fontweight='bold')
        ax.text(1.2, 4, '4', fontsize=16, ha='center', va='center', fontweight='bold')
        
        if not self.current_piece:
            return
        
        # Clear click areas
        self.note_click_areas = []
        
        # Calculate proper measures based on actual note durations (4/4 time = 1.0 duration per measure)
        melody = self.current_piece['melody']
        measure_duration = 1.0  # 4/4 time signature = 1.0 total duration per measure
        
        # Calculate note positions based on cumulative durations
        current_x = 1.8  # Start position
        measure_start_x = current_x
        current_duration_in_measure = 0.0
        measure_width_base = 3.4
        measure_count = 0
        note_positions = []
        
        for i, note_info in enumerate(melody):
            duration = note_info.get('duration', 0.25)
            
            # Check if this note would exceed the measure duration
            if current_duration_in_measure + duration > measure_duration and current_duration_in_measure > 0:
                # Draw measure line
                measure_end_x = measure_start_x + measure_width_base
                ax.axvline(x=measure_end_x, color='black', linewidth=2, ymin=0.3, ymax=0.7)
                
                # Start new measure
                measure_count += 1
                measure_start_x = measure_end_x
                current_x = measure_start_x + 0.3  # Small offset from measure line
                current_duration_in_measure = 0.0
            
            # Calculate x position based on duration within measure
            # Give half notes more space than quarter notes
            if duration >= 0.5:  # Half note or longer
                note_width = 0.6
            else:  # Quarter note or shorter
                note_width = 0.4
                
            x_pos = current_x + note_width * 0.5
            note_positions.append((i, x_pos))
            
            current_x += note_width
            current_duration_in_measure += duration
        
        # Draw final measure line if needed
        if current_duration_in_measure > 0:
            final_measure_end = current_x + 0.3
            ax.axvline(x=final_measure_end, color='black', linewidth=2, ymin=0.3, ymax=0.7)
        
        # Draw reference notes with proper spacing based on duration
        for i, note_info in enumerate(melody):
            # Find x position from our calculated positions
            x_pos = None
            for note_idx, pos in note_positions:
                if note_idx == i:
                    x_pos = pos
                    break
            
            if x_pos is None:
                continue
            
            # Get note position on staff
            y_pos = self._midi_to_staff_position(note_info['pitch'])
            
            # Note color (blue for reference)
            note_color = 'lightblue'
            
            # Draw note based on duration
            duration = note_info.get('duration', 1.0)
            note_symbol = self._get_note_symbol(duration)
            
            # Draw note head
            if note_symbol['filled']:
                circle = plt.Circle((x_pos, y_pos), 0.12, color=note_color, 
                                  ec='black', linewidth=1.5, picker=True)
            else:
                circle = plt.Circle((x_pos, y_pos), 0.12, color='white', 
                                  ec='black', linewidth=2, picker=True)
            ax.add_patch(circle)
            
            # Draw note stem (if needed)
            if note_symbol['has_stem']:
                stem_direction = -1 if y_pos > 5 else 1
                stem_height = 1.2
                stem_x = x_pos + (0.1 if stem_direction == 1 else -0.1)
                stem_start_y = y_pos + (0.1 if stem_direction == 1 else -0.1)
                stem_end_y = stem_start_y + stem_direction * stem_height
                
                ax.plot([stem_x, stem_x], [stem_start_y, stem_end_y], 
                       color='black', linewidth=2)
                
                # Draw flags for eighth notes and shorter
                if note_symbol['flags'] > 0:
                    flag_y = stem_end_y
                    for f in range(note_symbol['flags']):
                        flag_points = np.array([[stem_x, stem_x + 0.3, stem_x], 
                                              [flag_y - f*0.3, flag_y - f*0.3 - 0.15, flag_y - f*0.3 - 0.25]])
                        ax.plot(flag_points[0], flag_points[1], color='black', linewidth=2)
            
            # Add ledger lines if needed
            self._add_ledger_lines(ax, x_pos, y_pos)
            
            # Store click area for reference notes
            self.note_click_areas.append({
                'note_index': i,
                'x': x_pos,
                'y': y_pos,
                'radius': 0.15,
                'axes': ax,
                'is_reference': True
            })
            
            # Add note number label
            ax.text(x_pos, 2.3, str(i + 1), ha='center', va='center', 
                   fontsize=8, fontweight='bold', alpha=0.7)
        
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_traditional_performance_staff(self, ax, title):
        """Draw traditional staff showing actual performance data"""
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 10)
        
        # Draw staff infrastructure
        staff_lines = [3, 4, 5, 6, 7]
        for line in staff_lines:
            ax.axhline(y=line, color='black', linewidth=1.5, alpha=0.8)
        
        # Draw treble clef and time signature
        ax.text(0.5, 5, 'G', fontsize=32, ha='center', va='center', fontweight='bold')
        ax.text(1.2, 6, '4', fontsize=16, ha='center', va='center', fontweight='bold')
        ax.text(1.2, 4, '4', fontsize=16, ha='center', va='center', fontweight='bold')
        
        if not self.current_analysis:
            ax.text(8, 5, "No performance data available", ha='center', va='center', 
                   fontsize=12, style='italic', alpha=0.7)
            ax.set_aspect('equal')
            ax.axis('off')
            return
        
        # Get analysis data
        analysis_data = self.current_analysis.get("standard_analysis", {})
        note_details = analysis_data.get("note_details", [])
        
        if not note_details:
            ax.text(8, 5, "No analysis data available", ha='center', va='center', 
                   fontsize=12, style='italic', alpha=0.7)
            ax.set_aspect('equal')
            ax.axis('off')
            return
        
        # Calculate measure layout based on durations (same as reference)
        melody = self.current_piece['melody']
        measure_duration = 1.0  # 4/4 time signature
        
        # Calculate note positions based on cumulative durations
        current_x = 1.8
        measure_start_x = current_x
        current_duration_in_measure = 0.0
        measure_width_base = 3.4
        note_positions = []
        
        for i, note_info in enumerate(melody):
            duration = note_info.get('duration', 0.25)
            
            # Check if this note would exceed the measure duration
            if current_duration_in_measure + duration > measure_duration and current_duration_in_measure > 0:
                # Draw measure line
                measure_end_x = measure_start_x + measure_width_base
                ax.axvline(x=measure_end_x, color='black', linewidth=2, ymin=0.3, ymax=0.7)
                
                # Start new measure
                measure_start_x = measure_end_x
                current_x = measure_start_x + 0.3
                current_duration_in_measure = 0.0
            
            # Calculate x position based on duration
            if duration >= 0.5:  # Half note or longer
                note_width = 0.6
            else:  # Quarter note or shorter
                note_width = 0.4
                
            x_pos = current_x + note_width * 0.5
            note_positions.append((i, x_pos))
            
            current_x += note_width
            current_duration_in_measure += duration
        
        # Draw final measure line if needed
        if current_duration_in_measure > 0:
            final_measure_end = current_x + 0.3
            ax.axvline(x=final_measure_end, color='black', linewidth=2, ymin=0.3, ymax=0.7)
        
        # Draw performance notes based on note_details
        for i, note_detail in enumerate(note_details):
            # Find x position from our calculated positions
            x_pos = None
            for note_idx, pos in note_positions:
                if note_idx == i:
                    x_pos = pos
                    break
            
            if x_pos is None:
                # Fallback positioning if not found
                x_pos = 2.0 + i * 0.8
            
            actual_pitch = note_detail.get('actual_pitch')
            note_type = note_detail.get('note_type', 'matched')
            
            if actual_pitch and actual_pitch not in ['MISSED', 'N/A']:
                try:
                    # Convert pitch to staff position
                    pitch_hz = librosa.note_to_hz(actual_pitch)
                    pitch_midi = librosa.hz_to_midi(pitch_hz)
                    y_pos = self._midi_to_staff_position(pitch_midi)
                    
                    # Color based on note type and accuracy
                    if note_type == 'matched':
                        pitch_dev = note_detail.get('pitch_deviation_cents', 0)
                        if isinstance(pitch_dev, (int, float)) and abs(pitch_dev) < 20:
                            note_color = 'green'
                        else:
                            note_color = 'yellow'
                    elif note_type == 'extra':
                        note_color = 'orange'
                    else:
                        note_color = 'red'
                    
                    # Draw note
                    circle = plt.Circle((x_pos, y_pos), 0.12, color=note_color, 
                                      ec='black', linewidth=1.5, picker=True)
                    ax.add_patch(circle)
                    
                    # Draw stem
                    stem_direction = -1 if y_pos > 5 else 1
                    stem_height = 1.2
                    stem_x = x_pos + (0.1 if stem_direction == 1 else -0.1)
                    stem_start_y = y_pos + (0.1 if stem_direction == 1 else -0.1)
                    stem_end_y = stem_start_y + stem_direction * stem_height
                    
                    ax.plot([stem_x, stem_x], [stem_start_y, stem_end_y], 
                           color='black', linewidth=2)
                    
                    # Add ledger lines
                    self._add_ledger_lines(ax, x_pos, y_pos)
                    
                    # Store click area
                    self.note_click_areas.append({
                        'note_index': i,
                        'x': x_pos,
                        'y': y_pos,
                        'radius': 0.15,
                        'axes': ax,
                        'is_performance': True
                    })
                    
                    # Add label
                    ax.text(x_pos, 2.3, f"P{i + 1}", ha='center', va='center', 
                           fontsize=8, fontweight='bold', alpha=0.7)
                    
                except Exception as e:
                    print(f"Error drawing performance note {i}: {e}")
                    continue
            else:
                # Draw X for missed note
                expected_pitch = note_detail.get('expected_pitch')
                if expected_pitch and expected_pitch != 'N/A':
                    try:
                        pitch_hz = librosa.note_to_hz(expected_pitch)
                        pitch_midi = librosa.hz_to_midi(pitch_hz)
                        y_pos = self._midi_to_staff_position(pitch_midi)
                        
                        # Draw X
                        ax.plot([x_pos - 0.1, x_pos + 0.1], [y_pos - 0.1, y_pos + 0.1], 
                               'r-', linewidth=3)
                        ax.plot([x_pos - 0.1, x_pos + 0.1], [y_pos + 0.1, y_pos - 0.1], 
                               'r-', linewidth=3)
                    except Exception:
                        continue
        
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_comparison_view(self, ax):
        """Draw comparison view showing alignment and mistakes"""
        ax.set_title("Analysis Summary", fontsize=14, fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        
        if not self.current_analysis:
            ax.text(5, 3, "No analysis data for comparison", ha='center', va='center', 
                   fontsize=12, style='italic', alpha=0.7)
            ax.axis('off')
            return
        
        # Display analysis summary
        analysis_data = self.current_analysis.get("standard_analysis", {})
        overall = analysis_data.get("overall_assessment", {})
        
        summary_text = []
        if "completion_rate" in overall:
            summary_text.append(f"Completion: {overall['completion_rate']}%")
        if "missed_notes" in overall:
            summary_text.append(f"Missed Notes: {overall['missed_notes']}")
        if "extra_notes" in overall:
            summary_text.append(f"Extra Notes: {overall['extra_notes']}")
        if "pitch_accuracy" in overall:
            summary_text.append(f"Pitch Accuracy: {overall['pitch_accuracy']}%")
        if "timing_accuracy" in overall:
            summary_text.append(f"Timing Accuracy: {overall['timing_accuracy']}%")
        
        y_pos = 5
        for text in summary_text:
            ax.text(1, y_pos, text, fontsize=12, ha='left', va='center')
            y_pos -= 0.6
        
        # Add color legend
        legend_y = 2
        ax.text(6, legend_y + 1, "Legend:", fontsize=12, fontweight='bold')
        colors = [('green', 'Correct'), ('yellow', 'Slight error'), ('red', 'Wrong note'), ('orange', 'Extra note')]
        for i, (color, label) in enumerate(colors):
            circle = plt.Circle((6.5, legend_y - i*0.4), 0.1, color=color, ec='black')
            ax.add_patch(circle)
            ax.text(7, legend_y - i*0.4, label, fontsize=10, va='center')
        
        ax.axis('off')
    
    def _draw_grid_view(self, ax, title, reference=True):
        """Draw FL Studio-style grid view"""
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if not self.current_piece:
            return
        
        # Set up grid based on actual note durations
        melody = self.current_piece['melody']
        
        # Calculate total time in beats
        total_beats = sum(note['duration'] * 4 for note in melody)  # Convert to quarter note beats
        
        # Find pitch range
        pitches = [note['pitch'] for note in melody]
        min_pitch = min(pitches) - 2
        max_pitch = max(pitches) + 2
        
        ax.set_xlim(0, total_beats)
        ax.set_ylim(min_pitch, max_pitch)
        
        # Draw grid lines (every quarter note)
        for beat in range(int(total_beats) + 1):
            ax.axvline(x=beat, color='gray', alpha=0.3, linewidth=0.5)
        
        # Draw measure lines (every 4 beats)
        for measure in range(0, int(total_beats) + 1, 4):
            ax.axvline(x=measure, color='black', alpha=0.6, linewidth=1)
        
        for pitch in range(int(min_pitch), int(max_pitch) + 1):
            ax.axhline(y=pitch, color='gray', alpha=0.3, linewidth=0.5)
        
        # Draw notes as rectangles with proper durations
        current_time = 0
        for i, note in enumerate(melody):
            duration_beats = note['duration'] * 4  # Convert to quarter note beats
            pitch = note['pitch']
            
            # Color based on analysis if available
            if reference:
                color = 'lightblue'
            else:
                color = self._get_note_color_for_grid(i)
            
            # Draw note rectangle with actual duration
            rect = plt.Rectangle((current_time, pitch - 0.4), duration_beats, 0.8, 
                               facecolor=color, edgecolor='black', linewidth=1, picker=True)
            ax.add_patch(rect)
            
            # Add note label
            ax.text(current_time + duration_beats/2, pitch, f"{i+1}", 
                   ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Store click area
            self.note_click_areas.append({
                'note_index': i,
                'x': current_time + duration_beats/2,
                'y': pitch,
                'width': duration_beats,
                'height': 0.8,
                'axes': ax,
                'is_reference': reference
            })
            
            current_time += duration_beats
        
        ax.set_xlabel('Time (quarter note beats)')
        ax.set_ylabel('MIDI Pitch')
        ax.grid(True, alpha=0.3)
    
    def _draw_grid_comparison(self, ax):
        """Draw grid comparison view"""
        ax.set_title("Grid Comparison", fontsize=14, fontweight='bold')
        
        if not self.current_piece or not self.current_analysis:
            ax.text(0.5, 0.5, "No data for grid comparison", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, style='italic', alpha=0.7)
            ax.axis('off')
            return
        
        # Show both reference and performance on same grid
        melody = self.current_piece['melody']
        total_beats = sum(note['duration'] * 4 for note in melody)
        
        pitches = [note['pitch'] for note in melody]
        min_pitch = min(pitches) - 2
        max_pitch = max(pitches) + 2
        
        ax.set_xlim(0, total_beats)
        ax.set_ylim(min_pitch - 1, max_pitch + 1)
        
        # Draw grid
        for beat in range(int(total_beats) + 1):
            ax.axvline(x=beat, color='gray', alpha=0.3, linewidth=0.5)
        for measure in range(0, int(total_beats) + 1, 4):
            ax.axvline(x=measure, color='black', alpha=0.6, linewidth=1)
        
        # Draw reference (top half of each pitch line)
        current_time = 0
        for i, note in enumerate(melody):
            duration_beats = note['duration'] * 4
            pitch = note['pitch']
            
            rect = plt.Rectangle((current_time, pitch - 0.1), duration_beats, 0.4, 
                               facecolor='lightblue', edgecolor='blue', linewidth=1, alpha=0.7)
            ax.add_patch(rect)
            current_time += duration_beats
        
        # Draw performance (bottom half of each pitch line)
        if self.current_analysis:
            analysis_data = self.current_analysis.get("standard_analysis", {})
            note_details = analysis_data.get("note_details", [])
            
            current_time = 0
            for i, note in enumerate(melody):
                duration_beats = note['duration'] * 4
                
                if i < len(note_details):
                    detail = note_details[i]
                    actual_pitch = detail.get('actual_pitch')
                    
                    if actual_pitch and actual_pitch not in ['MISSED', 'N/A']:
                        try:
                            actual_midi = librosa.hz_to_midi(librosa.note_to_hz(actual_pitch))
                            color = self._get_note_color_for_grid(i)
                            
                            rect = plt.Rectangle((current_time, actual_midi - 0.5), duration_beats, 0.4, 
                                               facecolor=color, edgecolor='black', linewidth=1, alpha=0.8)
                            ax.add_patch(rect)
                        except:
                            pass
                
                current_time += duration_beats
        
        ax.set_xlabel('Time (quarter note beats)')
        ax.set_ylabel('MIDI Pitch')
        ax.text(0.02, 0.95, 'Blue = Reference, Colors = Performance', 
               transform=ax.transAxes, fontsize=10, va='top')
        ax.grid(True, alpha=0.3)
    
    def _get_note_symbol(self, duration):
        """Get note symbol properties based on duration"""
        if duration >= 2.0:  # Half note or longer
            return {'filled': False, 'has_stem': True, 'flags': 0}
        elif duration >= 1.0:  # Quarter note
            return {'filled': True, 'has_stem': True, 'flags': 0}
        elif duration >= 0.5:  # Eighth note
            return {'filled': True, 'has_stem': True, 'flags': 1}
        else:  # Sixteenth note or shorter
            return {'filled': True, 'has_stem': True, 'flags': 2}
    
    def _midi_to_staff_position(self, midi_pitch):
        """Convert MIDI pitch to staff position (treble clef)"""
        # G4 (MIDI 67) is on the second line of the treble staff (position 4)
        # Each semitone is 0.5 staff position
        relative_pitch = midi_pitch - 67
        staff_pos = 4 + relative_pitch * 0.5
        return staff_pos
    
    def _add_ledger_lines(self, ax, x_pos, y_pos):
        """Add ledger lines above/below staff when needed"""
        # Staff lines are at y=3,4,5,6,7
        if y_pos < 3:  # Below staff
            for line_y in np.arange(2.5, y_pos - 0.25, -1):
                if line_y % 1 == 0.5:  # Only at half-integer positions
                    ax.plot([x_pos - 0.15, x_pos + 0.15], [line_y, line_y], 
                           color='black', linewidth=1.5)
        elif y_pos > 7:  # Above staff
            for line_y in np.arange(7.5, y_pos + 0.75, 1):
                if line_y % 1 == 0.5:  # Only at half-integer positions
                    ax.plot([x_pos - 0.15, x_pos + 0.15], [line_y, line_y], 
                           color='black', linewidth=1.5)
    
    def _get_note_color_for_grid(self, note_index):
        """Get color for grid note based on analysis"""
        if not self.current_analysis:
            return 'gray'
        
        analysis_data = self.current_analysis.get("standard_analysis", {})
        note_details = analysis_data.get("note_details", [])
        
        if note_index < len(note_details):
            detail = note_details[note_index]
            note_type = detail.get('note_type', 'matched')
            
            if note_type == 'matched':
                pitch_dev = detail.get('pitch_deviation_cents', 0)
                if isinstance(pitch_dev, (int, float)) and abs(pitch_dev) < 20:
                    return 'green'
                else:
                    return 'yellow'
            elif note_type == 'extra':
                return 'orange'
            else:
                return 'red'
        
        return 'gray'
    
    def _on_canvas_click(self, event):
        """Handle clicks on the canvas"""
        if event.inaxes is None:
            return
        
        # Check if click is near any note
        for note_area in self.note_click_areas:
            if note_area['axes'] == event.inaxes:
                # Check if click is within note area
                dx = event.xdata - note_area['x']
                dy = event.ydata - note_area['y']
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance <= note_area['radius']:
                    # Note clicked!
                    note_index = note_area['note_index']
                    is_performance = note_area.get('is_performance', False)
                    is_reference = note_area.get('is_reference', False)
                    
                    print(f"🎵 Clicked note {note_index + 1} ({'Performance' if is_performance else 'Reference'})")
                    
                    # Call the callback with proper note selection
                    self.on_note_click(note_index, is_performance, is_reference)
                    break
    
    def _default_note_click(self, note_index, is_performance=False, is_reference=False):
        """Default note click handler"""
        note_type = "Performance" if is_performance else "Reference" if is_reference else "Unknown"
        print(f"Clicked {note_type} note {note_index + 1}")

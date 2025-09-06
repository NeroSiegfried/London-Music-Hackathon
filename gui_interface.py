#!/usr/bin/env python3
"""
ABRSM Music Analysis GUI

A comprehensive graphical interface for the ABRSM AI Music Feedback System
that provides intuitive, note-by-note analysis visualization and interaction.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import json
import threading
import os
import sys
from pathlib import Path

# Import our analysis modules
try:
    from enhanced_main import MusicAnalyzer, PIECES, get_feedback_from_llm
    from sheet_music_visualizer import SheetMusicVisualizer
    from time_signature_analyzer import TimeSignatureAnalyzer
    from polyphonic_analyzer import PolyphonicAnalyzer
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import analysis modules: {e}")
    MODULES_AVAILABLE = False

class ABRSMAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ABRSM AI Music Feedback System - Interactive Analysis")
        self.root.geometry("1400x900")
        
        # Analysis state
        self.current_analysis = None
        self.current_audio_file = None
        self.selected_note_index = None
        
        # Setup GUI
        self.setup_styles()
        self.create_widgets()
        self.create_menu()
        
    def setup_styles(self):
        """Configure GUI styling"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Custom colors for musical interface
        style.configure('Musical.TFrame', background='#f8f8f8')
        style.configure('Header.TLabel', font=('Arial', 14, 'bold'), background='#e8e8e8')
        style.configure('Note.TLabel', font=('Arial', 10), background='white')
        
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Audio File", command=self.open_audio_file)
        file_menu.add_command(label="Load Demo", command=self.load_demo)
        file_menu.add_separator()
        file_menu.add_command(label="Export Analysis", command=self.export_analysis)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Run Full Analysis", command=self.run_full_analysis)
        analysis_menu.add_command(label="Generate AI Feedback", command=self.generate_ai_feedback)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Sheet Music View", command=self.show_sheet_music)
        view_menu.add_command(label="Timing Analysis", command=self.show_timing_analysis)
        view_menu.add_command(label="Note Details", command=self.show_note_details)
    
    def create_widgets(self):
        """Create main GUI widgets"""
        
        # Main container
        main_frame = ttk.Frame(self.root, style='Musical.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top section - File selection and controls
        self.create_control_section(main_frame)
        
        # Middle section - Analysis visualization (notebook with tabs)
        self.create_analysis_section(main_frame)
        
        # Bottom section - Note-by-note details
        self.create_note_details_section(main_frame)
        
    def create_control_section(self, parent):
        """Create file selection and analysis controls"""
        control_frame = ttk.LabelFrame(parent, text="Audio File & Analysis Controls", style='Musical.TFrame')
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File selection row
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(file_frame, text="Audio File:").pack(side=tk.LEFT)
        self.file_var = tk.StringVar(value="No file selected")
        ttk.Label(file_frame, textvariable=self.file_var, background='white', relief='sunken').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        ttk.Button(file_frame, text="Browse", command=self.open_audio_file).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(file_frame, text="Demo", command=self.load_demo).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Analysis options row
        options_frame = ttk.Frame(control_frame)
        options_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Label(options_frame, text="Piece:").pack(side=tk.LEFT)
        self.piece_var = tk.StringVar(value="twinkle")
        piece_combo = ttk.Combobox(options_frame, textvariable=self.piece_var, values=list(PIECES.keys()), state='readonly')
        piece_combo.pack(side=tk.LEFT, padx=(5, 20))
        
        # Analysis button
        self.analyze_btn = ttk.Button(options_frame, text="🎵 Analyze Performance", command=self.run_analysis, state='disabled')
        self.analyze_btn.pack(side=tk.LEFT, padx=(20, 0))
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(options_frame, textvariable=self.progress_var).pack(side=tk.RIGHT)
        self.progress_bar = ttk.Progressbar(options_frame, mode='indeterminate')
        self.progress_bar.pack(side=tk.RIGHT, padx=(0, 10))
        
    def create_analysis_section(self, parent):
        """Create tabbed analysis visualization section"""
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Tab 1: Overview
        self.create_overview_tab()
        
        # Tab 2: Sheet Music
        self.create_sheet_music_tab()
        
        # Tab 3: Timing Analysis
        self.create_timing_tab()
        
        # Tab 4: Note Details
        self.create_note_analysis_tab()
        
        # Tab 5: AI Feedback
        self.create_feedback_tab()
        
    def create_overview_tab(self):
        """Create overview analysis tab"""
        overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(overview_frame, text="📊 Overview")
        
        # Split into summary and visualizations
        summary_frame = ttk.LabelFrame(overview_frame, text="Performance Summary")
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.summary_text = scrolledtext.ScrolledText(summary_frame, height=8, state='disabled')
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Quick stats
        stats_frame = ttk.LabelFrame(overview_frame, text="Quick Statistics")
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.stats_frame = stats_frame  # Store reference for updating
        
    def create_sheet_music_tab(self):
        """Create sheet music visualization tab"""
        sheet_frame = ttk.Frame(self.notebook)
        self.notebook.add(sheet_frame, text="🎼 Sheet Music")
        
        # Create matplotlib figure for sheet music
        self.sheet_fig = Figure(figsize=(12, 8), dpi=100)
        self.sheet_canvas = FigureCanvasTkAgg(self.sheet_fig, sheet_frame)
        self.sheet_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar for zooming/panning
        toolbar_frame = ttk.Frame(sheet_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.sheet_canvas, toolbar_frame)
        toolbar.update()
        
    def create_timing_tab(self):
        """Create timing analysis tab"""
        timing_frame = ttk.Frame(self.notebook)
        self.notebook.add(timing_frame, text="⏱️ Timing")
        
        # Create matplotlib figure for timing
        self.timing_fig = Figure(figsize=(12, 6), dpi=100)
        self.timing_canvas = FigureCanvasTkAgg(self.timing_fig, timing_frame)
        self.timing_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Timing controls
        timing_controls = ttk.Frame(timing_frame)
        timing_controls.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(timing_controls, text="Time Signature:").pack(side=tk.LEFT)
        self.time_sig_var = tk.StringVar(value="4/4")
        ttk.Label(timing_controls, textvariable=self.time_sig_var, background='white', relief='sunken').pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(timing_controls, text="Tempo:").pack(side=tk.LEFT)
        self.tempo_var = tk.StringVar(value="100 BPM")
        ttk.Label(timing_controls, textvariable=self.tempo_var, background='white', relief='sunken').pack(side=tk.LEFT, padx=(5, 0))
        
    def create_note_analysis_tab(self):
        """Create detailed note-by-note analysis tab"""
        note_frame = ttk.Frame(self.notebook)
        self.notebook.add(note_frame, text="🎵 Note Analysis")
        
        # Create treeview for note details
        columns = ('Note', 'Expected Pitch', 'Detected Pitch', 'Pitch Error', 'Expected Time', 'Detected Time', 'Timing Error', 'Accuracy')
        self.note_tree = ttk.Treeview(note_frame, columns=columns, show='headings', height=15)
        
        # Configure column headings and widths
        for col in columns:
            self.note_tree.heading(col, text=col)
            self.note_tree.column(col, width=100, anchor='center')
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(note_frame, orient=tk.VERTICAL, command=self.note_tree.yview)
        h_scrollbar = ttk.Scrollbar(note_frame, orient=tk.HORIZONTAL, command=self.note_tree.xview)
        self.note_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.note_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind selection event
        self.note_tree.bind('<<TreeviewSelect>>', self.on_note_select)
        
    def create_feedback_tab(self):
        """Create AI feedback tab"""
        feedback_frame = ttk.Frame(self.notebook)
        self.notebook.add(feedback_frame, text="🤖 AI Feedback")
        
        # API key section
        api_frame = ttk.LabelFrame(feedback_frame, text="AI Configuration")
        api_frame.pack(fill=tk.X, padx=10, pady=10)
        
        api_controls = ttk.Frame(api_frame)
        api_controls.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(api_controls, text="Google API Key:").pack(side=tk.LEFT)
        self.api_key_var = tk.StringVar(value=os.environ.get("GOOGLE_API_KEY", ""))
        api_entry = ttk.Entry(api_controls, textvariable=self.api_key_var, show="*", width=40)
        api_entry.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        
        ttk.Button(api_controls, text="Generate Feedback", command=self.generate_ai_feedback).pack(side=tk.RIGHT, padx=(10, 0))
        
        # Feedback display
        feedback_display = ttk.LabelFrame(feedback_frame, text="AI-Generated Feedback")
        feedback_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.feedback_text = scrolledtext.ScrolledText(feedback_display, height=15, wrap=tk.WORD, state='disabled')
        self.feedback_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_note_details_section(self, parent):
        """Create bottom section for selected note details"""
        details_frame = ttk.LabelFrame(parent, text="Selected Note Details")
        details_frame.pack(fill=tk.X)
        
        # Create grid for note details
        self.details_grid = ttk.Frame(details_frame)
        self.details_grid.pack(fill=tk.X, padx=10, pady=10)
        
        # Initialize with placeholder
        ttk.Label(self.details_grid, text="Select a note from the analysis to see detailed information here.", 
                 font=('Arial', 10, 'italic')).pack()
        
    def open_audio_file(self):
        """Open file dialog to select audio file"""
        filetypes = [
            ('Audio files', '*.wav *.mp3 *.flac *.m4a'),
            ('WAV files', '*.wav'),
            ('MP3 files', '*.mp3'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=filetypes
        )
        
        if filename:
            self.current_audio_file = filename
            self.file_var.set(os.path.basename(filename))
            self.analyze_btn.config(state='normal')
            
    def load_demo(self):
        """Load demo audio file"""
        try:
            # Create demo audio if it doesn't exist
            if not os.path.exists("demo_performance.wav"):
                self.progress_var.set("Creating demo audio...")
                self.progress_bar.start()
                
                # Run demo creation in thread to prevent GUI freezing
                def create_demo():
                    os.system("python enhanced_main.py --create-demo-only")
                    self.root.after(0, lambda: self.demo_created())
                
                threading.Thread(target=create_demo, daemon=True).start()
            else:
                self.demo_created()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load demo: {e}")
            
    def demo_created(self):
        """Called when demo audio is ready"""
        self.progress_bar.stop()
        self.progress_var.set("Ready")
        
        if os.path.exists("demo_performance.wav"):
            self.current_audio_file = "demo_performance.wav"
            self.file_var.set("demo_performance.wav")
            self.analyze_btn.config(state='normal')
            messagebox.showinfo("Demo Ready", "Demo audio file loaded successfully!")
        else:
            messagebox.showerror("Error", "Failed to create demo audio file")
            
    def run_analysis(self):
        """Run comprehensive analysis in background thread"""
        if not self.current_audio_file:
            messagebox.showerror("Error", "Please select an audio file first")
            return
            
        if not MODULES_AVAILABLE:
            messagebox.showerror("Error", "Analysis modules not available")
            return
            
        # Start analysis in background thread
        self.progress_var.set("Analyzing...")
        self.progress_bar.start()
        self.analyze_btn.config(state='disabled')
        
        def analyze():
            try:
                analyzer = MusicAnalyzer(piece_key=self.piece_var.get())
                
                # Create reference if needed
                analyzer.create_reference_data()
                
                # Run enhanced analysis
                f0, times, onsets, enhanced_analysis = analyzer.analyze_with_enhancements(
                    self.current_audio_file,
                    generate_visualizations=False,  # We'll generate our own
                    detect_polyphony=True,
                    analyze_timing=True
                )
                
                # Store results
                self.current_analysis = enhanced_analysis
                
                # Update GUI in main thread
                self.root.after(0, lambda: self.analysis_complete())
                
            except Exception as e:
                self.root.after(0, lambda: self.analysis_error(str(e)))
                
        threading.Thread(target=analyze, daemon=True).start()
        
    def run_full_analysis(self):
        """Run analysis from menu"""
        self.run_analysis()
        
    def analysis_complete(self):
        """Called when analysis is complete"""
        self.progress_bar.stop()
        self.progress_var.set("Analysis complete")
        self.analyze_btn.config(state='normal')
        
        if self.current_analysis:
            self.update_all_views()
            messagebox.showinfo("Success", "Analysis completed successfully!")
        
    def analysis_error(self, error_msg):
        """Called when analysis encounters an error"""
        self.progress_bar.stop()
        self.progress_var.set("Analysis failed")
        self.analyze_btn.config(state='normal')
        messagebox.showerror("Analysis Error", f"Analysis failed: {error_msg}")
        
    def update_all_views(self):
        """Update all GUI views with current analysis data"""
        if not self.current_analysis:
            return
            
        self.update_overview()
        self.update_sheet_music()
        self.update_timing_analysis()
        self.update_note_tree()
        
    def update_overview(self):
        """Update overview tab"""
        if not self.current_analysis:
            return
            
        standard_analysis = self.current_analysis.get("standard_analysis", {})
        enhanced_features = self.current_analysis.get("enhanced_features", {})
        
        # Update summary text
        self.summary_text.config(state='normal')
        self.summary_text.delete(1.0, tk.END)
        
        summary = f"""
📊 PERFORMANCE ANALYSIS SUMMARY

Piece: {standard_analysis.get('piece_title', 'Unknown')}
Audio File: {os.path.basename(self.current_audio_file) if self.current_audio_file else 'Unknown'}

Overall Assessment:
• Completion Rate: {standard_analysis.get('overall_assessment', {}).get('completion_rate', 'N/A')}%
• Pitch Accuracy: {standard_analysis.get('overall_assessment', {}).get('pitch_accuracy', 'N/A')}%
• Timing Accuracy: {standard_analysis.get('overall_assessment', {}).get('timing_accuracy', 'N/A')}%
• Notes Detected: {len(standard_analysis.get('note_details', []))}
• Expected Notes: {standard_analysis.get('analysis_metadata', {}).get('expected_notes', 'N/A')}

Enhanced Features:
"""
        
        if 'timing_analysis' in enhanced_features:
            timing = enhanced_features['timing_analysis']
            summary += f"""
• Time Signature: {timing.get('detected_time_signature', 'Unknown')}
• Tempo: {timing.get('tempo_bpm', 'Unknown')} BPM
• Beat Consistency: {timing.get('beat_analysis', {}).get('beat_consistency', 'N/A'):.1f}%
"""
        
        if 'polyphonic_analysis' in enhanced_features:
            poly = enhanced_features['polyphonic_analysis']
            complexity = poly.get('complexity_score', {})
            summary += f"""
• Polyphonic Content Detected
• Complexity Score: {complexity.get('score', 'N/A')}/100
• Average Simultaneous Notes: {complexity.get('avg_simultaneous_notes', 'N/A')}
"""
        
        self.summary_text.insert(1.0, summary)
        self.summary_text.config(state='disabled')
        
    def update_sheet_music(self):
        """Update sheet music visualization"""
        if not self.current_analysis:
            return
            
        # Clear previous plot
        self.sheet_fig.clear()
        
        try:
            # Create sheet music visualization
            visualizer = SheetMusicVisualizer()
            
            # Get reference melody and performance data
            piece_key = self.piece_var.get()
            reference_melody = PIECES[piece_key]['melody']
            
            # Simplified sheet music display for GUI
            ax = self.sheet_fig.add_subplot(111)
            ax.set_title(f"Sheet Music Analysis: {PIECES[piece_key]['title']}", fontsize=14, fontweight='bold')
            
            # Draw staff lines
            staff_width = 12
            staff_lines = [0, 0.5, 1, 1.5, 2]
            for line_pos in staff_lines:
                ax.plot([0, staff_width], [line_pos, line_pos], 'k-', linewidth=1)
            
            # Draw notes from analysis
            note_details = self.current_analysis.get("standard_analysis", {}).get("note_details", [])
            
            x_position = 0.5
            for i, note_detail in enumerate(note_details):
                expected_pitch = note_detail.get('expected_pitch', '')
                detected_pitch = note_detail.get('detected_pitch', '')
                pitch_error = note_detail.get('pitch_deviation_cents', 0)
                timing_error = note_detail.get('timing_deviation_ms', 0)
                
                # Determine note color based on accuracy
                if isinstance(pitch_error, (int, float)) and isinstance(timing_error, (int, float)):
                    if abs(pitch_error) < 20 and abs(timing_error) < 50:
                        color = 'green'
                    elif abs(pitch_error) < 50 and abs(timing_error) < 100:
                        color = 'orange'
                    else:
                        color = 'red'
                else:
                    color = 'gray'  # Missed notes
                
                # Simple note representation
                y_pos = 1 + (i % 3) * 0.5  # Simplified positioning
                circle = plt.Circle((x_position, y_pos), 0.08, color=color, zorder=10)
                ax.add_patch(circle)
                
                # Add note label
                ax.text(x_position, y_pos - 0.3, expected_pitch, ha='center', va='center', fontsize=8)
                if detected_pitch != expected_pitch and detected_pitch != 'MISSED':
                    ax.text(x_position, y_pos + 0.3, f"({detected_pitch})", ha='center', va='center', fontsize=7, color=color)
                
                x_position += 0.8
                if x_position > staff_width:
                    break
            
            # Add legend
            ax.text(0, -0.8, "🟢 Good (±20¢/±50ms)  🟠 Fair (±50¢/±100ms)  🔴 Needs Work", fontsize=10)
            
            ax.set_xlim(-0.5, staff_width + 0.5)
            ax.set_ylim(-1, 3)
            ax.set_aspect('equal')
            ax.axis('off')
            
        except Exception as e:
            ax = self.sheet_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Sheet music visualization error: {e}", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
        
        self.sheet_canvas.draw()
        
    def update_timing_analysis(self):
        """Update timing analysis visualization"""
        if not self.current_analysis:
            return
            
        # Clear previous plot
        self.timing_fig.clear()
        
        try:
            enhanced_features = self.current_analysis.get("enhanced_features", {})
            
            if 'timing_analysis' in enhanced_features:
                timing = enhanced_features['timing_analysis']
                
                # Update time signature and tempo displays
                time_sig = timing.get('detected_time_signature', (4, 4))
                self.time_sig_var.set(f"{time_sig[0]}/{time_sig[1]}")
                self.tempo_var.set(f"{timing.get('tempo_bpm', 100)} BPM")
                
                # Create timing visualization
                ax1 = self.timing_fig.add_subplot(211)
                ax2 = self.timing_fig.add_subplot(212)
                
                # Plot 1: Timing deviations
                note_details = self.current_analysis.get("standard_analysis", {}).get("note_details", [])
                note_numbers = []
                timing_deviations = []
                
                for i, note in enumerate(note_details):
                    timing_dev = note.get('timing_deviation_ms', 0)
                    if isinstance(timing_dev, (int, float)):
                        note_numbers.append(i + 1)
                        timing_deviations.append(timing_dev)
                
                if note_numbers:
                    bars = ax1.bar(note_numbers, timing_deviations, 
                                  color=['green' if abs(x) < 50 else 'orange' if abs(x) < 100 else 'red' 
                                        for x in timing_deviations], alpha=0.7)
                    ax1.set_title('Timing Deviations by Note')
                    ax1.set_xlabel('Note Number')
                    ax1.set_ylabel('Timing Error (ms)')
                    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    ax1.grid(True, alpha=0.3)
                
                # Plot 2: Beat analysis
                beat_analysis = timing.get('beat_analysis', {})
                strong_acc = beat_analysis.get('strong_beat_accuracy', 0)
                weak_acc = beat_analysis.get('weak_beat_accuracy', 0)
                consistency = beat_analysis.get('beat_consistency', 0)
                
                categories = ['Strong Beats\n(1, 3)', 'Weak Beats\n(2, 4)', 'Overall\nConsistency']
                values = [strong_acc, weak_acc, consistency]
                colors = ['darkblue', 'lightblue', 'purple']
                
                ax2.bar(categories, values, color=colors, alpha=0.7)
                ax2.set_title('Beat Pattern Analysis')
                ax2.set_ylabel('Accuracy Score')
                ax2.grid(True, alpha=0.3)
                
            else:
                ax = self.timing_fig.add_subplot(111)
                ax.text(0.5, 0.5, "No timing analysis data available", ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        
        except Exception as e:
            ax = self.timing_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Timing analysis error: {e}", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
        
        self.timing_fig.tight_layout()
        self.timing_canvas.draw()
        
    def update_note_tree(self):
        """Update note-by-note analysis tree"""
        # Clear existing items
        for item in self.note_tree.get_children():
            self.note_tree.delete(item)
            
        if not self.current_analysis:
            return
            
        note_details = self.current_analysis.get("standard_analysis", {}).get("note_details", [])
        
        for note in note_details:
            note_num = note.get('note_index', 'N/A')
            expected_pitch = note.get('expected_pitch', 'N/A')
            detected_pitch = note.get('detected_pitch', 'N/A')
            pitch_error = note.get('pitch_deviation_cents', 'N/A')
            expected_time = note.get('expected_time', 'N/A')
            detected_time = note.get('detected_time', 'N/A')
            timing_error = note.get('timing_deviation_ms', 'N/A')
            
            # Calculate accuracy score
            accuracy = "Perfect"
            if isinstance(pitch_error, (int, float)) and isinstance(timing_error, (int, float)):
                if abs(pitch_error) < 20 and abs(timing_error) < 50:
                    accuracy = "Excellent"
                elif abs(pitch_error) < 50 and abs(timing_error) < 100:
                    accuracy = "Good"
                else:
                    accuracy = "Needs Work"
            elif pitch_error == "MISSED" or timing_error == "MISSED":
                accuracy = "Missed"
            
            # Format values for display
            pitch_error_str = f"{pitch_error}¢" if isinstance(pitch_error, (int, float)) else str(pitch_error)
            timing_error_str = f"{timing_error}ms" if isinstance(timing_error, (int, float)) else str(timing_error)
            expected_time_str = f"{expected_time:.2f}s" if isinstance(expected_time, (int, float)) else str(expected_time)
            detected_time_str = f"{detected_time:.2f}s" if isinstance(detected_time, (int, float)) else str(detected_time)
            
            # Insert into tree
            item_id = self.note_tree.insert('', tk.END, values=(
                note_num, expected_pitch, detected_pitch, pitch_error_str,
                expected_time_str, detected_time_str, timing_error_str, accuracy
            ))
            
            # Color code rows based on accuracy
            if accuracy == "Excellent":
                self.note_tree.set(item_id, 'Note', f"♪ {note_num}")
            elif accuracy == "Good":
                self.note_tree.set(item_id, 'Note', f"♫ {note_num}")
            elif accuracy == "Needs Work":
                self.note_tree.set(item_id, 'Note', f"♪? {note_num}")
            elif accuracy == "Missed":
                self.note_tree.set(item_id, 'Note', f"✗ {note_num}")
        
    def on_note_select(self, event):
        """Handle note selection in tree"""
        selection = self.note_tree.selection()
        if selection:
            item = selection[0]
            values = self.note_tree.item(item, 'values')
            
            if values:
                self.show_note_details(values)
    
    def show_note_details(self, note_values=None):
        """Show detailed information for selected note"""
        # Clear previous details
        for widget in self.details_grid.winfo_children():
            widget.destroy()
            
        if not note_values:
            ttk.Label(self.details_grid, text="Select a note from the analysis to see detailed information here.", 
                     font=('Arial', 10, 'italic')).pack()
            return
            
        # Create detailed view
        note_num, expected_pitch, detected_pitch, pitch_error, expected_time, detected_time, timing_error, accuracy = note_values
        
        # Title
        title_frame = ttk.Frame(self.details_grid)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(title_frame, text=f"Note {note_num} - Detailed Analysis", 
                 font=('Arial', 12, 'bold')).pack(side=tk.LEFT)
        
        # Create two columns
        left_frame = ttk.Frame(self.details_grid)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        right_frame = ttk.Frame(self.details_grid)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Left column - Pitch analysis
        pitch_group = ttk.LabelFrame(left_frame, text="Pitch Analysis")
        pitch_group.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(pitch_group, text=f"Expected: {expected_pitch}").pack(anchor='w', padx=5, pady=2)
        ttk.Label(pitch_group, text=f"Detected: {detected_pitch}").pack(anchor='w', padx=5, pady=2)
        ttk.Label(pitch_group, text=f"Error: {pitch_error}").pack(anchor='w', padx=5, pady=2)
        
        # Right column - Timing analysis
        timing_group = ttk.LabelFrame(right_frame, text="Timing Analysis")
        timing_group.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(timing_group, text=f"Expected: {expected_time}").pack(anchor='w', padx=5, pady=2)
        ttk.Label(timing_group, text=f"Detected: {detected_time}").pack(anchor='w', padx=5, pady=2)
        ttk.Label(timing_group, text=f"Error: {timing_error}").pack(anchor='w', padx=5, pady=2)
        
        # Overall assessment
        assessment_frame = ttk.Frame(self.details_grid)
        assessment_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(assessment_frame, text="Overall Assessment:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        
        # Color code the accuracy
        accuracy_colors = {
            "Excellent": "green",
            "Good": "blue", 
            "Needs Work": "orange",
            "Missed": "red",
            "Perfect": "darkgreen"
        }
        
        accuracy_label = ttk.Label(assessment_frame, text=accuracy, font=('Arial', 10, 'bold'))
        accuracy_label.pack(side=tk.LEFT, padx=(10, 0))
        
    def generate_ai_feedback(self):
        """Generate AI feedback using current analysis"""
        if not self.current_analysis:
            messagebox.showerror("Error", "Please run analysis first")
            return
            
        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showerror("Error", "Please enter your Google API key")
            return
            
        # Generate feedback in background thread
        self.progress_var.set("Generating AI feedback...")
        self.progress_bar.start()
        
        def generate():
            try:
                standard_analysis = self.current_analysis.get("standard_analysis", {})
                enhanced_analysis = self.current_analysis
                
                feedback = get_feedback_from_llm(
                    json.dumps(standard_analysis, indent=2),
                    api_key,
                    enhanced_analysis
                )
                
                self.root.after(0, lambda: self.feedback_generated(feedback))
                
            except Exception as e:
                self.root.after(0, lambda: self.feedback_error(str(e)))
                
        threading.Thread(target=generate, daemon=True).start()
        
    def feedback_generated(self, feedback):
        """Called when AI feedback is generated"""
        self.progress_bar.stop()
        self.progress_var.set("Ready")
        
        # Display feedback
        self.feedback_text.config(state='normal')
        self.feedback_text.delete(1.0, tk.END)
        self.feedback_text.insert(1.0, feedback)
        self.feedback_text.config(state='disabled')
        
        # Switch to feedback tab
        self.notebook.select(4)  # Index of feedback tab
        
    def feedback_error(self, error_msg):
        """Called when AI feedback generation fails"""
        self.progress_bar.stop()
        self.progress_var.set("Ready")
        messagebox.showerror("Feedback Error", f"Failed to generate feedback: {error_msg}")
        
    def show_sheet_music(self):
        """Switch to sheet music tab"""
        self.notebook.select(1)
        
    def show_timing_analysis(self):
        """Switch to timing analysis tab"""
        self.notebook.select(2)
        
    def export_analysis(self):
        """Export analysis results to file"""
        if not self.current_analysis:
            messagebox.showerror("Error", "No analysis to export")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Export Analysis",
            defaultextension=".json",
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.current_analysis, f, indent=2)
                messagebox.showinfo("Success", f"Analysis exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")

def main():
    """Main function to run the GUI"""
    if not MODULES_AVAILABLE:
        print("Error: Required modules not available. Please run from the correct directory.")
        return
        
    root = tk.Tk()
    app = ABRSMAnalysisGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()

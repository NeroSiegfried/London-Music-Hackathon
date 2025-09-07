#!/usr/bin/env python3
"""
Enhanced ABRSM Music Analysis GUI with Detailed Note Analysis and Mistake Detection

This enhanced version includes:
- Detailed note-by-note interactive analysis
- Mistake detection and retry pattern recognition
- Performance diff analysis
- Enhanced visualization and user interaction
- Real-time note selection and playback
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import numpy as np
import json
import threading
import os
import sys
import time
import traceback
import re
from pathlib import Path
import pygame
import librosa
import soundfile as sf

# Import existing modules
try:
    from enhanced_main_fixed import MusicAnalyzer, PIECES
    MUSIC_ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import MusicAnalyzer: {e}")
    MUSIC_ANALYZER_AVAILABLE = False
    MusicAnalyzer = None
    PIECES = {}

try:
    from sheet_music_visualizer import Music21SheetVisualizer, create_visual_analysis
    SHEET_MUSIC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import sheet music visualizer: {e}")
    SHEET_MUSIC_AVAILABLE = False
    Music21SheetVisualizer = None

try:
    from time_signature_analyzer import TimeSignatureAnalyzer
    TIME_ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import time signature analyzer: {e}")
    TIME_ANALYZER_AVAILABLE = False
    TimeSignatureAnalyzer = None

try:
    from polyphonic_analyzer import PolyphonicAnalyzer
    POLYPHONIC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import polyphonic analyzer: {e}")
    POLYPHONIC_AVAILABLE = False
    PolyphonicAnalyzer = None

class ImprovedPolyphonicAnalyzer:
    """Optional polyphonic analyzer - enhances existing analysis without replacing it"""
    
    def __init__(self):
        self.sample_rate = 22050
        
    def enhance_analysis(self, audio_path, existing_analysis, reference_mxl=None):
        """Add polyphonic detection to existing analysis without replacing it"""
        try:
            import librosa
            import numpy as np
            
            print("� Adding polyphonic enhancement to existing analysis...")
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Simple polyphonic detection using CQT
            C = np.abs(librosa.cqt(y=y, sr=sr, fmin=librosa.note_to_hz('C2'), 
                                   n_bins=72, bins_per_octave=12))
            
            # Find strong frequency components
            threshold = np.percentile(C.flatten(), 85)
            strong_frames = np.any(C > threshold, axis=1)
            polyphonic_notes = np.sum(strong_frames)
            
            # Add to existing analysis
            if 'enhanced_features' not in existing_analysis:
                existing_analysis['enhanced_features'] = {}
                
            existing_analysis['enhanced_features']['polyphonic_detection'] = {
                'detected_harmonics': int(polyphonic_notes),
                'analysis_method': 'cqt_enhancement',
                'confidence': 0.7
            }
            
            print(f"🎵 Polyphonic enhancement added: {polyphonic_notes} harmonic components detected")
            return existing_analysis
            
        except Exception as e:
            print(f"⚠️ Polyphonic enhancement failed: {e}")
            return existing_analysis
    
try:
    IMPROVED_POLYPHONIC_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import time signature analyzer: {e}")
    TIME_ANALYZER_AVAILABLE = False
    TimeSignatureAnalyzer = None

try:
    from polyphonic_analyzer import PolyphonicAnalyzer
    POLYPHONIC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import polyphonic analyzer: {e}")
    POLYPHONIC_AVAILABLE = False
    PolyphonicAnalyzer = None

try:
    from interactive_sheet_music import InteractiveSheetMusic
    INTERACTIVE_SHEET_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import interactive sheet music: {e}")
    INTERACTIVE_SHEET_AVAILABLE = False
    InteractiveSheetMusic = None

MODULES_AVAILABLE = (MUSIC_ANALYZER_AVAILABLE and SHEET_MUSIC_AVAILABLE and 
                    TIME_ANALYZER_AVAILABLE and POLYPHONIC_AVAILABLE)

class EnhancedABRSMGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced ABRSM AI Music Feedback System - Competition Ready")
        self.root.geometry("1920x1080")
        
        # Make window resizable and set minimum size
        self.root.resizable(True, True)
        self.root.minsize(1600, 1000)
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
        # Initialize improved polyphonic analyzer
        if IMPROVED_POLYPHONIC_AVAILABLE:
            self.improved_analyzer = ImprovedPolyphonicAnalyzer()
        else:
            self.improved_analyzer = None
        
        # Analysis state
        self.current_analysis = None
        self.current_audio_file = None
        self.selected_note_index = None
        self.mistake_patterns = []
        self.performance_sections = []
        
        # Setup enhanced GUI
        self.setup_styles()
        self.load_available_midi_files()  # Load MIDI files dynamically
        self.create_enhanced_widgets()
        self.create_enhanced_menu()
        
    def setup_styles(self):
        """Configure enhanced GUI styling"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Enhanced colors for professional musical interface
        style.configure('Musical.TFrame', background='#f5f5f5')
        style.configure('Header.TLabel', font=('Arial', 16, 'bold'), background='#e0e0e0')
        style.configure('SubHeader.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0')
        style.configure('Note.TLabel', font=('Arial', 10), background='white')
        style.configure('Excellent.TLabel', foreground='darkgreen', font=('Arial', 10, 'bold'))
        style.configure('Good.TLabel', foreground='blue', font=('Arial', 10, 'bold'))
        style.configure('Poor.TLabel', foreground='orange', font=('Arial', 10, 'bold'))
        style.configure('Missed.TLabel', foreground='red', font=('Arial', 10, 'bold'))
        
    def load_available_midi_files(self):
        """Dynamically load all MIDI files from the midi folder"""
        self.available_pieces = {}
        midi_folder = "midi"
        
        if os.path.exists(midi_folder):
            for filename in os.listdir(midi_folder):
                if filename.endswith('.mid') or filename.endswith('.midi'):
                    # Extract piece name from filename
                    piece_name = filename.replace('.mid', '').replace('.midi', '').replace('_reference', '')
                    piece_key = piece_name.lower().replace(' ', '_').replace('-', '_')
                    
                    # Create a basic piece structure
                    self.available_pieces[piece_key] = {
                        'title': piece_name.replace('_', ' ').title(),
                        'midi_file': os.path.join(midi_folder, filename),
                        'melody': []  # Will be populated from MIDI if needed
                    }
        
        # Merge with existing PIECES if available
        if MODULES_AVAILABLE and hasattr(sys.modules[__name__], 'PIECES'):
            self.available_pieces.update(PIECES)
        
        print(f"✓ Loaded {len(self.available_pieces)} available pieces: {list(self.available_pieces.keys())}")
        
    def get_available_pieces(self):
        """Get list of available piece keys"""
        return list(self.available_pieces.keys())
        
    def create_enhanced_menu(self):
        """Create enhanced menu bar with additional options"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Audio File", command=self.open_audio_file)
        file_menu.add_command(label="Load Demo", command=self.load_demo)
        file_menu.add_separator()
        file_menu.add_command(label="Load XML Template", command=self.load_xml_template)
        file_menu.add_separator()
        file_menu.add_command(label="Export Analysis", command=self.export_analysis)
        file_menu.add_command(label="Export Detailed Report", command=self.export_detailed_report)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Run Full Analysis", command=self.run_full_analysis)
        analysis_menu.add_command(label="Detect Mistake Patterns", command=self.detect_mistake_patterns)
        analysis_menu.add_command(label="Analyze Performance Sections", command=self.analyze_performance_sections)
        analysis_menu.add_command(label="Generate AI Feedback", command=self.generate_enhanced_feedback)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Sheet Music View", command=self.show_sheet_music)
        view_menu.add_command(label="Mistake Analysis", command=self.show_mistake_analysis)
        view_menu.add_command(label="Performance Diff", command=self.show_performance_diff)
        view_menu.add_command(label="Note Details", command=self.show_note_details)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Play Selected Note", command=self.play_selected_note)
        tools_menu.add_command(label="Play Reference Audio", command=self.play_reference_audio)
        tools_menu.add_command(label="Play Performance Section", command=self.play_performance_section)
    
    def create_enhanced_widgets(self):
        """Create enhanced GUI widgets with detailed analysis capabilities"""
        
        # Create main paned window for resizable sections
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Controls and analysis
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=2)
        
        # Right panel - Detailed note analysis
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        # Create sections
        self.create_control_section(left_frame)
        self.create_enhanced_analysis_section(left_frame)
        self.create_detailed_note_panel(right_frame)
        
    def create_control_section(self, parent):
        """Create enhanced control section"""
        control_frame = ttk.LabelFrame(parent, text="🎼 Audio Analysis Controls")
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # File selection
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(file_frame, text="Audio File:").pack(side=tk.LEFT)
        self.file_label = ttk.Label(file_frame, text="No file selected", background='white', relief='sunken')
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        ttk.Button(file_frame, text="Browse", command=self.open_audio_file).pack(side=tk.RIGHT, padx=(10, 0))
        
        # Piece selection
        piece_frame = ttk.Frame(control_frame)
        piece_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Label(piece_frame, text="Reference Piece:").pack(side=tk.LEFT)
        self.piece_var = tk.StringVar(value=list(self.available_pieces.keys())[0] if self.available_pieces else "twinkle")
        piece_combo = ttk.Combobox(piece_frame, textvariable=self.piece_var, 
                                   values=list(self.available_pieces.keys()), state='readonly')
        piece_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Analysis buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(button_frame, text="🎵 Load Demo", 
                  command=self.load_demo).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="🔍 Analyze Performance", 
                  command=self.run_full_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🎯 Detect Mistakes", 
                  command=self.detect_mistake_patterns).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📊 Performance Diff", 
                  command=self.show_performance_diff).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="⚙️ Batch Process", 
                  command=self.batch_process_csv).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 10))
        
    def create_enhanced_analysis_section(self, parent):
        """Create enhanced tabbed analysis visualization section"""
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Enhanced tabs
        self.create_overview_tab()
        self.create_note_by_note_tab()
        self.create_mistake_analysis_tab()
        self.create_performance_diff_tab()
        self.create_sheet_music_tab()
        self.create_timing_tab()
        self.create_feedback_tab()
        
    def create_note_by_note_tab(self):
        """Create detailed note-by-note analysis tab"""
        note_frame = ttk.Frame(self.notebook)
        self.notebook.add(note_frame, text="🎵 Notes")
        
        # Top section - note overview
        overview_frame = ttk.LabelFrame(note_frame, text="Performance Overview")
        overview_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create metrics display
        metrics_frame = ttk.Frame(overview_frame)
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.accuracy_labels = {}
        metrics = ['Excellent', 'Good', 'Poor', 'Missed']
        colors = ['darkgreen', 'blue', 'orange', 'red']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            frame = ttk.Frame(metrics_frame)
            frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            ttk.Label(frame, text=f"{metric}:", font=('Arial', 10, 'bold')).pack()
            self.accuracy_labels[metric] = ttk.Label(frame, text="0", 
                                                    font=('Arial', 14, 'bold'), 
                                                    foreground=color)
            self.accuracy_labels[metric].pack()
        
        # Note table with enhanced details
        table_frame = ttk.LabelFrame(note_frame, text="Detailed Note Analysis")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Create enhanced treeview with better spacing
        columns = ('Note', 'Expected', 'Detected', 'Pitch Error (¢)', 'Timing Error (ms)', 
                  'Duration', 'Dynamics', 'Accuracy', 'Issues')
        self.enhanced_note_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Configure columns with better widths for readability
        col_widths = [80, 120, 120, 140, 160, 110, 110, 120, 180]
        for col, width in zip(columns, col_widths):
            self.enhanced_note_tree.heading(col, text=col)
            self.enhanced_note_tree.column(col, width=width, anchor='center', minwidth=width)
        
        # Configure row height for better readability
        style = ttk.Style()
        style.configure("Treeview", rowheight=35)  # Increased row height significantly
        
        # Add scrollbars
        v_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.enhanced_note_tree.yview)
        h_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.enhanced_note_tree.xview)
        self.enhanced_note_tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        # Pack treeview and scrollbars
        self.enhanced_note_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind events
        self.enhanced_note_tree.bind('<<TreeviewSelect>>', self.on_enhanced_note_select)
        self.enhanced_note_tree.bind('<Double-1>', self.on_note_double_click)
        
    def create_mistake_analysis_tab(self):
        """Create mistake pattern analysis tab"""
        mistake_frame = ttk.Frame(self.notebook)
        self.notebook.add(mistake_frame, text="🎯 Mistakes")
        
        # Split into two sections
        paned = ttk.PanedWindow(mistake_frame, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top - mistake patterns
        pattern_frame = ttk.LabelFrame(paned, text="Detected Mistake Patterns")
        paned.add(pattern_frame, weight=1)
        
        self.mistake_tree = ttk.Treeview(pattern_frame, 
                                        columns=('Type', 'Location', 'Severity', 'Description'),
                                        show='headings', height=8)
        
        for col in ['Type', 'Location', 'Severity', 'Description']:
            self.mistake_tree.heading(col, text=col)
            self.mistake_tree.column(col, width=120, anchor='center')
        
        self.mistake_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bind click event for mistake details
        self.mistake_tree.bind('<<TreeviewSelect>>', self.on_mistake_select)
        self.mistake_tree.bind('<Double-1>', self.on_mistake_double_click)
        
        # Bottom - retry analysis
        retry_frame = ttk.LabelFrame(paned, text="Retry Pattern Analysis")
        paned.add(retry_frame, weight=1)
        
        self.retry_text = scrolledtext.ScrolledText(retry_frame, height=10, state='disabled')
        self.retry_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_performance_diff_tab(self):
        """Create performance difference analysis tab"""
        diff_frame = ttk.Frame(self.notebook)
        self.notebook.add(diff_frame, text="📊 Diff")
        
        # Create matplotlib figure for diff visualization
        self.diff_fig = Figure(figsize=(12, 8), dpi=100)
        self.diff_canvas = FigureCanvasTkAgg(self.diff_fig, diff_frame)
        self.diff_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add controls
        controls_frame = ttk.Frame(diff_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(controls_frame, text="Analysis Type:").pack(side=tk.LEFT)
        self.diff_type_var = tk.StringVar(value="pitch_timing")
        diff_combo = ttk.Combobox(controls_frame, textvariable=self.diff_type_var,
                                 values=['pitch_timing', 'rhythm_patterns', 'section_analysis'],
                                 state='readonly')
        diff_combo.pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Button(controls_frame, text="Update Diff", 
                  command=self.update_performance_diff).pack(side=tk.LEFT, padx=5)
        
    def create_detailed_note_panel(self, parent):
        """Create detailed note analysis panel on the right side"""
        detail_frame = ttk.LabelFrame(parent, text="🔍 Selected Note Details")
        detail_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Note info section
        info_frame = ttk.Frame(detail_frame)
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.note_info_label = ttk.Label(info_frame, text="Select a note to see details", 
                                        font=('Arial', 12, 'bold'))
        self.note_info_label.pack()
        
        # Audio playback controls
        audio_frame = ttk.LabelFrame(detail_frame, text="Audio Playback")
        audio_frame.pack(fill=tk.X, padx=10, pady=10)
        
        playback_controls = ttk.Frame(audio_frame)
        playback_controls.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(playback_controls, text="▶ Play Note", 
                  command=self.play_selected_note).pack(side=tk.LEFT, padx=5)
        ttk.Button(playback_controls, text="▶ Play Reference", 
                  command=self.play_reference_note).pack(side=tk.LEFT, padx=5)
        ttk.Button(playback_controls, text="⏸ Stop", 
                  command=self.stop_playback).pack(side=tk.LEFT, padx=5)
        
        # Detailed analysis
        analysis_frame = ttk.LabelFrame(detail_frame, text="Detailed Analysis")
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.detail_text = scrolledtext.ScrolledText(analysis_frame, height=15, state='disabled')
        self.detail_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Visualization canvas for note-specific analysis
        viz_frame = ttk.LabelFrame(detail_frame, text="Note Visualization")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.note_fig = Figure(figsize=(6, 4), dpi=100)
        self.note_canvas = FigureCanvasTkAgg(self.note_fig, viz_frame)
        self.note_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_overview_tab(self):
        """Create enhanced overview tab"""
        overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(overview_frame, text="📊 Overview")
        
        # Performance summary with visual indicators
        summary_frame = ttk.LabelFrame(overview_frame, text="Performance Summary")
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.summary_text = scrolledtext.ScrolledText(summary_frame, height=10, state='disabled')
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Visual metrics
        metrics_frame = ttk.LabelFrame(overview_frame, text="Performance Metrics")
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.metrics_fig = Figure(figsize=(12, 6), dpi=100)
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, metrics_frame)
        self.metrics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_sheet_music_tab(self):
        """Create enhanced interactive sheet music tab with proper third-party visualization"""
        sheet_frame = ttk.Frame(self.notebook)
        self.notebook.add(sheet_frame, text="🎼 Sheet")
        
        # Create sheet music controls
        controls_frame = ttk.Frame(sheet_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(controls_frame, text="🔄 Refresh Sheet Music", 
                  command=self.refresh_sheet_music).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="💾 Export Sheet", 
                  command=self.export_sheet_music).pack(side=tk.LEFT, padx=5)
        
        # Time signature selection for songs without MIDI
        ttk.Label(controls_frame, text="Time Signature:").pack(side=tk.LEFT, padx=(20, 5))
        self.time_sig_var = tk.StringVar(value="4/4")
        time_sig_combo = ttk.Combobox(controls_frame, textvariable=self.time_sig_var, 
                                     values=["4/4", "3/4", "2/4", "6/8", "9/8", "12/8"], 
                                     state='readonly', width=8)
        time_sig_combo.pack(side=tk.LEFT, padx=5)
        
        # Sheet music display area
        self.sheet_canvas_frame = ttk.Frame(sheet_frame)
        self.sheet_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create matplotlib figure for sheet music
        self.sheet_fig = Figure(figsize=(12, 8), dpi=100)
        self.sheet_canvas = FigureCanvasTkAgg(self.sheet_fig, self.sheet_canvas_frame)
        self.sheet_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        sheet_toolbar = NavigationToolbar2Tk(self.sheet_canvas, self.sheet_canvas_frame)
        sheet_toolbar.update()
        
        # Initialize empty sheet
        self.refresh_sheet_music()
    
    def refresh_sheet_music(self):
        """Refresh sheet music display with template vs performance comparison"""
        try:
            self.sheet_fig.clear()
            
            if self.current_analysis and MODULES_AVAILABLE:
                # Create two subplots: template and performance
                gs = self.sheet_fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
                
                # Get current piece info
                piece_key = self.piece_var.get()
                piece_info = PIECES.get(piece_key, {})
                reference_melody = piece_info.get('melody', [])
                
                if reference_melody and self.current_analysis:
                    # Get analysis data
                    if 'standard_analysis' in self.current_analysis:
                        analysis_data = self.current_analysis['standard_analysis']
                    else:
                        analysis_data = self.current_analysis
                    
                    note_details = analysis_data.get('note_details', [])
                    
                    # Template sheet music (top)
                    ax1 = self.sheet_fig.add_subplot(gs[0])
                    self._draw_sheet_music(ax1, reference_melody, note_details, "template")
                    ax1.set_title("Template (Expected Performance)", fontsize=12, fontweight='bold')
                    
                    # Performance sheet music (bottom) 
                    ax2 = self.sheet_fig.add_subplot(gs[1])
                    self._draw_sheet_music(ax2, reference_melody, note_details, "performance")
                    ax2.set_title("Your Performance", fontsize=12, fontweight='bold')
                    
                    # Add overall legend
                    legend_elements = [
                        mpatches.Patch(facecolor='red', label='Missed Notes'),
                        mpatches.Patch(facecolor='orange', label='Timing Issues'),
                        mpatches.Patch(facecolor='blue', label='Pitch Issues'), 
                        mpatches.Patch(facecolor='green', label='Correct Notes'),
                        mpatches.Patch(facecolor='purple', label='Extra Notes')
                    ]
                    self.sheet_fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
                    
                else:
                    # Fallback display
                    ax = self.sheet_fig.add_subplot(111)
                    ax.text(0.5, 0.5, f"No melody data available for {piece_key}\nLoad a piece with melody data", 
                           ha='center', va='center', fontsize=12, transform=ax.transAxes)
                    ax.axis('off')
            else:
                # No analysis available
                ax = self.sheet_fig.add_subplot(111)
                ax.text(0.5, 0.5, "Run analysis to display sheet music\nwith performance differences", 
                       ha='center', va='center', fontsize=12, transform=ax.transAxes)
                ax.set_title("Sheet Music Visualization", fontsize=14, fontweight='bold')
                ax.axis('off')
            
            self.sheet_canvas.draw()
            
        except Exception as e:
            print(f"Error refreshing sheet music: {e}")
            import traceback
            traceback.print_exc()
            # Show error message on canvas
            self.sheet_fig.clear()
            ax = self.sheet_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error generating sheet music:\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10, color='red')
            ax.set_title("Sheet Music Error", fontsize=14)
            ax.axis('off')
            self.sheet_canvas.draw()
    
    def _draw_sheet_music(self, ax, melody_data, note_details, view_type):
        """Draw sheet music notation with color coding"""
        # Draw staff lines
        staff_y = 2
        staff_width = len(melody_data) * 0.8 + 2
        
        for i in range(5):
            y = staff_y + i * 0.5
            ax.plot([0.5, staff_width], [y, y], color='black', linewidth=1)
        
        # Draw treble clef
        ax.text(0.7, staff_y + 1, '𝄞', fontsize=30, va='center', ha='center')
        
        # Draw notes
        x_pos = 1.5
        note_spacing = 0.8
        
        for i, note_data in enumerate(melody_data):
            # Get analysis info for this note
            note_detail = None
            if i < len(note_details):
                note_detail = note_details[i]
            
            # Calculate note position on staff
            midi_pitch = note_data['pitch']
            # C4 (MIDI 60) is on ledger line below staff
            y_pos = staff_y + (midi_pitch - 60) * 0.25
            
            # Determine color based on view type and analysis
            if view_type == "template":
                # Template view: show missed notes in red, others in black
                if note_detail and note_detail.get('timing_deviation_ms') == 'MISSED':
                    color = 'red'
                else:
                    color = 'black'
            else:
                # Performance view: color code based on accuracy
                if note_detail:
                    if note_detail.get('timing_deviation_ms') == 'MISSED':
                        continue  # Don't draw missed notes in performance view
                    elif isinstance(note_detail.get('timing_deviation_ms'), (int, float)) and abs(note_detail.get('timing_deviation_ms', 0)) > 100:
                        color = 'orange'  # Timing issues
                    elif isinstance(note_detail.get('pitch_deviation_cents'), (int, float)) and abs(note_detail.get('pitch_deviation_cents', 0)) > 50:
                        color = 'blue'  # Pitch issues
                    else:
                        color = 'green'  # Correct
                else:
                    color = 'purple'  # Extra notes
            
            # Draw note head
            circle = plt.Circle((x_pos, y_pos), 0.12, color=color, fill=True)
            ax.add_patch(circle)
            
            # Draw stem
            stem_height = 1.5 if y_pos < staff_y + 2 else -1.5
            stem_x = x_pos + (0.12 if stem_height > 0 else -0.12)
            ax.plot([stem_x, stem_x], [y_pos, y_pos + stem_height], color=color, linewidth=2)
            
            # Add ledger lines if needed
            if y_pos < staff_y - 0.1:
                for ledger_y in np.arange(staff_y - 0.5, y_pos - 0.1, -0.5):
                    ax.plot([x_pos - 0.2, x_pos + 0.2], [ledger_y, ledger_y], color='black', linewidth=1)
            elif y_pos > staff_y + 2.1:
                for ledger_y in np.arange(staff_y + 2.5, y_pos + 0.1, 0.5):
                    ax.plot([x_pos - 0.2, x_pos + 0.2], [ledger_y, ledger_y], color='black', linewidth=1)
            
            x_pos += note_spacing
        
        # Set axis properties
        ax.set_xlim(0, staff_width + 0.5)
        ax.set_ylim(staff_y - 2, staff_y + 4.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def extract_melody_from_midi(self, piece_key):
        """Extract melody from MIDI file if available"""
        try:
            if piece_key in self.available_pieces and 'midi_file' in self.available_pieces[piece_key]:
                midi_path = self.available_pieces[piece_key]['midi_file']
                if os.path.exists(midi_path):
                    import mido
                    mid = mido.MidiFile(midi_path)
                    
                    melody = []
                    current_time = 0.0
                    ticks_per_beat = mid.ticks_per_beat
                    tempo = 500000  # Default 120 BPM
                    
                    for track in mid.tracks:
                        for msg in track:
                            current_time += msg.time
                            if msg.type == 'set_tempo':
                                tempo = msg.tempo
                            elif msg.type == 'note_on' and msg.velocity > 0:
                                # Convert MIDI to melody format
                                time_in_beats = current_time / ticks_per_beat
                                melody.append({
                                    'pitch': msg.note,
                                    'duration': 0.25,  # Default quarter note
                                    'time': time_in_beats
                                })
                    
                    return melody[:16]  # Limit to first 16 notes for display
            
            return None
            
        except Exception as e:
            print(f"Error extracting melody from MIDI: {e}")
            return None
    
    def export_sheet_music(self):
        """Export current sheet music as PNG"""
        try:
            if hasattr(self, 'sheet_fig'):
                filename = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                    title="Export Sheet Music"
                )
                if filename:
                    self.sheet_fig.savefig(filename, dpi=300, bbox_inches='tight')
                    messagebox.showinfo("Success", f"Sheet music exported to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export sheet music:\n{str(e)}")
        
    def on_sheet_note_click(self, note_index, is_performance=False, is_reference=False):
        """Handle clicks on sheet music notes with proper selection and playback options"""
        self.selected_note_index = note_index
        self.selected_note_type = 'performance' if is_performance else 'reference' if is_reference else 'reference'
        
        print(f"Selected {self.selected_note_type} note {note_index + 1}")
        
        # Switch to the notes tab and show details
        self.notebook.select(1)  # Notes tab
        
        # Update the note tree selection (basic tree)
        if hasattr(self, 'note_tree'):
            # Clear current selection
            for item in self.note_tree.selection():
                self.note_tree.selection_remove(item)
            
            # Select the clicked note
            children = self.note_tree.get_children()
            if note_index < len(children):
                item_id = children[note_index]
                self.note_tree.selection_set(item_id)
                self.note_tree.see(item_id)
                self.note_tree.focus(item_id)
                
                # Get the note data and show detailed analysis
                item_values = self.note_tree.item(item_id, 'values')
                if item_values:
                    self.show_detailed_note_analysis(item_values)
        
        # Update the enhanced note tree selection
        if hasattr(self, 'enhanced_note_tree'):
            # Clear current selection
            for item in self.enhanced_note_tree.selection():
                self.enhanced_note_tree.selection_remove(item)
            
            # Select the clicked note
            children = self.enhanced_note_tree.get_children()
            if note_index < len(children):
                item_id = children[note_index]
                self.enhanced_note_tree.selection_set(item_id)
                self.enhanced_note_tree.see(item_id)
                self.enhanced_note_tree.focus(item_id)
        
        # Update the note visualization to highlight selected note
        self.update_note_visualization()
        
        # Update the sheet music visualization to highlight selected note
        self.update_sheet_music_visualization()
        
        # Add playback controls if not already present
        self.update_playback_controls()
    
    def update_playback_controls(self):
        """Update playback controls with play-from-selection option"""
        if not hasattr(self, 'playback_controls_frame'):
            try:
                # Find a suitable parent for playback controls
                # Look for existing control frames
                control_parent = None
                for widget in self.root.winfo_children():
                    if isinstance(widget, ttk.PanedWindow):
                        # Try to get the first pane
                        try:
                            control_parent = widget.winfo_children()[0]
                            break
                        except:
                            continue
                    elif isinstance(widget, ttk.Frame):
                        control_parent = widget
                        break
                
                if control_parent is None:
                    # Create a simple frame if we can't find a good parent
                    control_parent = ttk.Frame(self.root)
                    control_parent.pack(fill=tk.X, padx=10, pady=5)
                
                # Create playback controls frame
                self.playback_controls_frame = ttk.LabelFrame(control_parent, text="🔊 Playback Controls")
                self.playback_controls_frame.pack(fill=tk.X, padx=10, pady=5)
                
                # Play from selection checkbox
                self.play_from_selection = tk.BooleanVar(value=False)
                ttk.Checkbutton(self.playback_controls_frame, 
                              text="Play from selection to end",
                              variable=self.play_from_selection).pack(side=tk.LEFT, padx=5)
                
                # Play both checkbox
                self.play_both = tk.BooleanVar(value=False)
                ttk.Checkbutton(self.playback_controls_frame, 
                              text="Play both (reference + performance)",
                              variable=self.play_both).pack(side=tk.LEFT, padx=5)
                
                # Playback buttons
                ttk.Button(self.playback_controls_frame, 
                         text="▶ Play Performance", 
                         command=self.play_performance_from_selection).pack(side=tk.LEFT, padx=5)
                
                ttk.Button(self.playback_controls_frame, 
                         text="▶ Play Reference", 
                         command=self.play_reference_from_selection).pack(side=tk.LEFT, padx=5)
                
                ttk.Button(self.playback_controls_frame, 
                         text="⏹ Stop", 
                         command=self.stop_playback).pack(side=tk.LEFT, padx=5)
            except Exception as e:
                print(f"Could not create playback controls: {e}")
                # Just skip if we can't create the controls
    
    def play_performance_from_selection(self):
        """Play performance audio from selected note to end"""
        if not self.current_audio_file or self.selected_note_index is None:
            return
        
        try:
            if self.play_both.get():
                # Play both performance and reference
                self.play_both_from_selection()
                return
                
            # Load audio
            y, sr = librosa.load(self.current_audio_file)
            
            if self.play_from_selection.get() and hasattr(self, 'current_analysis'):
                # Get performance notes data (the correct source)
                analysis_data = self.current_analysis.get("standard_analysis", {})
                performance_notes = analysis_data.get("performance_notes", [])
                
                if self.selected_note_index < len(performance_notes):
                    perf_note = performance_notes[self.selected_note_index]
                    start_time = perf_note.get('onset', 0)
                    start_sample = int(start_time * sr)
                    y = y[start_sample:]
            
            # Save and play
            temp_path = "temp_performance_playback.wav"
            sf.write(temp_path, y, sr)
            
            pygame.mixer.music.stop()
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            
        except Exception as e:
            print(f"Error playing performance: {e}")
    
    def play_reference_from_selection(self):
        """Play reference audio from selected note to end"""
        if self.selected_note_index is None:
            return
        
        try:
            if self.play_both.get():
                # Play both performance and reference
                self.play_both_from_selection()
                return
                
            # Get reference melody from current analysis or MIDI template
            melody = []
            if self.current_analysis and 'reference_melody' in self.current_analysis:
                melody = self.current_analysis['reference_melody']
            else:
                # Fallback to PIECES dictionary
                piece_key = self.piece_var.get()
                piece_info = PIECES.get(piece_key, {})
                melody = piece_info.get('melody', [])
            
            # Generate reference tones from selected note to end
            start_idx = self.selected_note_index if self.play_from_selection.get() else 0
            melody_to_play = melody[start_idx:]
            
            if melody_to_play:
                piece_info_to_play = {
                    'title': 'Reference',
                    'melody': melody_to_play
                }
                # Generate and play reference audio
                self.generate_and_play_reference(piece_info_to_play)
            else:
                # If no melody available, try to play the current reference MIDI file
                midi_template = self.piece_var.get()
                if midi_template and midi_template != "Select template":
                    midi_path = os.path.join("midi", midi_template)
                    if os.path.exists(midi_path):
                        # Convert MIDI to audio and play
                        reference_audio_path = midi_path.replace('.mid', '_reference.wav').replace('.midi', '_reference.wav')
                        if os.path.exists(reference_audio_path):
                            pygame.mixer.music.stop()
                            pygame.mixer.music.load(reference_audio_path)
                            pygame.mixer.music.play()
                        else:
                            print(f"Reference audio not found: {reference_audio_path}")
            
        except Exception as e:
            print(f"Error playing reference: {e}")
            
    def play_both_from_selection(self):
        """Play both reference and performance simultaneously from selected note"""
        if self.selected_note_index is None:
            return
            
        try:
            import threading
            import time
            
            # Start both playbacks in separate threads
            def play_performance():
                self.play_both.set(False)  # Temporarily disable to avoid recursion
                self.play_performance_from_selection()
                self.play_both.set(True)
                
            def play_reference():
                time.sleep(0.1)  # Small delay to sync
                self.play_both.set(False)  # Temporarily disable to avoid recursion
                self.play_reference_from_selection()
                self.play_both.set(True)
            
            # Start both threads
            perf_thread = threading.Thread(target=play_performance)
            ref_thread = threading.Thread(target=play_reference)
            
            perf_thread.start()
            ref_thread.start()
            
        except Exception as e:
            print(f"Error playing both: {e}")
    
    def stop_playback(self):
        """Stop current playback"""
        try:
            pygame.mixer.music.stop()
        except Exception as e:
            print(f"Error stopping playback: {e}")
        
    def create_timing_tab(self):
        """Create enhanced timing analysis tab"""
        timing_frame = ttk.Frame(self.notebook)
        self.notebook.add(timing_frame, text="⏱️ Timing")
        
        self.timing_fig = Figure(figsize=(12, 6), dpi=100)
        self.timing_canvas = FigureCanvasTkAgg(self.timing_fig, timing_frame)
        self.timing_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def update_timing_display(self):
        """Update the timing analysis display"""
        if not self.current_analysis:
            return
            
        try:
            self.timing_fig.clear()
            ax = self.timing_fig.add_subplot(111)
            
            # Get timing data from analysis
            standard_analysis = self.current_analysis.get("standard_analysis", {})
            note_details = standard_analysis.get("note_details", [])
            
            if not note_details:
                ax.text(0.5, 0.5, "No timing data available", ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
            else:
                # Extract timing deviations
                note_numbers = []
                timing_devs = []
                
                for i, note in enumerate(note_details):
                    timing_dev = note.get('timing_deviation_ms', 0)
                    if isinstance(timing_dev, (int, float)):
                        note_numbers.append(i + 1)
                        timing_devs.append(timing_dev)
                
                if note_numbers:
                    # Create timing deviation plot
                    colors = ['red' if abs(dev) > 100 else 'orange' if abs(dev) > 50 else 'green' for dev in timing_devs]
                    ax.bar(note_numbers, timing_devs, color=colors, alpha=0.7)
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='±50ms threshold')
                    ax.axhline(y=-50, color='orange', linestyle='--', alpha=0.5)
                    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='±100ms threshold')
                    ax.axhline(y=-100, color='red', linestyle='--', alpha=0.5)
                    
                    ax.set_xlabel('Note Number')
                    ax.set_ylabel('Timing Deviation (ms)')
                    ax.set_title('Timing Analysis: Note-by-Note Deviations')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, "No valid timing data found", ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            
            self.timing_canvas.draw()
            
        except Exception as e:
            print(f"Error updating timing display: {e}")
            ax = self.timing_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            self.timing_canvas.draw()
        
    def create_feedback_tab(self):
        """Create enhanced AI feedback tab"""
        feedback_frame = ttk.Frame(self.notebook)
        self.notebook.add(feedback_frame, text="🤖 AI")
        
        # Enhanced API configuration
        api_frame = ttk.LabelFrame(feedback_frame, text="AI Configuration")
        api_frame.pack(fill=tk.X, padx=10, pady=10)
        
        api_controls = ttk.Frame(api_frame)
        api_controls.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(api_controls, text="Google API Key:").pack(side=tk.LEFT)
        self.api_key_var = tk.StringVar(value=os.environ.get("GOOGLE_API_KEY", ""))
        api_entry = ttk.Entry(api_controls, textvariable=self.api_key_var, show="*", width=40)
        api_entry.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        
        ttk.Button(api_controls, text="Generate Enhanced Feedback", 
                  command=self.generate_enhanced_feedback).pack(side=tk.RIGHT, padx=(10, 0))
        
        # Enhanced feedback display with sections
        feedback_notebook = ttk.Notebook(feedback_frame)
        feedback_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Technical feedback
        tech_frame = ttk.Frame(feedback_notebook)
        feedback_notebook.add(tech_frame, text="Technical Analysis")
        self.tech_feedback_text = scrolledtext.ScrolledText(tech_frame, wrap=tk.WORD, state='disabled')
        self.tech_feedback_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Musical feedback
        musical_frame = ttk.Frame(feedback_notebook)
        feedback_notebook.add(musical_frame, text="Musical Interpretation")
        self.musical_feedback_text = scrolledtext.ScrolledText(musical_frame, wrap=tk.WORD, state='disabled')
        self.musical_feedback_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Practice suggestions
        practice_frame = ttk.Frame(feedback_notebook)
        feedback_notebook.add(practice_frame, text="Practice Suggestions")
        self.practice_feedback_text = scrolledtext.ScrolledText(practice_frame, wrap=tk.WORD, state='disabled')
        self.practice_feedback_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Event handlers and core functionality
    def on_enhanced_note_select(self, event):
        """Handle enhanced note selection"""
        selection = self.enhanced_note_tree.selection()
        if selection:
            item = selection[0]
            values = self.enhanced_note_tree.item(item, 'values')
            if values:
                self.selected_note_index = int(values[0]) - 1  # Convert to 0-based index
                self.show_detailed_note_analysis(values)
                
                # Update basic note tree selection
                if hasattr(self, 'note_tree'):
                    # Clear current selection
                    for tree_item in self.note_tree.selection():
                        self.note_tree.selection_remove(tree_item)
                    
                    # Select the corresponding note
                    children = self.note_tree.get_children()
                    if self.selected_note_index < len(children):
                        item_id = children[self.selected_note_index]
                        self.note_tree.selection_set(item_id)
                        self.note_tree.see(item_id)
                        self.note_tree.focus(item_id)
                
                # Update visualizations
                self.update_note_visualization()
                self.update_sheet_music_visualization()
    
    def on_mistake_select(self, event):
        """Handle mistake pattern selection"""
        selection = self.mistake_tree.selection()
        if selection:
            item = selection[0]
            values = self.mistake_tree.item(item, 'values')
            if values:
                mistake_type, location, severity, description = values
                # Show detailed information about this mistake pattern
                self.show_mistake_details(mistake_type, location, severity, description)
    
    def on_mistake_double_click(self, event):
        """Handle mistake pattern double-click to jump to relevant notes"""
        selection = self.mistake_tree.selection()
        if selection:
            item = selection[0]
            values = self.mistake_tree.item(item, 'values')
            if values:
                mistake_type, location, severity, description = values
                # Extract note numbers from location string
                import re
                note_matches = re.findall(r'\d+', location)
                if note_matches:
                    # Jump to the first note mentioned
                    note_index = int(note_matches[0]) - 1  # Convert to 0-based
                    self.selected_note_index = note_index
                    self.notebook.select(1)  # Switch to Notes tab
                    
                    # Update all tab selections
                    self.sync_note_selection_across_tabs(note_index)
    
    def show_mistake_details(self, mistake_type, location, severity, description):
        """Show detailed analysis of selected mistake pattern"""
        details = f"""
Mistake Pattern: {mistake_type}
Location: {location}  
Severity: {severity}

Description: {description}

Recommendations:
"""
        
        if "Sharp" in mistake_type:
            details += """
• Check your tuning and intonation
• Practice with a tuner to develop better pitch awareness
• Focus on listening to reference pitches before playing
• Work on ear training exercises"""
        elif "Flat" in mistake_type:
            details += """
• Check your instrument's tuning
• Work on breath support (for wind instruments) or bow pressure (for strings)
• Practice scales with a tuner
• Focus on maintaining consistent embouchure/technique"""
        elif "Rushing" in mistake_type:
            details += """
• Practice with a metronome at slower tempos
• Focus on listening to the beat rather than rushing ahead
• Work on subdivision exercises
• Record yourself playing to develop tempo awareness"""
        elif "Dragging" in mistake_type:
            details += """
• Practice with a metronome to develop steady tempo
• Work on maintaining energy throughout long notes
• Focus on anticipating beat subdivisions
• Practice clapping rhythms before playing"""
        elif "Retry" in mistake_type:
            details += """
• Work on building confidence through slow practice
• Practice difficult passages in small sections
• Focus on mental preparation before starting
• Develop a consistent pre-performance routine"""
        
        # Update the detail text area
        if hasattr(self, 'detail_text'):
            self.detail_text.config(state='normal')
            self.detail_text.delete(1.0, tk.END)
            self.detail_text.insert(1.0, details)
            self.detail_text.config(state='disabled')
    
    def sync_note_selection_across_tabs(self, note_index):
        """Synchronize note selection across all tabs and visualizations"""
        self.selected_note_index = note_index
        
        # Update basic note tree
        if hasattr(self, 'note_tree'):
            for item in self.note_tree.selection():
                self.note_tree.selection_remove(item)
            children = self.note_tree.get_children()
            if note_index < len(children):
                item_id = children[note_index]
                self.note_tree.selection_set(item_id)
                self.note_tree.see(item_id)
                self.note_tree.focus(item_id)
        
        # Update enhanced note tree
        if hasattr(self, 'enhanced_note_tree'):
            for item in self.enhanced_note_tree.selection():
                self.enhanced_note_tree.selection_remove(item)
            children = self.enhanced_note_tree.get_children()
            if note_index < len(children):
                item_id = children[note_index]
                self.enhanced_note_tree.selection_set(item_id)
                self.enhanced_note_tree.see(item_id)
                self.enhanced_note_tree.focus(item_id)
        
        # Update visualizations
        self.update_note_visualization()
        self.update_sheet_music_visualization()
        
        # Update playback controls
        self.update_playback_controls()
    
    def on_note_double_click(self, event):
        """Handle note double-click for playback"""
        self.play_selected_note()
    
    def show_detailed_note_analysis(self, note_values):
        """Show detailed analysis for selected note"""
        if not self.current_analysis:
            return
            
        note_num, expected, detected, pitch_error, timing_error, duration, dynamics, accuracy, issues = note_values
        
        # Update note info label
        self.note_info_label.config(text=f"Note {note_num}: {expected} → {detected}")
        
        # Update detailed analysis text
        self.detail_text.config(state='normal')
        self.detail_text.delete(1.0, tk.END)
        
        analysis_text = f"""
DETAILED NOTE ANALYSIS - Note {note_num}

📊 BASIC METRICS
Expected Pitch: {expected}
Detected Pitch: {detected}
Pitch Error: {pitch_error}
Timing Error: {timing_error}
Duration: {duration}
Dynamics: {dynamics}
Overall Accuracy: {accuracy}

⚠️  IDENTIFIED ISSUES
{issues if issues != 'None' else 'No significant issues detected'}

🎯 ANALYSIS DETAILS
"""
        
        # Add detailed analysis from current_analysis
        note_details = self.current_analysis.get("standard_analysis", {}).get("note_details", [])
        if self.selected_note_index < len(note_details):
            note_detail = note_details[self.selected_note_index]
            
            # Pitch analysis
            analysis_text += f"""
Pitch Analysis:
- Expected frequency: {self._pitch_to_frequency(expected):.2f} Hz
- Detected frequency: {self._pitch_to_frequency(detected):.2f} Hz
- Deviation in cents: {pitch_error}
- Pitch stability: {'Good' if pitch_error != 'MISSED' and abs(float(pitch_error.replace('¢', ''))) < 20 else 'Needs work' if pitch_error != 'MISSED' else 'Note missed'}

Timing Analysis:
- Expected timing: {note_detail.get('expected_time', 'N/A')} seconds
- Detected timing: {note_detail.get('detected_time', 'N/A')} seconds
- Timing precision: {'Excellent' if timing_error != 'MISSED' and abs(float(timing_error.replace('ms', ''))) < 50 else 'Needs work' if timing_error != 'MISSED' else 'Note missed'}

Musical Context:
- Note position in phrase: {self._get_phrase_position(int(note_num))}
- Harmonic function: {self._get_harmonic_function(expected)}
- Difficulty level: {self._assess_note_difficulty(note_detail)}
"""
        
        self.detail_text.insert(1.0, analysis_text)
        self.detail_text.config(state='disabled')
        
        # Update visualization
        self.update_note_visualization()
    
    def detect_mistake_patterns(self):
        """Detect and analyze mistake patterns in the performance"""
        if not self.current_analysis:
            messagebox.showwarning("No Analysis", "Please run analysis first")
            return
            
        self.progress.start()
        
        def analyze_mistakes():
            try:
                # Analyze note details for patterns
                note_details = self.current_analysis.get("standard_analysis", {}).get("note_details", [])
                
                # Pattern detection
                patterns = []
                
                # 1. Pitch pattern mistakes
                pitch_errors = []
                for i, note in enumerate(note_details):
                    pitch_dev = note.get('pitch_deviation_cents', 0)
                    if isinstance(pitch_dev, (int, float)) and abs(pitch_dev) > 20:
                        pitch_errors.append((i, pitch_dev))
                
                if len(pitch_errors) > 2:
                    # Check for systematic sharp/flat tendencies
                    sharp_count = sum(1 for _, dev in pitch_errors if dev > 0)
                    flat_count = len(pitch_errors) - sharp_count
                    
                    if sharp_count > flat_count * 2:
                        patterns.append({
                            'type': 'Systematic Sharp',
                            'location': f"Notes {', '.join([str(i+1) for i, _ in pitch_errors[:3]])}...",
                            'severity': 'Medium',
                            'description': 'Consistent tendency to play notes too high'
                        })
                    elif flat_count > sharp_count * 2:
                        patterns.append({
                            'type': 'Systematic Flat',
                            'location': f"Notes {', '.join([str(i+1) for i, _ in pitch_errors[:3]])}...",
                            'severity': 'Medium',
                            'description': 'Consistent tendency to play notes too low'
                        })
                
                # 2. Timing pattern mistakes
                timing_errors = []
                for i, note in enumerate(note_details):
                    timing_dev = note.get('timing_deviation_ms', 0)
                    if isinstance(timing_dev, (int, float)) and abs(timing_dev) > 50:
                        timing_errors.append((i, timing_dev))
                
                if len(timing_errors) > 2:
                    early_count = sum(1 for _, dev in timing_errors if dev < 0)
                    late_count = len(timing_errors) - early_count
                    
                    if early_count > late_count * 2:
                        patterns.append({
                            'type': 'Rushing',
                            'location': f"Notes {', '.join([str(i+1) for i, _ in timing_errors[:3]])}...",
                            'severity': 'High',
                            'description': 'Consistent tendency to play notes early'
                        })
                    elif late_count > early_count * 2:
                        patterns.append({
                            'type': 'Dragging',
                            'location': f"Notes {', '.join([str(i+1) for i, _ in timing_errors[:3]])}...",
                            'severity': 'High',
                            'description': 'Consistent tendency to play notes late'
                        })
                
                # 3. Section-based mistakes (potential retries)
                self._detect_retry_patterns(note_details, patterns)
                
                self.mistake_patterns = patterns
                
                # Update UI
                self.root.after(0, self.update_mistake_display)
                
            except Exception as e:
                print(f"Error in mistake analysis: {e}")
            finally:
                self.root.after(0, self.progress.stop)
        
        threading.Thread(target=analyze_mistakes, daemon=True).start()
    
    def _detect_retry_patterns(self, note_details, patterns):
        """Detect potential retry patterns in performance"""
        # Look for sequences where timing suddenly resets or jumps backwards
        retry_candidates = []
        
        for i in range(1, len(note_details)):
            current_time = note_details[i].get('detected_time', 0)
            prev_time = note_details[i-1].get('detected_time', 0)
            
            if isinstance(current_time, (int, float)) and isinstance(prev_time, (int, float)):
                # Check for significant backward jump in timing
                if current_time < prev_time - 1.0:  # 1 second threshold
                    retry_candidates.append(i)
        
        if retry_candidates:
            patterns.append({
                'type': 'Potential Retries',
                'location': f"Around notes {', '.join([str(i+1) for i in retry_candidates])}",
                'severity': 'Low',
                'description': f'Detected {len(retry_candidates)} potential restart(s) or correction(s)'
            })
    
    def update_mistake_display(self):
        """Update the mistake analysis display"""
        # Clear existing items
        for item in self.mistake_tree.get_children():
            self.mistake_tree.delete(item)
        
        # Add detected patterns
        for pattern in self.mistake_patterns:
            self.mistake_tree.insert('', tk.END, values=(
                pattern['type'],
                pattern['location'],
                pattern['severity'],
                pattern['description']
            ))
        
        # Update retry analysis text
        self.retry_text.config(state='normal')
        self.retry_text.delete(1.0, tk.END)
        
        retry_analysis = """
RETRY PATTERN ANALYSIS

This analysis looks for sections where the performer may have:
- Made a mistake and tried again
- Stopped and restarted from an earlier point
- Corrected timing or pitch mid-performance

"""
        
        retry_patterns = [p for p in self.mistake_patterns if p['type'] == 'Potential Retries']
        if retry_patterns:
            retry_analysis += f"Detected {len(retry_patterns)} potential retry pattern(s).\n\n"
            for pattern in retry_patterns:
                retry_analysis += f"• {pattern['description']} at {pattern['location']}\n"
        else:
            retry_analysis += "No clear retry patterns detected. This suggests a continuous performance without major restarts."
        
        self.retry_text.insert(1.0, retry_analysis)
        self.retry_text.config(state='disabled')
    
    def update_performance_diff(self):
        """Update performance difference visualization"""
        if not self.current_analysis:
            return
        
        self.diff_fig.clear()
        
        diff_type = self.diff_type_var.get()
        note_details = self.current_analysis.get("standard_analysis", {}).get("note_details", [])
        
        if diff_type == "pitch_timing":
            self._create_pitch_timing_diff(note_details)
        elif diff_type == "rhythm_patterns":
            self._create_rhythm_pattern_diff(note_details)
        elif diff_type == "section_analysis":
            self._create_section_analysis_diff(note_details)
        
        self.diff_fig.tight_layout()
        self.diff_canvas.draw()
    
    def _create_pitch_timing_diff(self, note_details):
        """Create pitch vs timing difference visualization"""
        if not note_details:
            return
            
        ax = self.diff_fig.add_subplot(111)
        
        pitch_errors = []
        timing_errors = []
        note_numbers = []
        
        for i, note in enumerate(note_details):
            pitch_dev = note.get('pitch_deviation_cents', 0)
            timing_dev = note.get('timing_deviation_ms', 0)
            
            if isinstance(pitch_dev, (int, float)) and isinstance(timing_dev, (int, float)):
                pitch_errors.append(pitch_dev)
                timing_errors.append(timing_dev)
                note_numbers.append(i + 1)
        
        if note_numbers:
            # Create scatter plot
            colors = []
            for pitch_err, timing_err in zip(pitch_errors, timing_errors):
                if abs(pitch_err) < 25 and abs(timing_err) < 50:
                    colors.append('green')
                elif abs(pitch_err) < 50 and abs(timing_err) < 100:
                    colors.append('blue')
                elif abs(pitch_err) < 100 and abs(timing_err) < 200:
                    colors.append('orange')
                else:
                    colors.append('red')
            
            scatter = ax.scatter(timing_errors, pitch_errors, c=colors, alpha=0.7, s=50)
            
            # Add note numbers as labels
            for i, (x, y, note_num) in enumerate(zip(timing_errors, pitch_errors, note_numbers)):
                ax.annotate(str(note_num), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax.set_xlabel('Timing Error (ms)')
            ax.set_ylabel('Pitch Error (cents)')
            ax.set_title('Performance Accuracy: Pitch vs Timing')
            ax.grid(True, alpha=0.3)
            
            # Add accuracy zones
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.axhspan(-25, 25, alpha=0.1, color='green', label='Excellent pitch')
            ax.axvspan(-50, 50, alpha=0.1, color='green', label='Excellent timing')
            
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No pitch/timing data available', 
                   ha='center', va='center', transform=ax.transAxes)
            for p, t in zip(pitch_errors, timing_errors):
                if abs(p) < 20 and abs(t) < 50:
                    colors.append('green')
                elif abs(p) < 50 and abs(t) < 100:
                    colors.append('orange')
                else:
                    colors.append('red')
            
            scatter = ax.scatter(timing_errors, pitch_errors, c=colors, alpha=0.7, s=100)
            
            # Add note numbers as labels
            for i, (t, p, n) in enumerate(zip(timing_errors, pitch_errors, note_numbers)):
                ax.annotate(str(n), (t, p), xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax.set_xlabel('Timing Error (ms)')
            ax.set_ylabel('Pitch Error (cents)')
            ax.set_title('Performance Deviation Analysis')
            ax.grid(True, alpha=0.3)
            
            # Add target zone
            ax.axhspan(-20, 20, alpha=0.2, color='green', label='Excellent Zone')
            ax.axvspan(-50, 50, alpha=0.2, color='green')
            
            ax.legend()
    
    # Audio playback methods
    def play_selected_note(self):
        """Play the selected note from the performance with improved extraction"""
        if self.selected_note_index is None or not self.current_audio_file:
            messagebox.showwarning("No Selection", "Please select a note first")
            return

        try:
            # Load audio and extract note section
            y, sr = librosa.load(self.current_audio_file)
            
            # Get note timing from analysis
            note_details = self.current_analysis.get("standard_analysis", {}).get("note_details", [])
            if self.selected_note_index < len(note_details):
                note = note_details[self.selected_note_index]
                start_time = note.get('actual_time', 0)
                
                print(f"🎵 Playing note {self.selected_note_index + 1} at time {start_time:.2f}s")
                
                # Improved note extraction with onset-based boundaries
                # Get all onsets to find natural note boundaries
                onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
                
                # Find the onset closest to our note time
                if len(onsets) > 0:
                    onset_idx = np.argmin(np.abs(onsets - start_time))
                    note_start = onsets[onset_idx]
                    
                    # Find next onset for note end, or use default duration
                    if onset_idx + 1 < len(onsets):
                        note_end = onsets[onset_idx + 1]
                    else:
                        note_end = note_start + 1.0  # Default 1 second
                    
                    # Ensure reasonable note length (0.3 to 2 seconds)
                    note_duration = note_end - note_start
                    if note_duration < 0.3:
                        note_end = note_start + 0.5
                    elif note_duration > 2.0:
                        note_end = note_start + 1.0
                else:
                    # Fallback if no onsets detected
                    note_start = max(0, start_time - 0.1)
                    note_end = start_time + 0.8
                
                # Extract audio segment
                start_sample = int(note_start * sr)
                end_sample = int(note_end * sr)
                
                # Ensure samples are within bounds
                start_sample = max(0, start_sample)
                end_sample = min(len(y), end_sample)
                
                note_audio = y[start_sample:end_sample]
                
                # Apply fade in/out to avoid clicks
                fade_samples = int(0.01 * sr)  # 10ms fade
                if len(note_audio) > 2 * fade_samples:
                    # Fade in
                    note_audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
                    # Fade out
                    note_audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                
                # Save temporary file and play
                import soundfile as sf
                temp_path = "temp_note.wav"
                sf.write(temp_path, note_audio, sr)
                
                # Stop any current playback
                pygame.mixer.music.stop()
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                
        except Exception as e:
            messagebox.showerror("Playback Error", f"Could not play note: {e}")

    def play_reference_note(self):
        """Play the reference note for the selected note"""
        if self.selected_note_index is None:
            messagebox.showwarning("No Selection", "Please select a note first")
            return
            
        try:
            # Get the expected pitch for the selected note from analysis
            note_details = self.current_analysis.get("standard_analysis", {}).get("note_details", [])
            if self.selected_note_index < len(note_details):
                note = note_details[self.selected_note_index]
                expected_pitch = note.get('expected_pitch', 'C4')
                
                # Convert pitch name to MIDI if needed
                if isinstance(expected_pitch, str):
                    try:
                        import music21
                        pitch_midi = music21.pitch.Pitch(expected_pitch).midi
                    except:
                        pitch_midi = 60  # Default to middle C
                else:
                    pitch_midi = expected_pitch
                
                print(f"🎵 Playing reference note: {expected_pitch} (MIDI {pitch_midi})")
                duration = 1.0  # Default duration
                
                # Generate reference tone
                freq = librosa.midi_to_hz(pitch_midi)
                sr = 22050
                t = np.linspace(0, duration, int(sr * duration))
                
                # Create a pleasant sine wave with harmonics
                audio = np.sin(2 * np.pi * freq * t)
                audio += 0.3 * np.sin(2 * np.pi * freq * 2 * t)  # Second harmonic
                audio += 0.1 * np.sin(2 * np.pi * freq * 3 * t)  # Third harmonic
                
                # Apply envelope for natural sound
                attack = int(0.05 * sr)  # 50ms attack
                decay = int(0.1 * sr)    # 100ms decay
                sustain_level = 0.7
                release = int(0.2 * sr)  # 200ms release
                
                envelope = np.ones(len(audio))
                if len(audio) > attack:
                    envelope[:attack] = np.linspace(0, 1, attack)
                if len(audio) > attack + decay:
                    envelope[attack:attack + decay] = np.linspace(1, sustain_level, decay)
                if len(audio) > release:
                    envelope[-release:] = np.linspace(envelope[-release], 0, release)
                
                audio *= envelope
                audio *= 0.3  # Reduce volume
                
                # Save and play
                import soundfile as sf
                temp_path = "temp_reference.wav"
                sf.write(temp_path, audio, sr)
                
                pygame.mixer.music.stop()
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                
        except Exception as e:
            messagebox.showerror("Reference Error", f"Could not play reference note: {e}")

    def play_reference_audio(self):
        """Play the full reference audio"""
        try:
            reference_file = f"{self.piece_var.get()}_reference.wav"
            if os.path.exists(reference_file):
                pygame.mixer.music.stop()
                pygame.mixer.music.load(reference_file)
                pygame.mixer.music.play()
            else:
                # Generate reference if it doesn't exist
                if MODULES_AVAILABLE:
                    analyzer = MusicAnalyzer(piece_key=self.piece_var.get())
                    if analyzer.create_reference_data():
                        pygame.mixer.music.load(reference_file)
                        pygame.mixer.music.play()
                    else:
                        messagebox.showerror("Error", "Could not generate reference audio")
                else:
                    messagebox.showerror("Error", "Analysis modules not available")
        except Exception as e:
            messagebox.showerror("Error", f"Could not play reference audio: {str(e)}")
    
    def play_performance_section(self):
        """Play a selected section of the performance"""
        if hasattr(self, 'current_audio_file') and self.current_audio_file:
            try:
                pygame.mixer.music.load(self.current_audio_file)
                pygame.mixer.music.play()
                self.update_status("Playing performance section...")
            except Exception as e:
                messagebox.showerror("Error", f"Could not play performance: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No audio file loaded.")
    
    def stop_playback(self):
        """Stop audio playback"""
        pygame.mixer.music.stop()
    
    # Utility methods
    def _pitch_to_frequency(self, pitch_name):
        """Convert pitch name to frequency"""
        try:
            return librosa.note_to_hz(pitch_name)
        except:
            return 440.0  # Default A4
    
    def _get_phrase_position(self, note_num):
        """Determine position of note within musical phrase"""
        # Simple heuristic based on note position
        if note_num <= 4:
            return "Beginning of phrase"
        elif note_num <= 8:
            return "Middle of phrase"
        else:
            return "End of phrase"
    
    def _get_harmonic_function(self, pitch):
        """Get harmonic function of the pitch"""
        # Simplified harmonic analysis
        note_map = {
            'C': 'Tonic', 'D': 'Supertonic', 'E': 'Mediant',
            'F': 'Subdominant', 'G': 'Dominant', 'A': 'Submediant', 'B': 'Leading tone'
        }
        base_note = pitch[0] if pitch else 'C'
        return note_map.get(base_note, 'Unknown')
    
    def _assess_note_difficulty(self, note_detail):
        """Assess the difficulty level of the note"""
        pitch_error = note_detail.get('pitch_deviation_cents', 0)
        timing_error = note_detail.get('timing_deviation_ms', 0)
        
        if isinstance(pitch_error, (int, float)) and isinstance(timing_error, (int, float)):
            total_error = abs(pitch_error) / 10 + abs(timing_error) / 50
            if total_error < 3:
                return "Easy"
            elif total_error < 6:
                return "Moderate"
            else:
                return "Difficult"
        return "Unknown"
    
    # Placeholder methods for interface completeness
    def open_audio_file(self):
        """Open audio file dialog"""
        filetypes = [("Audio files", "*.wav *.mp3 *.flac *.aac")]
        filename = filedialog.askopenfilename(title="Select Audio File", filetypes=filetypes)
        if filename:
            self.current_audio_file = filename
            self.file_label.config(text=os.path.basename(filename))
    
    def load_demo(self):
        """Load demo file"""
        demo_file = "audio/demo_performance.wav"  # Updated path
        if os.path.exists(demo_file):
            self.current_audio_file = demo_file
            self.file_label.config(text=demo_file)
            
            # Create a larger popup window for demo loaded message
            popup = tk.Toplevel(self.root)
            popup.title("Demo Loaded")
            popup.geometry("400x200")
            popup.resizable(False, False)
            
            # Center the popup
            popup.transient(self.root)
            popup.grab_set()
            
            label = tk.Label(popup, text="Demo file loaded successfully!\n\nClick 'Analyze' to run the analysis.",
                           font=('Arial', 12), wraplength=350, justify='center')
            label.pack(pady=20)
            
            button_frame = tk.Frame(popup)
            button_frame.pack(pady=10)
            
            ok_button = tk.Button(button_frame, text="OK", command=popup.destroy, width=10)
            ok_button.pack(side=tk.LEFT, padx=5)
            
            analyze_button = tk.Button(button_frame, text="Analyze Now", 
                                     command=lambda: [popup.destroy(), self.run_full_analysis()], width=12)
            analyze_button.pack(side=tk.LEFT, padx=5)
            
        else:
            messagebox.showerror("Demo Error", "Demo file not found")
    
    def run_full_analysis(self):
        """Run full analysis of the performance"""
        if not self.current_audio_file:
            messagebox.showwarning("No File", "Please select an audio file first")
            return
        
        self.progress.start()
        
        def analyze():
            try:
                if MODULES_AVAILABLE:
                    analyzer = MusicAnalyzer(piece_key=self.piece_var.get())
                    
                    # Use original analysis method (your DWT algorithm)
                    result = analyzer.analyze_with_enhancements(
                        self.current_audio_file, 
                        generate_visualizations=False,  # Disable to prevent threading issues
                        detect_polyphony=True,          # Enable polyphony detection
                        analyze_timing=True             # Enable timing analysis
                    )
                    
                    # The method returns (f0, times, onsets, enhanced_analysis)
                    if result and len(result) == 4:
                        f0, times, onsets, enhanced_analysis = result
                        self.current_analysis = enhanced_analysis
                        
                        # Optional: Try polyphonic enhancement if available
                        if hasattr(self, 'improved_analyzer') and self.improved_analyzer:
                            try:
                                print("� Adding polyphonic enhancement...")
                                enhanced_analysis = self.improved_analyzer.enhance_analysis(
                                    self.current_audio_file, enhanced_analysis
                                )
                            except Exception as e:
                                print(f"⚠️ Polyphonic enhancement failed: {e}")
                    else:
                        messagebox.showerror("Analysis Error", "Analysis returned unexpected format")
                        return
                    
                    # Update UI
                    self.root.after(0, self.update_all_displays)
                else:
                    messagebox.showerror("Error", "Analysis modules not available")
            except Exception as e:
                messagebox.showerror("Analysis Error", f"Analysis failed: {e}")
            finally:
                self.root.after(0, self.progress.stop)
        
        threading.Thread(target=analyze, daemon=True).start()
    
    def update_all_displays(self):
        """Update all display elements with current analysis"""
        if not self.current_analysis:
            return
        
        self.update_enhanced_note_tree()
        self.update_overview_metrics()
        self.update_performance_diff()
        self.update_sheet_music_visualization()
        self.update_timing_display()
        
        # Automatically detect mistake patterns after analysis
        if hasattr(self, 'current_analysis') and self.current_analysis:
            self.detect_mistake_patterns()
        
    def update_sheet_music_visualization(self):
        """Update the interactive sheet music visualization with analysis results"""
        if not self.current_analysis:
            return
            
        try:
            # Clear the figure first
            self.sheet_fig.clear()
            
            # Get the current piece information
            piece_key = self.piece_var.get()
            piece_info = PIECES.get(piece_key, {})
            
            # Get the analysis data
            standard_analysis = self.current_analysis.get("standard_analysis", {})
            note_details = standard_analysis.get("note_details", [])
            
            if not note_details:
                # Show a message if no analysis data
                ax = self.sheet_fig.add_subplot(111)
                ax.text(0.5, 0.5, "No analysis data available\nRun analysis first", 
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            else:
                # Create two subplots for template vs performance comparison
                ax1 = self.sheet_fig.add_subplot(2, 1, 1)
                ax2 = self.sheet_fig.add_subplot(2, 1, 2)
                
                # Extract template melody from piece info
                template_melody = piece_info.get('melody', [])
                
                if template_melody:
                    # Plot template (reference)
                    ax1.set_title("Reference Template", fontsize=12, fontweight='bold')
                    times = [note.get('start_time', 0) for note in template_melody]
                    pitches = []
                    durations = [note.get('duration', 0.5) for note in template_melody]
                    
                    # Convert pitches to MIDI numbers
                    for note in template_melody:
                        pitch = note.get('pitch', 60)
                        if isinstance(pitch, str):
                            try:
                                import music21
                                pitch = music21.pitch.Pitch(pitch).midi
                            except:
                                pitch = 60  # Default to middle C
                        pitches.append(pitch)
                    
                    # Create horizontal bars for note durations
                    for i, (time, pitch, duration) in enumerate(zip(times, pitches, durations)):
                        ax1.barh(pitch, duration, left=time, height=0.8, 
                                alpha=0.7, color='blue', label='Template' if i == 0 else "")
                    
                    # Plot performance
                    ax2.set_title("Your Performance", fontsize=12, fontweight='bold')
                    perf_times = []
                    perf_pitches = []
                    perf_colors = []
                    
                    for detail in note_details:
                        if 'actual_pitch' in detail and 'actual_time' in detail:
                            # Convert pitch names to MIDI numbers if needed
                            actual_pitch = detail['actual_pitch']
                            if isinstance(actual_pitch, str):
                                try:
                                    import music21
                                    actual_pitch = music21.pitch.Pitch(actual_pitch).midi
                                except:
                                    actual_pitch = 60  # Default to middle C
                            
                            perf_times.append(detail['actual_time'])
                            perf_pitches.append(actual_pitch)
                            
                            # Color code by pitch accuracy
                            pitch_error = abs(detail.get('pitch_deviation_cents', 100))
                            if pitch_error < 50:  # Less than 50 cents
                                perf_colors.append('green')
                            elif pitch_error < 100:  # Less than 100 cents
                                perf_colors.append('orange')
                            else:
                                perf_colors.append('red')
                    
                    # Plot performance notes with color coding
                    for i, (time, pitch, color) in enumerate(zip(perf_times, perf_pitches, perf_colors)):
                        duration = 0.5  # Default duration for visualization
                        ax2.barh(pitch, duration, left=time, height=0.8, 
                                alpha=0.7, color=color)
                    
                    # Set labels and formatting
                    ax1.set_ylabel('MIDI Pitch')
                    ax2.set_ylabel('MIDI Pitch') 
                    ax2.set_xlabel('Time (seconds)')
                    
                    # Add legend for performance colors
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='green', alpha=0.7, label='Excellent'),
                        Patch(facecolor='orange', alpha=0.7, label='Good'),
                        Patch(facecolor='red', alpha=0.7, label='Poor/Missed')
                    ]
                    ax2.legend(handles=legend_elements, loc='upper right')
                    
                    # Set reasonable axis limits
                    all_pitches = pitches + perf_pitches
                    if all_pitches:
                        pitch_min, pitch_max = min(all_pitches), max(all_pitches)
                        pitch_range = pitch_max - pitch_min
                        ax1.set_ylim(pitch_min - pitch_range*0.1, pitch_max + pitch_range*0.1)
                        ax2.set_ylim(pitch_min - pitch_range*0.1, pitch_max + pitch_range*0.1)
                    
                    all_times = times + perf_times
                    if all_times:
                        time_max = max(all_times + [sum(durations)])
                        ax1.set_xlim(0, time_max * 1.1)
                        ax2.set_xlim(0, time_max * 1.1)
                
                else:
                    # No template available - show performance only
                    ax = self.sheet_fig.add_subplot(111)
                    ax.text(0.5, 0.5, "Template melody not available\nShowing performance analysis only", 
                           ha='center', va='center', fontsize=12, transform=ax.transAxes)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
            
            # Adjust layout and refresh
            self.sheet_fig.tight_layout()
            self.sheet_canvas.draw()
            
        except Exception as e:
            print(f"Error updating sheet music: {e}")
            # Show error message
            self.sheet_fig.clear()
            ax = self.sheet_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error displaying sheet music:\n{str(e)}", 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            self.sheet_canvas.draw()
            
        except Exception as e:
            print(f"Error updating sheet music: {e}")
            # Show error message
            ax = self.sheet_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error displaying sheet music:\n{str(e)}", 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            self.sheet_canvas.draw()
        
    def update_enhanced_note_tree(self):
        """Update the enhanced note tree with detailed information"""
        # Clear existing items
        for item in self.enhanced_note_tree.get_children():
            self.enhanced_note_tree.delete(item)
        
        if not self.current_analysis:
            return
        
        note_details = self.current_analysis.get("standard_analysis", {}).get("note_details", [])
        accuracy_counts = {'Excellent': 0, 'Good': 0, 'Poor': 0, 'Missed': 0}
        
        for note in note_details:
            note_num = note.get('note_index', 'N/A')
            expected_pitch = note.get('expected_pitch', 'N/A')
            detected_pitch = note.get('actual_pitch', 'N/A')
            pitch_error = note.get('pitch_deviation_cents', 'N/A')
            timing_error = note.get('timing_deviation_ms', 'N/A')
            expected_time = note.get('expected_time', 'N/A')
            actual_time = note.get('actual_time', 'N/A')
            
            # Enhanced analysis
            duration = "Normal"  # Placeholder
            dynamics = "mf"      # Placeholder
            
            # Determine accuracy and issues
            issues = []
            if isinstance(pitch_error, (int, float)):
                if abs(pitch_error) > 50:
                    issues.append("Pitch")
                pitch_error_str = f"{pitch_error}¢"
            else:
                pitch_error_str = str(pitch_error)
                
            if isinstance(timing_error, (int, float)):
                if abs(timing_error) > 100:
                    issues.append("Timing")
                timing_error_str = f"{timing_error}ms"
            else:
                timing_error_str = str(timing_error)
            
            # Calculate accuracy
            if isinstance(pitch_error, (int, float)) and isinstance(timing_error, (int, float)):
                if abs(pitch_error) < 20 and abs(timing_error) < 50:
                    accuracy = "Excellent"
                elif abs(pitch_error) < 50 and abs(timing_error) < 100:
                    accuracy = "Good"
                else:
                    accuracy = "Poor"
            else:
                accuracy = "Missed"
                issues.append("Missing")
            
            accuracy_counts[accuracy] += 1
            
            issues_str = ", ".join(issues) if issues else "None"
            
            # Insert into tree
            self.enhanced_note_tree.insert('', tk.END, values=(
                note_num, expected_pitch, detected_pitch, pitch_error_str,
                timing_error_str, duration, dynamics, accuracy, issues_str
            ))
        
        # Update accuracy labels
        for accuracy, count in accuracy_counts.items():
            self.accuracy_labels[accuracy].config(text=str(count))
    
    def update_overview_metrics(self):
        """Update overview metrics visualization"""
        if not self.current_analysis:
            return
        
        self.metrics_fig.clear()
        
        # Create subplots for different metrics
        ax1 = self.metrics_fig.add_subplot(221)
        ax2 = self.metrics_fig.add_subplot(222)
        ax3 = self.metrics_fig.add_subplot(223)
        ax4 = self.metrics_fig.add_subplot(224)
        
        note_details = self.current_analysis.get("standard_analysis", {}).get("note_details", [])
        
        # Plot 1: Pitch accuracy over time
        note_nums = []
        pitch_errors = []
        for note in note_details:
            if isinstance(note.get('pitch_deviation_cents'), (int, float)):
                note_nums.append(note.get('note_index', 0))
                pitch_errors.append(abs(note.get('pitch_deviation_cents', 0)))
        
        if note_nums:
            ax1.plot(note_nums, pitch_errors, 'b-o', markersize=4)
            ax1.set_title('Pitch Accuracy')
            ax1.set_ylabel('Error (cents)')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Timing accuracy over time
        timing_errors = []
        for note in note_details:
            if isinstance(note.get('timing_deviation_ms'), (int, float)):
                timing_errors.append(abs(note.get('timing_deviation_ms', 0)))
        
        if len(timing_errors) == len(note_nums):
            ax2.plot(note_nums, timing_errors, 'r-o', markersize=4)
            ax2.set_title('Timing Accuracy')
            ax2.set_ylabel('Error (ms)')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Overall accuracy distribution
        overall_assessment = self.current_analysis.get("standard_analysis", {}).get("overall_assessment", {})
        metrics = ['Completion Rate', 'Pitch Accuracy', 'Timing Accuracy']
        values = [
            overall_assessment.get('completion_rate', 0),
            overall_assessment.get('pitch_accuracy', 0),
            overall_assessment.get('timing_accuracy', 0)
        ]
        
        ax3.bar(metrics, values, color=['green', 'blue', 'orange'])
        ax3.set_title('Overall Performance')
        ax3.set_ylabel('Percentage')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Accuracy distribution pie chart
        accuracy_counts = {'Excellent': 0, 'Good': 0, 'Poor': 0, 'Missed': 0}
        for note in note_details:
            pitch_error = note.get('pitch_deviation_cents', 0)
            timing_error = note.get('timing_deviation_ms', 0)
            
            if isinstance(pitch_error, (int, float)) and isinstance(timing_error, (int, float)):
                if abs(pitch_error) < 20 and abs(timing_error) < 50:
                    accuracy_counts['Excellent'] += 1
                elif abs(pitch_error) < 50 and abs(timing_error) < 100:
                    accuracy_counts['Good'] += 1
                else:
                    accuracy_counts['Poor'] += 1
            else:
                accuracy_counts['Missed'] += 1
        
        labels = []
        sizes = []
        colors = ['darkgreen', 'blue', 'orange', 'red']
        plot_colors = []
        
        for label, color in zip(accuracy_counts.keys(), colors):
            if accuracy_counts[label] > 0:
                labels.append(f"{label}: {accuracy_counts[label]}")
                sizes.append(accuracy_counts[label])
                plot_colors.append(color)
        
        if sizes:
            ax4.pie(sizes, labels=labels, colors=plot_colors, autopct='%1.1f%%')
            ax4.set_title('Note Quality Distribution')
        
        self.metrics_fig.tight_layout()
        self.metrics_canvas.draw()
    
    def update_note_visualization(self):
        """Update note-specific visualization with actual waveform and spectrum using improved segmentation"""
        if self.selected_note_index is None or not self.current_analysis or not self.current_audio_file:
            return
        
        try:
            # Load audio data
            y, sr = librosa.load(self.current_audio_file)
            
            # Get performance notes from analysis (the correct source)
            analysis_data = self.current_analysis.get("standard_analysis", {})
            note_details = analysis_data.get("note_details", [])
            
            if self.selected_note_index >= len(note_details):
                return
                
            note = note_details[self.selected_note_index]
            start_time = note.get('actual_time', 0)
            
            # Use the same improved segmentation as play_selected_note
            onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
            
            # Find the onset closest to our note time
            if len(onsets) > 0:
                onset_idx = np.argmin(np.abs(onsets - start_time))
                note_start = onsets[onset_idx]
                
                # Find next onset for note end, or use default duration
                if onset_idx + 1 < len(onsets):
                    note_end = onsets[onset_idx + 1]
                else:
                    note_end = note_start + 1.0  # Default 1 second
                
                # Ensure reasonable note length (0.3 to 2 seconds)
                note_duration = note_end - note_start
                if note_duration < 0.3:
                    note_end = note_start + 0.5
                elif note_duration > 2.0:
                    note_end = note_start + 1.0
            else:
                # Fallback if no onsets detected
                note_start = max(0, start_time - 0.1)
                note_end = start_time + 0.8
            
            # Extract audio segment using the same logic as playback
            start_sample = int(note_start * sr)
            end_sample = int(note_end * sr)
            
            # Ensure samples are within bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(y), end_sample)
            
            note_audio = y[start_sample:end_sample]
            
            if len(note_audio) < sr * 0.1:  # Too short
                note_audio = y[start_sample:start_sample + int(sr * 0.5)]  # Use 0.5 seconds
            
            # Clear and create subplots
            self.note_fig.clear()
            ax1 = self.note_fig.add_subplot(2, 2, 1)
            ax2 = self.note_fig.add_subplot(2, 2, 2)
            ax3 = self.note_fig.add_subplot(2, 2, 3)
            ax4 = self.note_fig.add_subplot(2, 2, 4)
            
            # Plot 1: Waveform with onset boundaries marked
            time_axis = np.linspace(note_start, note_start + len(note_audio) / sr, len(note_audio))
            ax1.plot(time_axis, note_audio)
            ax1.axvline(x=note_start, color='green', linestyle='--', label='Note Start')
            ax1.axvline(x=note_start + len(note_audio) / sr, color='red', linestyle='--', label='Note End')
            ax1.set_title(f'Note {self.selected_note_index + 1} Waveform (Onset-Based)')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Spectrum
            if len(note_audio) > 0:
                fft = np.fft.fft(note_audio)
                freq = np.fft.fftfreq(len(note_audio), 1/sr)
                magnitude = np.abs(fft)
                
                # Only plot positive frequencies up to 2000 Hz
                positive_freq_mask = (freq > 0) & (freq < 2000)
                ax2.plot(freq[positive_freq_mask], magnitude[positive_freq_mask])
                ax2.set_title('Frequency Spectrum')
                ax2.set_xlabel('Frequency (Hz)')
                ax2.set_ylabel('Magnitude')
                ax2.grid(True, alpha=0.3)
                
                # Mark expected frequency
                expected_pitch = note.get('expected_pitch', 'C4')
                expected_freq = librosa.note_to_hz(expected_pitch)
                ax2.axvline(x=expected_freq, color='red', linestyle='--', 
                           label=f'Expected: {expected_pitch} ({expected_freq:.1f} Hz)')
                ax2.legend()
            
            # Plot 3: Pitch tracking with improved analysis
            try:
                f0, voiced_flag, voiced_probs = librosa.pyin(note_audio, 
                                                           fmin=librosa.note_to_hz('C2'), 
                                                           fmax=librosa.note_to_hz('C7'),
                                                           sr=sr)
                times = librosa.frames_to_time(np.arange(len(f0)), sr=sr) + note_start
                
                # Only plot voiced segments
                voiced_f0 = np.copy(f0)
                voiced_f0[~voiced_flag] = np.nan
                
                ax3.plot(times, voiced_f0, 'b-', alpha=0.7, label='Detected Pitch')
                ax3.set_title('Pitch Tracking (Voiced Only)')
                ax3.set_xlabel('Time (s)')
                ax3.set_ylabel('Fundamental Frequency (Hz)')
                ax3.grid(True, alpha=0.3)
                
                # Add expected pitch line
                expected_pitch = note.get('expected_pitch', 'C4')
                expected_freq = librosa.note_to_hz(expected_pitch)
                ax3.axhline(y=expected_freq, color='r', linestyle='--', 
                           label=f'Expected: {expected_pitch} ({expected_freq:.1f} Hz)')
                
                # Add detected pitch (if available)
                detected_freq = note.get('raw_detected_freq')
                if detected_freq:
                    ax3.axhline(y=detected_freq, color='g', linestyle='--', 
                               label=f'Detected: {detected_freq:.1f} Hz')
                
                ax3.legend()
                
            except Exception as e:
                ax3.text(0.5, 0.5, f'Pitch tracking failed:\n{str(e)}', 
                        ha='center', va='center', transform=ax3.transAxes)
            
            # Plot 4: Spectogram
            try:
                D = librosa.stft(note_audio)
                DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                img = librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='hz', ax=ax4,
                                             x_coords=np.linspace(note_start, note_start + len(note_audio)/sr, DB.shape[1]))
                ax4.set_title('Spectrogram')
                self.note_fig.colorbar(img, ax=ax4, format='%+2.0f dB')
            except Exception as e:
                ax4.text(0.5, 0.5, f'Spectrogram failed:\n{str(e)}', 
                        ha='center', va='center', transform=ax4.transAxes)
            
            self.note_fig.tight_layout()
            self.note_canvas.draw()
            
        except Exception as e:
            # Fallback to simple visualization
            self.note_fig.clear()
            ax = self.note_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Note {self.selected_note_index + 1}\nVisualization Error:\n{str(e)}", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Note {self.selected_note_index + 1} Analysis")
            self.note_fig.tight_layout()
            self.note_canvas.draw()
    
    # Additional placeholder methods
    def analyze_performance_sections(self):
        """Analyze performance by sections"""
        pass
    
    def generate_enhanced_feedback(self):
        """Generate enhanced AI feedback"""
        pass
    
    def export_detailed_report(self):
        """Export detailed analysis report"""
        pass
    
    def show_sheet_music(self):
        """Show sheet music tab"""
        self.notebook.select(4)  # Sheet music tab index
    
    def show_mistake_analysis(self):
        """Show mistake analysis tab"""
        self.notebook.select(2)  # Mistake analysis tab index
    
    def show_performance_diff(self):
        """Show performance diff tab"""
        self.notebook.select(3)  # Performance diff tab index
    
    def show_note_details(self):
        """Show note details tab"""
        self.notebook.select(1)  # Note-by-note tab index
    
    def toggle_note_numbers(self):
        """Toggle note numbers on sheet music"""
        pass
    
    def highlight_mistakes(self):
        """Highlight mistakes on sheet music"""
        pass
    
    def export_analysis(self):
        """Export basic analysis"""
        pass
    
    def batch_process_csv(self):
        """Batch process all entries in the CSV file with non-empty J columns"""
        import csv
        from datetime import datetime
        
        csv_file = "data/abrsm_lmth25.csv"
        if not os.path.exists(csv_file):
            messagebox.showerror("Error", f"CSV file not found: {csv_file}")
            return
        
        # Create progress dialog
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Batch Processing")
        progress_window.geometry("500x300")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Progress text area
        progress_text = scrolledtext.ScrolledText(progress_window, height=15, width=60)
        progress_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Progress bar
        batch_progress = ttk.Progressbar(progress_window, mode='determinate')
        batch_progress.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        def log_progress(message):
            progress_text.insert(tk.END, message + "\n")
            progress_text.see(tk.END)
            progress_window.update()
        
        def process_batch():
            try:
                # Read CSV file
                import csv
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
                log_progress(f"✓ Loaded CSV with {len(data)} entries")
                
                # Filter entries with non-empty feedback (column J = feedback)
                valid_entries = [row for row in data if row.get('feedback', '').strip()]
                log_progress(f"✓ Found {len(valid_entries)} entries with feedback")
                
                batch_progress['maximum'] = len(valid_entries)
                training_data = []
                
                for idx, row in valid_entries.iterrows():
                    try:
                        performance_id = row['performance_id']
                        log_progress(f"Processing {performance_id}...")
                        
                        # Look for audio files
                        audio1_path = f"audio/{performance_id}_1.mp3"
                        audio2_path = f"audio/{performance_id}_2.mp3"
                        
                        piece1_features = None
                        piece2_features = None
                        
                        # Process piece 1
                        if os.path.exists(audio1_path):
                            piece1_features = self.analyze_audio_file(
                                audio1_path, 
                                row['title_piece_1'], 
                                row['composer_piece_1']
                            )
                            log_progress(f"  ✓ Analyzed piece 1: {row['title_piece_1']}")
                        else:
                            log_progress(f"  ⚠️  Audio file not found: {audio1_path}")
                        
                        # Process piece 2
                        if os.path.exists(audio2_path):
                            piece2_features = self.analyze_audio_file(
                                audio2_path, 
                                row['title_piece_2'], 
                                row['composer_piece_2']
                            )
                            log_progress(f"  ✓ Analyzed piece 2: {row['title_piece_2']}")
                        else:
                            log_progress(f"  ⚠️  Audio file not found: {audio2_path}")
                        
                        # Create training data entry
                        training_entry = {
                            'performance_id': performance_id,
                            'age_group': row['age_group'],
                            'ability_group': row['ability_group'],
                            'title_piece_1': row['title_piece_1'],
                            'composer_piece_1': row['composer_piece_1'],
                            'title_piece_2': row['title_piece_2'],
                            'composer_piece_2': row['composer_piece_2'],
                            'mark': row['mark'],
                            'feedback': row['feedback'],
                            'piece1_features': piece1_features,
                            'piece2_features': piece2_features
                        }
                        
                        training_data.append(training_entry)
                        batch_progress['value'] = len(training_data)
                        batch_progress.update()
                        
                    except Exception as e:
                        log_progress(f"  ❌ Error processing {performance_id}: {str(e)}")
                        continue
                
                # Save training data
                output_file = "data/training_data.json"
                with open(output_file, 'w') as f:
                    json.dump(training_data, f, indent=2, default=str)
                
                log_progress(f"\n✅ Batch processing complete!")
                log_progress(f"✓ Processed {len(training_data)} entries")
                log_progress(f"✓ Training data saved to: {output_file}")
                
            except Exception as e:
                log_progress(f"❌ Batch processing failed: {str(e)}")
        
        # Start processing in a separate thread
        threading.Thread(target=process_batch, daemon=True).start()
    
    def analyze_audio_file(self, audio_path, title, composer):
        """Analyze a single audio file and return features"""
        try:
            # Find matching MIDI template
            template_key = None
            midi_path = None
            used_midi = True
            
            # Try to find matching MIDI file
            for piece_key, piece_info in self.available_pieces.items():
                if (title.lower() in piece_info['title'].lower() or 
                    piece_info['title'].lower() in title.lower()):
                    template_key = piece_key
                    if 'midi_file' in piece_info:
                        midi_path = piece_info['midi_file']
                    break
            
            # If no MIDI found, use MP3 template analysis (stub for now)
            if not midi_path:
                used_midi = False
                return self.analyze_with_mp3_template(audio_path, title, composer)
            
            # Perform analysis with MIDI template
            if MODULES_AVAILABLE:
                analyzer = MusicAnalyzer(piece_key=template_key)
                results = analyzer.compare_performances(audio_path)
                
                if results:
                    features = {
                        'title': title,
                        'composer': composer,
                        'used_midi_template': used_midi,
                        'midi_template_path': midi_path,
                        'analysis_results': results,
                        'audio_path': audio_path
                    }
                    return features
            
            return None
            
        except Exception as e:
            print(f"Error analyzing {audio_path}: {e}")
            return None
    
    def analyze_with_mp3_template(self, audio_path, title, composer):
        """Stub for MP3 template analysis - to be implemented later"""
        from datetime import datetime
        return {
            'title': title,
            'composer': composer,
            'used_midi_template': False,
            'mp3_template_analysis': True,
            'audio_path': audio_path,
            'analysis_results': {
                'note': 'MP3 template analysis not yet implemented',
                'timestamp': str(datetime.now())
            }
        }
    
    def load_xml_template(self):
        """Load an XML template file and add it to available pieces"""
        try:
            # File dialog for XML files
            filetypes = [
                ("XML files", "*.xml *.mxl *.musicxml"),
                ("All files", "*.*")
            ]
            filename = filedialog.askopenfilename(
                title="Select XML Template File", 
                filetypes=filetypes
            )
            
            if not filename:
                return
            
            # Get template name from user
            template_name = tk.simpledialog.askstring(
                "Template Name", 
                "Enter a name for this template:",
                initialvalue=os.path.splitext(os.path.basename(filename))[0]
            )
            
            if not template_name:
                return
            
            # Parse the XML file using our musicxml_parser
            from musicxml_parser import extract_melody_from_musicxml, get_musicxml_tempo
            
            print(f"📋 Loading XML template: {filename}")
            melody_notes = extract_melody_from_musicxml(filename)
            
            if not melody_notes:
                messagebox.showerror("Error", "Could not extract melody from XML file")
                return
            
            # Get tempo from XML or use default
            tempo = get_musicxml_tempo(filename)
            
            # Create template entry
            template_key = template_name.lower().replace(' ', '_')
            new_template = {
                'title': template_name,
                'composer': 'Custom',
                'tempo': tempo,
                'time_signature': (4, 4),  # Default
                'melody': melody_notes,
                'source_file': filename
            }
            
            # Add to PIECES dictionary (import it to modify)
            global PIECES
            PIECES[template_key] = new_template
            
            # Update the piece selection dropdown
            if hasattr(self, 'piece_var'):
                current_values = list(self.piece_combo['values'])
                if template_key not in current_values:
                    new_values = current_values + [template_key]
                    self.piece_combo['values'] = new_values
                    self.piece_var.set(template_key)  # Select the new template
            
            messagebox.showinfo(
                "Success", 
                f"Template '{template_name}' loaded successfully!\n"
                f"Notes: {len(melody_notes)}\n"
                f"Tempo: {tempo} BPM\n"
                f"Duration: {melody_notes[-1]['start_time'] + melody_notes[-1]['duration']:.1f}s"
            )
            
            print(f"✅ Template '{template_name}' added to pieces ({len(melody_notes)} notes)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load XML template: {e}")
            print(f"❌ XML template loading failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedABRSMGUI(root)
    root.mainloop()

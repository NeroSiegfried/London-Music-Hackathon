# Musical Performance Analyzer - Simplified Architecture

A modern, streamlined system for analyzing musical performances against reference pieces using advanced audio processing and machine learning techniques.

## 🎯 New Simplified Architecture

This application has been completely refactored to follow a clean three-stage pipeline:

### 1. Data Extraction & Mapping
- **Audio Feature Extraction**: Uses librosa for robust onset detection and chroma analysis
- **Ground-Truth Mapping**: Creates detailed structural maps from melody data or MusicXML files
- **Template Generation**: Builds infallible reference templates with timing and pitch information

### 2. Sequence Alignment  
- **Dynamic Time Warping (DTW)**: Aligns performance events with template events
- **Pitch Set Comparison**: Uses Jaccard distance for chord/note comparison
- **Error Classification**: Identifies matched notes, added notes, and missed notes

### 3. Contextual Analysis & Visualization
- **Performance Metrics**: Calculates completion rate, pitch accuracy, timing accuracy
- **Measure-by-Measure Analysis**: Provides tempo analysis per measure
- **Visual Feedback**: Generates charts showing timing and pitch deviations

## ✨ Features

- **Multiple Interface Options**:
  - Command-line interface for batch processing
  - Modern GUI with tabbed interface for interactive analysis
  - Backwards compatibility with existing code

- **Comprehensive Analysis**:
  - Note-by-note pitch and timing analysis
  - Tempo consistency measurement
  - Visual performance feedback
  - JSON export of detailed results

- **Built-in Reference Pieces**:
  - Twinkle, Twinkle, Little Star
  - Mary Had a Little Lamb
  - Extensible to custom pieces via melody data or MusicXML

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd musical-performance-analyzer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Test installation**:
   ```bash
   python test_new_architecture.py
   ```

### Launch GUI (Recommended)

```bash
python launch_analyzer.py
```

### Command Line Analysis

```bash
python launch_analyzer.py performance.wav --piece twinkle --tempo 100 --output my_analysis
```

## 📖 Usage Guide

### GUI Interface

The GUI provides an intuitive interface with three main tabs:

1. **Summary Tab**: Overall performance metrics and assessment
2. **Note Details Tab**: Note-by-note breakdown with timing and pitch accuracy
3. **Visualization Tab**: Charts showing performance deviations

**Steps to analyze:**
1. Click "Browse" to select your audio file
2. Choose reference piece and set tempo
3. Click "Analyze Performance"
4. Review results in the tabs
5. Export results if needed

### Python API

```python
from main_analyzer import PerformanceAnalyzer, PIECES

# Create analyzer
analyzer = PerformanceAnalyzer()

# Set reference piece
analyzer.set_piece_from_melody(PIECES["twinkle"], tempo_bpm=100)

# Analyze performance
results = analyzer.analyze_performance("performance.wav", "output_prefix")

# Access results
assessment = results['standard_analysis']['overall_assessment']
print(f"Completion Rate: {assessment['completion_rate']:.1f}%")
print(f"Pitch Accuracy: {assessment['pitch_accuracy']:.1f}%")
print(f"Timing Accuracy: {assessment['timing_accuracy']:.1f}%")
```

## 🏗️ Architecture Overview

### Core Files

- **`main_analyzer.py`**: Core analysis engine with DTW alignment
- **`gui_analyzer.py`**: Modern GUI interface with tabbed display
- **`launch_analyzer.py`**: Universal launcher for GUI or CLI
- **`test_new_architecture.py`**: Comprehensive test suite

### Analysis Pipeline

```
Audio File → Feature Extraction → Template Mapping → DTW Alignment → Contextual Analysis → Results + Visualization
```

### Key Components

1. **`extract_musical_features()`**: Extracts onset times and pitch sets from audio
2. **`build_template_map_from_melody()`**: Creates reference templates from melody data
3. **`align_sequences()`**: DTW alignment between performance and template
4. **`analyze_performance()`**: Generates comprehensive analysis results
5. **`PerformanceAnalyzer`**: Main class orchestrating the analysis

## 📊 Output Format

Results are provided in comprehensive JSON format:

```json
{
  "summary": {
    "overall_tempo_ratio": 1.05,
    "tempo_consistency_std_dev": 0.12,
    "notes_matched": 24,
    "notes_added": 2,
    "notes_missed": 1
  },
  "standard_analysis": {
    "overall_assessment": {
      "completion_rate": 95.8,
      "pitch_accuracy": 87.5,
      "timing_accuracy": 79.2
    },
    "note_details": [
      {
        "note_index": 1,
        "expected_pitch": "C",
        "actual_pitch": "C",
        "expected_time": 0.0,
        "detected_time": 0.05,
        "pitch_deviation_cents": 0,
        "timing_deviation_ms": 50
      }
    ]
  },
  "measure_by_measure_analysis": [...],
  "alignment_details": {...}
}
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python test_new_architecture.py

# Run with pytest for detailed output
pytest test_new_architecture.py -v

# Test specific components
pytest test_new_architecture.py::TestFeatureExtraction -v
```

Tests cover:
- Audio feature extraction
- Template map generation  
- DTW sequence alignment
- Performance analysis
- Integration testing with demo files

## 📦 Dependencies

### Core Requirements
- `librosa>=0.10.0` - Audio analysis and feature extraction
- `numpy>=1.21.0` - Numerical computations
- `scipy>=1.7.0` - Scientific computing
- `matplotlib>=3.5.0` - Visualization
- `music21>=8.0.0` - Music notation and analysis
- `mido>=1.2.10` - MIDI file handling

### Development Tools
- `pytest>=7.0.0` - Testing framework
- `rapidfuzz>=2.0.0` - Fuzzy string matching

## 📁 File Structure

```
├── main_analyzer.py          # Core analysis engine (NEW)
├── gui_analyzer.py           # Modern GUI interface (NEW)
├── launch_analyzer.py        # Universal launcher (NEW)
├── test_new_architecture.py  # Comprehensive tests (NEW)
├── requirements.txt          # Updated dependencies
├── README_new.md            # This documentation
│
├── demo_performance.wav      # Sample audio for testing
├── twinkle_reference.wav     # Reference audio files
├── mary_reference.wav
│
└── Legacy files (preserved for compatibility):
    ├── enhanced_main_fixed.py
    ├── enhanced_gui_interface.py
    ├── README.md (original)
    └── Other utilities...
```

## 🔄 Migration from Legacy Code

The new architecture provides backwards compatibility:

```python
# Old way (still works)
from enhanced_main_fixed import MusicAnalyzer
analyzer = MusicAnalyzer("twinkle", tempo=100)

# New way (recommended)
from main_analyzer import PerformanceAnalyzer, PIECES
analyzer = PerformanceAnalyzer()
analyzer.set_piece_from_melody(PIECES["twinkle"], 100)
```

## 🎨 Customization

### Adding New Reference Pieces

```python
# Add to PIECES dictionary in main_analyzer.py
PIECES["your_song"] = {
    "title": "Your Song Title",
    "melody": [
        {'pitch': 60, 'duration': 0.25},  # C quarter note
        {'pitch': 62, 'duration': 0.25},  # D quarter note
        # ... more notes
    ]
}
```

### Custom Analysis Parameters

```python
analyzer = PerformanceAnalyzer()
analyzer.set_piece_from_melody(piece_info, tempo_bpm=120)

# Analyze with custom output
results = analyzer.analyze_performance(
    "performance.wav", 
    output_prefix="custom_analysis"
)
```

## 🚀 Future Enhancements

- Support for custom MusicXML reference pieces
- Real-time analysis during recording
- Advanced visualization options (staff notation, piano roll)
- Multiple performance comparison
- Machine learning-based feedback suggestions
- Web interface for remote analysis

## 🤝 Contributing

1. Follow the three-stage pipeline architecture
2. Add tests for new features in `test_new_architecture.py`
3. Update documentation for API changes
4. Ensure backwards compatibility when possible

## 📄 License

[Add your license information here]

## 🆚 Comparison: Old vs New Architecture

| Aspect | Old Architecture | New Architecture |
|--------|------------------|------------------|
| **Complexity** | 1800+ lines GUI, complex interdependencies | Clean 3-stage pipeline, ~600 lines total |
| **Maintainability** | Monolithic, hard to extend | Modular, easy to test and extend |
| **Performance** | Multiple redundant computations | Efficient DTW-based alignment |
| **Testing** | Limited test coverage | Comprehensive test suite |
| **Documentation** | Sparse inline comments | Full API documentation |
| **Extensibility** | Hard to add new pieces | Simple melody/MusicXML addition |
| **Dependencies** | Heavy, many unused packages | Minimal, focused dependencies |

The new architecture provides the same functionality with significantly improved maintainability, performance, and extensibility.

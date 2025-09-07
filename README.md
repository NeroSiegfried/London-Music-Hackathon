# Enhanced ABRSM Music Analysis System

## 🎵 Overview

This is a comprehensive music performance analysis system designed for ABRSM (Associated Board of the Royal Schools of Music) style assessments. The system provides detailed feedback on musical performances including pitch accuracy, timing analysis, and mistake detection.

## ✨ Key Features

### 🎧 Advanced Audio Analysis
- **Variable hop length** for raw audio analysis
- Adaptive processing for different musical content
- High-precision pitch detection using PYIN algorithm
- Multi-resolution timing analysis

### 🎹 Polyphonic Support
- **DWT (Dynamic Time Warping) algorithm** with flexibility for polyphonic pieces
- Handles chord notes detected in different orders
- Robust alignment for complex musical textures
- Automatic polyphonic content detection

### �� Interactive Sheet Music
- **Visual sheet music** with performance difference highlighting
- Color-coded accuracy indicators:
  - 🟢 Green: Correct notes
  - 🔵 Blue: Pitch issues  
  - 🟡 Orange: Timing issues
  - 🔴 Red: Missed notes
  - 🟣 Purple: Extra notes
- Template vs performance comparison views

### 🔊 Audio Playback
- **Template vs performance note playback**
- Individual note segment extraction
- Synchronized reference and performance audio
- Accurate timing preservation
- Interactive note selection and playback

### 📊 Comprehensive Analysis
- Note-by-note accuracy assessment
- Mistake pattern detection
- Performance timing analysis
- AI-powered feedback generation
- Detailed performance metrics

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Activate the virtual environment
source test_env/bin/activate

# Install dependencies (if needed)
pip install librosa matplotlib tkinter pygame soundfile numpy scipy music21
```

### 2. Launch the GUI
```bash
python3 launch_gui.py
```

### 3. Run Tests
```bash
# Basic system test
python3 test_system_integration.py

# Comprehensive demo of all features
python3 comprehensive_demo.py
```

## 📁 Core Files

### Main Components
- `enhanced_main_fixed.py` - Core music analysis engine
- `enhanced_gui_interface.py` - Interactive GUI interface
- `launch_gui.py` - GUI launcher script

### Analysis Modules
- `audio_digitizer.py` - Audio processing and digitization
- `polyphonic_analyzer.py` - Polyphonic music analysis
- `musicxml_parser.py` - MusicXML file parsing
- `sheet_music_visualizer.py` - Sheet music visualization
- `time_signature_analyzer.py` - Rhythm and timing analysis
- `interactive_sheet_music.py` - Interactive music notation

### Testing & Demo
- `test_system_integration.py` - System integration tests
- `comprehensive_demo.py` - Feature demonstration script

## 🎯 Usage Examples

### Basic Performance Analysis
```python
from enhanced_main_fixed import MusicAnalyzer

analyzer = MusicAnalyzer()
result = analyzer.analyze_with_enhancements("performance.wav")
```

### GUI Mode
1. Launch the GUI: `python3 launch_gui.py`
2. Click "Browse" to select an audio file
3. Choose a reference piece from the dropdown
4. Click "🔍 Analyze Performance"
5. Explore results in different tabs:
   - 📊 Overview: Performance summary
   - 🎵 Notes: Note-by-note analysis
   - 🎯 Mistakes: Error patterns
   - 📊 Diff: Performance comparison
   - 🎼 Sheet: Visual sheet music
   - ⏱️ Timing: Rhythm analysis
   - 🤖 AI: Generated feedback

## 🎼 Supported Formats

### Audio Input
- WAV files (recommended)
- MP3 files
- Other formats supported by librosa

### Reference Templates
- Built-in pieces: "Twinkle Twinkle Little Star", "Mary Had a Little Lamb"
- MusicXML files (.mxl, .xml)
- MIDI files (.mid, .midi)

## ⚙️ Technical Details

### Variable Hop Length Analysis
The system uses adaptive hop lengths for different analysis phases:
- **256 samples**: Detailed pitch tracking
- **512 samples**: Standard note detection  
- **1024 samples**: Rhythm pattern analysis
- **2048 samples**: Long-term structure analysis

### DWT Algorithm Flexibility
Dynamic Time Warping provides:
- Flexible note alignment
- Compensation for timing variations
- Handling of chord note order differences
- Robust tempo fluctuation tolerance

### Sheet Music Visualization
- Matplotlib-based rendering
- Interactive note selection
- Real-time highlighting
- Export capabilities

## 🧪 Testing

The system includes comprehensive testing:

```bash
# Run all tests
python3 test_system_integration.py

# Feature demonstration
python3 comprehensive_demo.py
```

Test coverage includes:
- ✅ Core imports and initialization
- ✅ Audio analysis functionality  
- ✅ GUI creation and components
- ✅ Polyphonic analysis features
- ✅ Audio playback system
- ✅ Code documentation quality

## 📚 Code Quality

The codebase maintains high standards:
- **47.6% documentation ratio** (comments + docstrings)
- Comprehensive error handling
- Clear variable naming conventions
- Modular architecture
- Type hints for better understanding

## 🎵 Sample Workflow

1. **Load Audio**: Import performance recording
2. **Select Template**: Choose reference piece or load MusicXML
3. **Analyze**: Run comprehensive analysis with variable hop length
4. **Review Results**: Examine note-by-note accuracy
5. **Visual Feedback**: View sheet music with performance diffs
6. **Listen**: Play back individual notes or sections
7. **Practice**: Use feedback to improve performance

## 🔧 Troubleshooting

### Common Issues
- **Import errors**: Ensure virtual environment is activated
- **Audio playback**: Check pygame/soundfile installation
- **GUI not starting**: Verify tkinter is available
- **Analysis errors**: Ensure audio file is not corrupted

### Performance Tips
- Use WAV files for best analysis quality
- Ensure good recording quality (minimal background noise)
- Provide clear note articulation for better detection
- Use appropriate reference templates

## 📈 Performance Metrics

The system provides detailed metrics:
- **Overall Score**: 0-100 composite score
- **Note Accuracy**: Percentage of correctly played notes
- **Timing Score**: Rhythmic precision assessment  
- **Pitch Score**: Intonation accuracy measurement
- **Individual Note Analysis**: Per-note feedback

## 🎯 Advanced Features

### Mistake Pattern Detection
- Automatic identification of common errors
- Retry pattern recognition
- Performance section analysis
- Detailed improvement suggestions

### AI-Powered Feedback
- Contextual performance advice
- Technical improvement recommendations
- Musical interpretation guidance
- Practice strategy suggestions

---

**🎵 Ready to analyze music performances!** 

Launch the system with `python3 launch_gui.py` and start exploring the comprehensive music analysis capabilities.

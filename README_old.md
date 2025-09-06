# ABRSM AI Music Feedback System 🎼

**Enhanced Competition-Ready Version - London Music Technology Hackathon 2025**

A comprehensive AI system that analyzes music performances and provides constructive feedback, combining advanced audio signal processing with large language models to support music education. Now featuring **sheet music visualization**, **time signature analysis**, and **polyphonic music support**.

## 🎯 Challenge Brief

ABRSM (The Associated Board of the Royal Schools of Music) challenged us to explore how AI can support music education by building a system that generates feedback on music performances, combining audio analysis and language modeling to create scalable tools for music assessment.

## ✨ Features

### **Core Analysis Engine**
- **Real-time Audio Analysis**: Extracts pitch, timing, and rhythmic information
- **Multiple Pieces Support**: "Twinkle, Twinkle, Little Star" and "Mary Had a Little Lamb"
- **AI-Powered Feedback**: Uses Google's Gemini API for personalized feedback
- **Self-Contained References**: Automatically generates MIDI and audio references

### **🆕 Enhanced Features**
- **📊 Sheet Music Visualization**: Visual comparison showing reference vs performance on staff notation
- **⏱️ Time Signature Analysis**: Detects time signatures, analyzes beat patterns, and timing compensation
- **🎹 Polyphonic Music Support**: Handles multiple simultaneous notes (piano, chords, harmonies)
- **📈 Advanced Visualizations**: Timing charts, rhythm analysis, and difference highlighting
- **🔍 Voice Leading Analysis**: For complex harmonic content
- **🎯 Complexity Scoring**: Automatic difficulty assessment

### **Professional Features**
- **Multiple Analysis Modes**: Simple, Enhanced, or Custom configurations
- **Demo Mode**: Built-in demo functionality for easy testing
- **Professional CLI**: Extensive command-line options
- **Error Resilience**: Comprehensive error handling and validation

## 🚀 Quick Start

### **🖥️ GUI Interface (Recommended)**
```bash
# Setup and launch interactive GUI
source venv/bin/activate
python launch_gui.py
```

**GUI Features:**
- **📊 Interactive Analysis**: Visual overview of performance metrics
- **🎼 Sheet Music View**: Reference vs performance comparison with error highlighting  
- **⏱️ Timing Analysis**: Beat patterns and tempo analysis
- **🎵 Note-by-Note Analysis**: Detailed breakdown of each note with accuracy scoring
- **🤖 AI Feedback**: Integrated LLM feedback generation
- **📁 File Management**: Easy audio file loading and analysis export

### Demo Mode (Command Line)
```bash
# Setup
source venv/bin/activate
export GOOGLE_API_KEY='your_api_key_here'

# Run enhanced demo with all features
python enhanced_main.py --enhanced --demo
```

### Enhanced Analysis (Your Own Recording)
```bash
# Full analysis with visualizations
python enhanced_main.py --enhanced my_performance.wav

# Simple analysis only
python enhanced_main.py --simple my_performance.wav

# Analysis without visualizations
python enhanced_main.py --no-visualizations my_performance.wav
```

## 🛠 Installation

1. **Quick Setup:**
   ```bash
   ./setup.sh
   ```

2. **Manual Setup:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Get Google AI Studio API Key:**
   - Visit [Google AI Studio](https://aistudio.google.com/)
   - Create a new API key
   - Set environment variable: `export GOOGLE_API_KEY='your_key_here'`

## 📊 How It Works

### **1. Reference Generation**
- Creates "perfect" MIDI and synthesized audio versions
- Enhanced synthesis with harmonics and natural envelopes
- Multiple pieces supported with easy extensibility

### **2. Performance Analysis**
- **Pitch Extraction**: librosa's PYIN algorithm for robust pitch tracking
- **Onset Detection**: Multiple methods for rhythm analysis
- **Polyphonic Detection**: Handles multiple simultaneous pitches
- **Time Signature Detection**: Automatic rhythm pattern recognition

### **3. Intelligent Comparison**
- **Smart Alignment**: Matches performed notes with reference notes
- **Multi-dimensional Analysis**: Pitch (cents), timing (ms), rhythm patterns
- **Voice Separation**: Melody vs harmony analysis for complex pieces
- **Beat Analysis**: Strong vs weak beat accuracy

### **4. Enhanced Visualization**
- **Sheet Music Generation**: Visual staff notation with differences highlighted
- **Timing Charts**: Beat consistency and rhythm pattern analysis
- **Difference Maps**: Color-coded accuracy indicators

### **5. AI Feedback Generation**
- **Context-Aware Prompts**: Incorporates timing and complexity analysis
- **Educational Focus**: Constructive, encouraging feedback
- **Technical Translation**: Converts analysis data to musical language

## 📈 Technical Architecture

```
Audio Input → Multi-Modal Analysis → Feature Extraction → Visualization Generation
     ↓              ↓                     ↓                     ↓
Monophonic     Polyphonic         Time Signature      Sheet Music + Charts
Analysis    →   Analysis      →    Analysis      →     Visual Comparison
     ↓              ↓                     ↓                     ↓
         Comprehensive JSON Report → Enhanced LLM Prompt → Natural Language Feedback
```

## 🎵 Analysis Capabilities

### **Monophonic Analysis** (Single melody line)
- ✅ Pitch accuracy (measured in cents)
- ✅ Timing precision (measured in milliseconds)  
- ✅ Note completion rate
- ✅ Rhythm pattern adherence

### **🆕 Polyphonic Analysis** (Multiple simultaneous notes)
- ✅ Chord progression detection
- ✅ Voice leading analysis
- ✅ Melody vs harmony separation
- ✅ Complexity scoring (0-100)
- ✅ Multi-pitch onset detection

### **🆕 Time Signature Analysis**
- ✅ Automatic time signature detection (4/4, 3/4, 2/4, etc.)
- ✅ Strong vs weak beat accuracy
- ✅ Timing compensation detection
- ✅ Beat consistency scoring
- ✅ Rhythmic difficulty assessment

### **🆕 Visual Analysis**
- ✅ Sheet music notation generation
- ✅ Reference vs performance comparison
- ✅ Color-coded accuracy indicators
- ✅ Timing deviation charts
- ✅ Beat pattern visualizations

## � Supported Content

### **Built-in Pieces**
- **Twinkle, Twinkle, Little Star** (`--piece twinkle`) - Beginner level
- **Mary Had a Little Lamb** (`--piece mary`) - Elementary level

### **Music Complexity Support**
- **Simple Melodies**: Single note sequences ✅
- **Piano Pieces**: Multiple simultaneous notes ✅
- **Chord Progressions**: Harmonic analysis ✅
- **Mixed Textures**: Melody + accompaniment ✅

*New pieces can be easily added by extending the PIECES dictionary*

## 🏆 Competition Advantages

### **Creativity & Innovation (25%)**
- ✅ Novel AI + traditional music analysis combination
- ✅ Self-generating reference system for portability
- ✅ First music feedback system with visual sheet music diff
- ✅ Polyphonic analysis capability unique in hackathon context

### **Technical Execution (25%)**
- ✅ Industry-standard audio processing (librosa, scipy)
- ✅ Robust error handling and comprehensive validation
- ✅ Professional CLI with extensive options
- ✅ Modular architecture enabling easy extension

### **Impact & Usefulness (25%)**
- ✅ Directly addresses ABRSM's scalable assessment needs
- ✅ Handles both simple and complex musical content
- ✅ Visual feedback aids learning and engagement
- ✅ Ready for integration into existing educational platforms

### **Presentation (25%)**
- ✅ Comprehensive documentation with examples
- ✅ Interactive demo mode requiring no external files
- ✅ Professional project structure and code quality
- ✅ Multiple visualization outputs for compelling demonstrations

## 🔧 Advanced Usage

### **Command Line Options**
```bash
# Enhanced analysis with all features
python enhanced_main.py --enhanced my_performance.wav

# Specific piece analysis
python enhanced_main.py --piece mary --enhanced my_performance.wav

# Simple analysis (original functionality)
python enhanced_main.py --simple my_performance.wav

# Analysis without LLM feedback
python enhanced_main.py --no-llm my_performance.wav

# Skip visualizations (faster processing)
python enhanced_main.py --no-visualizations my_performance.wav

# Demo mode
python enhanced_main.py --demo

# Create demo audio file only
python enhanced_main.py --create-demo-only
```

### **Integration Example**
```python
from enhanced_main import MusicAnalyzer

analyzer = MusicAnalyzer(piece_key="twinkle")
f0, times, onsets, enhanced_analysis = analyzer.analyze_with_enhancements(
    "performance.wav", 
    generate_visualizations=True,
    detect_polyphony=True,
    analyze_timing=True
)
```

## 📝 Example Enhanced Output

```
🎼 ABRSM AI Music Feedback System
   Analyzing: Twinkle, Twinkle, Little Star
   Audio: demo_performance.wav
   Mode: Enhanced Analysis with Visualizations
==================================================

� ENHANCED ANALYSIS FEATURES
==================================================

⏱️  Time Signature Analysis:
   • Detected: (4, 4)
   • Tempo: 100 BPM
   • Beat Consistency: 85.2%
   • Difficulty: Beginner - Most common time signature
   • Timing chart saved: timing_analysis.png

🎹 Polyphonic Content Detected:
   • Complexity: 45.2/100
   • Difficulty: Intermediate - Multi-voice texture
   • Avg Simultaneous Notes: 2.3
   • Chord Changes: 4

🎼 Sheet Music Analysis:
   • Visual comparison saved: sheet_music_analysis.png

🎯 YOUR PERSONALIZED FEEDBACK
==================================================
Great job on your performance! You maintained a steady rhythm throughout most of the piece and your pitch accuracy was quite good. I particularly noticed your strong beat placement in the first half.

For your next practice session, focus on the timing in measures 3-4 where you rushed slightly, and work on the pitch accuracy in the final phrase where a couple notes were a bit flat. The visual analysis shows exactly where to concentrate your practice.

Keep up the excellent work - your musical expression is developing beautifully!

📁 Generated Files:
   • timing_analysis.png
   • sheet_music_analysis.png
```

## 🚀 Future Enhancements

- **Real-time Analysis**: Live feedback during performance
- **Extended Repertoire**: Classical pieces, scales, arpeggios
- **Web Interface**: Browser-based GUI for easier access
- **Advanced Metrics**: Dynamics, articulation, expression analysis
- **Multi-instrument Support**: Guitar, violin, voice recognition
- **Progress Tracking**: Long-term improvement monitoring
- **Collaborative Features**: Teacher-student interaction tools

## 👥 Team

**Victor Nabasu** - Full Stack Developer & Music Technology Enthusiast
- Audio signal processing implementation
- AI integration and prompt engineering  
- Visualization system development
- CLI and user experience design

## 📄 Technical Requirements

- **Python 3.8+**
- **Audio Libraries**: librosa, scipy, numpy
- **Visualization**: matplotlib  
- **AI Integration**: Google Gemini API
- **Audio Formats**: WAV, MP3, FLAC supported
- **Output Formats**: JSON reports, PNG visualizations

## 📄 License

This project is shared with ABRSM as per the challenge requirements. Code is available for educational and research purposes.

---

*Built for the London Music Technology Hackathon 2025 - ABRSM Challenge*  
*Advancing music education through AI innovation* 🎵

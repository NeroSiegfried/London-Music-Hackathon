# ABRSM GUI User Guide 🎼

## Understanding the Analytics

This guide explains how to interpret the analysis results in the ABRSM AI Music Feedback GUI.

## 📊 Overview Tab

The **Overview** tab provides a high-level summary of your performance:

### Performance Summary
- **Completion Rate**: Percentage of expected notes that were detected
- **Pitch Accuracy**: How close your pitches were to the expected notes (measured in cents, where 100 cents = 1 semitone)
- **Timing Accuracy**: How close your note timing was to the expected rhythm (measured in milliseconds)
- **Notes Detected vs Expected**: Shows if you played all the notes

### Enhanced Features
- **Time Signature**: The detected rhythm pattern (e.g., 4/4, 3/4)
- **Tempo**: Your playing speed in Beats Per Minute (BPM)
- **Beat Consistency**: How steady your rhythm was (0-100%)
- **Polyphonic Analysis**: For complex music with multiple simultaneous notes

## 🎼 Sheet Music Tab

The **Sheet Music** tab shows your performance visually on a musical staff:

### Color Coding
- 🟢 **Green Notes**: Excellent accuracy (±20 cents pitch, ±50ms timing)
- 🟠 **Orange Notes**: Good accuracy (±50 cents pitch, ±100ms timing)  
- 🔴 **Red Notes**: Needs improvement (larger errors)
- ⚫ **Gray Notes**: Missed or undetected notes

### Reading the Display
- **Top Line**: Expected notes from the reference
- **Bottom Annotations**: Your actual played notes (if different)
- **Staff Lines**: Standard musical notation (treble clef)

## ⏱️ Timing Tab

The **Timing** tab analyzes your rhythmic performance:

### Timing Deviations Chart
- **X-axis**: Note number in sequence
- **Y-axis**: Timing error in milliseconds
- **Green bars**: Good timing (within ±50ms)
- **Orange bars**: Fair timing (±50-100ms)
- **Red bars**: Poor timing (>±100ms)

### Beat Pattern Analysis
- **Strong Beats (1,3)**: Accuracy on emphasized beats in 4/4 time
- **Weak Beats (2,4)**: Accuracy on lighter beats
- **Overall Consistency**: How steady your tempo was throughout

### Time Signature Detection
Shows the detected rhythm pattern and your average tempo.

## 🎵 Note Analysis Tab

The **Note Analysis** tab provides detailed note-by-note breakdown:

### Table Columns
- **Note**: Sequence number with accuracy icon
- **Expected Pitch**: The pitch you should have played (e.g., C4, G4)
- **Detected Pitch**: The pitch that was detected from your performance
- **Pitch Error**: Difference in cents (+ = too high, - = too low)
- **Expected Time**: When the note should occur (seconds)
- **Detected Time**: When your note was detected (seconds)
- **Timing Error**: Difference in milliseconds (+ = late, - = early)
- **Accuracy**: Overall assessment (Excellent/Good/Needs Work/Missed)

### Note Icons
- ♪ **Single Note**: Standard note
- ♫ **Beamed Notes**: Connected notes
- ♪? **Questionable**: Uncertain detection
- ✗ **Missed**: Note not detected

### Interactive Features
- **Click any row** to see detailed analysis in the bottom panel
- **Sort columns** by clicking headers
- **Color coding** matches the accuracy assessment

## 🤖 AI Feedback Tab

The **AI Feedback** tab generates personalized recommendations:

### Setup
1. Enter your Google API key (required for AI analysis)
2. Click "Generate Feedback" after running analysis

### Feedback Content
- **Technical Assessment**: Pitch and timing analysis
- **Musical Interpretation**: Phrasing and expression
- **Practice Suggestions**: Specific areas to focus on
- **Encouragement**: Positive reinforcement for good aspects

## 🔍 Selected Note Details

When you click on a note in the Note Analysis tab, the bottom panel shows:

### Pitch Analysis
- **Expected vs Detected**: Exact pitch comparison
- **Cent Deviation**: Precise tuning accuracy
- **Musical Context**: How this affects the overall melody

### Timing Analysis  
- **Expected vs Detected**: Precise timing comparison
- **Millisecond Deviation**: Exact rhythmic accuracy
- **Beat Context**: How this fits in the rhythm pattern

### Overall Assessment
Color-coded accuracy rating with specific feedback.

## 📁 File Management

### Loading Audio Files
- **Browse**: Select any audio file (.wav, .mp3, .flac, .m4a)
- **Demo**: Load built-in demo performance for testing
- **Supported Formats**: Most common audio formats

### Exporting Results
- **Export Analysis**: Save complete analysis as JSON file
- **Include Visualizations**: Charts and graphs are automatically saved as PNG files

## 💡 Tips for Best Results

### Recording Quality
- **Clear Audio**: Minimize background noise
- **Good Microphone**: Use a decent quality microphone
- **Steady Tempo**: Try to maintain consistent rhythm
- **Single Instrument**: Works best with solo performances

### Understanding Accuracy Thresholds
- **Pitch Accuracy**: ±20 cents is considered excellent (professional level)
- **Timing Accuracy**: ±50ms is considered excellent (human perception threshold)
- **These are strict standards** - don't be discouraged by "Needs Work" ratings

### Practice Recommendations
1. **Focus on Red Notes**: These need the most attention
2. **Work on Timing**: Use a metronome for rhythm practice
3. **Check Pitch**: Use a tuner for intonation practice
4. **Record Regularly**: Track your progress over time

## 🎯 Understanding the ABRSM Context

This tool is designed for **educational assessment**, similar to ABRSM grade examinations:

### What ABRSM Looks For
- **Technical Accuracy**: Correct notes and timing
- **Musical Understanding**: Appropriate phrasing and expression
- **Consistency**: Reliable performance throughout
- **Progress**: Improvement over time

### How This Tool Helps
- **Objective Measurement**: Precise pitch and timing analysis
- **Visual Feedback**: Easy-to-understand charts and graphs
- **AI Insights**: Personalized recommendations for improvement
- **Practice Tracking**: Monitor your development

## 🚀 Getting Started

1. **Launch the GUI**: Run `./start_gui.sh` or `python launch_gui.py`
2. **Load Demo**: Click "Demo" to try with sample audio
3. **Run Analysis**: Click "🎵 Analyze Performance"
4. **Explore Tabs**: Check each tab to understand your performance
5. **Generate Feedback**: Get AI recommendations for improvement
6. **Practice**: Use the insights to focus your practice time

## 🆘 Troubleshooting

### Common Issues
- **"No file selected"**: Click Browse or Demo to load audio
- **"Analysis modules not available"**: Ensure you're in the correct directory
- **"Missing dependencies"**: Run `pip install -r requirements.txt`
- **GUI won't start**: Install tkinter with `sudo apt-get install python3-tk` (Linux)

### Performance Issues
- **Slow analysis**: Large audio files take longer to process
- **Memory usage**: Close other applications for better performance
- **Visual glitches**: Update matplotlib and tkinter packages

### Getting Help
- Check the console output for detailed error messages
- Ensure your virtual environment is activated
- Try the command-line tools first to verify setup
- Review the README.md for installation instructions

---

**Happy practicing! 🎵**

*This tool is designed to support your musical journey with objective, detailed feedback to help you improve your performances.*

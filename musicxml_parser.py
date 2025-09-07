#!/usr/bin/env python3
"""
Music XML Parser for extracting melody from MusicXML files
"""

import music21
import numpy as np

def extract_melody_from_musicxml(musicxml_path):
    """
    Extract melody notes from a MusicXML file using music21
    
    Args:
        musicxml_path (str): Path to the MusicXML file
        
    Returns:
        list: List of note dictionaries with pitch, start_time, duration
    """
    try:
        # Load the score
        score = music21.converter.parse(musicxml_path)
        
        # Flatten to get all notes
        notes = score.flat.notes
        
        melody_notes = []
        
        for note_or_chord in notes:
            if isinstance(note_or_chord, music21.note.Note):
                # Single note
                pitch_midi = note_or_chord.pitch.midi
                start_time = float(note_or_chord.offset)
                duration = float(note_or_chord.duration.quarterLength)
                
                melody_notes.append({
                    'pitch': pitch_midi,
                    'start_time': start_time,
                    'duration': duration,
                    'velocity': 80  # Default velocity
                })
                
            elif isinstance(note_or_chord, music21.chord.Chord):
                # Chord - take the highest note (melody line)
                pitches = [p.midi for p in note_or_chord.pitches]
                highest_pitch = max(pitches)
                start_time = float(note_or_chord.offset)
                duration = float(note_or_chord.duration.quarterLength)
                
                melody_notes.append({
                    'pitch': highest_pitch,
                    'start_time': start_time, 
                    'duration': duration,
                    'velocity': 80
                })
        
        # Sort by start time
        melody_notes.sort(key=lambda x: x['start_time'])
        
        # Convert quarter note times to seconds (assuming 120 BPM default)
        tempo = 120  # Default tempo
        beat_duration = 60.0 / tempo  # Duration of one quarter note in seconds
        
        for note in melody_notes:
            note['start_time'] *= beat_duration
            note['duration'] *= beat_duration
            
        print(f"✅ Extracted {len(melody_notes)} notes from MusicXML")
        if melody_notes:
            total_duration = melody_notes[-1]['start_time'] + melody_notes[-1]['duration']
            print(f"   Total duration: {total_duration:.1f}s")
            pitch_range = [min(n['pitch'] for n in melody_notes), max(n['pitch'] for n in melody_notes)]
            print(f"   Pitch range: {pitch_range[0]} to {pitch_range[1]} ({music21.pitch.Pitch(pitch_range[0]).name} to {music21.pitch.Pitch(pitch_range[1]).name})")
        
        return melody_notes
        
    except Exception as e:
        print(f"❌ Failed to extract melody from MusicXML: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_musicxml_tempo(musicxml_path):
    """Extract tempo information from MusicXML file"""
    try:
        score = music21.converter.parse(musicxml_path)
        
        # Look for tempo markings
        tempo_markings = score.flat.getElementsByClass(music21.tempo.TempoIndication)
        
        if tempo_markings:
            # Get the first tempo marking
            tempo = tempo_markings[0]
            if hasattr(tempo, 'number'):
                return float(tempo.number)
            elif hasattr(tempo, 'getQuarterBPM'):
                return float(tempo.getQuarterBPM())
        
        # Default tempo if none found
        return 120.0
        
    except Exception as e:
        print(f"⚠️ Could not extract tempo from MusicXML: {e}")
        return 120.0

if __name__ == "__main__":
    # Test the parser
    import sys
    if len(sys.argv) > 1:
        musicxml_file = sys.argv[1]
        notes = extract_melody_from_musicxml(musicxml_file)
        print(f"\nExtracted {len(notes)} notes:")
        for i, note in enumerate(notes[:10]):  # Show first 10 notes
            pitch_name = music21.pitch.Pitch(note['pitch']).name
            print(f"  {i+1}: {pitch_name} (MIDI {note['pitch']}) at {note['start_time']:.2f}s for {note['duration']:.2f}s")
        if len(notes) > 10:
            print(f"  ... and {len(notes) - 10} more notes")
    else:
        print("Usage: python3 musicxml_parser.py <musicxml_file>")

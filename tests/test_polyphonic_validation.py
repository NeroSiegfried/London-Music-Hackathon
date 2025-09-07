#!/usr/bin/env python3
"""
Polyphonic Analysis Validation Test
Ensures polyphonic analyzer gives sensible accuracy values
"""

import os
import sys
import numpy as np
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_polyphonic_accuracy():
    """Test that polyphonic analysis gives reasonable accuracy values"""
    print("🎹 Testing Polyphonic Analysis Accuracy...")
    
    try:
        from polyphonic_analyzer import PolyphonicAnalyzer
        from enhanced_main_fixed import MusicAnalyzer
        
        # Initialize analyzers
        poly_analyzer = PolyphonicAnalyzer()
        music_analyzer = MusicAnalyzer(piece_key="ninettes_musette")
        
        # Check for test audio
        test_files = [
            "audio/demo_performance.wav",
            "demo_performance.wav"
        ]
        
        test_audio = None
        for f in test_files:
            if os.path.exists(f):
                test_audio = f
                break
        
        if not test_audio:
            print("⚠️  Creating test audio for polyphonic analysis...")
            # Create test audio
            test_audio = "tests/audio/polyphonic_test.wav"
            success = create_polyphonic_test_audio(music_analyzer, test_audio)
            if not success:
                print("❌ Could not create test audio")
                return False
        
        print(f"Testing with audio: {test_audio}")
        
        # Run polyphonic analysis
        result = poly_analyzer.analyze_polyphonic_performance(test_audio)
        
        if result:
            print("✅ Polyphonic analysis completed")
            
            # Check for reasonable accuracy values
            if 'accuracy_metrics' in result:
                metrics = result['accuracy_metrics']
                
                # Check note detection accuracy
                if 'note_detection_accuracy' in metrics:
                    accuracy = metrics['note_detection_accuracy']
                    print(f"Note detection accuracy: {accuracy:.1%}")
                    
                    # Validate reasonable range (30% to 95%)
                    if accuracy < 0.30:
                        print(f"⚠️  Accuracy suspiciously low: {accuracy:.1%}")
                        return False
                    elif accuracy > 0.95:
                        print(f"⚠️  Accuracy suspiciously high: {accuracy:.1%}")
                        return False
                    else:
                        print(f"✅ Accuracy in reasonable range: {accuracy:.1%}")
                
                # Check timing accuracy
                if 'timing_accuracy' in metrics:
                    timing_acc = metrics['timing_accuracy']
                    print(f"Timing accuracy: {timing_acc:.1%}")
                    
                    if 0.20 <= timing_acc <= 0.90:
                        print(f"✅ Timing accuracy reasonable: {timing_acc:.1%}")
                    else:
                        print(f"⚠️  Timing accuracy out of range: {timing_acc:.1%}")
                
                # Check pitch accuracy
                if 'pitch_accuracy' in metrics:
                    pitch_acc = metrics['pitch_accuracy']
                    print(f"Pitch accuracy: {pitch_acc:.1%}")
                    
                    if 0.40 <= pitch_acc <= 0.95:
                        print(f"✅ Pitch accuracy reasonable: {pitch_acc:.1%}")
                    else:
                        print(f"⚠️  Pitch accuracy out of range: {pitch_acc:.1%}")
                
                return True
            else:
                print("❌ No accuracy metrics in polyphonic analysis result")
                return False
        else:
            print("❌ Polyphonic analysis returned no result")
            return False
            
    except ImportError as e:
        print(f"❌ Could not import polyphonic analyzer: {e}")
        return False
    except Exception as e:
        print(f"❌ Polyphonic analysis test failed: {e}")
        traceback.print_exc()
        return False

def create_polyphonic_test_audio(analyzer, output_path):
    """Create test audio with polyphonic content"""
    try:
        import librosa
        from scipy.io import wavfile
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create a polyphonic test piece (melody + harmony)
        melody = analyzer.piece_info['melody'][:8]  # First 8 notes
        
        seconds_per_beat = 60.0 / analyzer.tempo
        total_duration = sum(n['duration'] for n in melody) * 4 * seconds_per_beat
        
        wav_data = np.zeros(int(total_duration * analyzer.sample_rate))
        current_time = 0.0
        
        # Add melody line
        for note in melody:
            frequency = librosa.midi_to_hz(note['pitch'])
            duration_samples = int(note['duration'] * 4 * seconds_per_beat * analyzer.sample_rate)
            
            t = np.linspace(0., duration_samples / analyzer.sample_rate, duration_samples, endpoint=False)
            note_wave = 0.5 * np.sin(2 * np.pi * frequency * t)
            
            # Add envelope
            envelope = np.exp(-1.0 * t / (duration_samples / analyzer.sample_rate))
            note_wave *= envelope
            
            start_sample = int(current_time * analyzer.sample_rate)
            end_sample = start_sample + len(note_wave)
            if end_sample <= len(wav_data):
                wav_data[start_sample:end_sample] += note_wave
            
            current_time += note['duration'] * 4 * seconds_per_beat
        
        # Add harmony line (thirds and fifths)
        current_time = 0.0
        for note in melody:
            # Add harmony note (major third)
            harmony_pitch = note['pitch'] + 4  # Major third
            frequency = librosa.midi_to_hz(harmony_pitch)
            duration_samples = int(note['duration'] * 4 * seconds_per_beat * analyzer.sample_rate)
            
            t = np.linspace(0., duration_samples / analyzer.sample_rate, duration_samples, endpoint=False)
            harmony_wave = 0.3 * np.sin(2 * np.pi * frequency * t)
            
            # Add envelope
            envelope = np.exp(-1.2 * t / (duration_samples / analyzer.sample_rate))
            harmony_wave *= envelope
            
            start_sample = int(current_time * analyzer.sample_rate)
            end_sample = start_sample + len(harmony_wave)
            if end_sample <= len(wav_data):
                wav_data[start_sample:end_sample] += harmony_wave
            
            current_time += note['duration'] * 4 * seconds_per_beat
        
        # Normalize and save
        if np.max(np.abs(wav_data)) > 0:
            wav_data = wav_data / np.max(np.abs(wav_data)) * 0.8
        
        wavfile.write(output_path, analyzer.sample_rate, (wav_data * 32767).astype(np.int16))
        print(f"✅ Created polyphonic test audio: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating polyphonic test audio: {e}")
        return False

def test_polyphonic_robustness():
    """Test polyphonic analysis with different scenarios"""
    print("\n🎹 Testing Polyphonic Analysis Robustness...")
    
    try:
        from polyphonic_analyzer import PolyphonicAnalyzer
        
        analyzer = PolyphonicAnalyzer()
        
        # Test different audio scenarios
        test_scenarios = [
            ("Empty audio", create_empty_audio),
            ("Single note", create_single_note_audio),
            ("Complex polyphony", create_complex_polyphonic_audio),
        ]
        
        passed_tests = 0
        total_tests = len(test_scenarios)
        
        for scenario_name, audio_creator in test_scenarios:
            print(f"\nTesting: {scenario_name}")
            
            try:
                test_file = f"tests/audio/poly_test_{scenario_name.lower().replace(' ', '_')}.wav"
                
                # Create test audio
                success = audio_creator(test_file)
                if not success:
                    print(f"❌ Could not create test audio for {scenario_name}")
                    continue
                
                # Run analysis
                result = analyzer.analyze_polyphonic_performance(test_file)
                
                if result is None:
                    print(f"⚠️  {scenario_name}: Analysis returned None (acceptable for some cases)")
                    passed_tests += 1
                elif isinstance(result, dict):
                    print(f"✅ {scenario_name}: Analysis completed successfully")
                    
                    # Check for reasonable structure
                    if 'accuracy_metrics' in result:
                        metrics = result['accuracy_metrics']
                        if 'note_detection_accuracy' in metrics:
                            accuracy = metrics['note_detection_accuracy']
                            if 0 <= accuracy <= 1:
                                print(f"   Accuracy: {accuracy:.1%} (valid range)")
                                passed_tests += 1
                            else:
                                print(f"   ❌ Accuracy out of range: {accuracy}")
                        else:
                            print(f"   ⚠️  No note detection accuracy metric")
                            passed_tests += 1
                    else:
                        print(f"   ⚠️  No accuracy metrics (may be acceptable)")
                        passed_tests += 1
                else:
                    print(f"❌ {scenario_name}: Unexpected result type: {type(result)}")
                
            except Exception as e:
                print(f"❌ {scenario_name} test failed: {e}")
        
        success_rate = passed_tests / total_tests
        print(f"\nPolyphonic robustness test: {passed_tests}/{total_tests} passed ({success_rate:.1%})")
        
        return success_rate >= 0.7
        
    except ImportError as e:
        print(f"❌ Could not import polyphonic analyzer: {e}")
        return False

def create_empty_audio(output_path):
    """Create empty/silent audio for testing"""
    try:
        from scipy.io import wavfile
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 1 second of silence
        sample_rate = 22050
        duration = 1.0
        wav_data = np.zeros(int(duration * sample_rate))
        
        wavfile.write(output_path, sample_rate, (wav_data * 32767).astype(np.int16))
        return True
    except Exception as e:
        print(f"Error creating empty audio: {e}")
        return False

def create_single_note_audio(output_path):
    """Create single note audio for testing"""
    try:
        import librosa
        from scipy.io import wavfile
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Single C4 note
        sample_rate = 22050
        duration = 1.0
        frequency = librosa.midi_to_hz(60)  # C4
        
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        wav_data = 0.7 * np.sin(2 * np.pi * frequency * t)
        
        # Add envelope
        envelope = np.exp(-0.5 * t / duration)
        wav_data *= envelope
        
        wavfile.write(output_path, sample_rate, (wav_data * 32767).astype(np.int16))
        return True
    except Exception as e:
        print(f"Error creating single note audio: {e}")
        return False

def create_complex_polyphonic_audio(output_path):
    """Create complex polyphonic audio for testing"""
    try:
        import librosa
        from scipy.io import wavfile
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        sample_rate = 22050
        duration = 2.0
        
        # Create chord progression: C major -> F major -> G major -> C major
        chord_progression = [
            [60, 64, 67],  # C major (C, E, G)
            [65, 69, 72],  # F major (F, A, C)
            [67, 71, 74],  # G major (G, B, D)
            [60, 64, 67],  # C major (C, E, G)
        ]
        
        chord_duration = duration / len(chord_progression)
        wav_data = np.zeros(int(duration * sample_rate))
        
        for i, chord in enumerate(chord_progression):
            chord_start = i * chord_duration
            chord_samples = int(chord_duration * sample_rate)
            
            for pitch in chord:
                frequency = librosa.midi_to_hz(pitch)
                t = np.linspace(0, chord_duration, chord_samples, endpoint=False)
                note_wave = 0.3 * np.sin(2 * np.pi * frequency * t)
                
                # Add envelope
                envelope = np.exp(-0.5 * t / chord_duration)
                note_wave *= envelope
                
                start_idx = int(chord_start * sample_rate)
                end_idx = start_idx + len(note_wave)
                if end_idx <= len(wav_data):
                    wav_data[start_idx:end_idx] += note_wave
        
        # Normalize
        if np.max(np.abs(wav_data)) > 0:
            wav_data = wav_data / np.max(np.abs(wav_data)) * 0.8
        
        wavfile.write(output_path, sample_rate, (wav_data * 32767).astype(np.int16))
        return True
    except Exception as e:
        print(f"Error creating complex polyphonic audio: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Polyphonic Analysis Validation Suite")
    print("=" * 50)
    
    # Run accuracy test
    accuracy_passed = test_polyphonic_accuracy()
    
    # Run robustness test
    robustness_passed = test_polyphonic_robustness()
    
    print("\n" + "=" * 50)
    print("📊 POLYPHONIC TEST RESULTS")
    print("=" * 50)
    
    print(f"Accuracy Test: {'✅ PASS' if accuracy_passed else '❌ FAIL'}")
    print(f"Robustness Test: {'✅ PASS' if robustness_passed else '❌ FAIL'}")
    
    overall_success = accuracy_passed and robustness_passed
    print(f"\nOverall: {'🎉 ALL TESTS PASSED' if overall_success else '⚠️  SOME TESTS FAILED'}")
    
    if overall_success:
        print("✅ Polyphonic analysis is working correctly with reasonable accuracy values")
    else:
        print("❌ Polyphonic analysis needs attention")
    
    sys.exit(0 if overall_success else 1)

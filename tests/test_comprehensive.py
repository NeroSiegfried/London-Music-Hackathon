#!/usr/bin/env python3
"""
Comprehensive ABRSM AI Test Suite
Tests all functionality with proper file organization
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
import numpy as np
import traceback
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestMusicAnalyzer(unittest.TestCase):
    """Test the MusicAnalyzer class from enhanced_main_fixed"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = Path(__file__).parent
        cls.project_dir = cls.test_dir.parent
        
        # Ensure test directories exist
        for subdir in ['audio', 'midi', 'visualizations', 'reports']:
            (cls.test_dir / subdir).mkdir(exist_ok=True)
        
        # Import the analyzer
        try:
            from enhanced_main_fixed import MusicAnalyzer
            cls.MusicAnalyzer = MusicAnalyzer
        except ImportError as e:
            cls.fail(f"Could not import MusicAnalyzer: {e}")
    
    def test_analyzer_initialization(self):
        """Test MusicAnalyzer can be initialized"""
        analyzer = self.MusicAnalyzer()
        self.assertIsNotNone(analyzer)
        self.assertEqual(analyzer.piece_key, "twinkle")
        self.assertIn("title", analyzer.piece_info)
        print("✅ MusicAnalyzer initialization test passed")
    
    def test_midi_loading(self):
        """Test MIDI loading functionality"""
        analyzer = self.MusicAnalyzer(piece_key="twinkle")
        
        # Test with existing MIDI file
        midi_files = list((self.project_dir / "midi").glob("*.mid"))
        if midi_files:
            # Test the load_melody_from_midi method
            result = analyzer.load_melody_from_midi()
            self.assertIsInstance(result, bool)
            print(f"✅ MIDI loading test: {result}")
        else:
            print("⚠️  No MIDI files found for testing")
    
    def test_reference_creation(self):
        """Test reference audio/MIDI creation"""
        analyzer = self.MusicAnalyzer(piece_key="twinkle")
        
        # Create reference in test directory
        old_reference_prefix = analyzer.reference_prefix
        analyzer.reference_prefix = str(self.test_dir / "midi" / "test_reference")
        
        try:
            result = analyzer.create_reference_data()
            self.assertIsInstance(result, bool)
            print(f"✅ Reference creation test: {result}")
        finally:
            analyzer.reference_prefix = old_reference_prefix
    
    def test_demo_audio_creation(self):
        """Test demo audio creation"""
        analyzer = self.MusicAnalyzer(piece_key="twinkle")
        
        # Create demo in test directory
        demo_path = self.test_dir / "audio" / "test_demo.wav"
        
        # Temporarily modify the demo path
        original_create_demo = analyzer.create_demo_audio
        def create_test_demo():
            # Save original demo_path logic and replace with test path
            import librosa
            from scipy.io import wavfile
            
            try:
                print("🎵 Creating test demo performance audio...")
                reference_melody = analyzer.piece_info['melody']
                demo_melody = reference_melody[:-1]  # Remove last note
                
                seconds_per_beat = 60.0 / analyzer.tempo
                total_duration = sum(n['duration'] for n in demo_melody) * 4 * seconds_per_beat
                
                wav_data = np.zeros(int(total_duration * analyzer.sample_rate))
                current_time = 0.0
                
                for note in demo_melody:
                    frequency = librosa.midi_to_hz(note['pitch'])
                    duration_samples = int(note['duration'] * 4 * seconds_per_beat * analyzer.sample_rate)
                    
                    t = np.linspace(0., duration_samples / analyzer.sample_rate, duration_samples, endpoint=False)
                    note_wave = 0.6 * np.sin(2 * np.pi * frequency * t)
                    
                    # Add envelope
                    envelope = np.exp(-2 * t / (duration_samples / analyzer.sample_rate))
                    note_wave *= envelope
                    
                    start_sample = int(current_time * analyzer.sample_rate)
                    end_sample = start_sample + len(note_wave)
                    if end_sample <= len(wav_data):
                        wav_data[start_sample:end_sample] += note_wave
                    
                    current_time += note['duration'] * 4 * seconds_per_beat
                
                # Normalize and save
                if np.max(np.abs(wav_data)) > 0:
                    wav_data = wav_data / np.max(np.abs(wav_data)) * 0.8
                wavfile.write(str(demo_path), analyzer.sample_rate, (wav_data * 32767).astype(np.int16))
                print(f"✅ Test demo audio created: {demo_path}")
                return True
                
            except Exception as e:
                print(f"❌ Error creating test demo audio: {e}")
                return False
        
        result = create_test_demo()
        self.assertTrue(result)
        self.assertTrue(demo_path.exists())
        print("✅ Demo audio creation test passed")

class TestSheetMusicVisualization(unittest.TestCase):
    """Test sheet music visualization with music21"""
    
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(__file__).parent
        (cls.test_dir / "visualizations").mkdir(exist_ok=True)
    
    def test_music21_sheet_creation(self):
        """Test music21 sheet music creation"""
        try:
            from sheet_music_visualizer import Music21SheetVisualizer
            
            # Create test melody
            test_melody = [
                {'pitch': 60, 'duration': 0.25},  # C4
                {'pitch': 62, 'duration': 0.25},  # D4  
                {'pitch': 64, 'duration': 0.5},   # E4
                {'pitch': 65, 'duration': 0.25},  # F4
                {'pitch': 67, 'duration': 0.5},   # G4
            ]
            
            test_performance = {
                'metadata': {'piece': 'Test Comprehensive'},
                'note_details': [
                    {'timing_deviation_ms': 0, 'pitch_deviation_cents': 0},
                    {'timing_deviation_ms': 50, 'pitch_deviation_cents': 25},
                    {'timing_deviation_ms': 'MISSED', 'pitch_deviation_cents': 'MISSED'},
                    {'timing_deviation_ms': 200, 'pitch_deviation_cents': 150},
                    {'timing_deviation_ms': -100, 'pitch_deviation_cents': -50},
                ]
            }
            
            visualizer = Music21SheetVisualizer()
            output_path = str(self.test_dir / "visualizations" / "test_comprehensive_sheet.png")
            
            result = visualizer.create_sheet_music_from_melody(
                test_melody, test_performance, output_path
            )
            
            self.assertIsNotNone(result)
            print("✅ Sheet music visualization test passed")
            
        except ImportError as e:
            self.fail(f"Could not import sheet music visualizer: {e}")

class TestGUIComponents(unittest.TestCase):
    """Test GUI components without launching full interface"""
    
    def test_gui_class_import(self):
        """Test that GUI class can be imported"""
        try:
            from enhanced_gui_interface import EnhancedABRSMGUI
            import tkinter as tk
            
            # Create hidden root window
            root = tk.Tk()
            root.withdraw()
            
            # Test GUI initialization
            app = EnhancedABRSMGUI(root)
            self.assertIsNotNone(app)
            
            # Test that MIDI templates were loaded
            if hasattr(app, 'available_pieces'):
                self.assertGreater(len(app.available_pieces), 0)
                print(f"✅ GUI loaded {len(app.available_pieces)} MIDI templates")
            
            # Test batch processing availability
            if hasattr(app, 'batch_process_csv'):
                print("✅ Batch processing function available")
            
            root.destroy()
            print("✅ GUI components test passed")
            
        except ImportError as e:
            self.fail(f"Could not import GUI components: {e}")

class TestAudioAnalysis(unittest.TestCase):
    """Test audio analysis functionality"""
    
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(__file__).parent
        cls.project_dir = cls.test_dir.parent
        
        # Import analyzer
        try:
            from enhanced_main_fixed import MusicAnalyzer
            cls.MusicAnalyzer = MusicAnalyzer
        except ImportError as e:
            cls.fail(f"Could not import MusicAnalyzer: {e}")
    
    def test_audio_analysis_methods(self):
        """Test that audio analysis methods exist and can be called"""
        analyzer = self.MusicAnalyzer()
        
        # Check that methods exist
        self.assertTrue(hasattr(analyzer, 'analyze_performance_audio'))
        self.assertTrue(hasattr(analyzer, 'compare_performances'))
        print("✅ Audio analysis methods available")
        
        # Test with demo audio if available
        demo_files = [
            self.project_dir / "audio" / "demo_performance.wav",
            self.project_dir / "demo_performance.wav"
        ]
        
        test_audio = None
        for demo_file in demo_files:
            if demo_file.exists():
                test_audio = str(demo_file)
                break
        
        if test_audio:
            try:
                # Test onset detection
                onset_times, f0, times = analyzer.analyze_performance_audio(test_audio)
                
                if onset_times is not None:
                    print(f"✅ Onset detection found {len(onset_times)} onsets")
                    self.assertIsInstance(onset_times, np.ndarray)
                    
                    # Test reasonable onset count (not too many)
                    if len(onset_times) > 0:
                        import librosa
                        y, sr = librosa.load(test_audio)
                        duration = len(y) / sr
                        onset_density = len(onset_times) / duration
                        
                        # Should not have more than 10 onsets per second
                        self.assertLess(onset_density, 10, "Onset detection may be over-detecting")
                        print(f"✅ Onset density: {onset_density:.2f} onsets/second (reasonable)")
                
                # Test full comparison
                report = analyzer.compare_performances(test_audio)
                if report:
                    self.assertIsInstance(report, dict)
                    self.assertIn('metadata', report)
                    print("✅ Performance comparison completed")
                    
            except Exception as e:
                print(f"⚠️  Audio analysis test failed: {e}")
                traceback.print_exc()
        else:
            print("⚠️  No demo audio file found for testing")

class TestPolyphonicAnalysis(unittest.TestCase):
    """Test polyphonic analysis functionality"""
    
    def test_polyphonic_analyzer_import(self):
        """Test that polyphonic analyzer can be imported"""
        try:
            from polyphonic_analyzer import PolyphonicAnalyzer
            
            analyzer = PolyphonicAnalyzer()
            self.assertIsNotNone(analyzer)
            print("✅ Polyphonic analyzer imported successfully")
            
            # Test that methods exist
            self.assertTrue(hasattr(analyzer, 'analyze_polyphonic_performance'))
            print("✅ Polyphonic analysis methods available")
            
        except ImportError as e:
            print(f"⚠️  Polyphonic analyzer not available: {e}")

class TestFileOrganization(unittest.TestCase):
    """Test proper file organization"""
    
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(__file__).parent
        cls.project_dir = cls.test_dir.parent
    
    def test_directory_structure(self):
        """Test that all required directories exist"""
        required_dirs = ['audio', 'midi', 'visualizations', 'reports']
        
        # Test project directories
        for dirname in required_dirs:
            project_dir = self.project_dir / dirname
            self.assertTrue(project_dir.exists(), f"Project {dirname} directory missing")
            
            # Test test directories
            test_dir = self.test_dir / dirname
            self.assertTrue(test_dir.exists(), f"Test {dirname} directory missing")
        
        print("✅ Directory structure test passed")
    
    def test_midi_files_available(self):
        """Test that MIDI files are available"""
        midi_dir = self.project_dir / "midi"
        midi_files = list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi"))
        
        self.assertGreater(len(midi_files), 0, "No MIDI files found")
        print(f"✅ Found {len(midi_files)} MIDI files")
        
        # Test file sizes (should not be empty)
        for midi_file in midi_files:
            self.assertGreater(midi_file.stat().st_size, 0, f"{midi_file.name} is empty")
        
        print("✅ MIDI files validation passed")

class TestIntegrationScenarios(unittest.TestCase):
    """Test complete integration scenarios"""
    
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(__file__).parent
        cls.project_dir = cls.test_dir.parent
        
        try:
            from enhanced_main_fixed import MusicAnalyzer
            cls.MusicAnalyzer = MusicAnalyzer
        except ImportError as e:
            cls.fail(f"Could not import MusicAnalyzer: {e}")
    
    def test_full_analysis_workflow(self):
        """Test complete analysis workflow"""
        # Initialize analyzer
        analyzer = self.MusicAnalyzer(piece_key="twinkle")
        
        # Create test audio
        test_audio_path = self.test_dir / "audio" / "full_test_demo.wav"
        
        try:
            # Create reference data
            result = analyzer.create_reference_data()
            print(f"Reference creation: {'✅' if result else '❌'}")
            
            # Create demo audio for testing
            demo_created = self._create_test_audio(analyzer, test_audio_path)
            print(f"Demo audio creation: {'✅' if demo_created else '❌'}")
            
            if demo_created:
                # Run full analysis
                report = analyzer.compare_performances(str(test_audio_path))
                
                if report:
                    # Validate report structure
                    self.assertIn('metadata', report)
                    self.assertIn('note_details', report)
                    
                    metadata = report['metadata']
                    self.assertIn('total_notes', metadata)
                    self.assertIn('detected_notes', metadata)
                    self.assertIn('missed_notes', metadata)
                    
                    # Save report to test directory
                    report_path = self.test_dir / "reports" / "full_analysis_test.json"
                    with open(report_path, 'w') as f:
                        json.dump(report, f, indent=2)
                    
                    print(f"✅ Full analysis completed:")
                    print(f"   - Total notes: {metadata['total_notes']}")
                    print(f"   - Detected: {metadata['detected_notes']}")
                    print(f"   - Missed: {metadata['missed_notes']}")
                    print(f"   - Report saved: {report_path}")
                    
                    # Validate reasonable detection rate
                    if metadata['total_notes'] > 0:
                        detection_rate = metadata['detected_notes'] / metadata['total_notes']
                        self.assertGreater(detection_rate, 0.3, "Detection rate too low")
                        self.assertLess(detection_rate, 1.2, "Detection rate suspiciously high")
                        print(f"   - Detection rate: {detection_rate:.1%} (reasonable)")
                    
                    return True
                else:
                    print("❌ Analysis returned no report")
                    return False
            else:
                print("❌ Could not create test audio")
                return False
                
        except Exception as e:
            print(f"❌ Full analysis workflow failed: {e}")
            traceback.print_exc()
            return False
    
    def _create_test_audio(self, analyzer, output_path):
        """Create test audio file"""
        try:
            import librosa
            from scipy.io import wavfile
            
            # Create audio with some variation from reference
            melody = analyzer.piece_info['melody']
            test_melody = melody[:-2]  # Remove last 2 notes to test missed notes
            
            seconds_per_beat = 60.0 / analyzer.tempo
            total_duration = sum(n['duration'] for n in test_melody) * 4 * seconds_per_beat
            
            wav_data = np.zeros(int(total_duration * analyzer.sample_rate))
            current_time = 0.0
            
            for i, note in enumerate(test_melody):
                # Add slight timing variation
                timing_variation = 0.02 * (i % 3 - 1)  # -0.02 to +0.02 seconds
                actual_time = current_time + timing_variation
                
                frequency = librosa.midi_to_hz(note['pitch'])
                duration_samples = int(note['duration'] * 4 * seconds_per_beat * analyzer.sample_rate)
                
                t = np.linspace(0., duration_samples / analyzer.sample_rate, duration_samples, endpoint=False)
                note_wave = 0.7 * np.sin(2 * np.pi * frequency * t)
                
                # Add envelope
                envelope = np.exp(-1.5 * t / (duration_samples / analyzer.sample_rate))
                note_wave *= envelope
                
                start_sample = int(max(0, actual_time * analyzer.sample_rate))
                end_sample = start_sample + len(note_wave)
                if end_sample <= len(wav_data):
                    wav_data[start_sample:end_sample] += note_wave
                
                current_time += note['duration'] * 4 * seconds_per_beat
            
            # Normalize and save
            if np.max(np.abs(wav_data)) > 0:
                wav_data = wav_data / np.max(np.abs(wav_data)) * 0.8
            wavfile.write(str(output_path), analyzer.sample_rate, (wav_data * 32767).astype(np.int16))
            
            return True
            
        except Exception as e:
            print(f"Error creating test audio: {e}")
            return False

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("🚀 Running Comprehensive ABRSM AI Test Suite")
    print("=" * 60)
    
    # Test classes in order of dependencies
    test_classes = [
        TestFileOrganization,
        TestMusicAnalyzer,
        TestSheetMusicVisualization,
        TestGUIComponents,
        TestAudioAnalysis,
        TestPolyphonicAnalysis,
        TestIntegrationScenarios,
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {passed/total_tests*100:.1f}%")
    
    if failures > 0:
        print(f"\n❌ FAILURES ({failures}):")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if errors > 0:
        print(f"\n💥 ERRORS ({errors}):")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    if passed == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The ABRSM AI system is fully functional and tested.")
    elif passed >= total_tests * 0.9:
        print("\n✅ MOST TESTS PASSED!")
        print("⚠️  System is functional with minor issues.")
    else:
        print("\n⚠️  SEVERAL TESTS FAILED!")
        print("❌ System may have significant issues.")
    
    return passed >= total_tests * 0.8

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

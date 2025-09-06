#!/usr/bin/env python3
"""
Comprehensive test suite for the new simplified architecture
Tests all components: feature extraction, alignment, analysis
"""

import pytest
import numpy as np
import os
import tempfile
import json
from unittest.mock import patch, MagicMock

from main_analyzer import (
    extract_musical_features,
    build_template_map_from_melody,
    align_sequences,
    analyze_performance,
    PerformanceAnalyzer,
    PIECES
)

class TestFeatureExtraction:
    """Test audio feature extraction"""
    
    def test_extract_features_with_valid_audio(self):
        """Test feature extraction with demo audio"""
        if os.path.exists("audio/demo_performance.wav"):
            features = extract_musical_features("audio/demo_performance.wav")
            assert isinstance(features, list)
            if features:  # Only test if features were extracted
                assert all('time' in event and 'pitches' in event for event in features)
                assert all(isinstance(event['time'], (int, float)) for event in features)
                assert all(isinstance(event['pitches'], set) for event in features)
        else:
            pytest.skip("audio/demo_performance.wav not found")
    
    def test_extract_features_invalid_file(self):
        """Test feature extraction with invalid file"""
        features = extract_musical_features("nonexistent_file.wav")
        assert features == []

class TestTemplateMapping:
    """Test template map generation"""
    
    def test_build_template_from_melody(self):
        """Test template map creation from melody data"""
        piece_info = PIECES["twinkle"]
        template_map = build_template_map_from_melody(piece_info, tempo_bpm=120)
        
        assert "measures" in template_map
        assert "notes" in template_map
        assert "seconds_per_beat" in template_map
        assert "total_duration" in template_map
        
        # Check note structure
        notes = template_map["notes"]
        assert len(notes) == len(piece_info["melody"])
        
        for note in notes:
            assert "pitches" in note
            assert "start_offset_sec" in note
            assert "duration_sec" in note
            assert "note_index" in note
            assert isinstance(note["pitches"], set)
    
    def test_template_timing_calculation(self):
        """Test that template timing calculations are correct"""
        piece_info = PIECES["twinkle"]
        tempo = 100  # BPM
        template_map = build_template_map_from_melody(piece_info, tempo_bpm=tempo)
        
        expected_seconds_per_beat = 60.0 / tempo
        assert abs(template_map["seconds_per_beat"] - expected_seconds_per_beat) < 0.001
        
        # Check that notes have increasing start times
        notes = template_map["notes"]
        start_times = [note["start_offset_sec"] for note in notes]
        assert start_times == sorted(start_times)

class TestSequenceAlignment:
    """Test DTW sequence alignment"""
    
    def test_align_identical_sequences(self):
        """Test alignment of identical sequences"""
        events = [
            {'time': 0.0, 'pitches': {0}},
            {'time': 1.0, 'pitches': {2}},
            {'time': 2.0, 'pitches': {4}}
        ]
        
        alignment = align_sequences(events, events)
        
        assert "matched_pairs" in alignment
        assert "keys_added" in alignment
        assert "keys_missed" in alignment
        
        assert len(alignment["matched_pairs"]) == len(events)
        assert len(alignment["keys_added"]) == 0
        assert len(alignment["keys_missed"]) == 0
    
    def test_align_with_missing_notes(self):
        """Test alignment when performance has missing notes"""
        template_events = [
            {'time': 0.0, 'pitches': {0}},
            {'time': 1.0, 'pitches': {2}},
            {'time': 2.0, 'pitches': {4}}
        ]
        
        performance_events = [
            {'time': 0.0, 'pitches': {0}},
            {'time': 2.0, 'pitches': {4}}  # Missing middle note
        ]
        
        alignment = align_sequences(performance_events, template_events)
        
        assert len(alignment["matched_pairs"]) == 2
        assert len(alignment["keys_missed"]) == 1
        assert len(alignment["keys_added"]) == 0
    
    def test_align_with_extra_notes(self):
        """Test alignment when performance has extra notes"""
        template_events = [
            {'time': 0.0, 'pitches': {0}},
            {'time': 2.0, 'pitches': {4}}
        ]
        
        performance_events = [
            {'time': 0.0, 'pitches': {0}},
            {'time': 1.0, 'pitches': {2}},  # Extra note
            {'time': 2.0, 'pitches': {4}}
        ]
        
        alignment = align_sequences(performance_events, template_events)
        
        assert len(alignment["matched_pairs"]) == 2
        assert len(alignment["keys_missed"]) == 0
        assert len(alignment["keys_added"]) == 1

class TestPerformanceAnalysis:
    """Test performance analysis functions"""
    
    def test_analyze_performance_structure(self):
        """Test that analysis returns correct structure"""
        # Create mock alignment and template
        alignment = {
            "matched_pairs": [
                ({'time': 0.0, 'pitches': {0}}, {'time': 0.0, 'pitches': {0}}),
                ({'time': 1.0, 'pitches': {2}}, {'time': 1.0, 'pitches': {2}})
            ],
            "keys_added": [],
            "keys_missed": []
        }
        
        template_map = {
            "measures": [{"measure_number": 1, "start_offset_sec": 0.0, "duration_sec": 2.0}],
            "notes": [
                {"pitches": {0}, "start_offset_sec": 0.0, "duration_sec": 0.5},
                {"pitches": {2}, "start_offset_sec": 1.0, "duration_sec": 0.5}
            ]
        }
        
        results = analyze_performance(alignment, template_map)
        
        # Check main structure
        assert "summary" in results
        assert "standard_analysis" in results
        assert "measure_by_measure_analysis" in results
        assert "alignment_details" in results
        
        # Check summary structure
        summary = results["summary"]
        assert "overall_tempo_ratio" in summary
        assert "tempo_consistency_std_dev" in summary
        assert "notes_matched" in summary
        assert "notes_added" in summary
        assert "notes_missed" in summary
        
        # Check standard analysis structure
        std_analysis = results["standard_analysis"]
        assert "overall_assessment" in std_analysis
        assert "note_details" in std_analysis
        
        assessment = std_analysis["overall_assessment"]
        assert "completion_rate" in assessment
        assert "pitch_accuracy" in assessment
        assert "timing_accuracy" in assessment

class TestPerformanceAnalyzer:
    """Test the main PerformanceAnalyzer class"""
    
    def test_analyzer_initialization(self):
        """Test analyzer can be created and configured"""
        analyzer = PerformanceAnalyzer()
        assert analyzer.piece_info is None
        assert analyzer.template_map is None
        
        # Set piece from melody
        piece_info = PIECES["twinkle"]
        analyzer.set_piece_from_melody(piece_info, 120)
        
        assert analyzer.piece_info == piece_info
        assert analyzer.tempo_bpm == 120
        assert analyzer.template_map is not None
    
    @patch('main_analyzer.extract_musical_features')
    def test_analyzer_workflow(self, mock_extract):
        """Test complete analyzer workflow"""
        # Mock the feature extraction to return predictable data
        mock_extract.return_value = [
            {'time': 0.0, 'pitches': {0}},
            {'time': 0.5, 'pitches': {0}},
            {'time': 1.0, 'pitches': {7}}
        ]
        
        analyzer = PerformanceAnalyzer()
        analyzer.set_piece_from_melody(PIECES["twinkle"], 100)
        
        # Create a temporary output file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        try:
            results = analyzer.analyze_performance(temp_audio_path, "test_output")
            
            # Verify results structure
            assert isinstance(results, dict)
            assert "summary" in results
            assert "standard_analysis" in results
            
            # Verify that output files would be created (mocked)
            mock_extract.assert_called_once_with(temp_audio_path)
            
        finally:
            # Clean up
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

class TestBackwardsCompatibility:
    """Test backwards compatibility with old interface"""
    
    def test_pieces_available(self):
        """Test that predefined pieces are available"""
        assert "twinkle" in PIECES
        assert "mary" in PIECES
        
        for piece_key, piece_info in PIECES.items():
            assert "title" in piece_info
            assert "melody" in piece_info
            assert isinstance(piece_info["melody"], list)
            
            for note in piece_info["melody"]:
                assert "pitch" in note
                assert "duration" in note

def test_integration_with_demo():
    """Integration test using demo file if available"""
    if not os.path.exists("audio/demo_performance.wav"):
        pytest.skip("audio/demo_performance.wav not found")
    
    analyzer = PerformanceAnalyzer()
    analyzer.set_piece_from_melody(PIECES["twinkle"], 100)
    
    try:
        results = analyzer.analyze_performance("audio/demo_performance.wav", "integration_test")
        
        # Basic validation
        assert isinstance(results, dict)
        assert "summary" in results
        assert "standard_analysis" in results
        
        # Check that files were created
        assert os.path.exists("integration_test.json")
        
        # Validate JSON structure
        with open("integration_test.json", 'r') as f:
            json_data = json.load(f)
            assert json_data == results
        
        print("Integration test passed - analysis completed successfully")
        
    except Exception as e:
        pytest.fail(f"Integration test failed: {e}")
    
    finally:
        # Clean up
        for file in ["integration_test.json", "integration_test_analysis.png"]:
            if os.path.exists(file):
                os.unlink(file)

if __name__ == "__main__":
    # Run tests
    print("Running comprehensive test suite for new architecture...")
    pytest.main([__file__, "-v"])

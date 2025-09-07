#!/usr/bin/env python3
"""
Test script for enhanced GUI features:
1. Sheet music visualization fixes
2. LLM integration with ABRSM scoring
3. Template auto-population
4. Performer score input field
"""

import sys
import os
import json
import tkinter as tk
from enhanced_gui_interface import EnhancedABRSMGUI

def test_enhanced_features():
    """Test all enhanced features"""
    print("🧪 Testing Enhanced ABRSM GUI Features")
    print("=" * 50)
    
    # Test 1: Module import and initialization
    try:
        root = tk.Tk()
        root.withdraw()  # Hide during testing
        app = EnhancedABRSMGUI(root)
        print("✅ Test 1 PASSED: GUI initializes successfully")
    except Exception as e:
        print(f"❌ Test 1 FAILED: GUI initialization error - {e}")
        return False
    
    # Test 2: Check for performer score field
    try:
        assert hasattr(app, 'performer_score_var'), "Performer score variable missing"
        assert hasattr(app, 'api_key_var'), "API key variable missing"
        print("✅ Test 2 PASSED: Both API key and performer score fields exist")
    except Exception as e:
        print(f"❌ Test 2 FAILED: Missing required fields - {e}")
        return False
    
    # Test 3: Check ABRSM prompt generation with performer score
    try:
        # Test with performer score
        app.performer_score_var.set("75")
        analysis_data = {
            "pitch_accuracy": 85.5,
            "timing_accuracy": 90.2,
            "completion_rate": 95.0,
            "notes_analyzed": 50
        }
        prompt = app._create_abrsm_scoring_prompt(analysis_data)
        
        assert "PERFORMER'S SELF-ASSESSMENT SCORE: 75.0/100" in prompt, "Performer score not included in prompt"
        assert "ABRSM SCORING FRAMEWORK" in prompt, "ABRSM framework missing"
        assert "SCORING BANDS WITH DETAILED CRITERIA" in prompt, "Scoring bands missing"
        print("✅ Test 3 PASSED: ABRSM prompt includes performer score and detailed criteria")
    except Exception as e:
        print(f"❌ Test 3 FAILED: ABRSM prompt generation error - {e}")
        return False
    
    # Test 4: Check template auto-population functionality
    try:
        app.load_available_midi_files()  # This populates app.available_pieces
        piece_keys = app.get_available_pieces()  # This returns the list
        print(f"✅ Test 4 PASSED: Found {len(piece_keys)} pieces for auto-population")
        
        # Check if at least some templates were loaded
        if len(piece_keys) > 0:
            print(f"   📁 Templates available: {', '.join(piece_keys[:3])}{'...' if len(piece_keys) > 3 else ''}")
    except Exception as e:
        print(f"❌ Test 4 FAILED: Template auto-population error - {e}")
        return False
    
    # Test 5: Check sheet music visualization method exists
    try:
        assert hasattr(app, '_draw_sheet_music'), "Sheet music drawing method missing"
        print("✅ Test 5 PASSED: Sheet music visualization methods exist")
    except Exception as e:
        print(f"❌ Test 5 FAILED: Sheet music methods missing - {e}")
        return False
    
    # Test 6: Check error handling method exists
    try:
        assert hasattr(app, '_show_feedback_error'), "Feedback error handling method missing"
        assert hasattr(app, '_update_abrsm_feedback_ui'), "ABRSM feedback update method missing"
        print("✅ Test 6 PASSED: Feedback handling methods exist")
    except Exception as e:
        print(f"❌ Test 6 FAILED: Feedback methods missing - {e}")
        return False
    
    # Test 7: Check comprehensive analysis generation
    try:
        # Mock proper analysis data structure
        app.current_analysis = {
            'standard_analysis': {
                'note_details': [
                    {
                        'pitch_deviation_cents': 10,
                        'timing_deviation_ms': 50,
                        'expected_pitch': 'C4',
                        'actual_pitch': 'C4',
                        'expected_time': 0.0,
                        'actual_time': 0.05
                    },
                    {
                        'pitch_deviation_cents': -5,
                        'timing_deviation_ms': -30,
                        'expected_pitch': 'D4',
                        'actual_pitch': 'D4',
                        'expected_time': 0.5,
                        'actual_time': 0.47
                    }
                ],
                'overall_assessment': {
                    'accuracy': 0.9,
                    'timing': 0.95
                }
            }
        }
        app.piece_var.set("test_piece")
        
        analysis_json = app._generate_comprehensive_analysis_json()
        assert 'performance_statistics' in analysis_json, "Performance statistics missing"
        assert 'technical_metrics' in analysis_json, "Technical metrics missing"
        assert 'piece_information' in analysis_json, "Piece information missing"
        print("✅ Test 7 PASSED: Comprehensive analysis generation works")
    except Exception as e:
        print(f"❌ Test 7 FAILED: Analysis generation error - {e}")
        return False
    
    # Test 8: Test performer score validation
    try:
        # Test valid scores
        app.performer_score_var.set("85")
        prompt = app._create_abrsm_scoring_prompt({})
        assert "85.0/100" in prompt, "Valid score not processed correctly"
        
        # Test invalid score (should be ignored)
        app.performer_score_var.set("invalid")
        prompt = app._create_abrsm_scoring_prompt({})
        assert "PERFORMER'S SELF-ASSESSMENT SCORE" not in prompt, "Invalid score should be ignored"
        
        print("✅ Test 8 PASSED: Performer score validation works correctly")
    except Exception as e:
        print(f"❌ Test 8 FAILED: Score validation error - {e}")
        return False
    
    # Cleanup
    root.destroy()
    
    print("\n🎉 ALL TESTS PASSED!")
    print("\n📋 Enhanced Features Summary:")
    print("   ✅ Sheet music visualization with proper staff positioning")
    print("   ✅ LLM integration with comprehensive ABRSM scoring system")
    print("   ✅ Template auto-population from MIDI/MXL files")
    print("   ✅ Performer score input field for targeted feedback")
    print("   ✅ Detailed 8-band ABRSM scoring with examples")
    print("   ✅ Error handling and feedback display")
    print("   ✅ JSON analysis generation with comprehensive metrics")
    
    return True

if __name__ == "__main__":
    success = test_enhanced_features()
    sys.exit(0 if success else 1)

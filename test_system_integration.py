#!/usr/bin/env python3
"""
Test script to verify the core functionality works
"""

import os
import sys

def test_core_imports():
    """Test that core modules can be imported"""
    print("🧪 Testing core imports...")
    
    try:
        from enhanced_main_fixed import MusicAnalyzer, PIECES
        print("✅ MusicAnalyzer imported successfully")
        print(f"   Available pieces: {list(PIECES.keys())}")
    except Exception as e:
        print(f"❌ Failed to import MusicAnalyzer: {e}")
        return False
    
    try:
        from enhanced_gui_interface import EnhancedABRSMGUI
        print("✅ GUI interface imported successfully")
    except Exception as e:
        print(f"❌ Failed to import GUI: {e}")
        return False
    
    return True

def test_analyzer_functionality():
    """Test basic analyzer functionality"""
    print("\n🧪 Testing analyzer functionality...")
    
    try:
        from enhanced_main_fixed import MusicAnalyzer
        analyzer = MusicAnalyzer()
        print("✅ MusicAnalyzer instance created successfully")
        
        # Test with available audio files
        test_files = [
            "test_5measures.wav",
            "audio/test_5measures.wav", 
            "audio/demo_performance.wav"
        ]
        
        test_file = None
        for file in test_files:
            if os.path.exists(file):
                test_file = file
                break
        
        if test_file:
            print(f"🎵 Testing with audio file: {test_file}")
            
            # Test basic analysis
            result = analyzer.analyze_performance_audio(test_file)
            
            # Also test the enhanced analysis if available
            try:
                enhanced_result = analyzer.analyze_with_enhancements(test_file, generate_visualizations=False)
                if enhanced_result:
                    result = enhanced_result
                    print("✅ Enhanced analysis completed successfully")
            except Exception as e:
                print(f"ℹ️ Enhanced analysis not available: {e}")
            
            if result is not None:
                print("✅ Audio analysis completed successfully")
                
                # Check if it's a dictionary with results
                if isinstance(result, dict):
                    # Print basic metrics
                    if "overall_score" in result:
                        print(f"   Overall Score: {result['overall_score']:.1f}/100")
                    if "note_accuracy" in result:
                        print(f"   Note Accuracy: {result['note_accuracy']:.1f}%")
                elif isinstance(result, (list, tuple)):
                    print(f"   Analysis returned {len(result)} results")
                else:
                    print(f"   Analysis returned: {type(result)}")
                
                return True
            else:
                print("❌ Analysis returned None")
                return False
        else:
            print("⚠️ No test audio files found, skipping analysis test")
            return True
            
    except Exception as e:
        print(f"❌ Analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_creation():
    """Test GUI creation without actually showing it"""
    print("\n🧪 Testing GUI creation...")
    
    try:
        import tkinter as tk
        from enhanced_gui_interface import EnhancedABRSMGUI
        
        # Create root but don't show it
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Create GUI instance
        gui = EnhancedABRSMGUI(root)
        print("✅ GUI created successfully")
        
        # Test if key components exist
        if hasattr(gui, 'current_analysis'):
            print("✅ GUI has analysis state")
        if hasattr(gui, 'available_pieces'):
            print(f"✅ GUI has {len(gui.available_pieces)} available pieces")
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ GUI creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting comprehensive test suite...\n")
    
    tests = [
        test_core_imports,
        test_analyzer_functionality,
        test_gui_creation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! The system is ready to use.")
        return True
    else:
        print(f"\n⚠️ {failed} test(s) failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Test complete GUI workflow with demo analysis
"""

import tkinter as tk
import time
import threading
from gui_interface import ABRSMAnalysisGUI

def test_gui_workflow():
    """Test the complete GUI workflow"""
    print("Creating GUI...")
    root = tk.Tk()
    gui = ABRSMAnalysisGUI(root)
    
    # Test sequence
    def run_test_sequence():
        try:
            print("Starting test sequence...")
            time.sleep(1)  # Let GUI initialize
            
            print("Loading demo file...")
            gui.load_demo()
            time.sleep(2)  # Wait for demo to load
            
            print("Current audio file:", gui.current_audio_file)
            if gui.current_audio_file:
                print("Running analysis...")
                gui.run_analysis()
                time.sleep(5)  # Wait for analysis to complete
                
                print("Checking analysis result...")
                if gui.current_analysis:
                    print("✓ Analysis completed successfully!")
                    print("✓ Standard analysis available:", 'standard_analysis' in gui.current_analysis)
                    print("✓ Enhanced features available:", 'enhanced_features' in gui.current_analysis)
                    
                    # Check if tabs have content
                    if hasattr(gui, 'summary_text'):
                        content = gui.summary_text.get(1.0, tk.END).strip()
                        if content and len(content) > 50:
                            print("✓ Overview tab has content")
                        else:
                            print("✗ Overview tab appears empty")
                else:
                    print("✗ No analysis result found")
            else:
                print("✗ No audio file loaded")
                
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Close GUI after test
            root.after(1000, root.quit)
    
    # Start test in background
    threading.Thread(target=run_test_sequence, daemon=True).start()
    
    # Run GUI for limited time
    root.after(10000, root.quit)  # Auto-close after 10 seconds
    try:
        root.mainloop()
    except:
        pass
    
    return gui

if __name__ == "__main__":
    print("Testing Complete GUI Workflow")
    print("=" * 40)
    gui = test_gui_workflow()
    print("\nTest completed.")

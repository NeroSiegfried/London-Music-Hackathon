#!/usr/bin/env python3
"""
Launch script for the Enhanced ABRSM Music Analysis GUI
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

def main():
    """Launch the Enhanced ABRSM GUI"""
    
    print("🎵 Launching Enhanced ABRSM Music Analysis System...")
    
    try:
        # Import the GUI
        from enhanced_gui_interface import EnhancedABRSMGUI
        
        # Create the main window
        root = tk.Tk()
        
        # Create the application
        app = EnhancedABRSMGUI(root)
        
        print("✅ GUI loaded successfully!")
        print("📝 Features available:")
        print("   • Audio performance analysis")
        print("   • Note-by-note comparison")
        print("   • Mistake pattern detection") 
        print("   • Interactive sheet music")
        print("   • Performance timing analysis")
        print("   • AI-powered feedback")
        print("\n🎯 Ready for music analysis!")
        
        # Start the GUI
        root.mainloop()
        
    except ImportError as e:
        error_msg = f"""
Failed to import required modules: {e}

Make sure you have:
1. Activated the virtual environment: source test_env/bin/activate
2. Installed all dependencies: pip install -r requirements.txt

If you're missing modules, try:
pip install librosa matplotlib tkinter pygame soundfile numpy scipy music21
"""
        print(error_msg)
        
        # Try to show GUI error if tkinter is available
        try:
            messagebox.showerror("Import Error", error_msg)
        except:
            pass
        
        sys.exit(1)
        
    except Exception as e:
        error_msg = f"Failed to start GUI: {e}"
        print(error_msg)
        
        try:
            messagebox.showerror("Startup Error", error_msg)
        except:
            pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()

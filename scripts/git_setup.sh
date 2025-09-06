#!/bin/bash

# ABRSM AI Music Feedback System - Git Setup Script
# London Music Technology Hackathon 2025

echo "🎵 ABRSM AI Music Feedback System - Git Setup"
echo "=============================================="

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install git first."
    exit 1
fi

# Check if we're already in a git repository
if [ -d ".git" ]; then
    echo "⚠️  Git repository already exists."
    read -p "Do you want to continue and add changes? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 1
    fi
else
    # Initialize git repository
    echo "📁 Initializing git repository..."
    git init
fi

# Add all files (respecting .gitignore)
echo "📂 Adding files to git..."
git add .

# Check if there are any changes to commit
if git diff --staged --quiet; then
    echo "ℹ️  No changes to commit."
else
    # Commit changes
    echo "💾 Committing changes..."
    git commit -m "ABRSM AI Music Feedback System - Competition Ready

- Enhanced GUI with note-by-note analysis
- Advanced mistake detection and performance diff
- Comprehensive LLM training documentation  
- Complete ABRSM song database (583 pieces)
- Competition-ready features for hackathon submission"
    
    echo "✅ Git repository ready!"
    echo ""
    echo "🔗 Next steps to push to GitHub:"
    echo "1. Create a new repository on GitHub.com"
    echo "2. Run: git remote add origin https://github.com/YOUR_USERNAME/abrsm-ai-music-feedback.git"
    echo "3. Run: git branch -M main"
    echo "4. Run: git push -u origin main"
    echo ""
    echo "📖 See GITHUB_SETUP.md for detailed instructions"
fi

echo ""
echo "🎯 Repository Structure:"
echo "✅ Source code and documentation included"
echo "✅ requirements.txt with all dependencies"
echo "✅ .gitignore excludes venv/ and large files"
echo "✅ Demo files included for immediate testing"
echo "✅ Ready for hackathon submission!"

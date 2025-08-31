"""Run the FloatChat Streamlit application."""

import subprocess
import sys
from pathlib import Path

def main():
    """Run Streamlit app."""
    app_path = Path(__file__).parent / "app" / "main.py"
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        str(app_path),
        "--server.port=8501",
        "--server.address=localhost"
    ])

if __name__ == "__main__":
    main()

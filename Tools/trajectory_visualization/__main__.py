import sys
import os

# Ensure the parent directory is in sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from Tools.trajectory_visualization.cli import main

if __name__ == "__main__":
    main()

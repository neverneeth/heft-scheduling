"""
Simple Sanity Check Example

This script demonstrates a quick sanity check of the scheduling framework.
It generates a random DAG, runs multiple algorithms, and visualizes the results.
"""

import sys
import os

# Add project root to path so we can import src as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import using the full package path
from src.utils.sanity_checker import quick_sanity_check


def main():
    """Run a simple sanity check."""
    print("Running HEFT Scheduling Framework Sanity Check\n")
    
    # Run sanity check with default parameters
    results = quick_sanity_check(
        num_tasks=9,
        num_processors=3,
        random_seed=42
    )
    
    print("\nSanity check complete!")
    print("Check the generated plots for visual comparison.")


if __name__ == "__main__":
    main()
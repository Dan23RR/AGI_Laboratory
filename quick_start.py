#!/usr/bin/env python3
"""
Quick Start Script for AGI Laboratory
=====================================

Simple script to start evolution with default settings.
"""

import subprocess
import sys
import os

print("🚀 AGI LABORATORY QUICK START")
print("="*50)

# Check Python version
if sys.version_info < (3, 8):
    print("❌ Python 3.8 or higher required!")
    sys.exit(1)

# Default configurations
configs = {
    "1": {
        "name": "Quick Test (10 generations)",
        "cmd": ["python", "launch_agi_lab.py", "--generations", "10", "--population", "20"]
    },
    "2": {
        "name": "Standard Evolution (100 generations)",
        "cmd": ["python", "launch_agi_lab.py", "--generations", "100", "--population", "50"]
    },
    "3": {
        "name": "Extended Evolution with Meta-Learning (500 generations)",
        "cmd": ["python", "launch_agi_lab.py", "--generations", "500", "--population", "100", "--meta-evolution"]
    },
    "4": {
        "name": "Trading Specialization",
        "cmd": ["python", "launch_agi_lab.py", "--task", "trading", "--generations", "200", "--population", "50"]
    },
    "5": {
        "name": "Research/Creativity Focus",
        "cmd": ["python", "launch_agi_lab.py", "--task", "research", "--generations", "200", "--population", "50"]
    },
    "6": {
        "name": "CPU-only Mode (slower)",
        "cmd": ["python", "launch_agi_lab.py", "--device", "cpu", "--generations", "50", "--population", "30"]
    }
}

# Show options
print("\nSelect evolution configuration:")
print("-"*50)
for key, config in configs.items():
    print(f"{key}. {config['name']}")
print("-"*50)

# Get user choice
while True:
    choice = input("\nEnter your choice (1-6) or 'q' to quit: ").strip()
    
    if choice.lower() == 'q':
        print("👋 Exiting...")
        sys.exit(0)
    
    if choice in configs:
        break
    
    print("❌ Invalid choice. Please try again.")

# Confirm
selected = configs[choice]
print(f"\n✅ Selected: {selected['name']}")
print(f"Command: {' '.join(selected['cmd'])}")

confirm = input("\nProceed? (y/n): ").strip().lower()
if confirm != 'y':
    print("❌ Cancelled")
    sys.exit(0)

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Launch
print("\n🔬 Launching AGI Evolution Laboratory...")
print("="*50)
print("Press Ctrl+C to stop at any time")
print("="*50)

try:
    # Run the evolution
    result = subprocess.run(selected['cmd'], check=True)
    
    if result.returncode == 0:
        print("\n✅ Evolution completed successfully!")
    else:
        print(f"\n⚠️ Evolution exited with code: {result.returncode}")
        
except subprocess.CalledProcessError as e:
    print(f"\n❌ Evolution failed: {e}")
except KeyboardInterrupt:
    print("\n\n⚠️ Evolution interrupted by user")
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")

print("\n📁 Check the checkpoints_* directory for results")
print("📊 Check evolution_log_*.log for detailed logs")
print("\n✅ Done!")
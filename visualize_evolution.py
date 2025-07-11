#!/usr/bin/env python3
"""
Visualize Evolution Progress
===========================
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch

def plot_evolution_history():
    """Plot fitness over generations from checkpoint"""
    # Find checkpoint
    checkpoint_files = list(Path('.').glob('checkpoint_*.pt'))
    if not checkpoint_files:
        print("No checkpoint files found!")
        return
    
    latest = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    checkpoint = torch.load(latest)
    
    # Extract history
    history = checkpoint.get('evolution_history', {})
    if not history:
        print("No evolution history found in checkpoint!")
        return
    
    generations = list(range(1, len(history.get('best_fitness', [])) + 1))
    best_fitness = history.get('best_fitness', [])
    avg_fitness = history.get('avg_fitness', [])
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness')
    plt.plot(generations, avg_fitness, 'g--', linewidth=1, label='Average Fitness')
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title('Evolution Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Mark improvements
    for i in range(1, len(best_fitness)):
        if best_fitness[i] > best_fitness[i-1]:
            plt.plot(generations[i], best_fitness[i], 'ro', markersize=8)
    
    plt.tight_layout()
    plt.savefig('evolution_progress.png')
    plt.show()
    
    print(f"✅ Plot saved as evolution_progress.png")
    print(f"📊 Final best fitness: {best_fitness[-1]:.4f}")
    print(f"📈 Improvement: {best_fitness[-1] - best_fitness[0]:.4f}")

if __name__ == "__main__":
    plot_evolution_history()
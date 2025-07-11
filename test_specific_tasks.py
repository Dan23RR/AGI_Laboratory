#!/usr/bin/env python3
"""
Test Evolved Model on Specific Tasks
===================================
"""

import torch
import numpy as np
from pathlib import Path
from evolution.mind_factory_v2 import MindFactoryV2, MindConfig

def load_evolved_mind():
    """Load the best evolved mind"""
    checkpoint_files = list(Path('.').glob('checkpoint_*.pt'))
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found!")
    
    latest = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    checkpoint = torch.load(latest)
    
    factory = MindFactoryV2()
    config = MindConfig(hidden_dim=512, output_dim=256, memory_fraction=0.5)
    mind = factory.create_mind_from_genome(checkpoint['best_genome'], config)
    
    return mind, checkpoint.get('best_fitness', 0)

def test_pattern_completion():
    """Test if model can complete patterns"""
    print("\n🧩 PATTERN COMPLETION TEST")
    print("-"*40)
    
    mind, fitness = load_evolved_mind()
    
    # Create a simple pattern: alternating high/low values
    pattern = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0]]).float()
    pattern = pattern.repeat(1, 16)  # Make it 128 dims
    
    # Show partial pattern and see if model completes it
    partial = pattern.clone()
    partial[0, 64:] = 0.5  # Hide second half
    
    output = mind.forward(partial)
    
    # Check if output learned the pattern
    correlation = torch.corrcoef(torch.stack([pattern[0, :output.shape[1]], output[0]]))[0, 1]
    
    print(f"Pattern correlation: {correlation:.4f}")
    print(f"Success: {'✅' if correlation > 0.5 else '❌'}")
    
    return correlation.item()

def test_sequence_memory():
    """Test if model can remember sequences"""
    print("\n🧠 SEQUENCE MEMORY TEST")
    print("-"*40)
    
    mind, _ = load_evolved_mind()
    mind.reset_state()
    
    # Create a sequence to remember
    sequence = [torch.randn(1, 128) for _ in range(5)]
    
    # Feed sequence
    print("Feeding sequence...")
    for i, x in enumerate(sequence):
        _ = mind.forward(x)
        print(f"  Step {i+1} fed")
    
    # Test recall with similar input
    recall_scores = []
    for i, original in enumerate(sequence):
        noise = torch.randn_like(original) * 0.1
        query = original + noise
        output = mind.forward(query)
        
        # Measure similarity to original
        similarity = torch.cosine_similarity(output, original, dim=1).item()
        recall_scores.append(similarity)
        print(f"  Recall {i+1}: {similarity:.4f}")
    
    avg_recall = np.mean(recall_scores)
    print(f"\nAverage recall: {avg_recall:.4f}")
    print(f"Success: {'✅' if avg_recall > 0.6 else '❌'}")
    
    return avg_recall

def test_logical_operations():
    """Test basic logical reasoning"""
    print("\n🔤 LOGICAL OPERATIONS TEST")
    print("-"*40)
    
    mind, _ = load_evolved_mind()
    
    # Create logical patterns (AND operation)
    test_cases = [
        (torch.tensor([[0, 0]]), torch.tensor([[0]])),  # 0 AND 0 = 0
        (torch.tensor([[0, 1]]), torch.tensor([[0]])),  # 0 AND 1 = 0
        (torch.tensor([[1, 0]]), torch.tensor([[0]])),  # 1 AND 0 = 0
        (torch.tensor([[1, 1]]), torch.tensor([[1]])),  # 1 AND 1 = 1
    ]
    
    correct = 0
    for input_vals, expected in test_cases:
        # Pad input to 128 dims
        padded_input = torch.zeros(1, 128)
        padded_input[0, :2] = input_vals
        
        output = mind.forward(padded_input)
        prediction = (output[0, 0] > 0.5).float()
        
        is_correct = (prediction == expected[0, 0]).item()
        correct += is_correct
        
        print(f"  {input_vals[0].numpy()} → {prediction.item():.0f} (expected {expected[0,0].item():.0f}) {'✅' if is_correct else '❌'}")
    
    accuracy = correct / len(test_cases)
    print(f"\nAccuracy: {accuracy:.2%}")
    
    return accuracy

def test_adaptation_speed():
    """Test how quickly model adapts to new patterns"""
    print("\n⚡ ADAPTATION SPEED TEST")
    print("-"*40)
    
    mind, _ = load_evolved_mind()
    
    # Create a new pattern the model hasn't seen
    new_pattern = torch.sin(torch.linspace(0, 4*np.pi, 128)).unsqueeze(0)
    
    adaptation_scores = []
    for epoch in range(10):
        output = mind.forward(new_pattern)
        similarity = torch.cosine_similarity(output[:, :128], new_pattern, dim=1).item()
        adaptation_scores.append(similarity)
        print(f"  Epoch {epoch+1}: similarity = {similarity:.4f}")
    
    # Check if model improved
    improvement = adaptation_scores[-1] - adaptation_scores[0]
    print(f"\nImprovement: {improvement:+.4f}")
    print(f"Success: {'✅' if improvement > 0.1 else '❌'}")
    
    return improvement

def run_all_tests():
    """Run all tests and show summary"""
    print("🧬 TESTING EVOLVED MODEL")
    print("="*50)
    
    results = {
        "Pattern Completion": test_pattern_completion(),
        "Sequence Memory": test_sequence_memory(),
        "Logical Operations": test_logical_operations(),
        "Adaptation Speed": test_adaptation_speed()
    }
    
    print("\n📊 SUMMARY")
    print("="*50)
    for test_name, score in results.items():
        status = "✅" if score > 0.5 else "⚠️" if score > 0 else "❌"
        print(f"{test_name:.<30} {score:>6.4f} {status}")
    
    overall = np.mean(list(results.values()))
    print(f"\nOVERALL SCORE: {overall:.4f}")
    
    return results

if __name__ == "__main__":
    run_all_tests()
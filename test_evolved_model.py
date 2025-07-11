#!/usr/bin/env python3
"""
Test Your Evolved Model
=====================
"""

import torch
import json
from pathlib import Path
from evolution.mind_factory_v2 import MindFactoryV2, MindConfig
from evolution.fitness.agi_fitness_metrics_v2 import AGIFitnessEvaluator

def load_best_genome():
    """Load the best genome from checkpoint"""
    # Find latest checkpoint
    checkpoint_files = list(Path('.').glob('checkpoint_*.pt'))
    if not checkpoint_files:
        print("❌ No checkpoint files found!")
        return None
    
    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    print(f"📂 Loading checkpoint: {latest_checkpoint}")
    
    checkpoint = torch.load(latest_checkpoint)
    best_genome = checkpoint['best_genome']
    print(f"✅ Loaded genome with fitness: {checkpoint.get('best_fitness', 'N/A')}")
    
    return best_genome

def test_evolved_model():
    """Test the evolved model on various tasks"""
    # Load genome
    genome = load_best_genome()
    if genome is None:
        return
    
    # Create mind from genome
    factory = MindFactoryV2()
    mind_config = MindConfig(
        hidden_dim=512,
        output_dim=256,
        memory_fraction=0.5
    )
    
    print("\n🧠 Creating mind from evolved genome...")
    mind = factory.create_mind_from_genome(genome, mind_config)
    
    # Create fitness evaluator
    evaluator = AGIFitnessEvaluator()
    
    print("\n📊 Running comprehensive evaluation...")
    fitness_score = evaluator.evaluate_complete(mind.model)
    
    print("\n🎯 EVALUATION RESULTS:")
    print("="*50)
    print(f"Generalization:  {fitness_score.generalization:.4f}")
    print(f"Reasoning:       {fitness_score.reasoning:.4f}")
    print(f"Adaptability:    {fitness_score.adaptability:.4f}")
    print(f"Creativity:      {fitness_score.creativity:.4f}")
    print(f"Emergence:       {fitness_score.emergence:.4f}")
    print(f"Consciousness:   {fitness_score.consciousness:.4f}")
    print(f"Efficiency:      {fitness_score.efficiency:.4f}")
    print(f"Robustness:      {fitness_score.robustness:.4f}")
    print("="*50)
    print(f"OVERALL SCORE:   {fitness_score.overall:.4f}")
    
    # Test specific capabilities
    print("\n🔬 Testing specific capabilities...")
    
    # Test 1: Pattern Recognition
    print("\n1️⃣ Pattern Recognition Test:")
    test_input = torch.randn(1, 128)
    with torch.no_grad():
        output = mind.forward(test_input)
    print(f"   Output shape: {output.shape}")
    print(f"   Output variance: {output.var().item():.4f}")
    
    # Test 2: Memory Persistence
    print("\n2️⃣ Memory Test:")
    memory_before = mind.get_state().get('working_memory')
    if memory_before is not None:
        print(f"   Memory size: {memory_before.shape}")
        print(f"   Memory utilization: {(memory_before.abs() > 0.1).float().mean():.2%}")
    
    # Test 3: Multi-step Reasoning
    print("\n3️⃣ Multi-step Reasoning:")
    sequence = [torch.randn(1, 128) for _ in range(5)]
    outputs = []
    for step, x in enumerate(sequence):
        with torch.no_grad():
            out = mind.forward(x)
            outputs.append(out)
        print(f"   Step {step+1}: Output norm = {out.norm().item():.4f}")
    
    # Show active modules
    print("\n🧩 Active Modules in Evolved Mind:")
    for i, (name, module) in enumerate(mind.modules.items()):
        print(f"   {i+1}. {name}")
    
    return mind, fitness_score

def interactive_test(mind):
    """Interactive testing mode"""
    print("\n🎮 INTERACTIVE TEST MODE")
    print("="*50)
    print("Commands:")
    print("  'test <n>' - Run n random inputs")
    print("  'state' - Show current mind state")
    print("  'reset' - Reset mind state")
    print("  'quit' - Exit")
    
    while True:
        cmd = input("\n> ").strip().lower()
        
        if cmd == 'quit':
            break
        elif cmd == 'state':
            state = mind.get_state()
            for key, tensor in state.items():
                if tensor is not None:
                    print(f"{key}: shape={tensor.shape}, mean={tensor.mean():.4f}")
        elif cmd == 'reset':
            mind.reset_state()
            print("✅ Mind state reset")
        elif cmd.startswith('test'):
            try:
                n = int(cmd.split()[1]) if len(cmd.split()) > 1 else 1
                for i in range(n):
                    x = torch.randn(1, 128)
                    out = mind.forward(x)
                    print(f"Test {i+1}: input_norm={x.norm():.4f}, output_norm={out.norm():.4f}")
            except:
                print("❌ Invalid command. Use 'test <n>'")
        else:
            print("❌ Unknown command")

if __name__ == "__main__":
    print("🧬 AGI Laboratory - Model Tester")
    print("="*50)
    
    # Test the evolved model
    mind, scores = test_evolved_model()
    
    # Optional: Enter interactive mode
    if mind is not None:
        response = input("\nEnter interactive test mode? (y/n): ")
        if response.lower() == 'y':
            interactive_test(mind)
    
    print("\n✅ Testing complete!")
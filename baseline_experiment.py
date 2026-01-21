import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import evaluate
import time
from hierarchical_summarizer import HierarchicalSummarizer

def get_device():
    """Select best available device for M3 Pro MacBook Pro"""
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) is available")
        return 'mps'
    elif torch.cuda.is_available():
        print("✅ CUDA is available")
        return 'cuda'
    else:
        print("⚠️  Using CPU fallback")
        return 'cpu'

def run_baseline_experiment(num_samples: int = 50, device: str = None):
    """
    Run the Baseline Experiment (Method B for the paper).
    
    Objective: 
    Measure the performance of SOTA Hierarchical Summarization 
    (Adaptive Semantic Chunking + PEGASUS) *without* Fact Verification.
    
    This establishes the 'strong baseline' that we aim to improve upon.
    """
    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    
    print(f"🚀 Starting Baseline Experiment")
    print(f"   Samples: {num_samples}")
    print(f"   Device:  {device}")
    
    # 1. Load Dataset
    print("Loading Multi-News dataset (Test Split)...")
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    
    # Select random samples
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    # Convert to list of dicts for easier handling
    data_samples = [dataset[int(i)] for i in indices]
    
    # 2. Initialize Model
    print("Initializing SOTA Hierarchical Summarizer...")
    summarizer = HierarchicalSummarizer(device=device)
    
    # 3. Load Metrics
    print("Loading ROUGE metric...")
    rouge = evaluate.load('rouge')
    
    results = []
    generated_summaries = []
    reference_summaries = []
    
    # 4. Run Inference
    print("\n⚡ Generating Summaries...")
    start_time = time.time()
    for item in tqdm(data_samples, desc="Processing"):
        doc = item['document']
        ref = item['summary']
        
        # Run Pipeline
        try:
            output = summarizer.summarize_document(doc)
            gen_summary = output['final_summary']
            
            # Save detailed log
            results.append({
                'reference': ref,
                'generated': gen_summary,
                'num_chunks': len(output['chunks']),
                'chunk_summaries': output['chunk_summaries']
            })
            
            generated_summaries.append(gen_summary)
            reference_summaries.append(ref)
            
        except Exception as e:
            print(f"Error processing doc: {e}")
            continue
    
    inference_time = time.time() - start_time
            
    # 5. Compute ROUGE
    print("\nComputing ROUGE Scores...")
    metrics = rouge.compute(
        predictions=generated_summaries, 
        references=reference_summaries,
        use_stemmer=True
    )
    
    # Extract float values - handle both Score objects and direct floats
    # The evaluate library returns different formats depending on version
    if hasattr(metrics['rouge1'], 'mid'):
        # Older format with .mid.fmeasure
        rouge1_score = metrics['rouge1'].mid.fmeasure
        rouge2_score = metrics['rouge2'].mid.fmeasure
        rougeL_score = metrics['rougeL'].mid.fmeasure
    else:
        # Newer format returns numpy.float64 directly
        rouge1_score = float(metrics['rouge1'])
        rouge2_score = float(metrics['rouge2'])
        rougeL_score = float(metrics['rougeL'])
    
    print("\n" + "="*50)
    print("BASELINE RESULTS (Multi-News Subset)")
    print("="*50)
    print(f"ROUGE-1: {rouge1_score*100:.2f}")
    print(f"ROUGE-2: {rouge2_score*100:.2f}")
    print(f"ROUGE-L: {rougeL_score*100:.2f}")
    print(f"\nTotal Inference Time: {inference_time:.2f}s")
    print(f"Avg Time per Sample: {inference_time/len(results):.2f}s")
    print("="*50)
    
    # Create results directory
    results_dir = '/Users/mrarnav69/Documents/Hierarchical-Multi-Document-Summarizer/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save Results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(results_dir, 'baseline_experiment_details.csv'), index=False)
    
    summary_stats = {
        'num_samples': len(results),
        'device': device,
        'inference_time': inference_time,
        'avg_time_per_sample': inference_time/len(results),
        'metrics': {
            'rouge1': float(rouge1_score),
            'rouge2': float(rouge2_score),
            'rougeL': float(rougeL_score)
        },
        'config': 'Hierarchical PEGASUS (SOTA Chunking, Optimized)'
    }
    with open(os.path.join(results_dir, 'baseline_metrics.json'), 'w') as f:
        json.dump(summary_stats, f, indent=2)
        
    print(f"\nSaved detailed outputs to '{results_dir}/baseline_experiment_details.csv'")
    print(f"Saved metrics to '{results_dir}/baseline_metrics.json'")

if __name__ == "__main__":
    # Auto-detect device (MPS on M3 Pro for best performance)
    run_baseline_experiment(num_samples=20)

import json
import os
import numpy as np
import scipy
import shutil
import subprocess

def run_conv(kernel, A, n_trials=10):
    # Clean up and create directories
    if os.path.exists("input"):
        shutil.rmtree("input")
    if os.path.exists("output"):
        shutil.rmtree("output")
    
    os.makedirs("input")
    os.makedirs("output")
    
    # Save input data
    np.save('input/A.npy', A)
    
    # Run the kernel with correct arguments
    cmd = [f'./{kernel}', '--input', 'input', '--output', 'output', '--trial-max', str(n_trials), '--time-max', 'inf']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running {kernel}:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Kernel {kernel} execution failed")
    
    B = np.load('output/B.npy')
    with open('output/measurements.json', 'r') as f:
        measurements = dict(json.load(f))
    measurements['B'] = B
    return measurements

if __name__ == "__main__":
    m, n = 100, 100
    input_data = np.random.rand(m, n).astype(np.float64)

    n_trials = 1000  # Reduced for interleaved testing
    
    baseline_times = []
    optimized_times = []
    
    for trial in range(n_trials):
        if trial % 100 == 0:
            print(f"Progress: {trial}/{n_trials}")
        
        # Randomly decide order for this trial to avoid systemic bias
        order = np.random.choice(['baseline_first', 'optimized_first'])
        
        if order == 'baseline_first':
            baseline_result = run_conv('conv_baseline', input_data, n_trials=1)
            baseline_times.append(baseline_result['time'])
            optimized_result = run_conv('conv_optimized', input_data, n_trials=1)
            optimized_times.append(optimized_result['time'])
        else:
            optimized_result = run_conv('conv_optimized', input_data, n_trials=1)
            optimized_times.append(optimized_result['time'])
            baseline_result = run_conv('conv_baseline', input_data, n_trials=1)
            baseline_times.append(baseline_result['time'])
    
    baseline_times = np.array(baseline_times)
    optimized_times = np.array(optimized_times)
    
    wins = np.sum(optimized_times < baseline_times)
    win_rate = wins / n_trials
    
    # Use binomial test (null hypothesis: win rate = 0.5)
    p_value = scipy.stats.binomtest(wins, n_trials, 0.5).pvalue
    
    print(f"baseline time: {np.min(baseline_times):.0f} ns")
    print(f"optimized time: {np.min(optimized_times):.0f} ns")
    print(f"speedup: {np.min(baseline_times) / np.min(optimized_times):.2f}x")
    print(f"optimized wins: {wins}/{n_trials} ({win_rate:.1%})")
    print(f"binomial test p-value: {p_value:.6f}")
    
    if p_value > 0.05:
        print("No significant difference detected.")
    else:
        print("Significant difference detected!")

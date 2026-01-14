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

    n_trials = 10000

    print("Running C benchmark...")
    baseline_results = run_conv('conv_baseline', input_data, n_trials=n_trials)
    optimized_results = run_conv('conv_optimized', input_data, n_trials=n_trials)

    optimized_trials = np.array(optimized_results['times'])
    baseline_trials = np.array(baseline_results['times'])
    win_count = np.sum(optimized_trials < baseline_trials)
    win_prob = win_count / n_trials
    p_value = scipy.stats.binomtest(win_count, n_trials, 0.5).pvalue
    
    print(f"baseline time: {baseline_results['time']} nanoseconds")
    print(f"optimized time: {optimized_results['time']} nanoseconds")
    print(f"Speedup: {baseline_results['time'] / optimized_results['time']:.2f}x")
    print(f"Measured win frequency: {win_prob:.2%}")
    print(f"P-value: {p_value}")

import json
import os
import numpy as np

def run_benchmark(kernel, input):
    os.mkdir("input")
    np.save('input/A.npy', input)
    os.mkdir("output")
    cmd = f'./{kernel} --input input/A.npy --output output/B.npy'
    os.system(cmd)
    output = np.load('output/B.npy')
    measurements = json.load(open('output/measurements.json'))
    measurements['output'] = output
    return measurements

if __name__ == "__main__":
    m, n = 512, 512
    input = np.random.rand(m, n).astype(np.float64)

    print("Running C benchmark...")
    baseline_results = run_benchmark('conv', input)
    print(f"C execution time: {baseline_results['time']} seconds")

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <getopt.h>
#include <iostream>
#include <string>
#include <vector>
#include "npy.hpp"
#include "json.hpp"
#include <fstream>
#include <cmath>

using json = nlohmann::json;

#define TIME_MAX 5.0
#define TRIAL_MAX 10000

template <typename Setup, typename Run>
long long benchmark(Setup setup, Run run, double time_max = TIME_MAX, int max_trials = TRIAL_MAX){
  auto time_total = std::chrono::high_resolution_clock::duration(0);
  auto time_min = std::chrono::high_resolution_clock::duration(0);
  // Initial run to avoid measuring setup overhead
  setup();
  run();
  int trial = 0;
  
  while(trial < max_trials){
    setup();
    auto tic = std::chrono::high_resolution_clock::now();
    run();
    auto toc = std::chrono::high_resolution_clock::now();
    if(toc < tic){
      exit(EXIT_FAILURE);
    }
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(toc-tic);
    trial++;
    if(trial == 1 || time < time_min){
      time_min = time;
    }
    time_total += time;
    if(time_total.count() * 1e-9 > time_max){
      break;
    }
  }
  return (long long) time_min.count();
}

// Forward declaration of the core C convolution function
extern "C" {
    void conv_kernel(const double* A, double* B, int m, int n);
}

// Main function - benchmark harness with inline argument parsing
int main(int argc, char **argv){
    // Define the long options
    static struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"input", required_argument, 0, 'i'},
        {"output", required_argument, 0, 'o'},
        {"time-max", required_argument, 0, 't'},
        {"trial-max", required_argument, 0, 'r'},
        {"verbose", no_argument, 0, 'v'},
        {0, 0, 0, 0}
    };

    // Parse the options
    int option_index = 0;
    int c;
    std::string input, output;
    bool verbose = false;
    double time_max = TIME_MAX;
    int max_trials = TRIAL_MAX;
    
    while ((c = getopt_long(argc, argv, "hi:o:v:t:r:", long_options, &option_index)) != -1) {
        switch (c) {
            case 'h':
                std::cout << "Options:" << std::endl;
                std::cout << "  -h, --help      Print this help message" << std::endl;
                std::cout << "  -i, --input     Specify the path for the inputs" << std::endl;
                std::cout << "  -o, --output    Specify the path for the outputs" << std::endl;
                std::cout << "  -v, --verbose   Print verbose output" << std::endl;
                std::cout << "  -t, --time-max  Maximum total time for benchmarking (seconds)" << std::endl;
                std::cout << "  -r, --trial-max Maximum number of trials for benchmarking" << std::endl;
                std::cout << "  --              Kernel-specific arguments" << std::endl;
                exit(0);
            case 'i':
                input = optarg;
                break;
            case 'o':
                output = optarg;
                break;
            case 'v':
                verbose = true;
                break;
            case 't':
                time_max = std::stod(optarg);
                break;
            case 'r':
                max_trials = std::stoi(optarg);
                break;
            case '?':
                break;
            default:
                abort();
        }
    }

    // Check that all required options are present
    if (input.empty() || output.empty()) {
        std::cerr << "Missing required option" << std::endl;
        exit(1);
    }

    // Print verbose output if requested
    if (verbose) {
        std::cout << "Input path: " << input << std::endl;
        std::cout << "Output path: " << output << std::endl;
        std::cout << "Max time: " << time_max << " seconds" << std::endl;
        std::cout << "Max trials: " << max_trials << std::endl;
    }

    // Load the input matrix A - inline npy load
    std::vector<double> A;
    std::vector<unsigned long> input_shape;
    bool input_fortran_order;
    npy::LoadArrayFromNumpy<double>(input + "/A.npy", input_shape, input_fortran_order, A);
    
    // Get dimensions from loaded shape
    int m = static_cast<int>(input_shape[0]);
    int n = static_cast<int>(input_shape[1]);
    
    if (verbose) {
        std::cout << "Matrix dimensions: " << m << "x" << n << std::endl;
    }
    
    // Create output vector B
    std::vector<double> B(m * n, 0.0);

    // Benchmark the convolution
    auto time = benchmark(
    []() {
        // Setup function - nothing to do here
    },
        [&A, &B, &m, &n]() {
            // Call the core C convolution function with array data
            conv_kernel(A.data(), B.data(), m, n);
        }
      , time_max, max_trials
    );

    // Save results as 2D array - inline npy store
    std::vector<unsigned long> output_shape = {static_cast<unsigned long>(m), static_cast<unsigned long>(n)};
    npy::SaveArrayAsNumpy(output + "/B.npy", false, output_shape.size(), output_shape.data(), B);

    json measurements;
    measurements["time"] = time;
    std::ofstream measurements_file(output + "/measurements.json");
    measurements_file << measurements;
    measurements_file.close();
    
    if (verbose) {
        std::cout << "Convolution completed. Time: " << time << " ns" << std::endl;
    }

    return 0;
}
#include "benchmark.hpp"
#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

// Forward declaration of the core C convolution function
extern "C" {
    void experiment_conv_impl(const double* A, double* B, int m, int n);
}

// Move npy functions implementation here
template <typename T>
std::vector<T> npy_load_vector(std::string fname) {
  std::vector<T> vec;
  std::vector<unsigned long> shape;
  bool fortran_order;
  npy::LoadArrayFromNumpy<T>(fname, shape, fortran_order, vec);
  return vec;
}

template <typename T>
void npy_store_vector(std::string fname, std::vector<T> vec, bool fortran_order) {
    std::vector<unsigned long> shape = {static_cast<unsigned long>(vec.size())};
    npy::SaveArrayAsNumpy(fname, fortran_order, shape.size(), shape.data(), vec);
}

// Explicit template instantiations
template std::vector<double> npy_load_vector<double>(std::string fname);
template void npy_store_vector<double>(std::string fname, std::vector<double> vec, bool fortran_order);

benchmark_params_t parse(int argc, char **argv) {
  // Define the long options
  static struct option long_options[] = {
    {"help", no_argument, 0, 'h'},
    {"input", required_argument, 0, 'i'},
    {"output", required_argument, 0, 'o'},
    {"verbose", no_argument, 0, 'v'},
    {0, 0, 0, 0}
  };

  // Parse the options
  int option_index = 0;
  int c;
  benchmark_params_t params;
  params.verbose = false;
  while ((c = getopt_long(argc, argv, "hi:o:v", long_options, &option_index)) != -1) {
    switch (c) {
      case 'h':
        std::cout << "Options:" << std::endl;
        std::cout << "  -h, --help      Print this help message" << std::endl;
        std::cout << "  -i, --input     Specify the path for the inputs" << std::endl;
        std::cout << "  -o, --output    Specify the path for the outputs" << std::endl;
        std::cout << "  -v, --verbose   Print verbose output" << std::endl;
        std::cout << "  --              Kernel-specific arguments" << std::endl;
        exit(0);
      case 'i':
        params.input = optarg;
        break;
      case 'o':
        params.output = optarg;
        break;
      case 'v':
        params.verbose = true;
        break;
      case '?':
        break;
      default:
        abort();
    }
  }

  // Check that all required options are present
  if (params.input.empty() || params.output.empty()) {
    std::cerr << "Missing required option" << std::endl;
    exit(1);
  }

  // Print verbose output if requested
  if (params.verbose) {
    std::cout << "Input path: " << params.input << std::endl;
    std::cout << "Output path: " << params.output << std::endl;
  }

  // Store the remaining command-line arguments
  params.argc = argc - optind + 1;
  params.argv = argv + optind - 1;

  return params;
}

// Implementation of the C wrapper that handles I/O and benchmarking
extern "C" {
    void experiment_conv_c(const char* input_path, const char* output_path, bool verbose) {
        // Load the input matrix A
        auto A = npy_load_vector<double>(fs::path(input_path)/"A.npy");
        
        // Assume A is a square matrix, calculate dimensions from vector size
        int total_size = A.size();
        int m = static_cast<int>(std::sqrt(total_size));
        int n = m; // Assuming square matrix
        
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
                experiment_conv_impl(A.data(), B.data(), m, n);
            }
        );

        // Save results
        npy_store_vector<double>(fs::path(output_path)/"B.npy", B, false);

        json measurements;
        measurements["time"] = time;
        std::ofstream measurements_file(fs::path(output_path)/"measurements.json");
        measurements_file << measurements;
        measurements_file.close();
        
        if (verbose) {
            std::cout << "Convolution completed. Time: " << time << " ns" << std::endl;
        }
    }
}

// Main function - benchmark harness
int main(int argc, char **argv){
    auto params = parse(argc, argv);
    experiment_conv_c(params.input.c_str(), params.output.c_str(), params.verbose);
    return 0;
}
#include "./benchmark.hpp"
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

void experiment_conv(benchmark_params_t params);

int main(int argc, char **argv){
    auto params = parse(argc, argv);
    experiment_conv(params);
    return 0;
}

void experiment_conv(benchmark_params_t params){
    // Load the input matrix A
    auto A = npy_load_vector<double>(fs::path(params.input)/"A.npy");
    
    // Assume A is a square matrix, calculate dimensions from vector size
    // For a proper implementation, you'd need to load the shape from the .npy file
    int total_size = A.size();
    int m = static_cast<int>(std::sqrt(total_size));
    int n = m; // Assuming square matrix
    
    // Create output vector B
    std::vector<double> B(m * n, 0.0);

    //perform convolution with 3x3 kernel

    auto time = benchmark(
    []() {
        // Setup function - nothing to do here
    },
        [&A, &B, &m, &n]() {
            // Clear B for each benchmark run
            std::fill(B.begin(), B.end(), 0.0);
            
            for(int x = 0; x < m; x++){
                for (int y = 0; y < n; y++){
                    for (int dx = -1; dx <= 1; dx++){
                        for (int dy = -1; dy <= 1; dy++){
                            if(x + dx >= 0 && x + dx < m && y + dy >= 0 && y + dy < n){
                                B[x*n + y] += A[(x+dx)*n + (y+dy)];
                            }
                        }
                    }
                    
                }
            }
        }
    );

    npy_store_vector<double>(fs::path(params.output)/"B.npy", B);

    json measurements;
    measurements["time"] = time;
    std::ofstream measurements_file(fs::path(params.output)/"measurements.json");
    measurements_file << measurements;
    measurements_file.close();
}
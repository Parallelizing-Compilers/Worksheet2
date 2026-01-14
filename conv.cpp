#include "benchmark.hpp"
#include <cmath>
#include <algorithm>

// C++ implementation function - simplified to just core computation
void experiment_conv_impl(const std::vector<double>& A, std::vector<double>& B, int m, int n) {
    // Clear B for the computation
    std::fill(B.begin(), B.end(), 0.0);
    
    // Perform convolution with 3x3 kernel
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
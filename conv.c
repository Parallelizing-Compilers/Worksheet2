#include <string.h>

// C implementation function - pure C with arrays
void experiment_conv_impl(const double* A, double* B, int m, int n) {
    // Clear B for the computation
    memset(B, 0, m * n * sizeof(double));
    
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
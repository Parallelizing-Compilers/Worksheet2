#include <string.h>
#include <stdlib.h>

void conv_kernel(const double* A, double* B, int m, int n) {
    // Clear B for the computation
    memset(B, 0, m * n * sizeof(double));

    double *tmp = malloc(sizeof(double) * m * n);
    memset(tmp, 0, m * n * sizeof(double));
    
    // Perform convolution with 3x3 kernel
    for(int x = 0; x < m; x++){
        for (int y = 0; y < n; y++){
            for (int dx = -1; dx <= 1; dx++){
                if(x + dx >= 0 && x + dx < m){
                    tmp[x*n + y] += A[(x+dx)*n + y];
                }
            }
        }
    }

    for (int x = 0; x < m; x++){
        for (int y = 0; y < n; y++){
            for (int dy = -1; dy <= 1; dy++){
                if(y + dy >= 0 && y + dy < n){
                    B[x*n + y] += tmp[x*n + (y+dy)];
                }
            }
        }
    }

    free(tmp);
}
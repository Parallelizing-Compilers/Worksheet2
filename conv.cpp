#include "./benchmark.hpp"
#include <sys/stat.h>
#include <iostream>
#include <cstdint>

namespace fs = std::filesystem;

template <typename T, typename I>
void experiment_conv(benchmark_params_t params);

int main(int argc, char **argv){
    auto params = parse(argc, argv);
    auto A_desc = json::parse(std::ifstream(fs::path(params.input)/"A.bspnpy"/"binsparse.json"))["binsparse"]; 
    auto x_desc = json::parse(std::ifstream(fs::path(params.input)/"x.bspnpy"/"binsparse.json"))["binsparse"]; 

    //print format
    if (A_desc["format"] != "CSR") {throw std::runtime_error("Only CSR format for A is supported");}
    if (x_desc["format"] != "DVEC") {throw std::runtime_error("Only dense format for x is supported");}
    if (A_desc["data_types"]["pointers_to_1"] == "int32" &&
        A_desc["data_types"]["values"] == "float64") {
            experiment_spmv_csr<double, int32_t>(params);
    } else if (A_desc["data_types"]["pointers_to_1"] == "int64" &&
        A_desc["data_types"]["values"] == "float64") {
            experiment_spmv_csr<double, int64_t>(params);
    } else {
        std::cout << "pointers_to_1_type: " << A_desc["data_types"]["pointers_to_1"] << std::endl;
        std::cout << "values_type: " << A_desc["data_types"]["values"] << std::endl;
        throw std::runtime_error("Unsupported data types");
    }

    return 0;
}

void experiment_conv(benchmark_params_t params){
    int m = A_desc["shape"][0];
    int n = A_desc["shape"][1];

    auto A = npy_load_vector<double>(fs::path(params.input)/"A.npy");

    B = malloc(m * n * sizeof(double));

    //perform an spmv of the matrix in c++

    auto time = benchmark(
    []() {
    },
        [&A_val, &m, &n]() {
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

    npy_store_vector<T>(fs::path(params.output)/"B.bspnpy", y_val);

    json measurements;
    measurements["time"] = time;
    std::ofstream measurements_file(fs::path(params.output)/"measurements.json");
    measurements_file << measurements;
    measurements_file.close();
}
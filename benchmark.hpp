#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <getopt.h>
#include <iostream>
#include <string>
#include <vector>
#include "npy.hpp"
#include "json.hpp"

using json = nlohmann::json;

#define TIME_MAX 5.0
#define TRIAL_MAX 10000

/*
  Benchmark a function `run` by running it multiple times and measuring the
  time. The function `setup` is called before each run to prepare the input
  data. The function returns the minimum time in nanoseconds of all the runs.
  Runs at most `TRIAL_MAX` times or until the total time exceeds `TIME_MAX`.
*/
template <typename Setup, typename Run>
long long benchmark(Setup setup, Run run){
  auto time_total = std::chrono::high_resolution_clock::duration(0);
  auto time_min = std::chrono::high_resolution_clock::duration(0);
  int trial = 0;
  while(trial < TRIAL_MAX){
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
    if(time_total.count() * 1e-9 > TIME_MAX){
      break;
    }
  }
  return (long long) time_min.count();
}

// Function declarations
template <typename T>
std::vector<T> npy_load_vector(std::string fname);

template <typename T>
void npy_store_vector(std::string fname, std::vector<T> vec, bool fortran_order = false);

void experiment(std::string input, std::string output, int verbose);

// Forward declaration for C convolution function
extern "C" {
    void experiment_conv_impl(const double* A, double* B, int m, int n);
}

struct benchmark_params_t {
  std::string input;
  std::string output;
  bool verbose;
  int argc;
  char **argv;
};

benchmark_params_t parse(int argc, char **argv);
CC = gcc
CXX = g++
CXXFLAGS += -std=c++20
CFLAGS += 
LDLIBS +=

all: conv_baseline conv_optimized

clean:
	rm -rf conv_baseline conv_optimized
	rm -rf *.o *.dSYM *.trace

conv_baseline.o: conv_baseline.c
	$(CC) $(CFLAGS) -c $< -o $@

conv_optimized.o: conv_optimized.c
	$(CC) $(CFLAGS) -c $< -o $@

benchmark.o: benchmark.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

conv_baseline: benchmark.o conv_baseline.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

conv_optimized: benchmark.o conv_optimized.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)
CC = gcc
CXX = g++
CXXFLAGS += -std=c++20
CFLAGS += 
LDLIBS +=

all: conv

clean:
	rm -rf conv
	rm -rf *.o *.dSYM *.trace

conv.o: conv.c
	$(CC) $(CFLAGS) -c $< -o $@

benchmark.o: benchmark.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

conv: benchmark.o conv.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)
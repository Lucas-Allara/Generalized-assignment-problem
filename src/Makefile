CXX ?= g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -MMD -MP -Iinclude -isystem $(SCIPDIR)/include -isystem $(SCIPDIR)/include/highs

# SCIP and HiGHS (only for 'make exact'): by default, the environment created by 'pixi install'.
SCIPDIR ?= ../.pixi/envs/default
SCIPLIBS = -L$(SCIPDIR)/lib -Wl,-rpath,$(abspath $(SCIPDIR)/lib) -lscip -lhighs

CORE = Instance.o ConstructiveHeuristics.o LocalSearch.o Metaheuristics.o Knapsack.o LagrangianBound.o VariableFixing.o EjectionChains.o Bundle.o
EXACT = MipModel.o Matheuristic.o BranchAndPrice.o ColumnGeneration.o

# heuristics, without SCIP
all: gap_simulator tests

gap_simulator: main.o $(CORE)
	$(CXX) $^ -o $@

tests: tests.o $(CORE)
	$(CXX) $^ -o $@

# exact methods, LP-based bounds and matheuristic, with SCIP and HiGHS
exact: gap_exact exact_tests

gap_exact: exact_main.o $(CORE) $(EXACT)
	$(CXX) $^ $(SCIPLIBS) -o $@

exact_tests: exact_tests.o $(CORE) $(EXACT)
	$(CXX) $^ $(SCIPLIBS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o *.d gap_simulator tests gap_exact exact_tests

.PHONY: all exact clean

-include $(wildcard *.d)

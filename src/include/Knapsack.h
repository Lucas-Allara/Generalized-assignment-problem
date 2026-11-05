#ifndef KNAPSACK_H
#define KNAPSACK_H

#include <vector>

namespace knapsack {

// 0-1 knapsack by dynamic programming over the capacity: chooses the items that
// maximize the total profit with total weight at most 'capacity'. Only items with
// positive profit and weight at most the capacity are considered. Returns the best
// profit and leaves the chosen items (indices into 'profit') in 'chosen'.
// Time O(k * capacity), where k is the number of items considered.
double solve(const std::vector<double>& profit, const std::vector<int>& weight,
             int capacity, std::vector<int>& chosen);

}

#endif

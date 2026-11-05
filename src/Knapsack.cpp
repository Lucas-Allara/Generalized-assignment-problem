#include "Knapsack.h"

using namespace std;

namespace knapsack {

double solve(const vector<double>& profit, const vector<int>& weight, int capacity,
             vector<int>& chosen) {
    chosen.clear();
    if (capacity < 0) return 0;
    // Buffers reused between calls (one per thread): the knapsack is solved once per
    // warehouse at every iteration of the subgradient method and of the pricing.
    thread_local vector<int> items;
    thread_local vector<double> best;
    thread_local vector<char> take;
    items.clear();
    for (int k = 0; k < (int)profit.size(); k++) {
        if (profit[k] > 0 && weight[k] <= capacity) items.push_back(k);
    }
    // best[c]: best profit with total weight at most c using the items seen so far;
    // take[t * (capacity + 1) + c]: whether item t is taken in the optimum for capacity c.
    int width = capacity + 1;
    best.assign(width, 0.0);
    take.assign(items.size() * width, 0);
    for (int t = 0; t < (int)items.size(); t++) {
        int w = weight[items[t]];
        double p = profit[items[t]];
        char* row = take.data() + (size_t)t * width;
        for (int c = capacity; c >= w; c--) {
            if (best[c - w] + p > best[c]) {
                best[c] = best[c - w] + p;
                row[c] = 1;
            }
        }
    }
    int c = capacity;
    for (int t = (int)items.size() - 1; t >= 0; t--) {
        if (take[(size_t)t * width + c]) {
            chosen.push_back(items[t]);
            c -= weight[items[t]];
        }
    }
    return best[capacity];
}

}

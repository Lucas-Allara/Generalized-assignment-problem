#include "VariableFixing.h"
#include "ColumnGeneration.h"
#include "LagrangianBound.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

using namespace std;

namespace fixing {

namespace {

const double INF = numeric_limits<double>::infinity();

double seconds_since(chrono::high_resolution_clock::time_point t0) {
    return chrono::duration_cast<chrono::milliseconds>(
        chrono::high_resolution_clock::now() - t0).count() / 1000.0;
}

// Builds the reduced instance of the current fixings of 'r': 'r.allowed[j][i]' says
// whether seller j may still go to warehouse i and 'r.forced[j]' is its forced
// warehouse (-1: free).
void build_reduced(const Instance& inst, Reduction& r) {
    int n = inst.sellers.size();
    int m = inst.warehouses.size();
    vector<int> capacity;
    for (const Warehouse& w : inst.warehouses) capacity.push_back(w.capacity);
    r.offset = 0;
    for (int j = 0; j < n; j++) {
        if (r.forced[j] < 0) continue;
        capacity[r.forced[j]] -= inst.demand(j, r.forced[j]);
        r.offset += inst.cost[j][r.forced[j]];
    }
    for (int i = 0; i < m; i++) {
        if (capacity[i] < 0) r.no_improving_solution = true;   // the forced sellers do not fit
    }
    r.instance = Instance();
    r.instance.penalty = inst.penalty;
    for (int i = 0; i < m; i++) r.instance.warehouses.push_back(Warehouse(i, max(0, capacity[i])));
    r.original.clear();
    for (int j = 0; j < n; j++) {
        if (r.forced[j] >= 0) continue;
        vector<int> demand(m);
        for (int i = 0; i < m; i++) demand[i] = r.allowed[j][i] ? inst.demand(j, i) : max(0, capacity[i]) + 1;
        int k = r.original.size();
        r.instance.sellers.push_back(Seller(k, demand, inst.cost[j]));
        r.instance.cost.push_back(inst.cost[j]);
        r.original.push_back(j);
    }
    r.fixed_to_one = 0;
    for (int j = 0; j < n; j++) r.fixed_to_one += r.forced[j] >= 0;
}

// Smallest cost that a solution must reach to be discarded: with integral costs,
// a cheaper solution is cheaper by at least 1.
double threshold(bool integral, double upper_bound) {
    return integral ? upper_bound - 1 + 1e-6 : upper_bound + 1e-9 * max(1.0, fabs(upper_bound));
}

// Fixes the pairs of the free sellers whose stored penalties exceed the
// threshold; returns the number of new fixings.
int fix_pairs(Reduction& r, double limit) {
    int n = r.allowed.size();
    int m = n > 0 ? r.allowed[0].size() : 0;
    vector<int> free_sellers;
    for (int j = 0; j < n; j++) {
        if (r.forced[j] < 0) free_sellers.push_back(j);
    }
    int changes = 0;
    for (int j : free_sellers) {
        for (int i = 0; i < m; i++) {
            if (!r.allowed[j][i]) continue;
            bool not_one = r.with[j][i] > limit;
            bool not_zero = r.without[j][i] > limit;
            if (not_one && not_zero) {
                r.no_improving_solution = true;   // neither value of x_ij is possible
            } else if (not_one) {
                r.allowed[j][i] = 0;
                r.fixed_to_zero++;
                changes++;
            } else if (not_zero && r.forced[j] != i) {
                if (r.forced[j] >= 0) r.no_improving_solution = true;   // forced into two warehouses
                r.forced[j] = i;
                changes++;
            }
        }
    }
    return changes;
}

}

bool integral_costs(const Instance& inst) {
    for (const auto& row : inst.cost)
        for (double c : row)
            if (c != floor(c)) return false;
    return inst.penalty == floor(inst.penalty);
}

double penalties(const Instance& inst, const vector<double>& lambda,
                 vector<vector<double>>& with, vector<vector<double>>& without) {
    int n = inst.sellers.size();
    int m = inst.warehouses.size();
    double value = 0;
    vector<double> u_term(n);
    for (int j = 0; j < n; j++) {
        u_term[j] = min(0.0, inst.penalty - lambda[j]);
        value += lambda[j] + u_term[j];
    }
    // knapsack values: best[i], and with item j forced in or out of knapsack i
    vector<double> best(m);
    vector<vector<double>> forced_in(m, vector<double>(n, -INF)), forced_out(m, vector<double>(n, 0.0));
    vector<double> F, G;
    for (int i = 0; i < m; i++) {
        int b = inst.warehouses[i].capacity;
        int width = b + 1;
        vector<int> items;
        for (int j = 0; j < n; j++) {
            if (inst.demand(j, i) <= b && lambda[j] - inst.cost[j][i] > 0) items.push_back(j);
        }
        int k = items.size();
        // F[t * width + c]: best profit of the first t items with weight at most c;
        // G[t * width + c]: best profit of the items t..k-1 with weight at most c.
        F.assign((size_t)(k + 1) * width, 0.0);
        G.assign((size_t)(k + 1) * width, 0.0);
        for (int t = 0; t < k; t++) {
            int w = inst.demand(items[t], i);
            double p = lambda[items[t]] - inst.cost[items[t]][i];
            const double* prev = &F[(size_t)t * width];
            double* next = &F[(size_t)(t + 1) * width];
            for (int c = 0; c <= b; c++) next[c] = (c >= w) ? max(prev[c], prev[c - w] + p) : prev[c];
        }
        for (int t = k - 1; t >= 0; t--) {
            int w = inst.demand(items[t], i);
            double p = lambda[items[t]] - inst.cost[items[t]][i];
            const double* prev = &G[(size_t)(t + 1) * width];
            double* next = &G[(size_t)t * width];
            for (int c = 0; c <= b; c++) next[c] = (c >= w) ? max(prev[c], prev[c - w] + p) : prev[c];
        }
        best[i] = F[(size_t)k * width + b];
        vector<bool> is_item(n, false);
        for (int t = 0; t < k; t++) {
            int j = items[t];
            is_item[j] = true;
            int w = inst.demand(j, i);
            double p = lambda[j] - inst.cost[j][i];
            const double* before = &F[(size_t)t * width];
            const double* after = &G[(size_t)(t + 1) * width];
            double out = -INF, in = -INF;
            for (int c = 0; c <= b; c++) out = max(out, before[c] + after[b - c]);
            for (int c = 0; c <= b - w; c++) in = max(in, before[c] + after[b - w - c]);
            forced_out[i][j] = out;
            forced_in[i][j] = p + in;
        }
        for (int j = 0; j < n; j++) {
            if (is_item[j]) continue;
            forced_out[i][j] = best[i];   // j is not in any optimal solution
            int w = inst.demand(j, i);
            if (w <= b) forced_in[i][j] = lambda[j] - inst.cost[j][i] + F[(size_t)k * width + (b - w)];
        }
        value -= best[i];
    }
    // loss of the knapsacks when j leaves all of them
    vector<double> leave(n, 0.0);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) leave[j] += best[i] - forced_out[i][j];
    with.assign(m, vector<double>(n, INF));
    without.assign(m, vector<double>(n, INF));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double loss_out = best[i] - forced_out[i][j];
            without[i][j] = value + loss_out;
            if (forced_in[i][j] == -INF) continue;   // j does not fit in i
            with[i][j] = value + (best[i] - forced_in[i][j]) + (leave[j] - loss_out) - u_term[j];
        }
    }
    return value;
}

Reduction reduce(const Instance& inst, double upper_bound, double seconds, int passes, Solution* best,
                 bool lagrangian_heuristic, const Reduction* from, bool bundle) {
    auto t0 = chrono::high_resolution_clock::now();
    int n = inst.sellers.size();
    int m = inst.warehouses.size();
    bool integral = integral_costs(inst);
    double ub = upper_bound;
    Reduction r;
    if (from != nullptr) {
        r = *from;
        r.upper_bound = min(r.upper_bound, ub);
        if (!r.with.empty()) fix_pairs(r, threshold(integral, ub));
    } else {
        r.allowed.assign(n, vector<char>(m, 0));
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                r.allowed[j][i] = inst.demand(j, i) <= inst.warehouses[i].capacity;
                r.pairs += r.allowed[j][i];
            }
        }
        r.forced.assign(n, -1);
        r.with.assign(n, vector<double>(m, -INF));
        r.without.assign(n, vector<double>(m, -INF));
    }

    for (int pass = 0; pass < passes; pass++) {
        build_reduced(inst, r);
        if (r.no_improving_solution || r.instance.sellers.empty()) break;
        double left = seconds > 0 ? seconds - seconds_since(t0) : 0;
        if (seconds > 0 && left <= 0) break;
        // the subgradient method gets an equal share of the time left for each pass
        double share = seconds > 0 ? max(1e-3, left / (passes - pass)) : 0;
        lagrangian::Result lr = lagrangian::subgradient(r.instance, ub - r.offset, 5000, share, 50,
                                                        nullptr, 100, lagrangian_heuristic ? 10 : 0);
        r.passes++;
        if (lr.has_solution) {
            Solution candidate = expand(inst, r, lr.solution);
            if (candidate.total_cost < ub - 1e-9) {
                ub = candidate.total_cost;
                if (best != nullptr) *best = candidate;
            }
        }
        if (lr.multipliers.empty()) break;
        if (bundle) {
            double left_now = seconds > 0 ? max(1e-3, seconds - seconds_since(t0)) : 0;
            column_generation::Result b = column_generation::bundle(r.instance, seconds > 0 ? min(share, left_now) : 0,
                                                                    {}, &lr.multipliers);
            if (!b.pi.empty() && b.bound > lr.bound) {
                lr.multipliers = b.pi;
                lr.bound = b.bound;
            }
        }
        r.bound = max(r.bound, lr.bound + r.offset);

        // A pair is fixed when its bound shows that no solution with that choice is
        // cheaper than the upper bound (with integral costs, by at least 1).
        vector<vector<double>> with, without;
        double value = penalties(r.instance, lr.multipliers, with, without) + r.offset;
        r.bound = max(r.bound, value);
        for (int k = 0; k < (int)r.original.size(); k++) {
            int j = r.original[k];
            for (int i = 0; i < m; i++) {
                r.with[j][i] = with[i][k] + r.offset;
                r.without[j][i] = without[i][k] + r.offset;
            }
        }
        int changes = fix_pairs(r, threshold(integral, ub));
        if (r.no_improving_solution || changes == 0) break;
    }
    build_reduced(inst, r);
    r.upper_bound = min(r.upper_bound, ub);
    r.seconds += seconds_since(t0);
    return r;
}

int refix(const Instance& inst, Reduction& r, double upper_bound) {
    auto t0 = chrono::high_resolution_clock::now();
    if (r.with.empty() || upper_bound >= r.upper_bound) return 0;
    r.upper_bound = upper_bound;
    int changes = fix_pairs(r, threshold(integral_costs(inst), upper_bound));
    if (changes > 0) build_reduced(inst, r);
    r.seconds += seconds_since(t0);
    return changes;
}

long long Refixer::ruled_out_pairs() const {
    long long count = 0;
    if (!started_) return 0;
    int m = inst_.warehouses.size();
    for (int j = 0; j < (int)r_.allowed.size(); j++) {
        for (int i = 0; i < m; i++) {
            if (inst_.demand(j, i) <= inst_.warehouses[i].capacity) count += ruled_out(j, i);
        }
    }
    return count;
}

bool Refixer::update(double upper_bound, Solution& best) {
    best = Solution();
    best.total_cost = 1e300;
    if (upper_bound >= tested_ - 1e-9 || r_.no_improving_solution) return false;
    tested_ = upper_bound;
    long long before = r_.fixed_to_zero + r_.fixed_to_one;
    double left = budget_ - r_.seconds;
    if (!started_) {
        started_ = true;
        r_ = reduce(inst_, upper_bound, max(1e-3, left), 3, &best, true, nullptr, bundle_);
        first_fixed_pct = r_.pairs > 0 ? 100.0 * r_.fixed_to_zero / r_.pairs : 0;
        last_run_ub_ = r_.upper_bound;
        version_++;
        tested_ = min(tested_, r_.upper_bound);
        return true;
    }
    incumbents++;
    refix(inst_, r_, upper_bound);
    double gap = last_run_ub_ - r_.bound;
    if (left > 0.5 && !r_.no_improving_solution && last_run_ub_ - upper_bound >= 0.1 * gap) {
        r_ = reduce(inst_, upper_bound, 0.5 * left, 2, &best, true, &r_, bundle_);
        last_run_ub_ = r_.upper_bound;
    }
    tested_ = min(tested_, r_.upper_bound);
    bool changed = r_.fixed_to_zero + r_.fixed_to_one != before || r_.no_improving_solution;
    if (changed) version_++;
    return changed;
}

Solution expand(const Instance& inst, const Reduction& reduction, const Solution& reduced) {
    int n = inst.sellers.size();
    Solution sol;
    sol.sellers_per_warehouse.assign(inst.warehouses.size(), {});
    sol.assigned_warehouse.assign(n, -1);
    for (int j = 0; j < n; j++) {
        if (j < (int)reduction.forced.size() && reduction.forced[j] >= 0) sol.assign(reduction.forced[j], j);
    }
    for (int k = 0; k < (int)reduction.original.size(); k++) {
        int i = k < (int)reduced.assigned_warehouse.size() ? reduced.assigned_warehouse[k] : -1;
        if (i >= 0) sol.assign(i, reduction.original[k]);
    }
    for (int j = 0; j < n; j++) {
        int i = sol.assigned_warehouse[j];
        sol.total_cost += i >= 0 ? inst.cost[j][i] : inst.penalty;
    }
    return sol;
}

}

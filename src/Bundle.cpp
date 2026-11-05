#include "ColumnGeneration.h"
#include "Knapsack.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>

// The bundle method needs no LP solver, so it lives apart from the column
// generation with HiGHS (it is part of the core library, used by the fixing).

using namespace std;

namespace column_generation {

namespace {

// A column of the bundle: sellers (sorted), cost, weight in the quadratic program
// and the last iteration in which its weight was positive.
struct BundleColumn {
    vector<int> sellers;
    double cost;
    double weight;
    int last_used;
};

}

Result bundle(const Instance& inst, double time_limit, const vector<pair<int, vector<int>>>& initial,
              const vector<double>* start) {
    auto t0 = chrono::high_resolution_clock::now();
    auto elapsed = [&] {
        return chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now() - t0).count() / 1000.0;
    };
    int n = inst.sellers.size();
    int m = inst.warehouses.size();
    double p = inst.penalty;
    Result result;

    // blocks: the columns of each warehouse; the empty column (always feasible)
    // starts with all the weight
    vector<vector<BundleColumn>> blocks(m);
    set<pair<int, vector<int>>> known;
    for (int i = 0; i < m; i++) blocks[i].push_back({{}, 0.0, 1.0, 0});
    auto add_column = [&](int i, vector<int> sellers, int iteration) {
        sort(sellers.begin(), sellers.end());
        if (sellers.empty() || !known.insert({i, sellers}).second) return;
        double cost = 0;
        for (int j : sellers) cost += inst.cost[j][i];
        result.columns.push_back({i, sellers});
        blocks[i].push_back({sellers, cost, 0.0, iteration});
    };
    for (const auto& [i, sellers] : initial) add_column(i, sellers, 0);

    // exact Lagrangian function: value, and the best knapsack of each warehouse
    vector<double> profit(n);
    vector<int> weight(n), chosen;
    vector<vector<int>> sets(m);
    vector<double> kp(m);
    auto evaluate = [&](const vector<double>& pi) {
        double value = 0;
        for (int j = 0; j < n; j++) value += min(pi[j], p);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                profit[j] = pi[j] - inst.cost[j][i];
                weight[j] = inst.demand(j, i);
            }
            kp[i] = knapsack::solve(profit, weight, inst.warehouses[i].capacity, chosen);
            sets[i] = chosen;
            value -= kp[i];
        }
        return value;
    };
    // value of the model (the bundle) at pi
    auto model = [&](const vector<double>& pi) {
        double value = 0;
        for (int j = 0; j < n; j++) value += min(pi[j], p);
        for (int i = 0; i < m; i++) {
            double best = 0;
            for (const BundleColumn& col : blocks[i]) {
                double v = -col.cost;
                for (int j : col.sellers) v += pi[j];
                best = max(best, v);
            }
            value -= best;
        }
        return value;
    };

    // initial center: the given point, or the smallest cost of each seller
    vector<double> center(n);
    if (start != nullptr && (int)start->size() == n) {
        center = *start;
    } else {
        for (int j = 0; j < n; j++) {
            double c = p;
            for (int i = 0; i < m; i++) c = min(c, inst.cost[j][i]);
            center[j] = c;
        }
    }
    double center_value = evaluate(center);
    for (int i = 0; i < m; i++) add_column(i, sets[i], 0);
    result.bound = center_value;
    result.pi = center;
    double t = 1.0;
    vector<double> u(n, 0.0), g(n, 1.0);   // g = 1 - u - A lambda (the empty columns cover nothing)
    vector<double> candidate(n);

    for (int it = 1;; it++) {
        if (time_limit > 0 && elapsed() >= time_limit) break;
        double before = elapsed();
        // quadratic program by block coordinate descent, until the optimality
        // conditions hold: in each warehouse, no column has a gradient below that
        // of a column with positive weight (by more than a tolerance)
        double tolerance = 1e-9 * max(1.0, fabs(center_value) / n);
        for (int sweep = 0; sweep < 2000; sweep++) {
            double largest = 0;   // largest violation of the optimality conditions
            for (int j = 0; j < n; j++) {
                double r = g[j] + u[j];
                double nu = min(1.0, max(0.0, r - (p - center[j]) / t));
                g[j] = r - nu;
                largest = max(largest, t * fabs(nu - u[j]));
                u[j] = nu;
            }
            for (int i = 0; i < m; i++) {
                vector<BundleColumn>& block = blocks[i];
                for (int step = 0; step < 100; step++) {
                    // gradient of each column: c(S) - center(S) - t g(S)
                    int a = -1, b = -1;
                    double ga = -1e300, gb = 1e300;
                    vector<double> grad(block.size());
                    for (int k = 0; k < (int)block.size(); k++) {
                        double v = block[k].cost;
                        for (int j : block[k].sellers) v -= center[j] + t * g[j];
                        grad[k] = v;
                        if (v < gb) { gb = v; b = k; }
                        if (block[k].weight > 1e-12 && v > ga) { ga = v; a = k; }
                    }
                    if (a < 0 || b < 0 || a == b) break;
                    if (step == 0) largest = max(largest, ga - gb);
                    if (ga - gb <= tolerance) break;
                    // |e_b - e_a|^2 = size of the symmetric difference
                    const vector<int>& sa = block[a].sellers;
                    const vector<int>& sb = block[b].sellers;
                    int common = 0;
                    for (size_t x = 0, y = 0; x < sa.size() && y < sb.size();) {
                        if (sa[x] == sb[y]) { common++; x++; y++; }
                        else if (sa[x] < sb[y]) x++;
                        else y++;
                    }
                    double norm = sa.size() + sb.size() - 2.0 * common;
                    double delta = norm > 0 ? min(block[a].weight, (ga - gb) / (t * norm)) : block[a].weight;
                    if (delta <= 0) break;
                    block[a].weight -= delta;
                    block[b].weight += delta;
                    for (int j : sb) g[j] -= delta;
                    for (int j : sa) g[j] += delta;
                }
            }
            if (largest <= tolerance) break;
            if (time_limit > 0 && elapsed() >= time_limit) break;
        }
        result.lp_seconds += elapsed() - before;
        for (int j = 0; j < n; j++) candidate[j] = center[j] + t * g[j];
        double predicted = model(candidate) - center_value;
        double value = evaluate(candidate);
        result.rounds = it;
        for (int i = 0; i < m; i++) add_column(i, sets[i], it);
        if (value > result.bound) {
            result.bound = value;
            result.pi = candidate;
        }
        result.lp_value = center_value + predicted;
        if (predicted <= 1e-8 * max(1.0, fabs(center_value))) {
            result.converged = true;
            break;
        }
        if (value - center_value >= 0.1 * predicted) {   // serious step
            if (value - center_value >= 0.5 * predicted) t = min(1e6, 2 * t);
            center = candidate;
            center_value = value;
            // the residual of the quadratic program refers to the old center: u and
            // the weights stay as a warm start, g is still 1 - u - A lambda
        } else {
            t = max(1e-6, 0.7 * t);   // null step
        }
        // drop the columns unused for 50 iterations (never the empty one)
        for (int i = 0; i < m; i++) {
            vector<BundleColumn>& block = blocks[i];
            for (BundleColumn& col : block)
                if (col.weight > 1e-12) col.last_used = it;
            vector<BundleColumn> kept;
            for (size_t k = 0; k < block.size(); k++)
                if (k == 0 || block[k].weight > 1e-12 || it - block[k].last_used < 50) kept.push_back(block[k]);
            block.swap(kept);
        }
    }
    for (int i = 0; i < m; i++)
        for (const BundleColumn& col : blocks[i])
            if (col.weight > 1e-9 && !col.sellers.empty()) result.active.push_back({i, col.sellers});
    // convexity duals of the best point: mu_i = -KP_i(pi)
    evaluate(result.pi);
    result.mu.assign(m, 0.0);
    for (int i = 0; i < m; i++) result.mu[i] = -kp[i];
    result.seconds = elapsed();
    return result;
}

}

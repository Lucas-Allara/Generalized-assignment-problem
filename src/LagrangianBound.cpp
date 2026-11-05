#include "LagrangianBound.h"
#include "ConstructiveHeuristics.h"
#include "Knapsack.h"
#include "Metaheuristics.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <set>

using namespace std;

namespace lagrangian {

double evaluate(const Instance& inst, const vector<double>& lambda, vector<double>& subgradient,
                vector<vector<int>>* sets) {
    int n = inst.sellers.size();
    int m = inst.warehouses.size();
    double value = 0;
    subgradient.assign(n, 1.0);
    for (int j = 0; j < n; j++) {
        value += lambda[j];
        // u_j: leaving the seller unassigned pays off when its multiplier exceeds the penalty.
        if (inst.penalty - lambda[j] < 0) {
            value += inst.penalty - lambda[j];
            subgradient[j] -= 1.0;
        }
    }
    vector<double> profit(n);
    vector<int> weight(n);
    vector<int> chosen;
    if (sets != nullptr) sets->assign(m, {});
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            profit[j] = lambda[j] - inst.cost[j][i];
            weight[j] = inst.demand(j, i);
        }
        value -= knapsack::solve(profit, weight, inst.warehouses[i].capacity, chosen);
        for (int j : chosen) subgradient[j] -= 1.0;
        if (sets != nullptr) (*sets)[i] = chosen;
    }
    return value;
}

Solution repair(const Instance& inst, const vector<vector<int>>& sets) {
    int n = inst.sellers.size();
    Solution sol;
    sol.sellers_per_warehouse.assign(inst.warehouses.size(), {});
    sol.assigned_warehouse.assign(n, -1);
    // cheapest warehouse among those whose knapsack chose the seller
    vector<int> best(n, -1);
    for (int i = 0; i < (int)sets.size(); i++) {
        for (int j : sets[i]) {
            if (best[j] < 0 || inst.cost[j][i] < inst.cost[j][best[j]]) best[j] = i;
        }
    }
    for (int j = 0; j < n; j++) {
        if (best[j] >= 0) sol.assign(best[j], j);
    }
    constructive::complete(inst, sol);
    return sol;
}

Result subgradient(const Instance& inst, double upper_bound, int max_iterations,
                   double seconds, int patience, vector<pair<int, vector<int>>>* columns,
                   int last_iterations, int heuristic_every) {
    auto start = chrono::high_resolution_clock::now();
    auto elapsed = [&] {
        return chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now() - start).count() / 1000.0;
    };
    int n = inst.sellers.size();

    // Initial multipliers: the second smallest cost of each seller, as in Fisher,
    // Jaikumar and Van Wassenhove (the smallest one if there is a single warehouse).
    vector<double> lambda(n);
    for (int j = 0; j < n; j++) {
        const auto& d = inst.sellers[j].distances;
        lambda[j] = d.size() > 1 ? d[1].first : d[0].first;
    }

    Result result;
    result.bound = -1e300;
    double mu = 2.0;
    int stall = 0;
    vector<double> g;
    vector<vector<int>> sets;
    deque<vector<vector<int>>> recent;   // knapsack solutions of the last iterations
    for (int it = 0; it < max_iterations; it++) {
        if (seconds > 0 && elapsed() >= seconds) break;
        bool heuristic = heuristic_every > 0 && it % heuristic_every == 0;
        bool need_sets = columns != nullptr || heuristic;
        double value = evaluate(inst, lambda, g, need_sets ? &sets : nullptr);
        if (heuristic) {
            Solution candidate = repair(inst, sets);
            if (!result.has_solution || candidate.total_cost < result.solution.total_cost) {
                result.solution = candidate;
                result.has_solution = true;
                upper_bound = min(upper_bound, candidate.total_cost);
            }
        }
        if (columns != nullptr) {
            recent.push_back(sets);
            if ((int)recent.size() > last_iterations) recent.pop_front();
        }
        result.iterations = it + 1;
        if (value > result.bound + 1e-9) {
            result.bound = value;
            result.multipliers = lambda;
            stall = 0;
        } else if (++stall >= patience) {
            mu /= 2;
            stall = 0;
            if (mu < 1e-4) break;
        }
        double norm2 = 0;
        for (double x : g) norm2 += x * x;
        if (norm2 == 0) {   // the relaxed solution satisfies the relaxed constraints
            result.optimal = true;
            if (heuristic_every > 0 && !heuristic) {
                // the knapsack solutions are then an optimal assignment
                evaluate(inst, lambda, g, &sets);
                Solution candidate = repair(inst, sets);
                if (!result.has_solution || candidate.total_cost < result.solution.total_cost) {
                    result.solution = candidate;
                    result.has_solution = true;
                }
            }
            break;
        }
        double step = mu * max(upper_bound - value, 1e-6) / norm2;
        for (int j = 0; j < n; j++) lambda[j] += step * g[j];
    }
    if (columns != nullptr) {
        set<pair<int, vector<int>>> distinct;
        for (const auto& iteration : recent) {
            for (int i = 0; i < (int)iteration.size(); i++) {
                if (iteration[i].empty()) continue;
                vector<int> sellers = iteration[i];
                sort(sellers.begin(), sellers.end());
                distinct.insert({i, sellers});
            }
        }
        columns->assign(distinct.begin(), distinct.end());
    }
    result.seconds = elapsed();
    return result;
}

Solution heuristic_solution(const Instance& inst, double subgradient_seconds, double total_seconds) {
    auto start = chrono::high_resolution_clock::now();
    Solution best = metaheuristics::best_constructive(inst);
    Result lr = subgradient(inst, best.total_cost, 5000, subgradient_seconds, 50, nullptr, 100, 10);
    if (lr.has_solution && lr.solution.total_cost < best.total_cost) best = lr.solution;
    double spent = chrono::duration_cast<chrono::milliseconds>(
        chrono::high_resolution_clock::now() - start).count() / 1000.0;
    metaheuristics::vnd(best, inst, total_seconds > 0 ? max(1e-3, total_seconds - spent) : 0);
    best.total_cost = metaheuristics::evaluate(best, inst);
    return best;
}

}

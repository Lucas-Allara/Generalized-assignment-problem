#include "ColumnGeneration.h"
#include "Knapsack.h"
#include <Highs.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>

using namespace std;

namespace column_generation {

Result root(const Instance& inst, double time_limit, Solver solver,
            const vector<pair<int, vector<int>>>& initial, const vector<double>* start) {
    if (solver == Solver::BUNDLE) return bundle(inst, time_limit, initial, start);
    auto t0 = chrono::high_resolution_clock::now();
    auto elapsed = [&] {
        return chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now() - t0).count() / 1000.0;
    };
    int n = inst.sellers.size();
    int m = inst.warehouses.size();
    Result result;

    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("threads", 1);
    if (solver == Solver::INTERIOR_POINT) {
        highs.setOptionValue("solver", "ipm");
        highs.setOptionValue("run_crossover", "off");
    } else {
        highs.setOptionValue("solver", "simplex");
    }
    // rows: assignment of each seller (= 1), then convexity of each warehouse (<= 1)
    for (int j = 0; j < n; j++) highs.addRow(1.0, 1.0, 0, nullptr, nullptr);
    for (int i = 0; i < m; i++) highs.addRow(-kHighsInf, 1.0, 0, nullptr, nullptr);
    for (int j = 0; j < n; j++) {
        HighsInt row = j;
        double one = 1.0;
        highs.addCol(inst.penalty, 0.0, kHighsInf, 1, &row, &one);
    }
    set<pair<int, vector<int>>> known;
    auto add_column = [&](int i, const vector<int>& sellers) {
        if (sellers.empty() || !known.insert({i, sellers}).second) return false;
        double cost = 0;
        vector<HighsInt> rows;
        for (int j : sellers) {
            cost += inst.cost[j][i];
            rows.push_back(j);
        }
        rows.push_back(n + i);
        vector<double> ones(rows.size(), 1.0);
        highs.addCol(cost, 0.0, kHighsInf, rows.size(), rows.data(), ones.data());
        result.columns.push_back({i, sellers});
        return true;
    };
    for (const auto& [i, sellers] : initial) add_column(i, sellers);

    vector<double> profit(n);
    vector<int> weight(n), chosen;
    while (true) {
        double left = time_limit > 0 ? time_limit - elapsed() : 1e9;
        if (left <= 0) break;
        highs.setOptionValue("time_limit", highs.getRunTime() + left);   // the limit of HiGHS is cumulative
        double before = elapsed();
        highs.run();
        if (highs.getModelStatus() != HighsModelStatus::kOptimal && solver == Solver::INTERIOR_POINT) {
            // without crossover the interior point method may stop short of a
            // certified optimum (small or degenerate LPs): that round runs again with
            // crossover
            highs.setOptionValue("run_crossover", "on");
            highs.run();
            highs.setOptionValue("run_crossover", "off");
            result.crossovers++;
        }
        result.lp_seconds += elapsed() - before;
        if (highs.getModelStatus() != HighsModelStatus::kOptimal) break;
        result.rounds++;
        result.lp_value = highs.getInfo().objective_function_value;
        const HighsSolution& solution = highs.getSolution();
        // reduced cost of a column = cost - sum of the row duals
        vector<double> pi(n), mu(m);
        for (int j = 0; j < n; j++) pi[j] = solution.row_dual[j];
        for (int i = 0; i < m; i++) mu[i] = min(0.0, solution.row_dual[n + i]);

        double bound = 0;
        for (int j = 0; j < n; j++) bound += pi[j] + min(0.0, inst.penalty - pi[j]);
        int added = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                profit[j] = pi[j] - inst.cost[j][i];
                weight[j] = inst.demand(j, i);
            }
            double value = knapsack::solve(profit, weight, inst.warehouses[i].capacity, chosen);
            double rc = -value - mu[i];
            bound += mu[i] + min(0.0, rc);
            if (rc < -1e-6) {
                sort(chosen.begin(), chosen.end());
                added += add_column(i, chosen);
            }
        }
        if (bound > result.bound) {
            result.bound = bound;
            result.pi = pi;
            result.mu = mu;
        }
        if (added == 0 || result.lp_value - result.bound <= 1e-6 * max(1.0, fabs(result.lp_value))) {
            result.converged = true;
            break;
        }
    }
    result.seconds = elapsed();
    return result;
}

}

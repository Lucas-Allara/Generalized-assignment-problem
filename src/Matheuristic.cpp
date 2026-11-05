#include "Matheuristic.h"
#include "EjectionChains.h"
#include "LagrangianBound.h"
#include "Metaheuristics.h"
#include "MipModel.h"
#include <algorithm>
#include <chrono>
#include <random>

using namespace std;

namespace matheuristic {

namespace {

// 'destroy' assigned sellers chosen at random.
vector<int> random_sellers(const Solution& sol, int destroy, mt19937& rng) {
    vector<int> assigned;
    for (int j = 0; j < (int)sol.assigned_warehouse.size(); j++) {
        if (sol.assigned_warehouse[j] >= 0) assigned.push_back(j);
    }
    shuffle(assigned.begin(), assigned.end(), rng);
    if ((int)assigned.size() > destroy) assigned.resize(destroy);
    return assigned;
}

// The sellers of the warehouses of a random seller, from its cheapest warehouse
// on, until at least 'destroy' sellers (or every warehouse) are taken.
vector<int> warehouse_sellers(const Instance& inst, const Solution& sol, int destroy, mt19937& rng) {
    int n = inst.sellers.size();
    int seed_seller = uniform_int_distribution<int>(0, n - 1)(rng);
    vector<int> freed;
    for (const auto& [distance, i] : inst.sellers[seed_seller].distances) {
        if ((int)freed.size() >= destroy) break;
        for (int j : sol.sellers_per_warehouse[i]) freed.push_back(j);
    }
    return freed;
}

}

Solution mip_lns(const Instance& inst, int iterations, int destroy, unsigned seed,
                 double seconds, double sub_time_limit, Destroy op, Start start_from, const Solution* initial) {
    mt19937 rng(seed);
    auto start = chrono::high_resolution_clock::now();
    auto elapsed = [&] {
        return chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now() - start).count() / 1000.0;
    };
    auto left = [&] { return seconds > 0 ? max(1e-3, seconds - elapsed()) : 0.0; };

    Solution best;
    if (initial != nullptr) {
        best = *initial;
        best.total_cost = metaheuristics::evaluate(best, inst);
    } else if (start_from == Start::LAGRANGIAN) {
        best = lagrangian::heuristic_solution(inst, 0.6 * seconds, seconds);
    } else {
        best = metaheuristics::initial_solution(inst, seconds > 0 ? seconds / 2 : 0);
        if (start_from == Start::LP_ROUNDING && (seconds <= 0 || elapsed() < seconds)) {
            Solution rounded = mip::lp_rounding(inst, left()).solution;
            metaheuristics::vnd(rounded, inst, left());
            rounded.total_cost = metaheuristics::evaluate(rounded, inst);
            if (rounded.total_cost < best.total_cost) best = rounded;
        }
    }

    int n_sellers = inst.sellers.size();
    for (int it = 0; seconds > 0 || it < iterations; it++) {
        double limit = sub_time_limit;
        if (seconds > 0) {
            if (elapsed() >= seconds) break;
            limit = min(limit, left());
        }
        bool by_warehouses = op == Destroy::WAREHOUSES
                          || (op == Destroy::MIXED && uniform_int_distribution<int>(0, 1)(rng) == 1);
        vector<int> freed = by_warehouses ? warehouse_sellers(inst, best, destroy, rng)
                                          : random_sellers(best, destroy, rng);
        for (int j = 0; j < n_sellers; j++) {
            if (best.assigned_warehouse[j] < 0) freed.push_back(j);
        }
        // Repair: SCIP reassigns the freed sellers optimally (or as well as it can
        // within the limit); the solution only changes if it improves.
        mip::reoptimize(inst, best, freed, limit);
    }
    best.total_cost = metaheuristics::evaluate(best, inst);
    return best;
}

Solution portfolio(const Instance& inst, double seconds, unsigned seed) {
    auto start = chrono::high_resolution_clock::now();
    auto elapsed = [&] {
        return chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now() - start).count() / 1000.0;
    };
    auto until = [&](double fraction) { return max(1e-3, fraction * seconds - elapsed()); };
    Solution best = mip::lp_rounding(inst, until(0.15)).solution;
    metaheuristics::vnd(best, inst, until(0.2));
    best.total_cost = metaheuristics::evaluate(best, inst);
    Solution lagrangian_start = lagrangian::heuristic_solution(inst, until(0.3), until(0.4));
    if (lagrangian_start.total_cost < best.total_cost) best = lagrangian_start;
    Solution tabu = ejection::search(inst, until(0.6), seed, &best);
    if (tabu.total_cost < best.total_cost) best = tabu;
    int n = inst.sellers.size();
    best = mip_lns(inst, 1000000000, min(100, n / 2), seed, until(1.0), 5.0, Destroy::WAREHOUSES, Start::LAGRANGIAN, &best);
    best.total_cost = metaheuristics::evaluate(best, inst);
    return best;
}

}

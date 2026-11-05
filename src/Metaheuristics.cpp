#include "Metaheuristics.h"
#include "ConstructiveHeuristics.h"
#include "LocalSearch.h"
#include <algorithm>
#include <chrono>
#include <random>

using namespace std;

namespace metaheuristics {

double evaluate(const Solution& sol, const Instance& inst) {
    int n_sellers = inst.sellers.size();
    double total = 0;
    for (int seller = 0; seller < n_sellers; seller++) {
        int warehouse = sol.assigned_warehouse[seller];
        if (warehouse < 0) total += inst.penalty;
        else total += inst.cost[seller][warehouse];
    }
    return total;
}

// Free capacity of each warehouse in the current solution.
static vector<int> free_capacity(const Solution& sol, const Instance& inst) {
    vector<int> capacity;
    for (int i = 0; i < (int)inst.warehouses.size(); i++) capacity.push_back(inst.warehouses[i].capacity);
    for (int seller = 0; seller < (int)inst.sellers.size(); seller++) {
        int warehouse = sol.assigned_warehouse[seller];
        if (warehouse >= 0) capacity[warehouse] -= inst.demand(seller, warehouse);
    }
    return capacity;
}

// Seconds elapsed since 'start'.
static double elapsed(chrono::high_resolution_clock::time_point start) {
    return chrono::duration_cast<chrono::milliseconds>(
        chrono::high_resolution_clock::now() - start).count() / 1000.0;
}

// VND (Variable Neighborhood Descent): applies the local search operators in
// order, repeating each while it improves, and restarts the pass whenever one of
// them improved. It ends in a local optimum with respect to every neighborhood.
void vnd(Solution& sol, const Instance& inst, double seconds) {
    const int Q_MAX = 10;   // relocate and swap move up to Q_MAX sellers at once
    auto start = chrono::high_resolution_clock::now();
    // whether a move may still be tried; each operator call applies at most one move
    auto in_time = [&] { return seconds <= 0 || elapsed(start) < seconds; };
    bool improved = true;
    while (improved && in_time()) {
        improved = false;
        while (in_time() && local_search::assign_unassigned(sol, inst)) improved = true;
        while (in_time() && local_search::assign_with_ejection(sol, inst)) improved = true;
        for (int q = 1; q <= Q_MAX; q++) {
            while (in_time() && local_search::relocate_multiple(sol, inst, q)) improved = true;
            while (in_time() && local_search::swap_multiple(sol, inst, q)) improved = true;
        }
        while (in_time() && local_search::cyclic_exchange(sol, inst, 5)) improved = true;
    }
}

Solution best_constructive(const Instance& inst) {
    // The cost-based constructives leave many sellers unassigned when the cheap
    // warehouses of a seller are those where its demand is largest (type e); the
    // demand-based measures assign them all, at a higher cost.
    vector<Solution> candidates = {
        constructive::nearest_neighbor(inst),
        constructive::regret_insertion(inst),
        constructive::regret_greedy(inst, constructive::Desirability::DEMAND),
        constructive::regret_greedy(inst, constructive::Desirability::RELATIVE_DEMAND)};
    Solution best = candidates[0];
    for (const Solution& candidate : candidates) {
        if (candidate.total_cost < best.total_cost) best = candidate;
    }
    return best;
}

Solution initial_solution(const Instance& inst, double seconds) {
    auto start = chrono::high_resolution_clock::now();
    Solution best = best_constructive(inst);
    double left = seconds > 0 ? max(1e-3, seconds - elapsed(start)) : 0;
    vnd(best, inst, left);
    best.total_cost = evaluate(best, inst);
    return best;
}

// Mixed perturbation: half random swaps and half random relocations. These are
// moves that may worsen the solution, precisely to escape from the local optimum.
// The mix turned out slightly better than swaps alone.
static void perturb(Solution& sol, const Instance& inst, int strength, mt19937& rng) {
    int n_sellers = inst.sellers.size();
    int n_warehouses = inst.warehouses.size();
    vector<int> capacity = free_capacity(sol, inst);
    uniform_int_distribution<int> pick_seller(0, n_sellers - 1);
    uniform_int_distribution<int> pick_warehouse(0, n_warehouses - 1);

    // First half: random swaps (exchange the warehouses of two sellers).
    int swaps = strength / 2;
    for (int t = 0; t < swaps; t++) {
        int seller1 = pick_seller(rng), seller2 = pick_seller(rng);
        if (seller1 == seller2) continue;
        int w1 = sol.assigned_warehouse[seller1], w2 = sol.assigned_warehouse[seller2];
        if (w1 < 0 || w2 < 0 || w1 == w2) continue;
        if (inst.demand(seller1, w2) <= capacity[w2] + inst.demand(seller2, w2)
            && inst.demand(seller2, w1) <= capacity[w1] + inst.demand(seller1, w1)) {
            sol.remove(w1, seller1); sol.remove(w2, seller2);
            sol.assign(w2, seller1); sol.assign(w1, seller2);
            capacity[w1] += inst.demand(seller1, w1) - inst.demand(seller2, w1);
            capacity[w2] += inst.demand(seller2, w2) - inst.demand(seller1, w2);
        }
    }
    // Second half: random relocations (move a seller to a random warehouse with room).
    for (int t = swaps; t < strength; t++) {
        int seller = pick_seller(rng);
        int current = sol.assigned_warehouse[seller];
        if (current < 0) continue;
        int target = pick_warehouse(rng);
        if (target == current) continue;
        if (capacity[target] >= inst.demand(seller, target)) {
            sol.remove(current, seller); sol.assign(target, seller);
            capacity[current] += inst.demand(seller, current);
            capacity[target] -= inst.demand(seller, target);
        }
    }
}

// ILS (Iterated Local Search): starts from a constructive solution + VND and
// repeats the cycle (perturb + VND), always keeping the best solution found.
Solution ils(const Instance& inst, int iterations, int strength, unsigned seed, double seconds) {
    mt19937 rng(seed);
    // With seconds > 0 it stops on time; otherwise, on the number of iterations.
    auto start = chrono::high_resolution_clock::now();
    Solution best = initial_solution(inst, seconds);

    for (int it = 0; seconds > 0 || it < iterations; it++) {
        if (seconds > 0 && elapsed(start) >= seconds) break;
        Solution candidate = best;
        perturb(candidate, inst, strength, rng);
        candidate.total_cost = evaluate(candidate, inst);
        vnd(candidate, inst, seconds > 0 ? max(1e-3, seconds - elapsed(start)) : 0);
        if (candidate.total_cost < best.total_cost) best = candidate;
    }
    return best;
}

// LNS (Large Neighborhood Search, ruin and recreate): at each iteration it
// unassigns 'destroy' random sellers and lets the VND reinsert them (through
// assign_unassigned) and optimize. It is a larger perturbation than that of the
// ILS. It accepts the result only if it improves.
Solution lns(const Instance& inst, int iterations, int destroy, unsigned seed, double seconds,
             const Solution* start_solution) {
    mt19937 rng(seed);
    auto start = chrono::high_resolution_clock::now();
    Solution best = start_solution != nullptr ? *start_solution : initial_solution(inst, seconds);

    int n_sellers = inst.sellers.size();
    for (int it = 0; seconds > 0 || it < iterations; it++) {
        if (seconds > 0 && elapsed(start) >= seconds) break;
        Solution candidate = best;
        // Destroy: unassign 'destroy' sellers chosen at random.
        vector<int> assigned;
        for (int seller = 0; seller < n_sellers; seller++) {
            if (candidate.assigned_warehouse[seller] >= 0) assigned.push_back(seller);
        }
        shuffle(assigned.begin(), assigned.end(), rng);
        for (int t = 0; t < destroy && t < (int)assigned.size(); t++) {
            int seller = assigned[t];
            candidate.remove(candidate.assigned_warehouse[seller], seller);
        }
        // Recreate and optimize: the VND reinserts the unassigned sellers and improves.
        vnd(candidate, inst, seconds > 0 ? max(1e-3, seconds - elapsed(start)) : 0);
        candidate.total_cost = evaluate(candidate, inst);
        if (candidate.total_cost < best.total_cost) best = candidate;
    }
    return best;
}

}

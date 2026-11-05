#include "ConstructiveHeuristics.h"
#include <algorithm>

using namespace std;

namespace constructive {

// Nearest neighbor heuristic.
// Goes through the sellers in order and assigns each one to the closest warehouse
// that still has enough capacity for its demand there.
Solution nearest_neighbor(const Instance& inst) {
    const vector<Warehouse>& warehouses = inst.warehouses;
    const vector<Seller>& sellers = inst.sellers;

    Solution sol;
    sol.sellers_per_warehouse.resize(warehouses.size());
    sol.assigned_warehouse.assign(sellers.size(), -1);

    // Remaining capacity of each warehouse, decreased as sellers are assigned.
    vector<int> remaining;
    for (int i = 0; i < (int)warehouses.size(); i++) {
        remaining.push_back(warehouses[i].capacity);
    }

    for (int j = 0; j < (int)sellers.size(); j++) {
        bool placed = false;

        // The distances are sorted, so the first feasible warehouse is the
        // closest available one.
        for (int t = 0; t < (int)sellers[j].distances.size(); t++) {
            double distance = sellers[j].distances[t].first;
            int warehouse = sellers[j].distances[t].second;

            if (remaining[warehouse] >= sellers[j].demand[warehouse]) {
                sol.assign(warehouse, j);
                sol.total_cost += distance;
                remaining[warehouse] -= sellers[j].demand[warehouse];
                placed = true;
                break;
            }
        }

        if (!placed) {
            sol.total_cost += inst.penalty;
        }
    }
    return sol;
}

// Regret insertion heuristic.
// At each step it picks the "most urgent" seller: the one whose cost difference
// between its best and second-best feasible warehouse is largest (largest regret).
// The idea is to serve first the sellers that would lose the most if their
// preferred warehouse filled up.
Solution regret_insertion(const Instance& inst) {
    const vector<Warehouse>& warehouses = inst.warehouses;
    const vector<Seller>& sellers = inst.sellers;

    int n_sellers = sellers.size();
    Solution sol;
    sol.sellers_per_warehouse.resize(warehouses.size());
    sol.assigned_warehouse.assign(n_sellers, -1);

    vector<int> remaining;
    for (int i = 0; i < (int)warehouses.size(); i++) {
        remaining.push_back(warehouses[i].capacity);
    }

    vector<bool> done(n_sellers, false);

    // One assignment per iteration until every seller is placed.
    for (int step = 0; step < n_sellers; step++) {
        int chosen_seller = -1;
        int chosen_warehouse = -1;
        double chosen_cost = 0;
        double largest_regret = -1;
        bool has_candidate = false;

        for (int j = 0; j < n_sellers; j++) {
            if (done[j]) continue;

            // The two closest feasible warehouses of this seller.
            int best_warehouse = -1;
            double best_cost = 0, second_cost = 0;
            bool has_second = false;

            for (int t = 0; t < (int)sellers[j].distances.size(); t++) {
                int warehouse = sellers[j].distances[t].second;
                if (remaining[warehouse] >= sellers[j].demand[warehouse]) {
                    if (best_warehouse == -1) {
                        best_warehouse = warehouse;
                        best_cost = sellers[j].distances[t].first;
                    } else {
                        second_cost = sellers[j].distances[t].first;
                        has_second = true;
                        break;
                    }
                }
            }

            // A seller without a feasible warehouse is left for later.
            if (best_warehouse == -1) continue;
            has_candidate = true;

            // The regret measures how much the seller would lose by not getting
            // its best option. A seller with a single feasible option gets the
            // highest priority through a very large value.
            double regret;
            if (has_second) {
                regret = second_cost - best_cost;
            } else {
                regret = 1e9;
            }

            if (regret > largest_regret) {
                largest_regret = regret;
                chosen_seller = j;
                chosen_warehouse = best_warehouse;
                chosen_cost = best_cost;
            }
        }

        // If no remaining seller has a feasible warehouse, the rest stay
        // unassigned and are penalized.
        if (!has_candidate) {
            for (int j = 0; j < n_sellers; j++) {
                if (!done[j]) sol.total_cost += inst.penalty;
            }
            break;
        }

        sol.assign(chosen_warehouse, chosen_seller);
        sol.total_cost += chosen_cost;
        remaining[chosen_warehouse] -= sellers[chosen_seller].demand[chosen_warehouse];
        done[chosen_seller] = true;
    }
    return sol;
}

Solution regret_greedy(const Instance& inst, Desirability desirability) {
    int n_sellers = inst.sellers.size();
    int n_warehouses = inst.warehouses.size();

    // f[j][i] and the warehouses of each seller sorted by increasing f (ties by
    // cost and then by index), so that the two best feasible ones are the first
    // two that fit.
    vector<vector<double>> f(n_sellers, vector<double>(n_warehouses));
    vector<vector<int>> order(n_sellers);
    for (int j = 0; j < n_sellers; j++) {
        for (int i = 0; i < n_warehouses; i++) {
            double d = inst.demand(j, i);
            if (desirability == Desirability::COST) f[j][i] = inst.cost[j][i];
            else if (desirability == Desirability::DEMAND) f[j][i] = d;
            else f[j][i] = d / inst.warehouses[i].capacity;
        }
        order[j].resize(n_warehouses);
        for (int i = 0; i < n_warehouses; i++) order[j][i] = i;
        sort(order[j].begin(), order[j].end(), [&](int a, int b) {
            if (f[j][a] != f[j][b]) return f[j][a] < f[j][b];
            if (inst.cost[j][a] != inst.cost[j][b]) return inst.cost[j][a] < inst.cost[j][b];
            return a < b;
        });
    }

    Solution sol;
    sol.sellers_per_warehouse.resize(n_warehouses);
    sol.assigned_warehouse.assign(n_sellers, -1);
    vector<int> remaining;
    for (const Warehouse& w : inst.warehouses) remaining.push_back(w.capacity);

    vector<bool> done(n_sellers, false);
    for (int step = 0; step < n_sellers; step++) {
        int chosen_seller = -1, chosen_warehouse = -1;
        double largest_regret = -1;
        for (int j = 0; j < n_sellers; j++) {
            if (done[j]) continue;
            int best = -1, second = -1;
            for (int i : order[j]) {
                if (remaining[i] < inst.demand(j, i)) continue;
                if (best == -1) best = i;
                else { second = i; break; }
            }
            if (best == -1) continue;   // it fits nowhere: it stays unassigned
            // a seller with a single feasible warehouse goes first
            double regret = second == -1 ? 1e18 : f[j][second] - f[j][best];
            if (regret > largest_regret) {
                largest_regret = regret;
                chosen_seller = j;
                chosen_warehouse = best;
            }
        }
        if (chosen_seller == -1) break;   // no remaining seller fits anywhere
        sol.assign(chosen_warehouse, chosen_seller);
        remaining[chosen_warehouse] -= inst.demand(chosen_seller, chosen_warehouse);
        done[chosen_seller] = true;
    }
    for (int j = 0; j < n_sellers; j++) {
        int i = sol.assigned_warehouse[j];
        sol.total_cost += i >= 0 ? inst.cost[j][i] : inst.penalty;
    }
    return sol;
}

void complete(const Instance& inst, Solution& sol) {
    int n_sellers = inst.sellers.size();
    vector<int> remaining;
    for (const Warehouse& w : inst.warehouses) remaining.push_back(w.capacity);
    for (int j = 0; j < n_sellers; j++) {
        int i = sol.assigned_warehouse[j];
        if (i >= 0) remaining[i] -= inst.demand(j, i);
    }
    // regret of each unassigned seller with the capacities at the start
    vector<pair<double, int>> order;
    for (int j = 0; j < n_sellers; j++) {
        if (sol.assigned_warehouse[j] >= 0) continue;
        double best = -1, second = -1;
        for (const auto& [distance, i] : inst.sellers[j].distances) {
            if (remaining[i] < inst.demand(j, i)) continue;
            if (best < 0) best = distance;
            else { second = distance; break; }
        }
        double regret = best < 0 ? -1 : (second < 0 ? 1e18 : second - best);
        order.push_back({-regret, j});
    }
    sort(order.begin(), order.end());
    for (const auto& [minus_regret, j] : order) {
        for (const auto& [distance, i] : inst.sellers[j].distances) {
            if (remaining[i] < inst.demand(j, i)) continue;
            sol.assign(i, j);
            remaining[i] -= inst.demand(j, i);
            break;
        }
    }
    sol.total_cost = 0;
    for (int j = 0; j < n_sellers; j++) {
        int i = sol.assigned_warehouse[j];
        sol.total_cost += i >= 0 ? inst.cost[j][i] : inst.penalty;
    }
}

}

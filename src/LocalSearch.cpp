#include "LocalSearch.h"
#include <algorithm>

using namespace std;

namespace local_search {

// Free capacity of each warehouse in a solution: its total capacity minus the
// demand that its assigned sellers place on it.
static vector<int> free_capacity(const Solution& sol, const Instance& inst) {
    vector<int> capacity;
    for (int i = 0; i < (int)inst.warehouses.size(); i++) {
        capacity.push_back(inst.warehouses[i].capacity);
    }
    for (int seller = 0; seller < (int)inst.sellers.size(); seller++) {
        int warehouse = sol.assigned_warehouse[seller];
        if (warehouse >= 0) capacity[warehouse] -= inst.demand(seller, warehouse);
    }
    return capacity;
}

// Tries to assign the sellers left unassigned (assigned_warehouse == -1) to
// their closest feasible warehouse. Assigning one removes its penalty of
// 3 * largest cost, so entering any warehouse always improves. Returns true if
// it assigned any. This is what reconsiders the unassigned sellers when
// relocate and swap free capacity.
bool assign_unassigned(Solution& sol, const Instance& inst) {
    const vector<Seller>& sellers = inst.sellers;
    int n_sellers = sellers.size();
    vector<int> capacity = free_capacity(sol, inst);

    bool changed = false;
    for (int seller = 0; seller < n_sellers; seller++) {
        if (sol.assigned_warehouse[seller] >= 0) continue;   // already assigned
        // The distances are sorted: the first feasible warehouse is the closest one.
        for (int t = 0; t < (int)sellers[seller].distances.size(); t++) {
            int warehouse = sellers[seller].distances[t].second;
            int demand = inst.demand(seller, warehouse);
            if (capacity[warehouse] >= demand) {
                sol.assign(warehouse, seller);
                sol.total_cost += sellers[seller].distances[t].first - inst.penalty;
                capacity[warehouse] -= demand;
                changed = true;
                break;
            }
        }
    }
    return changed;
}

// Ejection: an unassigned seller enters a warehouse k after a seller already in
// k moves to another warehouse with room. Since placing the unassigned seller
// removes its penalty of 3 * largest cost, this usually improves even if the
// displaced seller ends up somewhat worse. It places unassigned sellers in tight
// instances. Applies the first improvement found.
bool assign_with_ejection(Solution& sol, const Instance& inst) {
    const vector<Seller>& sellers = inst.sellers;
    const vector<vector<double>>& cost = inst.cost;
    int n_warehouses = inst.warehouses.size();
    vector<int> capacity = free_capacity(sol, inst);

    for (int seller = 0; seller < (int)sellers.size(); seller++) {
        if (sol.assigned_warehouse[seller] >= 0) continue;   // only unassigned sellers
        for (int t = 0; t < (int)sellers[seller].distances.size(); t++) {
            int k = sellers[seller].distances[t].second;
            int demand = inst.demand(seller, k);
            if (capacity[k] >= demand) continue;   // a direct entry is done by assign_unassigned
            int missing = demand - capacity[k];   // capacity to free in k
            for (int displaced : sol.sellers_per_warehouse[k]) {
                if (inst.demand(displaced, k) < missing) continue;   // removing it is not enough
                for (int other = 0; other < n_warehouses; other++) {
                    if (other == k || capacity[other] < inst.demand(displaced, other)) continue;
                    double delta = (cost[seller][k] - inst.penalty)
                                 + (cost[displaced][other] - cost[displaced][k]);
                    if (delta < 0) {
                        sol.remove(k, displaced);
                        sol.assign(other, displaced);
                        sol.assign(k, seller);
                        sol.total_cost += delta;
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

// Relocate (first improvement): moves a seller to a cheaper warehouse with room.
// Keeps the first move that improves.
bool relocate_first_improvement(Solution& sol, const Instance& inst) {
    const vector<vector<double>>& cost = inst.cost;
    int n_sellers = inst.sellers.size();
    int n_warehouses = inst.warehouses.size();
    vector<int> capacity = free_capacity(sol, inst);

    for (int seller = 0; seller < n_sellers; seller++) {
        int current = sol.assigned_warehouse[seller];
        if (current < 0) continue;
        for (int target = 0; target < n_warehouses; target++) {
            if (target == current) continue;
            if (cost[seller][target] < cost[seller][current]
                && inst.demand(seller, target) <= capacity[target]) {
                sol.remove(current, seller);
                sol.assign(target, seller);
                sol.total_cost += cost[seller][target] - cost[seller][current];
                return true;
            }
        }
    }
    return false;
}

// Relocate (best improvement): goes through every possible move and applies the
// one that reduces the cost the most.
bool relocate_best_improvement(Solution& sol, const Instance& inst) {
    const vector<vector<double>>& cost = inst.cost;
    int n_sellers = inst.sellers.size();
    int n_warehouses = inst.warehouses.size();
    vector<int> capacity = free_capacity(sol, inst);

    bool improved = false;
    double best_delta = 0;
    int best_seller = -1, best_origin = -1, best_target = -1;

    for (int seller = 0; seller < n_sellers; seller++) {
        int current = sol.assigned_warehouse[seller];
        if (current < 0) continue;
        for (int target = 0; target < n_warehouses; target++) {
            if (target == current) continue;
            if (cost[seller][target] < cost[seller][current]
                && inst.demand(seller, target) <= capacity[target]) {
                double delta = cost[seller][target] - cost[seller][current];
                if (!improved || delta < best_delta) {
                    improved = true;
                    best_delta = delta;
                    best_seller = seller;
                    best_origin = current;
                    best_target = target;
                }
            }
        }
    }

    if (improved) {
        sol.remove(best_origin, best_seller);
        sol.assign(best_target, best_seller);
        sol.total_cost += best_delta;
    }
    return improved;
}

// Swap (first improvement): exchanges the warehouses of two sellers if the
// exchange reduces the cost and both still fit. Applies the first one that improves.
bool swap_first_improvement(Solution& sol, const Instance& inst) {
    const vector<vector<double>>& cost = inst.cost;
    int n_sellers = inst.sellers.size();
    vector<int> capacity = free_capacity(sol, inst);

    for (int seller1 = 0; seller1 < n_sellers; seller1++) {
        int warehouse1 = sol.assigned_warehouse[seller1];
        if (warehouse1 < 0) continue;
        for (int seller2 = seller1 + 1; seller2 < n_sellers; seller2++) {
            int warehouse2 = sol.assigned_warehouse[seller2];
            if (warehouse2 < 0 || warehouse1 == warehouse2) continue;

            double delta = (cost[seller1][warehouse2] + cost[seller2][warehouse1])
                         - (cost[seller1][warehouse1] + cost[seller2][warehouse2]);
            // In the exchange, each warehouse recovers the demand of the seller that leaves.
            bool fits1 = inst.demand(seller1, warehouse2) <= capacity[warehouse2] + inst.demand(seller2, warehouse2);
            bool fits2 = inst.demand(seller2, warehouse1) <= capacity[warehouse1] + inst.demand(seller1, warehouse1);

            if (delta < 0 && fits1 && fits2) {
                sol.remove(warehouse1, seller1);
                sol.remove(warehouse2, seller2);
                sol.assign(warehouse2, seller1);
                sol.assign(warehouse1, seller2);
                sol.total_cost += delta;
                return true;
            }
        }
    }
    return false;
}

// Swap (best improvement): evaluates every feasible exchange and applies the one
// with the largest cost reduction.
bool swap_best_improvement(Solution& sol, const Instance& inst) {
    const vector<vector<double>>& cost = inst.cost;
    int n_sellers = inst.sellers.size();
    vector<int> capacity = free_capacity(sol, inst);

    bool improved = false;
    double best_delta = 0;
    int best_s1 = -1, best_w1 = -1, best_s2 = -1, best_w2 = -1;

    for (int seller1 = 0; seller1 < n_sellers; seller1++) {
        int warehouse1 = sol.assigned_warehouse[seller1];
        if (warehouse1 < 0) continue;
        for (int seller2 = seller1 + 1; seller2 < n_sellers; seller2++) {
            int warehouse2 = sol.assigned_warehouse[seller2];
            if (warehouse2 < 0 || warehouse1 == warehouse2) continue;

            double delta = (cost[seller1][warehouse2] + cost[seller2][warehouse1])
                         - (cost[seller1][warehouse1] + cost[seller2][warehouse2]);
            bool fits1 = inst.demand(seller1, warehouse2) <= capacity[warehouse2] + inst.demand(seller2, warehouse2);
            bool fits2 = inst.demand(seller2, warehouse1) <= capacity[warehouse1] + inst.demand(seller1, warehouse1);

            if (delta < 0 && fits1 && fits2) {
                if (!improved || delta < best_delta) {
                    improved = true;
                    best_delta = delta;
                    best_s1 = seller1;
                    best_w1 = warehouse1;
                    best_s2 = seller2;
                    best_w2 = warehouse2;
                }
            }
        }
    }

    if (improved) {
        sol.remove(best_w1, best_s1);
        sol.remove(best_w2, best_s2);
        sol.assign(best_w2, best_s1);
        sol.assign(best_w1, best_s2);
        sol.total_cost += best_delta;
    }
    return improved;
}

// Generalized relocate: moves up to 'count' profitable sellers from an origin
// warehouse to a target one (the ones with the largest gain that fit). count = 1
// is a simple relocate. Applies the first (origin, target) pair that improves.
bool relocate_multiple(Solution& sol, const Instance& inst, int count) {
    const vector<vector<double>>& cost = inst.cost;
    int n_warehouses = inst.warehouses.size();
    vector<int> capacity = free_capacity(sol, inst);

    for (int origin = 0; origin < n_warehouses; origin++) {
        if (sol.sellers_per_warehouse[origin].empty()) continue;
        for (int target = 0; target < n_warehouses; target++) {
            if (target == origin) continue;
            // Candidates: sellers of 'origin' that improve by moving to 'target',
            // stored with their gain (negative = profitable).
            vector<pair<double, int>> candidates;
            for (int seller : sol.sellers_per_warehouse[origin]) {
                double gain = cost[seller][target] - cost[seller][origin];
                if (gain < 0) candidates.push_back(make_pair(gain, seller));
            }
            if (candidates.empty()) continue;
            sort(candidates.begin(), candidates.end());   // most profitable first

            int room = capacity[target];
            double delta = 0;
            vector<int> to_move;
            for (int idx = 0; idx < (int)candidates.size() && (int)to_move.size() < count; idx++) {
                int seller = candidates[idx].second;
                int demand = inst.demand(seller, target);
                if (demand <= room) {
                    to_move.push_back(seller);
                    room -= demand;
                    delta += candidates[idx].first;
                }
            }
            if (!to_move.empty()) {
                for (int seller : to_move) {
                    sol.remove(origin, seller);
                    sol.assign(target, seller);
                }
                sol.total_cost += delta;
                return true;
            }
        }
    }
    return false;
}

// Generalized swap: applies up to 'count' successive exchanges (the best one each
// time) between a pair of warehouses. count = 1 is a simple swap. Applies the
// first pair of warehouses that improves.
bool swap_multiple(Solution& sol, const Instance& inst, int count) {
    const vector<vector<double>>& cost = inst.cost;
    int n_warehouses = inst.warehouses.size();
    vector<int> capacity = free_capacity(sol, inst);

    for (int warehouse1 = 0; warehouse1 < n_warehouses; warehouse1++) {
        for (int warehouse2 = warehouse1 + 1; warehouse2 < n_warehouses; warehouse2++) {
            double total_delta = 0;
            int done = 0;
            while (done < count) {
                // Best feasible exchange between these two warehouses.
                double best_delta = 0;
                int best_s1 = -1, best_s2 = -1;
                for (int seller1 : sol.sellers_per_warehouse[warehouse1]) {
                    for (int seller2 : sol.sellers_per_warehouse[warehouse2]) {
                        double delta = (cost[seller1][warehouse2] + cost[seller2][warehouse1])
                                     - (cost[seller1][warehouse1] + cost[seller2][warehouse2]);
                        if (delta < best_delta
                            && inst.demand(seller1, warehouse2) <= capacity[warehouse2] + inst.demand(seller2, warehouse2)
                            && inst.demand(seller2, warehouse1) <= capacity[warehouse1] + inst.demand(seller1, warehouse1)) {
                            best_delta = delta;
                            best_s1 = seller1;
                            best_s2 = seller2;
                        }
                    }
                }
                if (best_s1 < 0) break;   // no further improving exchange
                sol.remove(warehouse1, best_s1);
                sol.remove(warehouse2, best_s2);
                sol.assign(warehouse2, best_s1);
                sol.assign(warehouse1, best_s2);
                capacity[warehouse1] += inst.demand(best_s1, warehouse1) - inst.demand(best_s2, warehouse1);
                capacity[warehouse2] += inst.demand(best_s2, warehouse2) - inst.demand(best_s1, warehouse2);
                total_delta += best_delta;
                done++;
            }
            if (done > 0) {
                sol.total_cost += total_delta;
                return true;
            }
        }
    }
    return false;
}

// Cyclic exchange among 3 warehouses: rotates three sellers among three warehouses
// (seller1 -> warehouse2, seller2 -> warehouse3, seller3 -> warehouse1). These are
// moves that relocate and swap (2 warehouses only) cannot reach. To keep it
// tractable, the target warehouses are limited to the 'neighbors' closest ones of
// each seller (the distances are already sorted). An epsilon prevents accepting
// improvements that are floating-point noise (which could cause infinite loops).
// Applies the first improvement.
bool cyclic_exchange(Solution& sol, const Instance& inst, int neighbors) {
    const double EPS = 1e-7;
    const vector<Seller>& sellers = inst.sellers;
    const vector<vector<double>>& cost = inst.cost;
    int n_sellers = sellers.size();
    vector<int> capacity = free_capacity(sol, inst);

    for (int seller1 = 0; seller1 < n_sellers; seller1++) {
        int warehouse1 = sol.assigned_warehouse[seller1];
        if (warehouse1 < 0) continue;

        // warehouse2 ranges over the closest neighbors of seller1.
        int seen2 = 0;
        for (int a = 0; a < (int)sellers[seller1].distances.size() && seen2 < neighbors; a++) {
            int warehouse2 = sellers[seller1].distances[a].second;
            if (warehouse2 == warehouse1) continue;
            seen2++;
            for (int idx2 = 0; idx2 < (int)sol.sellers_per_warehouse[warehouse2].size(); idx2++) {
                int seller2 = sol.sellers_per_warehouse[warehouse2][idx2];

                // And warehouse3 over the closest neighbors of seller2.
                int seen3 = 0;
                for (int b = 0; b < (int)sellers[seller2].distances.size() && seen3 < neighbors; b++) {
                    int warehouse3 = sellers[seller2].distances[b].second;
                    if (warehouse3 == warehouse1 || warehouse3 == warehouse2) continue;
                    seen3++;
                    for (int idx3 = 0; idx3 < (int)sol.sellers_per_warehouse[warehouse3].size(); idx3++) {
                        int seller3 = sol.sellers_per_warehouse[warehouse3][idx3];
                        // Feasibility of the rotation s1->w2, s2->w3, s3->w1.
                        if (inst.demand(seller1, warehouse2) <= capacity[warehouse2] + inst.demand(seller2, warehouse2)
                            && inst.demand(seller2, warehouse3) <= capacity[warehouse3] + inst.demand(seller3, warehouse3)
                            && inst.demand(seller3, warehouse1) <= capacity[warehouse1] + inst.demand(seller1, warehouse1)) {
                            double delta = (cost[seller1][warehouse2] + cost[seller2][warehouse3] + cost[seller3][warehouse1])
                                         - (cost[seller1][warehouse1] + cost[seller2][warehouse2] + cost[seller3][warehouse3]);
                            if (delta < -EPS) {
                                sol.remove(warehouse1, seller1);
                                sol.remove(warehouse2, seller2);
                                sol.remove(warehouse3, seller3);
                                sol.assign(warehouse2, seller1);
                                sol.assign(warehouse3, seller2);
                                sol.assign(warehouse1, seller3);
                                sol.total_cost += delta;
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }
    return false;
}

}

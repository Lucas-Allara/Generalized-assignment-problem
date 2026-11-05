#ifndef CONSTRUCTIVE_HEURISTICS_H
#define CONSTRUCTIVE_HEURISTICS_H

#include "Instance.h"
#include "Solution.h"

namespace constructive {

// builds an initial solution by assigning each seller to its closest feasible
// warehouse.
Solution nearest_neighbor(const Instance& inst);

// builds an initial solution by regret insertion: at each step it assigns the
// seller with the largest difference between its best and second-best feasible
// warehouse.
Solution regret_insertion(const Instance& inst);

// desirability measures of the regret heuristic of Martello and Toth (1981): the
// smaller the value, the more desirable it is to assign seller j to warehouse i.
enum class Desirability {
    COST,              // c_ij (the regret insertion above)
    DEMAND,            // d_ij
    RELATIVE_DEMAND    // d_ij / b_i
};

// regret heuristic of Martello and Toth with a general desirability measure f_ij:
// at each step it assigns the seller with the largest difference between the
// second-best and the best value of f_ij among the warehouses where it still
// fits, to its best warehouse. With COST it builds the same solution as
// regret_insertion. The measures based on the demand fill the warehouses more
// slowly, which matters when the cheap warehouses of a seller are those where its
// demand is largest (the instances of type e).
Solution regret_greedy(const Instance& inst, Desirability desirability);

// completes a partial solution (capacity-feasible, with some sellers unassigned):
// the unassigned sellers are taken in decreasing order of their regret (second
// cheapest minus cheapest warehouse with room, computed once at the start) and
// each one goes to its cheapest warehouse with room at its turn; those that fit
// nowhere stay unassigned. Updates total_cost. Time O(u (m + log u)) for u unassigned
// sellers, so that it can be called often (Lagrangian heuristic).
void complete(const Instance& inst, Solution& sol);

}

#endif

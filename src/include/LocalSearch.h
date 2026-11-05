#ifndef LOCAL_SEARCH_H
#define LOCAL_SEARCH_H

#include "Instance.h"
#include "Solution.h"

namespace local_search {

bool relocate_first_improvement(Solution& sol, const Instance& inst);
bool relocate_best_improvement(Solution& sol, const Instance& inst);
bool swap_first_improvement(Solution& sol, const Instance& inst);
bool swap_best_improvement(Solution& sol, const Instance& inst);

// assigns the unassigned sellers to a feasible warehouse (removing their
// penalty). Returns true if it assigned any.
bool assign_unassigned(Solution& sol, const Instance& inst);

// assigns an unassigned seller by moving another one out of its warehouse
// (ejection).
bool assign_with_ejection(Solution& sol, const Instance& inst);

// generalized relocate and swap: move or exchange up to 'count' sellers at once
// (count = 1 is the simple operator).
bool relocate_multiple(Solution& sol, const Instance& inst, int count);
bool swap_multiple(Solution& sol, const Instance& inst, int count);

// cyclic exchange among 3 warehouses: rotates j1->i2, j2->i3, j3->i1. 'neighbors'
// limits the candidate warehouses to the closest ones of each seller.
bool cyclic_exchange(Solution& sol, const Instance& inst, int neighbors);

}

#endif

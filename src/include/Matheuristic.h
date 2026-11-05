#ifndef MATHEURISTIC_H
#define MATHEURISTIC_H

#include "Instance.h"
#include "Solution.h"

namespace matheuristic {

// destruction operators of the matheuristic (the unassigned sellers are always
// freed as well)
enum class Destroy {
    RANDOM,       // 'destroy' sellers chosen at random
    WAREHOUSES,   // every seller of a group of related warehouses: those of a random
                  // seller in increasing order of its cost (in the real instance,
                  // the closest ones), until 'destroy' sellers are freed
    MIXED         // one of the two, at random, at each iteration
};

// initial solution of the matheuristic, improved with the VND in every case
enum class Start {
    CONSTRUCTIVE,   // the best constructive solution (as the LNS)
    LP_ROUNDING,    // the best of that one and the rounding of the linear relaxation
    LAGRANGIAN      // the best of the constructive and the Lagrangian heuristic
                    // (subgradient method for at most 60% of the time; it
                    // usually converges before)
};

// LNS whose repair, instead of the VND, solves with SCIP the integer subproblem of
// the destroyed sellers: at each iteration it frees some sellers with the
// operator 'op', keeps the rest fixed and lets SCIP choose the best warehouses for
// the freed ones, with at most 'sub_time_limit' seconds per subproblem. The
// freeing of whole warehouses lets the subproblem exchange sellers among them,
// which a random set of sellers spread over many full warehouses does not allow.
// It starts from the solution given by 'start', or from 'initial' if given. If
// 'seconds' > 0, it stops on time (including the initial solution) instead of on
// the number of iterations.
Solution mip_lns(const Instance& inst, int iterations = 100, int destroy = 100, unsigned seed = 1,
                 double seconds = 0, double sub_time_limit = 5.0, Destroy op = Destroy::WAREHOUSES,
                 Start start = Start::LAGRANGIAN, const Solution* initial = nullptr);

// Portfolio start for the hard instances (type d above all): the rounding of the
// linear relaxation and the Lagrangian heuristic, both improved with the VND (on
// the large instances of type d the rounding is the best, since a basic solution
// splits at most m of the n sellers; on the small ones the Lagrangian heuristic),
// then the better of the two improved by the tabu search with ejection chains
// (ejection::search, 20% of the time) and by the MIP-LNS from its solution (the
// rest). The time includes everything.
Solution portfolio(const Instance& inst, double seconds, unsigned seed = 1);

}

#endif

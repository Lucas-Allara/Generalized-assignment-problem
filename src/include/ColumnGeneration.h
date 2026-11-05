#ifndef COLUMN_GENERATION_H
#define COLUMN_GENERATION_H

#include <utility>
#include <vector>
#include "Instance.h"

namespace column_generation {

// solver of the restricted master LP
enum class Solver {
    SIMPLEX,          // dual simplex of HiGHS: vertex duals, as SoPlex inside SCIP
    INTERIOR_POINT,   // interior point method of HiGHS without crossover: duals in the
                      // interior of the optimal face, which stabilizes the column
                      // generation (Gondzio, Gonzalez-Brevis and Munari, 2013)
    BUNDLE            // no LP: proximal bundle method on the same dual (see bundle)
};

struct Result {
    double bound = -1e300;        // best Lagrangian bound of the rounds (valid lower bound)
    double lp_value = 1e300;      // value of the last restricted master LP
    bool converged = false;       // no column with negative reduced cost was left
    int rounds = 0;
    int crossovers = 0;           // interior point rounds that needed crossover
    double seconds = 0;
    double lp_seconds = 0;        // time inside the LP solver
    std::vector<std::pair<int, std::vector<int>>> columns;   // every column of the master
    std::vector<double> pi, mu;   // duals of the best bound
    // bundle: the columns with positive weight in the last quadratic program (an
    // approximate optimal solution of the master LP)
    std::vector<std::pair<int, std::vector<int>>> active;
};

// Column generation of the master LP of the root of the branch-and-price, with
// HiGHS instead of SoPlex: the same set partitioning master (assignment rows = 1,
// convexity rows <= 1, a column u_j per seller with the penalty) and the same
// knapsack pricing, starting from 'initial' columns. Every round adds the column of
// negative reduced cost of each warehouse and computes the Lagrangian bound of the
// duals; it stops when no column has negative reduced cost, when the master LP
// value and the bound meet, or on time. The columns can seed the branch-and-price.
// 'start' (optional) is the initial dual point of the bundle method.
Result root(const Instance& inst, double time_limit, Solver solver,
            const std::vector<std::pair<int, std::vector<int>>>& initial,
            const std::vector<double>* start = nullptr);

// Proximal bundle method for the dual of the master, which is the Lagrangian
// dual L(pi) = sum_j min(pi_j, penalty) - sum_i KP_i(pi), with KP_i the knapsack
// of warehouse i for the profits pi_j - c_ij. Instead of the master LP, whose
// simplex method stalls in degenerate pivots on these set partitioning masters
// (thousands of pivots per round with the objective unchanged), each iteration
// solves the proximal master
//   max_pi  sum_j min(pi_j, p) - sum_i max_{S in B_i} (pi(S) - c_i(S)) - |pi - center|^2 / (2t)
// over the columns B_i generated so far. Its dual is a quadratic program over the
// convex weights of the columns of each warehouse and the u_j in [0, 1]: the
// master LP with the violation g = 1 - u - A lambda of the assignment rows
// penalized by (t/2)|g|^2 instead of forced to zero. It is solved by block
// coordinate descent: each u_j in closed form, and for each warehouse pairwise
// moves of weight between two of its columns with exact line search; the new
// point is center + t g. Its knapsacks give the exact value and new columns; the
// center moves when the gain is at least a tenth of the gain predicted by the
// model (serious step), and t doubles after good serious steps and shrinks after
// null steps. It stops when the predicted gain is below 1e-8 relative (the center
// is then optimal up to that tolerance) or on time. The bound is always valid.
// Bundle methods against the column generation with an LP master: Briant,
// Lemarechal, Meurdesoif, Michel, Perrot and Vanderbeck (2008).
Result bundle(const Instance& inst, double time_limit,
              const std::vector<std::pair<int, std::vector<int>>>& initial,
              const std::vector<double>* start = nullptr);

}

#endif

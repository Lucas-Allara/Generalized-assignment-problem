#ifndef BRANCH_AND_PRICE_H
#define BRANCH_AND_PRICE_H

#include <utility>
#include <vector>
#include "Instance.h"
#include "Solution.h"
#include "VariableFixing.h"

namespace branch_and_price {

// Techniques of the column generation and of the branching, each of which can be
// switched off (to measure what it contributes).
struct Options {
    // Only solutions cheaper than this are accepted (used after variable fixing,
    // where the reduced problem only contains the solutions cheaper than the
    // incumbent).
    double objective_limit = 1e300;
    // Early branching: the column generation of a node stops, and the node is
    // branched on, when more columns cannot raise its bound in integer terms
    // (integral costs) or when the bound has not improved for some rounds.
    bool early_termination = true;
    // More columns per round: the columns of the smoothed duals and of the LP
    // duals, and for each warehouse a second column without the least profitable
    // seller of the best one.
    bool extra_columns = true;
    // Columns that stay out of the basis leave the LP (SCIP keeps them in the
    // problem and prices them in again when their reduced cost becomes negative).
    bool column_aging = true;
    // Self-adjusting smoothing parameter (Pessoa, Sadykov, Uchoa and Vanderbeck,
    // 2018) instead of a fixed alpha = 0.8.
    bool automatic_smoothing = true;
    // Strong branching with Lagrangian bounds on the most fractional candidates.
    bool strong_branching = true;
    // Restricted master heuristic (price-and-branch) at the root.
    bool restricted_master_heuristic = true;
    // Re-fixing: Lagrangian variable fixing during the search (fixing::Refixer).
    // Every improvement of the incumbent rules out more pairs (i, j), which the
    // pricing skips and whose columns are fixed to zero, as if they were branching
    // decisions valid at every node; the improving solutions of the Lagrangian
    // heuristic go to SCIP.
    bool refixing = false;
    // Root started from the optimal duals of the master, computed by the bundle
    // method (column_generation::bundle, at most 20% of the time limit) from the
    // multipliers of the subgradient method, with its active columns. The fixing
    // of solve_with_fixing and the re-fixing then refine their multipliers with
    // the bundle method too.
    bool bundle_root = false;
    // Additional initial columns (warehouse, sorted sellers), for example those of
    // the column generation with an interior point method.
    const std::vector<std::pair<int, std::vector<int>>>* initial_columns = nullptr;
};

struct Result {
    Solution solution;            // best solution found (valid only if has_solution)
    bool has_solution = false;
    double primal = 1e300;        // cost of the best solution
    double dual = -1e300;         // lower bound
    double root_bound = -1e300;   // bound of the root node (column generation bound)
    bool optimal = false;
    bool no_better_solution = false;   // proved: nothing is cheaper than objective_limit
    double seconds = 0;
    long long nodes = 0;
    int columns = 0;              // columns generated, including the initial ones
    int heuristic_runs = 0;       // runs of the restricted master heuristic
    int heuristic_improvements = 0;   // runs that improved the incumbent
    int early_stops = 0;          // nodes whose column generation stopped early
    int mispricings = 0;          // rounds in which the smoothed duals gave no column
    long long lp_iterations = 0;  // simplex iterations of the master
    // statistics of the re-fixing (Options::refixing)
    long long pairs = 0;            // pairs (i, j) with d_ij <= b_i
    long long fixed_to_zero = 0;    // pairs fixed to 0 at the end
    int fixed_to_one = 0;           // sellers fixed to a warehouse at the end
    double first_fixed_pct = 0;     // percentage fixed to 0 with the first incumbent
    int refixing_incumbents = 0;    // improvements of the incumbent used for new fixings
    int lagrangian_solutions = 0;   // improving solutions of the Lagrangian heuristic
    double fixing_seconds = 0;
};

// Branch-and-price on the set partitioning formulation (Savelsbergh, 1997):
//   min  sum_p c_p lambda_p + penalty * sum_j u_j
//   s.t. sum_{p containing j} lambda_p + u_j = 1     for every seller j
//        sum_{p of warehouse i} lambda_p <= 1         for every warehouse i
//        lambda_p, u_j in {0, 1},
// where each column p is a set of sellers that fits in one warehouse and c_p is
// the cost of assigning them to it. The pricing problem is a 0-1 knapsack per
// warehouse, and the branching is on the original variables x_ij (seller j in
// warehouse i or not), which keeps the pricing problem a knapsack. 'start' gives
// the initial columns and the initial solution; without it, the best constructive
// solution improved with the VND is used. The multipliers of the subgradient
// method are the initial center of the dual stabilization of the root.
Result solve(const Instance& inst, double time_limit, const Solution* start = nullptr,
             const Options& options = Options());

// Branch-and-price on the instance reduced by Lagrangian variable fixing
// (fixing::reduce, with the cost of 'start' as the upper bound), which only keeps
// the solutions cheaper than 'start': the result is the best of both. The fixing
// takes at most a quarter of the time limit.
Result solve_with_fixing(const Instance& inst, double time_limit, const Solution& start,
                         const Options& options = Options(), fixing::Reduction* reduction = nullptr);

}

#endif

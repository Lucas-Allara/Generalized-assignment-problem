#ifndef MIP_MODEL_H
#define MIP_MODEL_H

#include <vector>
#include "Instance.h"
#include "Solution.h"
#include "VariableFixing.h"

namespace mip {

struct Result {
    Solution solution;            // best solution found (valid only if has_solution)
    bool has_solution = false;
    double primal = 1e300;        // cost of the best solution
    double dual = -1e300;         // lower bound
    bool optimal = false;
    bool no_better_solution = false;   // proved: nothing is cheaper than the objective limit
    double seconds = 0;
    long long nodes = 0;
};

// Compact integer model solved with SCIP:
//   min  sum_ij c_ij x_ij + penalty * sum_j u_j
//   s.t. sum_i x_ij + u_j = 1             for every seller j
//        sum_j d_ij x_ij <= b_i           for every warehouse i
//        x_ij, u_j in {0, 1},
// where u_j = 1 leaves seller j unassigned, with the same penalty as the
// heuristics. 'start' (optional) is given to SCIP as an initial solution. If
// 'objective_limit' is finite, only solutions cheaper than it are accepted.
Result solve(const Instance& inst, double time_limit, const Solution* start = nullptr,
             double objective_limit = 1e300);

// Compact model on the instance reduced by Lagrangian variable fixing
// (fixing::reduce, with the cost of 'start' as the upper bound, at most a quarter
// of the time limit), which only keeps the solutions cheaper than 'start': the
// result is the best of both, and its bound is valid for the original instance.
// 'reduction' (optional) receives the statistics of the fixing.
Result solve_with_fixing(const Instance& inst, double time_limit, const Solution& start,
                         fixing::Reduction* reduction = nullptr);

// Statistics of solve_with_refixing.
struct Refixing {
    long long pairs = 0;            // pairs (i, j) with d_ij <= b_i
    long long fixed_to_zero = 0;    // pairs fixed to 0 at the end
    int fixed_to_one = 0;           // sellers fixed to a warehouse at the end
    double first_fixed_pct = 0;     // percentage of the pairs fixed to 0 by the first
                                    // fixing, with the cost of 'start'
    int incumbents = 0;             // improvements of the incumbent used for new fixings
    int subgradient_runs = 0;       // passes of the subgradient method
    int lagrangian_solutions = 0;   // improving solutions of the Lagrangian heuristic
                                    // handed to SCIP
    double bound = -1e300;          // Lagrangian bound of the solutions cheaper than the
                                    // incumbent
    double seconds = 0;             // time spent in the fixing
    // Best solution of the Lagrangian heuristic (total_cost = +infinity if none):
    // SCIP may not commit it before the fixings prune every node.
    Solution lagrangian_best = no_solution();
    static Solution no_solution() { Solution s; s.total_cost = 1e300; return s; }
};

// Compact model with Lagrangian variable fixing inside SCIP (re-fixing): instead
// of fixing once with the cost of 'start' and solving the reduced instance, the
// whole model is solved with 'start' as its incumbent and a propagator fixes
// pairs globally every time the incumbent improves, with the penalties already
// computed (fixing::refix) and, when the upper bound has moved enough, with a new
// run of the subgradient method on the problem left by the fixings (at most a
// quarter of the time limit in all). The improving solutions of the Lagrangian
// heuristic go to SCIP. With a weak start almost nothing is fixed at first, but
// each better incumbent found by SCIP removes more pairs, and SCIP keeps its
// incumbent, its tree and its improvement heuristics (which the reduced instance
// of solve_with_fixing loses, since the incumbent is not a solution of it).
// With 'bundle', the multipliers of the fixing are refined by the bundle method.
Result solve_with_refixing(const Instance& inst, double time_limit, const Solution& start,
                           Refixing* stats = nullptr, bool bundle = false);

// Linear relaxation of the same model (x and u in [0, 1]); 'dual' is its optimum.
Result lp_relaxation(const Instance& inst, double time_limit);

// Rounding of the linear relaxation. In a basic optimal solution of the LP at most
// m sellers are split among several warehouses (Benders and van Nunen, 1983): the
// LP has n + m constraints, so at most n + m basic variables, and each seller
// needs at least one of them. The sellers with an integral LP value keep their
// warehouse, which respects every capacity, and the others are inserted with
// constructive::complete. The LP is solved without presolving and without
// heuristics, so that SCIP returns the basic solution of the simplex method.
struct Rounding {
    Solution solution;
    double lp_bound = -1e300;     // optimum of the LP
    int split = 0;                // sellers without an integral LP value
    double seconds = 0;
};
Rounding lp_rounding(const Instance& inst, double time_limit);

// Rounding of Shmoys and Tardos (1993). For each warehouse i, its sellers with
// x*_ij > 0 are sorted by decreasing demand and poured, in that order, into
// ceil(sum_j x*_ij) slots of size 1; the unassigned fractions u*_j go to a dummy
// warehouse in the same way. x* is then a fractional matching between sellers and
// slots that covers every seller, so a minimum cost matching that covers every
// seller costs at most the LP bound; the matching is solved as a linear program,
// whose basic solutions are integral (bipartite graph). In each warehouse the
// seller of slot k + 1 is not heavier than any seller of slot k, so the load is at
// most b_i + max_j d_ij: the assignment may exceed the capacities. 'repaired'
// removes from each overloaded warehouse its heaviest sellers until it fits and
// completes the solution with constructive::complete.
struct ShmoysTardos {
    Solution assignment;          // the rounded assignment (may exceed capacities)
    double cost = 0;              // its cost, at most the LP bound
    double lp_bound = -1e300;
    int overloaded = 0;           // warehouses above their capacity
    double largest_overload = 0;  // largest (load - b_i) / b_i
    bool within_bound = true;     // load_i <= b_i + max_{j: x*_ij > 0} d_ij for every i
    Solution repaired;            // feasible solution
    double seconds = 0;
};
ShmoysTardos shmoys_tardos(const Instance& inst, double time_limit);

// Root node of the compact model with the default settings of SCIP (presolving,
// cutting planes and heuristics): 'dual' is the bound at the end of the root.
Result root_node(const Instance& inst, double time_limit);

// Linear relaxation strengthened only with cover inequalities, separated by a
// separator of our own and without presolving, heuristics or other cutting planes.
// For a warehouse i, a cover is a set C of sellers with sum_{j in C} d_ij > b_i,
// and sum_{j in E(C)} x_ij <= |C| - 1 is valid, where the extension E(C) adds the
// sellers whose demand in i is at least the largest one of C. The most violated
// cover for an LP solution x* minimizes sum_{j in C} (1 - x*_ij), a 0-1 knapsack
// over the sellers with x*_ij > 0, like the pricing problem of the
// branch-and-price. Every cover inequality is valid for the convex hull of the
// knapsack of its warehouse, so the bound lies between the LP bound and the
// Lagrangian bound of the assignment constraints (Geoffrion, 1974).
struct CutResult {
    double bound = -1e300;
    int cuts = 0;                 // inequalities added
    int rounds = 0;               // separation rounds
    double seconds = 0;
};
CutResult cover_cuts(const Instance& inst, double time_limit);

// Linear relaxation strengthened with exact separation of the knapsack polytopes
// (the idea of Avella, Boccia and Vasilyev, 2010), with the same settings as
// cover_cuts. For a warehouse i and the LP solution x*, the most violated valid
// inequality pi x <= pi_0 of conv{x in {0,1}^n : sum_j d_ij x_j <= b_i}, with
// 0 <= pi_j <= 1, solves the separation LP
//   max sum_j pi_j x*_ij - pi_0   s.t.  sum_{j in s} pi_j <= pi_0 for every feasible set s,
// over the sellers with x*_ij > 0. Its constraints are generated as needed: the
// most violated one is a knapsack with profits pi (the pricing problem again), and
// the LP is solved with HiGHS. A positive optimum gives a violated inequality,
// valid with zero coefficients for the other sellers (any feasible set restricted
// to the support is feasible); a zero optimum proves that x* restricted to i lies
// in the convex hull. Separating every warehouse until no inequality is violated
// gives the bound of the Dantzig-Wolfe master, that is, the Lagrangian dual.
CutResult exact_knapsack_cuts(const Instance& inst, double time_limit);

// Branch-and-cut: the compact model with the default settings of SCIP plus the
// exact knapsack separator at the root ('start' and 'objective_limit' as in solve).
Result branch_and_cut(const Instance& inst, double time_limit, const Solution* start = nullptr,
                      double objective_limit = 1e300);

// Shadow prices of the capacities: the duals y_i <= 0 of the capacity constraints
// in the LP relaxation, solved with the simplex method of HiGHS. -y_i is the cost
// saved per unit of capacity added to warehouse i, as long as the optimal basis
// does not change; for the integer problem it is only an estimate.
struct CapacityDuals {
    std::vector<double> shadow_price;   // -y_i >= 0 for each warehouse
    double lp_bound = -1e300;
    bool solved = false;
};
CapacityDuals capacity_duals(const Instance& inst, double time_limit);

// Subproblem of the matheuristic: decides the warehouses of 'free_sellers' only,
// with the other sellers of 'sol' fixed (they use up part of the capacity). The
// current assignment of the free sellers is given as an initial solution, so the
// result is never worse. Applies the best solution found to 'sol' and returns the
// change in its cost (zero or negative).
double reoptimize(const Instance& inst, Solution& sol, const std::vector<int>& free_sellers,
                  double time_limit);

}

#endif

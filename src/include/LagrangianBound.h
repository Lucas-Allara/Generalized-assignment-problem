#ifndef LAGRANGIAN_BOUND_H
#define LAGRANGIAN_BOUND_H

#include <utility>
#include <vector>
#include "Instance.h"
#include "Solution.h"

namespace lagrangian {

struct Result {
    double bound = 0;                  // best lower bound found
    int iterations = 0;
    double seconds = 0;
    bool optimal = false;              // the Lagrangian solution was feasible (zero subgradient)
    std::vector<double> multipliers;   // multipliers of the best bound
    Solution solution;                 // best solution of the Lagrangian heuristic
    bool has_solution = false;         // (only if it was enabled)
};

// Lagrangian heuristic: turns the knapsack solutions of the relaxation (one set of
// sellers per warehouse, each within its capacity) into a feasible solution. A
// seller chosen by several warehouses stays in the cheapest of them, which only
// frees capacity in the others; the sellers chosen by none are then inserted with
// constructive::complete, and those that fit nowhere stay unassigned.
Solution repair(const Instance& inst, const std::vector<std::vector<int>>& sets);

// Value of the Lagrangian function for the given multipliers (one per seller) and
// the subgradient at that point. The assignment constraints
// sum_i x_ij + u_j = 1 are relaxed, which leaves one 0-1 knapsack per warehouse
// (Fisher, Jaikumar and Van Wassenhove, 1986):
//   L(lambda) = sum_j lambda_j - sum_i KP_i(lambda) + sum_j min(0, penalty - lambda_j),
// with KP_i(lambda) = max sum_j (lambda_j - c_ij) x_ij s.t. sum_j d_ij x_ij <= b_i.
// If 'sets' is given, it receives the knapsack solution of each warehouse.
double evaluate(const Instance& inst, const std::vector<double>& lambda,
                std::vector<double>& subgradient,
                std::vector<std::vector<int>>* sets = nullptr);

// Subgradient optimization of the multipliers with the Polyak step
// t = mu (upper_bound - L) / ||g||^2. mu starts at 2 and is halved after
// 'patience' iterations without improvement; it stops when mu < 1e-4, after
// 'max_iterations' or after 'seconds' (if positive). 'upper_bound' is the cost
// of a known solution: the closer it is to the optimum, the closer the bound gets
// to the Lagrangian dual (with a poor solution the steps are too long). If 'columns' is given, it receives the distinct knapsack
// solutions (warehouse, sellers) of the last 'last_iterations' iterations: they
// are good initial columns for the branch-and-price. If 'heuristic_every' > 0, the
// Lagrangian heuristic runs every that many iterations; its best solution is
// returned and its cost replaces 'upper_bound' in the step whenever it is lower.
Result subgradient(const Instance& inst, double upper_bound, int max_iterations = 2000,
                   double seconds = 0, int patience = 50,
                   std::vector<std::pair<int, std::vector<int>>>* columns = nullptr,
                   int last_iterations = 100, int heuristic_every = 0);

// Initial solution built from the relaxation: the best constructive solution
// gives the first upper bound, the subgradient method runs with the Lagrangian
// heuristic (every 10 iterations) for at most 'subgradient_seconds' (until it
// converges if zero), and the best of the constructive and the Lagrangian
// solutions is improved with the VND until 'total_seconds' have passed since the
// call (without limit if zero).
Solution heuristic_solution(const Instance& inst, double subgradient_seconds, double total_seconds);

}

#endif

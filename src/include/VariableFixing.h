#ifndef VARIABLE_FIXING_H
#define VARIABLE_FIXING_H

#include <vector>
#include "Instance.h"
#include "Solution.h"

namespace fixing {

// Lagrangian penalties of a multiplier vector lambda (relaxation of the assignment
// constraints, one knapsack per warehouse). For every pair (i, j):
//   with[i][j]    = Lagrangian bound of the problem with x_ij = 1, which also implies
//                   x_kj = 0 for k != i and u_j = 0 (j leaves the other knapsacks);
//   without[i][j] = Lagrangian bound of the problem with x_ij = 0.
// Both need the value of each knapsack with one item forced in or out. A forward
// and a backward dynamic program over the items give them all in O(k b_i) per
// warehouse (k items), the time of a single knapsack: forcing item t in or out
// combines the best values of the items before t and after t over every split of
// the capacity. Pairs that do not fit (d_ij > b_i) get +infinity. Returns L(lambda).
double penalties(const Instance& inst, const std::vector<double>& lambda,
                 std::vector<std::vector<double>>& with, std::vector<std::vector<double>>& without);

// Reduced problem: the pairs fixed by the penalties. x_ij is fixed to 0 when the
// bound with x_ij = 1 shows that no solution cheaper than 'upper_bound' assigns j
// to i, and seller j is fixed to warehouse i when the bound with x_ij = 0 shows it
// for every other choice. The reduced instance keeps only the free sellers, with
// the capacities left by the fixed ones, and marks the forbidden pairs with a
// demand larger than the capacity, so that every method of the project skips them.
// Only solutions cheaper than the upper bound survive (with integral costs, at
// least 1 cheaper): the optimum is the upper bound itself or the optimum of the
// reduced instance plus 'offset'.
struct Reduction {
    Instance instance;                // reduced instance
    std::vector<int> original;        // original index of each seller of the reduced instance
    std::vector<int> forced;          // forced warehouse of each original seller (-1: free)
    double offset = 0;                // cost of the forced assignments
    bool no_improving_solution = false;   // the fixings rule out every cheaper solution
    double upper_bound = 1e300;       // best upper bound known at the end
    double bound = -1e300;            // Lagrangian bound of the solutions cheaper than the
                                      // upper bound: the optimum is at least
                                      // min(upper_bound, bound)
    long long pairs = 0;              // pairs (i, j) with d_ij <= b_i in the original instance
    long long fixed_to_zero = 0;      // pairs fixed to 0
    int fixed_to_one = 0;             // sellers fixed to a warehouse
    int passes = 0;
    double seconds = 0;
    std::vector<std::vector<char>> allowed;   // allowed[j][i]: seller j may still go to warehouse i
    // Lagrangian bounds of the last pass with x_ij = 1 and with x_ij = 0, in the
    // original indices ([j][i], -infinity where none was computed). They stay valid
    // lower bounds for every solution that respects the fixings, so a lower upper
    // bound fixes more pairs with them without solving the relaxation again (refix).
    std::vector<std::vector<double>> with, without;
};

// Repeats, at most 'passes' times or until nothing new is fixed: the subgradient
// method (with the Lagrangian heuristic, which may lower the upper bound) on the
// current reduced instance, and the fixing of the pairs its penalties allow. Each
// pass runs on a smaller problem, whose Lagrangian bound is at least as high, so it
// may fix more pairs. 'best' receives the best solution known (the one of cost
// 'upper_bound' if given, or a better one found by the Lagrangian heuristic, which
// 'lagrangian_heuristic' = false switches off). With 'from', it continues a
// previous reduction: its fixings are kept (they remain valid for any lower upper
// bound), its penalties are first tested against the new upper bound (refix), and
// the statistics accumulate. With 'bundle', the multipliers of each pass are
// refined by the bundle method (column_generation::bundle) up to the optimal
// duals of the master, whose penalties are stronger: the subgradient method
// often stops short of them (on type d, far short).
Reduction reduce(const Instance& inst, double upper_bound, double seconds, int passes = 3,
                 Solution* best = nullptr, bool lagrangian_heuristic = true,
                 const Reduction* from = nullptr, bool bundle = false);

// Fixings that a lower upper bound allows with the penalties of the last pass of
// 'r', without running the subgradient method: x_ij is fixed to 0 when its bound
// with x_ij = 1 reaches the new upper bound (with integral costs, exceeds it minus
// 1), and seller j is fixed to i when its bound with x_ij = 0 does. Each test
// costs O(1), so it can run after every improvement of the incumbent. Updates 'r'
// (reduced instance included) and returns the number of new fixings.
int refix(const Instance& inst, Reduction& r, double upper_bound);

// Re-fixing during a search: keeps the reduction of the current incumbent and
// updates it whenever a cheaper incumbent appears. The first update runs reduce
// (three passes); each later one tests the stored penalties against the new
// upper bound (refix) and, when the upper bound has closed at least a tenth of
// the gap left at the last run of the subgradient method and the time budget
// allows it, runs two more passes on the problem left by the fixings (with half
// of the budget left). Used by the compact model and by the branch-and-price.
class Refixer {
public:
    Refixer(const Instance& inst, double budget, bool bundle = false)
        : inst_(inst), budget_(budget), bundle_(bundle) {}

    // New upper bound (cost of the incumbent). Returns whether the fixings
    // changed. If the Lagrangian heuristic found a solution cheaper than
    // 'upper_bound', 'best' receives it (otherwise best.total_cost = +infinity).
    bool update(double upper_bound, Solution& best);

    // Whether x_ij = 0 in every solution cheaper than the incumbent.
    bool ruled_out(int j, int i) const {
        return started_ && (!r_.allowed[j][i] || (r_.forced[j] >= 0 && r_.forced[j] != i));
    }
    int forced(int j) const { return started_ ? r_.forced[j] : -1; }
    bool no_improving_solution() const { return r_.no_improving_solution; }
    const Reduction& reduction() const { return r_; }
    long long ruled_out_pairs() const;   // pairs with d_ij <= b_i that are ruled out
    long long version() const { return version_; }   // number of updates that changed the fixings

    double first_fixed_pct = 0;   // percentage of the pairs fixed to 0 by the first update
    int incumbents = 0;           // later updates (improvements of the incumbent)

private:
    const Instance& inst_;
    double budget_;
    bool bundle_;
    Reduction r_;
    bool started_ = false;
    double tested_ = 1e300;
    double last_run_ub_ = 1e300;
    long long version_ = 0;
};

// Solution of the original instance from a solution of the reduced one.
Solution expand(const Instance& inst, const Reduction& reduction, const Solution& reduced);

// Whether the costs (and the penalty) are integers: then a cheaper solution is
// cheaper by at least 1.
bool integral_costs(const Instance& inst);

}

#endif

#ifndef METAHEURISTICS_H
#define METAHEURISTICS_H

#include "Instance.h"
#include "Solution.h"

namespace metaheuristics {

// recomputes the total cost of a solution from scratch: the distances of the
// assigned sellers plus the penalty of each unassigned one.
double evaluate(const Solution& sol, const Instance& inst);

// best of the constructive heuristics: nearest neighbor and the regret heuristic
// with the cost, the demand and the relative demand.
Solution best_constructive(const Instance& inst);

// best_constructive improved with the VND. If 'seconds' > 0, the VND stops after
// that time.
Solution initial_solution(const Instance& inst, double seconds = 0);

// local search: chains every neighborhood until a local optimum of all of them.
// If 'seconds' > 0, it stops after that time (between two moves), possibly before
// reaching the local optimum.
void vnd(Solution& sol, const Instance& inst, double seconds = 0);

// ILS: starts from a constructive solution + VND and repeats (perturb + VND),
// always keeping the best solution. 'strength' = number of random moves.
// If 'seconds' > 0, it stops on time instead of on the number of iterations; the
// time includes the construction of the initial solution.
Solution ils(const Instance& inst, int iterations = 100, int strength = 10, unsigned seed = 1, double seconds = 0);

// LNS (ruin and recreate): unassigns 'destroy' random sellers and lets the VND
// reinsert and optimize them. If 'seconds' > 0, it stops on time (including the
// construction of the initial solution).
// If 'start' is given, it starts from that solution instead of initial_solution.
Solution lns(const Instance& inst, int iterations = 100, int destroy = 50, unsigned seed = 1, double seconds = 0,
             const Solution* start = nullptr);

}

#endif

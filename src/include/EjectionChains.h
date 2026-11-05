#ifndef EJECTION_CHAINS_H
#define EJECTION_CHAINS_H

#include "Instance.h"
#include "Solution.h"

namespace ejection {

// Tabu search through infeasible assignments with ejection chains, inspired by
// Yagiura, Ibaraki and Glover (2004) for the GAP (a simpler variant with its own
// constants). The capacities may be exceeded: the search minimizes
//   cost + sum_i alpha_i * max(0, load_i - b_i),
// where the weights alpha_i adapt at every iteration (strategic oscillation):
// those of the overloaded warehouses grow by 5%, and all of them shrink by 2%
// when nothing is overloaded. On tight instances (type d) a search that never
// leaves the feasible region gets stuck, because every improving exchange first
// needs some free capacity; with the weights it crosses the boundary and comes
// back elsewhere. The moves are shift (a seller to another warehouse or
// unassigned), swap (two sellers of different warehouses) and ejection chains of
// length 2 (j moves to i', which ejects a seller of i' to a third warehouse or
// leaves it unassigned), restricted to the 'candidates' cheapest warehouses of
// each seller. Each iteration applies the best move of a random sample of
// 'sample' sellers by the weighted objective, even if it is worse; a seller may
// not return to the warehouse it left for a random tenure of 5 to 15 iterations,
// unless the move gives a feasible solution cheaper than the best one. Every
// feasible assignment visited is a candidate for the best solution, and so is,
// every 10 iterations, a greedy repair of the current infeasible one. After
// 'patience' iterations without a new best solution, it restarts from the best
// with 'kick' random shifts. Starts from 'start' if given (otherwise from the best
// constructive solution) and returns the best feasible solution, improved with
// the VND.
struct Options {
    int candidates = 20;     // cheapest warehouses considered for each seller
    int sample = 200;        // sellers whose moves are evaluated at each iteration
    int patience = 2000;     // iterations without a new best before a restart
    int kick = 10;           // random shifts of a restart
    long long iterations = 20000;   // when 'seconds' <= 0
};

Solution search(const Instance& inst, double seconds, unsigned seed = 1, const Solution* start = nullptr,
                const Options& options = Options());

}

#endif

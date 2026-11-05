#include "EjectionChains.h"
#include "Metaheuristics.h"
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>

using namespace std;

namespace ejection {

namespace {

double elapsed(chrono::high_resolution_clock::time_point t0) {
    return chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - t0).count() / 1000.0;
}

// State of the search: the warehouse of each seller (m = unassigned), the load of
// each warehouse, the members of each warehouse and the two parts of the
// objective. Unassigned sellers pay the penalty and use no capacity.
class Search {
public:
    Search(const Instance& inst, const Options& options, unsigned seed)
        : inst_(inst), n_(inst.sellers.size()), m_(inst.warehouses.size()), rng_(seed) {
        capacity_.resize(m_);
        for (int i = 0; i < m_; i++) capacity_[i] = inst.warehouses[i].capacity;
        // candidate warehouses of each seller: the cheapest ones where it fits alone
        candidates_.assign(n_, {});
        for (int j = 0; j < n_; j++) {
            vector<int> order;
            for (int i = 0; i < m_; i++)
                if (inst.demand(j, i) <= capacity_[i]) order.push_back(i);
            sort(order.begin(), order.end(), [&](int a, int b) { return inst.cost[j][a] < inst.cost[j][b]; });
            if ((int)order.size() > options.candidates) order.resize(options.candidates);
            candidates_[j] = order;
        }
        // initial weight: the typical saving in cost per unit of demand
        double spread = 0, demand = 0;
        for (int j = 0; j < n_; j++) {
            double lo = 1e300, hi = -1e300;
            for (int i = 0; i < m_; i++) {
                lo = min(lo, inst.cost[j][i]);
                hi = max(hi, inst.cost[j][i]);
                demand += inst.demand(j, i);
            }
            spread += hi - lo;
        }
        initial_weight_ = max(1e-3, spread / n_ / max(1.0, demand / (n_ * m_)));
        weight_.assign(m_, initial_weight_);
        tabu_until_.assign((size_t)n_ * (m_ + 1), 0);
    }

    void load(const Solution& sol) {
        warehouse_.assign(n_, m_);
        members_.assign(m_ + 1, {});
        position_.assign(n_, -1);
        load_.assign(m_, 0);
        cost_ = 0;
        excess_ = 0;
        for (int j = 0; j < n_; j++) {
            int i = sol.assigned_warehouse[j];
            place(j, i >= 0 ? i : m_);
        }
    }

    Solution solution() const {
        Solution sol;
        sol.sellers_per_warehouse.assign(m_, {});
        sol.assigned_warehouse.assign(n_, -1);
        for (int j = 0; j < n_; j++)
            if (warehouse_[j] < m_) sol.assign(warehouse_[j], j);
        sol.total_cost = cost_;
        return sol;
    }

    double cost() const { return cost_; }

    // One iteration of tabu search: the best move (shift, swap or ejection chain)
    // of a random sample of sellers by the weighted objective, even if it is worse,
    // except moves that send a seller back to a warehouse it left less than
    // 'tenure' iterations ago (unless the move gives a feasible solution cheaper
    // than 'best_cost').
    void tabu_step(long long iteration, double best_cost, int sample) {
        struct Move { int j, t, k, r; double delta; };   // k = -1: shift; r = -1: swap
        Move best{-1, -1, -1, -1, 1e300};
        vector<int> sellers(n_);
        for (int j = 0; j < n_; j++) sellers[j] = j;
        if (sample < n_) {
            for (int q = 0; q < sample; q++) swap(sellers[q], sellers[q + rng_() % (n_ - q)]);
            sellers.resize(sample);
        }
        auto allowed = [&](int j, int to, double delta, long long excess_after) {
            if (tabu_until_[(size_t)j * (m_ + 1) + to] <= iteration) return true;
            return excess_after == 0 && cost_ + delta < best_cost - 1e-9;   // aspiration (approximate)
        };
        for (int j : sellers) {
            int s = warehouse_[j];
            auto consider = [&](const Move& mv, double cost_delta, double pen, long long excess_after, int to) {
                double delta = cost_delta + pen;
                if (delta < best.delta - 1e-12 && allowed(j, to, cost_delta, excess_after)) {
                    best = mv;
                    best.delta = delta;
                }
            };
            // shifts
            for (int idx = -1; idx < (int)candidates_[j].size(); idx++) {
                int t = idx < 0 ? m_ : candidates_[j][idx];
                if (t == s) continue;
                long long ex = excess_ - (s < m_ ? over(s, load_[s]) - over(s, load_[s] - d(j, s)) : 0)
                             + (t < m_ ? over(t, load_[t] + d(j, t)) - over(t, load_[t]) : 0);
                consider({j, t, -1, -1, 0}, c(j, t) - c(j, s),
                         penalty_change(s, -d(j, s)) + penalty_change(t, d(j, t)), ex, t);
            }
            if (s == m_) continue;
            for (int t : candidates_[j]) {
                if (t == s) continue;
                for (int k : members_[t]) {
                    // swap
                    long long ls = load_[s] - d(j, s) + d(k, s), lt = load_[t] + d(j, t) - d(k, t);
                    long long ex = excess_ - over(s, load_[s]) - over(t, load_[t]) + over(s, ls) + over(t, lt);
                    consider({j, t, k, -1, 0}, c(j, t) - c(j, s) + c(k, s) - c(k, t),
                             weight_[s] * (over(s, ls) - over(s, load_[s])) + weight_[t] * (over(t, lt) - over(t, load_[t])),
                             ex, t);
                    // ejection chains: k leaves t for r
                    long long ls2 = load_[s] - d(j, s);
                    double base_pen = weight_[s] * (over(s, ls2) - over(s, load_[s]))
                                    + weight_[t] * (over(t, lt) - over(t, load_[t]));
                    long long base_ex = excess_ - over(s, load_[s]) - over(t, load_[t]) + over(s, ls2) + over(t, lt);
                    for (int idx = -1; idx < (int)candidates_[k].size(); idx++) {
                        int r = idx < 0 ? m_ : candidates_[k][idx];
                        if (r == s || r == t) continue;
                        long long ex2 = base_ex + (r < m_ ? over(r, load_[r] + d(k, r)) - over(r, load_[r]) : 0);
                        consider({j, t, k, r, 0}, c(j, t) - c(j, s) + c(k, r) - c(k, t),
                                 base_pen + penalty_change(r, d(k, r)), ex2, t);
                    }
                }
            }
        }
        if (best.j < 0) return;
        int s = warehouse_[best.j];
        int tenure = 5 + rng_() % 11;
        tabu_until_[(size_t)best.j * (m_ + 1) + s] = iteration + tenure;
        move(best.j, best.t);
        if (best.k >= 0) {
            int to = best.r < 0 ? s : best.r;
            tabu_until_[(size_t)best.k * (m_ + 1) + best.t] = iteration + tenure;
            move(best.k, to);
        }
    }

    bool feasible() const { return excess_ == 0; }

    // Every iteration: the weights of the overloaded warehouses grow by 5% and,
    // when nothing is overloaded, all of them shrink by 2%.
    void adapt_weights() {
        if (excess_ > 0) {
            for (int i = 0; i < m_; i++)
                if (load_[i] > capacity_[i]) weight_[i] *= 1.05;
        } else {
            for (double& w : weight_) w = max(1e-3 * initial_weight_, 0.98 * w);
        }
    }

    // Feasible solution from the current state, which is left unchanged: the
    // sellers of each overloaded warehouse leave it, the cheapest exit per unit of
    // demand first, for the cheapest warehouse with room (or unassigned) until it
    // fits. Returns its cost; 'out' receives the warehouse of each seller.
    double repair(vector<int>& out) const {
        out = warehouse_;
        vector<long long> load = load_;
        double cost = cost_;
        for (int i = 0; i < m_; i++) {
            while (load[i] > capacity_[i]) {
                int chosen = -1, target = m_;
                double best_rate = 1e300;
                for (int j : members_[i]) {
                    if (out[j] != i) continue;
                    // cheapest destination of j with room
                    int to = m_;
                    double to_cost = inst_.penalty;
                    for (int t : candidates_[j]) {
                        if (t != i && load[t] + inst_.demand(j, t) <= capacity_[t] && inst_.cost[j][t] < to_cost) {
                            to = t;
                            to_cost = inst_.cost[j][t];
                        }
                    }
                    double rate = (to_cost - inst_.cost[j][i]) / inst_.demand(j, i);
                    if (rate < best_rate) {
                        best_rate = rate;
                        chosen = j;
                        target = to;
                    }
                }
                if (chosen < 0) break;
                load[i] -= inst_.demand(chosen, i);
                cost += c(chosen, target) - inst_.cost[chosen][i];
                if (target < m_) load[target] += inst_.demand(chosen, target);
                out[chosen] = target;
            }
        }
        return cost;
    }

    Solution from(const vector<int>& assignment) const {
        Solution sol;
        sol.sellers_per_warehouse.assign(m_, {});
        sol.assigned_warehouse.assign(n_, -1);
        for (int j = 0; j < n_; j++)
            if (assignment[j] < m_) sol.assign(assignment[j], j);
        sol.total_cost = 0;
        for (int j = 0; j < n_; j++) sol.total_cost += c(j, assignment[j]);
        return sol;
    }

    // random shifts to candidate warehouses (a perturbation)
    void kick(int moves) {
        for (int t = 0; t < moves; t++) {
            int j = rng_() % n_;
            const vector<int>& c = candidates_[j];
            if (c.empty()) continue;
            int i = c[rng_() % c.size()];
            if (i != warehouse_[j]) move(j, i);
        }
    }

private:
    const Instance& inst_;
    int n_, m_;
    mt19937 rng_;
    vector<long long> capacity_;
    vector<vector<int>> candidates_;
    vector<double> weight_;
    double initial_weight_ = 1;
    vector<int> warehouse_;
    vector<vector<int>> members_;
    vector<int> position_;
    vector<long long> load_;
    double cost_ = 0;
    long long excess_ = 0;   // total excess, kept up to date
    vector<long long> tabu_until_;   // [j * (m + 1) + i]: iteration until which j may not return to i

    double c(int j, int i) const { return i < m_ ? inst_.cost[j][i] : inst_.penalty; }
    int d(int j, int i) const { return i < m_ ? inst_.demand(j, i) : 0; }
    long long over(int i, long long load) const { return max(0LL, load - capacity_[i]); }
    // change of the weighted excess of warehouse i when its load changes by 'delta'
    double penalty_change(int i, long long delta) const {
        if (i >= m_ || delta == 0) return 0;
        return weight_[i] * (over(i, load_[i] + delta) - over(i, load_[i]));
    }

    void place(int j, int i) {
        warehouse_[j] = i;
        position_[j] = members_[i].size();
        members_[i].push_back(j);
        cost_ += c(j, i);
        if (i < m_) {
            excess_ -= over(i, load_[i]);
            load_[i] += d(j, i);
            excess_ += over(i, load_[i]);
        }
    }

    void unplace(int j) {
        int i = warehouse_[j];
        vector<int>& list = members_[i];
        int last = list.back();
        list[position_[j]] = last;
        position_[last] = position_[j];
        list.pop_back();
        cost_ -= c(j, i);
        if (i < m_) {
            excess_ -= over(i, load_[i]);
            load_[i] -= d(j, i);
            excess_ += over(i, load_[i]);
        }
    }

    void move(int j, int i) {
        unplace(j);
        place(j, i);
    }
};

}

Solution search(const Instance& inst, double seconds, unsigned seed, const Solution* start, const Options& options) {
    auto t0 = chrono::high_resolution_clock::now();
    Solution best = start != nullptr ? *start : metaheuristics::best_constructive(inst);
    best.total_cost = metaheuristics::evaluate(best, inst);
    Search search(inst, options, seed);
    search.load(best);
    long long since = 0;   // iterations without a new best solution
    auto visit = [&] {
        if (search.cost() < best.total_cost - 1e-9) {
            best = search.solution();
            since = 0;
        }
    };
    int sample = min((int)inst.sellers.size(), options.sample);
    for (long long it = 0; seconds <= 0 ? it < options.iterations : elapsed(t0) < seconds; it++) {
        search.tabu_step(it, best.total_cost, sample);
        if (search.feasible()) {
            visit();
        } else if (it % 10 == 0) {
            vector<int> repaired;
            if (search.repair(repaired) < best.total_cost - 1e-9) {
                best = search.from(repaired);
                since = 0;
            }
        }
        search.adapt_weights();
        if (++since >= options.patience) {
            search.load(best);
            search.kick(options.kick);
            since = 0;
        }
    }
    metaheuristics::vnd(best, inst, seconds > 0 ? max(1e-3, seconds - elapsed(t0)) : 0);
    best.total_cost = metaheuristics::evaluate(best, inst);
    return best;
}

}

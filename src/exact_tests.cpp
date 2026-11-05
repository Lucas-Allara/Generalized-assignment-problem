// =============================================================================
//  exact_tests.cpp - tests and experiments of the exact methods, the lower
//  bounds and the matheuristic (they need SCIP)
//
//  BUILD (from src/):  make exact
//  RUN:
//    ./exact_tests check            the exact methods against brute force and
//                                   against each other on random instances
//    ./exact_tests e7 [threads]     relaxations, their heuristics and the exact methods
//    ./exact_tests e8 [threads]     tuning of the matheuristic
//    ./exact_tests e9 [threads]     LNS against the matheuristic on the 58 instances
//    ./exact_tests e11 [threads]    covers and exact knapsack cuts against the Lagrangian bound
//                                   (reads e7)
//    ./exact_tests e12 [threads]    rounding of Shmoys and Tardos against the simple one
//    ./exact_tests e13 [threads]    ablation of the techniques of the branch-and-price
//    ./exact_tests e14 [threads]    simplex, interior point and bundle method for the root master
//    ./exact_tests e15 [threads]    shadow prices and sensitivity of the real instance
//    ./exact_tests e16 [threads]    better start (MIP-LNS) and re-fixing against the fixing of E7
//                                   (reads e7)
//    ./exact_tests e17 [threads]    stronger heuristics for type d and the bundle master on the
//                                   instances still open (reads the table of gaps)
//    ./exact_tests refixing         only the correctness tests of the re-fixing
//    ./exact_tests bundle           only those of the bundle method, the tabu search and the portfolio
//    ./exact_tests gaps             true gaps of every method (reads e5, e7, e9 and, if present,
//                                   e16 and e17 for the best bounds)
//
//  The instances are distributed among 'threads' threads (default 8), each with
//  its own SCIP; the times are therefore measured with several runs in parallel.
//  The CSV files are written to src/results/.
// =============================================================================

#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "BranchAndPrice.h"
#include "ConstructiveHeuristics.h"
#include "EjectionChains.h"
#include "ColumnGeneration.h"
#include "Experiments.h"
#include "Instance.h"
#include "Knapsack.h"
#include "LagrangianBound.h"
#include "Matheuristic.h"
#include "Metaheuristics.h"
#include "MipModel.h"
#include "VariableFixing.h"

using namespace std;
using namespace experiments;

// time limit (seconds) of the compact model and of the branch-and-price in E7
static const double EXACT_TIME_LIMIT = 600;

// configuration of the matheuristic in E9, chosen with the tuning of E8: the
// Lagrangian start was the best on all five tuning instances (from the
// constructive solution the instances of type e stay far from the optimum, and on
// e801600 the LP takes too much of the budget for the rounding), and freeing the
// sellers of related warehouses, 100 at a time, had the lowest mean excess over the
// best configuration (0.10%, tied with the mixed operator). The size is capped at
// half of the sellers so that the sub-MIP remains a neighborhood.
static const matheuristic::Start MIP_LNS_START = matheuristic::Start::LAGRANGIAN;
static const matheuristic::Destroy MIP_LNS_OPERATOR = matheuristic::Destroy::WAREHOUSES;
static int mip_lns_destroy(int n) { return min(100, n / 2); }

static mutex g_print;
static int g_pass = 0, g_fail = 0;
static void check(bool ok, const string& desc) {
    lock_guard<mutex> lock(g_print);
    if (ok) { g_pass++; }
    else { g_fail++; cout << "  [FAIL] " << desc << "\n"; }
}

// Runs task(k) for k = 0..count-1 on 'threads' threads.
static void parallel_for(int count, int threads, const function<void(int)>& task) {
    atomic<int> next(0);
    vector<thread> pool;
    for (int t = 0; t < max(1, threads); t++) {
        pool.emplace_back([&] {
            for (int k = next++; k < count; k = next++) task(k);
        });
    }
    for (thread& th : pool) th.join();
}

// Results of the tasks already finished by an interrupted run: each task appends a
// line "key<TAB>value" to results/<name>.partial when it ends, and a new run of the
// same experiment reuses those values. The file is removed when the experiment ends.
class Checkpoint {
public:
    explicit Checkpoint(const string& name) : path_(output(name + ".partial")) {
        ifstream input(path_);
        string line;
        while (getline(input, line)) {
            size_t tab = line.find('\t');
            if (tab != string::npos) done_[line.substr(0, tab)] = line.substr(tab + 1);
        }
        if (!done_.empty()) cout << "  resuming: " << done_.size() << " tasks already done\n";
        out_.open(path_, ios::app);
    }
    bool get(const string& key, string& value) {
        lock_guard<mutex> lock(mutex_);
        auto it = done_.find(key);
        if (it == done_.end()) return false;
        value = it->second;
        return true;
    }
    void put(const string& key, const string& value) {
        lock_guard<mutex> lock(mutex_);
        done_[key] = value;
        out_ << key << '\t' << value << '\n';
        out_.flush();
    }
    void finish() {
        out_.close();
        remove(path_.c_str());
    }

private:
    string path_;
    map<string, string> done_;
    ofstream out_;
    mutex mutex_;
};

static bool integral_costs(const Instance& inst) {
    for (const auto& row : inst.cost)
        for (double c : row)
            if (c != floor(c)) return false;
    return true;
}

// -----------------------------------------------------------------------------
//  Correctness tests on random instances
// -----------------------------------------------------------------------------

// random instance with integer costs in [10, 50], demands in [5, 25] and each
// capacity equal to 'tight' times the average demand placed on a warehouse
static string random_instance(const string& path, int m, int n, double tight, mt19937& rng) {
    ofstream f(path);
    f << m << " " << n << "\n";
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) f << 10 + rng() % 41 << " ";
        f << "\n";
    }
    vector<vector<int>> d(m, vector<int>(n));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            d[i][j] = 5 + rng() % 21;
            f << d[i][j] << " ";
        }
        f << "\n";
    }
    for (int i = 0; i < m; i++) {
        int total = 0;
        for (int j = 0; j < n; j++) total += d[i][j];
        f << max(1, (int)(tight * total / m)) << " ";
    }
    f << "\n";
    return path;
}

// optimum by enumeration (each seller in a warehouse with room or unassigned)
static double brute_force(const Instance& inst) {
    int m = inst.warehouses.size(), n = inst.sellers.size();
    double best = 1e300;
    vector<int> capacity;
    for (const Warehouse& w : inst.warehouses) capacity.push_back(w.capacity);
    function<void(int, double)> search = [&](int j, double cost) {
        if (cost >= best) return;
        if (j == n) { best = cost; return; }
        search(j + 1, cost + inst.penalty);
        for (int i = 0; i < m; i++) {
            if (capacity[i] < inst.demand(j, i)) continue;
            capacity[i] -= inst.demand(j, i);
            search(j + 1, cost + inst.cost[j][i]);
            capacity[i] += inst.demand(j, i);
        }
    };
    search(0, 0);
    return best;
}

// every optimal assignment (warehouse of each seller, -1 if unassigned) by enumeration
static vector<vector<int>> all_optimal(const Instance& inst, double& optimum) {
    int m = inst.warehouses.size(), n = inst.sellers.size();
    optimum = brute_force(inst);
    vector<vector<int>> found;
    vector<int> capacity, assignment(n, -1);
    for (const Warehouse& w : inst.warehouses) capacity.push_back(w.capacity);
    function<void(int, double)> search = [&](int j, double cost) {
        if (cost > optimum + 1e-9) return;
        if (j == n) { found.push_back(assignment); return; }
        assignment[j] = -1;
        search(j + 1, cost + inst.penalty);
        for (int i = 0; i < m; i++) {
            if (capacity[i] < inst.demand(j, i)) continue;
            capacity[i] -= inst.demand(j, i);
            assignment[j] = i;
            search(j + 1, cost + inst.cost[j][i]);
            capacity[i] += inst.demand(j, i);
        }
        assignment[j] = -1;
    };
    search(0, 0);
    return found;
}

// cheapest solution with seller j in warehouse i (forced = true) or not in it
static double brute_force_restricted(const Instance& inst, int fi, int fj, bool forced) {
    int m = inst.warehouses.size(), n = inst.sellers.size();
    double best = 1e300;
    vector<int> capacity;
    for (const Warehouse& w : inst.warehouses) capacity.push_back(w.capacity);
    function<void(int, double)> search = [&](int j, double cost) {
        if (cost >= best) return;
        if (j == n) { best = cost; return; }
        if (!(j == fj && forced)) search(j + 1, cost + inst.penalty);
        for (int i = 0; i < m; i++) {
            if (j == fj && forced && i != fi) continue;
            if (j == fj && !forced && i == fi) continue;
            if (capacity[i] < inst.demand(j, i)) continue;
            capacity[i] -= inst.demand(j, i);
            search(j + 1, cost + inst.cost[j][i]);
            capacity[i] += inst.demand(j, i);
        }
    };
    search(0, 0);
    return best;
}

// Lagrangian variable fixing against enumeration
static void test_fixing(const string& dir, mt19937& rng) {
    const double EPS = 1e-6;
    bool penalties_ok = true, valid_ok = true, optimal_ok = true, solve_ok = true;
    int found_count = 0;
    long long fixed_pairs = 0, fixed_sellers = 0;
    for (int t = 0; t < 120; t++) {
        int m = 1 + rng() % 3, n = 1 + rng() % 7;
        double tight = (t % 3 == 0) ? 0.35 : (t % 3 == 1 ? 0.6 : 0.9);
        Instance inst(random_instance(dir + "/fixing" + to_string(t), m, n, tight, rng));
        // penalties of random multipliers against the knapsacks by enumeration
        vector<double> lambda(n);
        for (double& l : lambda) l = rng() % 61 + (rng() % 2 ? 0.5 : 0.0);
        vector<vector<double>> with, without;
        double value = fixing::penalties(inst, lambda, with, without);
        auto knapsack_value = [&](int i, int in, int out) {   // best with item 'in' forced, 'out' excluded
            double best = -1e300;
            for (int mask = 0; mask < (1 << n); mask++) {
                if (in >= 0 && !(mask >> in & 1)) continue;
                if (out >= 0 && (mask >> out & 1)) continue;
                int w = 0;
                double p = 0;
                for (int j = 0; j < n; j++)
                    if (mask >> j & 1) { w += inst.demand(j, i); p += lambda[j] - inst.cost[j][i]; }
                if (w <= inst.warehouses[i].capacity) best = max(best, p);
            }
            return best;
        };
        double expected = 0;
        vector<double> kp(m);
        for (int j = 0; j < n; j++) expected += lambda[j] + min(0.0, inst.penalty - lambda[j]);
        for (int i = 0; i < m; i++) { kp[i] = knapsack_value(i, -1, -1); expected -= kp[i]; }
        penalties_ok = penalties_ok && fabs(value - expected) < 1e-6;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double out = expected + kp[i] - knapsack_value(i, -1, j);
                penalties_ok = penalties_ok && fabs(without[i][j] - out) < 1e-6;
                double in_value = knapsack_value(i, j, -1);
                if (in_value < -1e299) {
                    penalties_ok = penalties_ok && with[i][j] > 1e299;
                } else {
                    double in = expected + kp[i] - in_value - min(0.0, inst.penalty - lambda[j]);
                    for (int k = 0; k < m; k++)
                        if (k != i) in += kp[k] - knapsack_value(k, -1, j);
                    penalties_ok = penalties_ok && fabs(with[i][j] - in) < 1e-6;
                    // and the bound is valid for the solutions with x_ij = 1
                    valid_ok = valid_ok && with[i][j] <= brute_force_restricted(inst, i, j, true) + EPS;
                }
                valid_ok = valid_ok && without[i][j] <= brute_force_restricted(inst, i, j, false) + EPS;
            }
        }
        // With the upper bound OPT + 1 (and without the Lagrangian heuristic, which
        // would lower it to OPT), every optimal solution survives the fixing.
        double optimum;
        vector<vector<int>> optimal = all_optimal(inst, optimum);
        fixing::Reduction r = fixing::reduce(inst, optimum + 1, 0, 3, nullptr, false);
        bool found = r.upper_bound < optimum + 0.5;
        bool survives = found || !r.no_improving_solution;
        for (const vector<int>& a : optimal) {
            if (found) break;
            for (int j = 0; j < n; j++) {
                if (r.forced[j] >= 0 && a[j] != r.forced[j]) survives = false;
                if (r.forced[j] < 0 && a[j] >= 0) {
                    int k = find(r.original.begin(), r.original.end(), j) - r.original.begin();
                    if (r.instance.demand(k, a[j]) > r.instance.warehouses[a[j]].capacity) survives = false;
                }
            }
        }
        optimal_ok = optimal_ok && survives && r.upper_bound >= optimum - EPS && (found || r.bound <= optimum + EPS);
        found_count += found;
        fixed_pairs += r.fixed_to_zero;
        fixed_sellers += r.fixed_to_one;
        // the exact methods with fixing find the optimum from a heuristic start
        Solution start = metaheuristics::initial_solution(inst);
        mip::Result rm = mip::solve_with_fixing(inst, 60, start);
        branch_and_price::Result rb = branch_and_price::solve_with_fixing(inst, 60, start);
        solve_ok = solve_ok && rm.optimal && fabs(rm.primal - optimum) < EPS && feasible(rm.solution, inst)
                && fabs(evaluate(rm.solution, inst) - optimum) < EPS && rm.dual <= optimum + EPS
                && rb.optimal && fabs(rb.primal - optimum) < EPS && feasible(rb.solution, inst)
                && fabs(evaluate(rb.solution, inst) - optimum) < EPS;
    }
    check(penalties_ok, "fixing: penalties = knapsacks by enumeration");
    check(valid_ok, "fixing: every penalty bounds the cheapest solution with that choice");
    check(optimal_ok, "fixing: every optimal solution survives with upper bound OPT + 1");
    cout << "  fixing: " << fixed_pairs << " pairs fixed to 0 and " << fixed_sellers
         << " sellers fixed with upper bound OPT + 1 (" << found_count << " cases with a lower upper bound)\n";
    check(solve_ok, "fixing: compact model and branch-and-price with fixing = brute force");
}

// Re-fixing: the penalties kept by a reduction fix more pairs with a lower upper
// bound (refix) or with more passes from it (reduce with 'from'), and neither
// removes an optimal solution; the compact model and the branch-and-price with
// re-fixing find the optimum from a heuristic start and from the empty solution
// (every seller unassigned, so that the fixing starts with nothing and every
// improvement of the incumbent fixes more).
static void test_refixing(const string& dir, mt19937& rng) {
    const double EPS = 1e-6;
    bool survive_ok = true, monotone_ok = true, solve_ok = true;
    long long refixed = 0;
    int mip_incumbents = 0, bnp_incumbents = 0;
    for (int t = 0; t < 120; t++) {
        int m = 1 + rng() % 3, n = 1 + rng() % 7;
        double tight = (t % 3 == 0) ? 0.35 : (t % 3 == 1 ? 0.6 : 0.9);
        Instance inst(random_instance(dir + "/refixing" + to_string(t), m, n, tight, rng));
        double optimum;
        vector<vector<int>> optimal = all_optimal(inst, optimum);
        // every optimal solution respects the fixings of 'r' (upper bound OPT + 1)
        auto survives = [&](const fixing::Reduction& r) {
            if (r.no_improving_solution) return false;
            for (const vector<int>& a : optimal) {
                for (int j = 0; j < n; j++) {
                    if (r.forced[j] >= 0 && a[j] != r.forced[j]) return false;
                    if (a[j] >= 0 && !r.allowed[j][a[j]]) return false;
                }
            }
            return true;
        };
        double loose = optimum + 1 + rng() % 40;
        fixing::Reduction r = fixing::reduce(inst, loose, 0, 3, nullptr, false);
        long long before = r.fixed_to_zero + r.fixed_to_one;
        if (!r.no_improving_solution) {
            fixing::refix(inst, r, optimum + 1);
            survive_ok = survive_ok && survives(r);
            monotone_ok = monotone_ok && r.fixed_to_zero + r.fixed_to_one >= before;
            refixed += r.fixed_to_zero + r.fixed_to_one - before;
            fixing::Reduction more = fixing::reduce(inst, optimum + 1, 0, 2, nullptr, false, &r);
            survive_ok = survive_ok && survives(more);
            monotone_ok = monotone_ok && more.fixed_to_zero >= r.fixed_to_zero && more.passes >= r.passes;
        }
        // the methods with re-fixing, from two starts
        Solution empty;
        empty.sellers_per_warehouse.assign(m, {});
        empty.assigned_warehouse.assign(n, -1);
        empty.total_cost = n * inst.penalty;
        int kind = 0;
        for (const Solution& start : {metaheuristics::initial_solution(inst), empty}) {
            string name = "refixing" + to_string(t) + (kind++ == 0 ? " (heuristic start)" : " (empty start)");
            mip::Refixing stats;
            mip::Result rm = mip::solve_with_refixing(inst, 60, start, &stats);
            branch_and_price::Options o;
            o.refixing = true;
            branch_and_price::Result rb = branch_and_price::solve(inst, 60, &start, o);
            mip_incumbents += stats.incumbents;
            bnp_incumbents += rb.refixing_incumbents;
            bool mip_ok = rm.optimal && fabs(rm.primal - optimum) < EPS && feasible(rm.solution, inst)
                       && fabs(evaluate(rm.solution, inst) - optimum) < EPS && rm.dual <= optimum + EPS;
            bool bnp_ok = rb.optimal && fabs(rb.primal - optimum) < EPS && feasible(rb.solution, inst)
                       && fabs(evaluate(rb.solution, inst) - optimum) < EPS && rb.dual <= optimum + EPS;
            if (!mip_ok || !bnp_ok) {
                lock_guard<mutex> lock(g_print);
                cout << "    " << name << ": optimum " << optimum << ", MIP " << rm.primal << "/" << rm.dual
                     << (rm.optimal ? " optimal" : "") << ", B&P " << rb.primal << "/" << rb.dual
                     << (rb.optimal ? " optimal" : "") << "\n";
            }
            solve_ok = solve_ok && mip_ok && bnp_ok;
        }
    }
    // Medium instances from the empty start, where SCIP improves the incumbent
    // several times, so that the fixings change during the search: the optimum of
    // the compact model without fixing is the reference.
    bool medium_ok = true;
    int medium_incumbents = 0, medium_bnp_incumbents = 0;
    for (int t = 0; t < 12; t++) {
        int m = 3 + rng() % 3, n = 14 + rng() % 12;
        double tight = (t % 2 == 0) ? 0.6 : 0.9;
        Instance inst(random_instance(dir + "/refixing_medium" + to_string(t), m, n, tight, rng));
        mip::Result reference = mip::solve(inst, 120);
        if (!reference.optimal) continue;
        Solution empty;
        empty.sellers_per_warehouse.assign(m, {});
        empty.assigned_warehouse.assign(n, -1);
        empty.total_cost = n * inst.penalty;
        mip::Refixing stats;
        mip::Result rm = mip::solve_with_refixing(inst, 120, empty, &stats);
        branch_and_price::Options o;
        o.refixing = true;
        branch_and_price::Result rb = branch_and_price::solve(inst, 120, &empty, o);
        medium_incumbents += stats.incumbents;
        medium_bnp_incumbents += rb.refixing_incumbents;
        bool ok = rm.optimal && fabs(rm.primal - reference.primal) < EPS && feasible(rm.solution, inst)
               && fabs(evaluate(rm.solution, inst) - rm.primal) < EPS
               && rb.optimal && fabs(rb.primal - reference.primal) < EPS && feasible(rb.solution, inst)
               && fabs(evaluate(rb.solution, inst) - rb.primal) < EPS;
        if (!ok) {
            lock_guard<mutex> lock(g_print);
            cout << "    refixing_medium" << t << ": MIP " << reference.primal << ", re-fixing MIP " << rm.primal
                 << "/" << rm.dual << ", B&P " << rb.primal << "/" << rb.dual << "\n";
        }
        medium_ok = medium_ok && ok;
    }
    check(medium_ok, "re-fixing: compact model and branch-and-price with re-fixing = compact model (medium instances)");
    cout << "  re-fixing on medium instances: " << medium_incumbents << " (MIP) and " << medium_bnp_incumbents
         << " (branch-and-price) incumbents used\n";
    check(survive_ok, "re-fixing: refix and further passes keep every optimal solution");
    check(monotone_ok, "re-fixing: the fixings only grow");
    check(solve_ok, "re-fixing: compact model and branch-and-price with re-fixing = brute force");
    cout << "  re-fixing: " << refixed << " new fixings from a lower upper bound; "
         << mip_incumbents << " (MIP) and " << bnp_incumbents << " (branch-and-price) incumbents used\n";
}

// Bundle method, tabu search with ejection chains and portfolio start: the bundle
// bound lies between the LP bound and the optimum and, when the column generation
// with HiGHS converges, equals its master LP value; the branch-and-price with the
// bundle root (with and without re-fixing) and the compact model with re-fixing
// and bundle multipliers find the optimum; the solutions of the tabu search and
// of the portfolio are feasible, cost what they report and are not worse than
// their start.
static void test_bundle(const string& dir, mt19937& rng) {
    const double EPS = 1e-6;
    bool bound_ok = true, master_ok = true, exact_ok = true, heuristic_ok = true;
    int compared = 0;
    for (int t = 0; t < 120; t++) {
        int m = 1 + rng() % 4, n = 1 + rng() % 8;
        double tight = (t % 3 == 0) ? 0.35 : (t % 3 == 1 ? 0.6 : 0.9);
        Instance inst(random_instance(dir + "/bundle" + to_string(t), m, n, tight, rng));
        double optimum = brute_force(inst);
        double lp = mip::lp_relaxation(inst, 60).dual;
        column_generation::Result b = column_generation::bundle(inst, 30, {});
        bound_ok = bound_ok && b.converged && b.bound >= lp - 1e-6 * max(1.0, fabs(lp)) && b.bound <= optimum + EPS;
        column_generation::Result cg = column_generation::root(inst, 30, column_generation::Solver::SIMPLEX, {});
        if (cg.converged) {
            compared++;
            master_ok = master_ok && fabs(cg.lp_value - b.bound) <= 1e-5 * max(1.0, fabs(cg.lp_value));
        }
        Solution start = metaheuristics::initial_solution(inst);
        for (bool refixing : {false, true}) {
            branch_and_price::Options o;
            o.bundle_root = true;
            o.refixing = refixing;
            branch_and_price::Result r = branch_and_price::solve(inst, 60, &start, o);
            exact_ok = exact_ok && r.optimal && fabs(r.primal - optimum) < EPS && feasible(r.solution, inst)
                    && fabs(evaluate(r.solution, inst) - optimum) < EPS && r.dual <= optimum + EPS;
        }
        mip::Result rm = mip::solve_with_refixing(inst, 60, start, nullptr, true);
        exact_ok = exact_ok && rm.optimal && fabs(rm.primal - optimum) < EPS && feasible(rm.solution, inst);
        Solution tabu = ejection::search(inst, 0, 1 + t, &start);
        Solution mix = matheuristic::portfolio(inst, 2, 1 + t);
        heuristic_ok = heuristic_ok && feasible(tabu, inst) && fabs(evaluate(tabu, inst) - tabu.total_cost) < EPS
                    && tabu.total_cost <= start.total_cost + EPS && tabu.total_cost >= optimum - EPS
                    && feasible(mix, inst) && fabs(evaluate(mix, inst) - mix.total_cost) < EPS
                    && mix.total_cost >= optimum - EPS;
    }
    check(bound_ok, "bundle: converged, LP bound <= bundle bound <= optimum");
    check(master_ok, "bundle: bound = master LP value of the column generation with HiGHS (" + to_string(compared) + " cases)");
    check(exact_ok, "bundle: branch-and-price with bundle root (and re-fixing), MIP with re-fixing and bundle = brute force");
    check(heuristic_ok, "tabu search and portfolio: feasible, consistent costs, not worse than the start");
}

static void test_check() {
    cout << "== Correctness tests of the exact methods ==\n";
    const double EPS = 1e-6;
    char dir_template[] = "/tmp/gap_check_XXXXXX";
    string dir = mkdtemp(dir_template);
    mt19937 rng(12345);

    // knapsack against enumeration
    bool knapsack_ok = true;
    for (int trial = 0; trial < 20000; trial++) {
        int n = rng() % 13, capacity = rng() % 40;
        vector<double> profit(n);
        vector<int> weight(n);
        for (int k = 0; k < n; k++) {
            profit[k] = (int)(rng() % 41) - 10 + (rng() % 2 ? 0.5 : 0.0);
            weight[k] = 1 + rng() % 20;
        }
        double best = 0;
        for (int mask = 0; mask < (1 << n); mask++) {
            int w = 0;
            double p = 0;
            for (int k = 0; k < n; k++)
                if (mask >> k & 1) { w += weight[k]; p += profit[k]; }
            if (w <= capacity && p > best) best = p;
        }
        vector<int> chosen;
        double value = knapsack::solve(profit, weight, capacity, chosen);
        int w = 0;
        double p = 0;
        for (int k : chosen) { w += weight[k]; p += profit[k]; }
        knapsack_ok = knapsack_ok && fabs(value - best) < 1e-9 && fabs(p - value) < 1e-9 && w <= capacity;
    }
    check(knapsack_ok, "knapsack = enumeration on 20000 random cases");
    test_fixing(dir, rng);
    test_refixing(dir, rng);
    test_bundle(dir, rng);

    // tiny instances: every method against brute force
    for (int t = 0; t < 150; t++) {
        int m = 1 + rng() % 3, n = 1 + rng() % 8;
        double tight = (t % 3 == 0) ? 0.35 : (t % 3 == 1 ? 0.6 : 0.9);
        string name = "tiny" + to_string(t);
        Instance inst(random_instance(dir + "/" + name, m, n, tight, rng));
        double optimum = brute_force(inst);

        mip::Result r = mip::solve(inst, 60);
        check(r.optimal && fabs(r.primal - optimum) < EPS && feasible(r.solution, inst)
              && fabs(evaluate(r.solution, inst) - optimum) < EPS, name + ": compact model = brute force");

        branch_and_price::Result b = branch_and_price::solve(inst, 60);
        check(b.optimal && fabs(b.primal - optimum) < EPS && feasible(b.solution, inst)
              && fabs(evaluate(b.solution, inst) - optimum) < EPS, name + ": branch-and-price = brute force");

        // every technique of the branch-and-price switched off, one at a time and all
        for (int variant = 0; variant < 7; variant++) {
            branch_and_price::Options o;
            if (variant == 0 || variant == 6) o.early_termination = false;
            if (variant == 1 || variant == 6) o.extra_columns = false;
            if (variant == 2 || variant == 6) o.column_aging = false;
            if (variant == 3 || variant == 6) o.automatic_smoothing = false;
            if (variant == 4 || variant == 6) o.strong_branching = false;
            if (variant == 5 || variant == 6) o.restricted_master_heuristic = false;
            branch_and_price::Result v = branch_and_price::solve(inst, 60, nullptr, o);
            check(v.optimal && fabs(v.primal - optimum) < EPS && feasible(v.solution, inst),
                  name + ": branch-and-price variant " + to_string(variant) + " = brute force");
        }

        // without an initial solution the root is not pruned before the column
        // generation converges, so its bound is the Lagrangian dual
        Solution empty;
        empty.sellers_per_warehouse.assign(m, {});
        empty.assigned_warehouse.assign(n, -1);
        empty.total_cost = n * inst.penalty;
        branch_and_price::Result full = branch_and_price::solve(inst, 60, &empty);
        check(full.optimal && fabs(full.primal - optimum) < EPS, name + ": branch-and-price from scratch = brute force");

        mip::Result lp = mip::lp_relaxation(inst, 60);
        lagrangian::Result lr = lagrangian::subgradient(inst, optimum, 3000);
        check(lp.dual <= optimum + EPS && lr.bound <= optimum + EPS, name + ": LP and Lagrangian bounds <= optimum");
        // the root may still be pruned once ceil(bound) reaches the incumbent (integer costs)
        check(full.root_bound >= lp.dual - EPS && full.root_bound <= optimum + EPS
              && lr.bound <= ceil(full.root_bound - EPS) + EPS,
              name + ": LP <= column generation bound, Lagrangian <= column generation bound");

        // every start and destruction operator of the matheuristic
        for (auto start : {matheuristic::Start::CONSTRUCTIVE, matheuristic::Start::LP_ROUNDING,
                           matheuristic::Start::LAGRANGIAN}) {
            for (auto op : {matheuristic::Destroy::RANDOM, matheuristic::Destroy::WAREHOUSES,
                            matheuristic::Destroy::MIXED}) {
                Solution mh = matheuristic::mip_lns(inst, 5, 3, 1, 0, 5.0, op, start);
                check(feasible(mh, inst) && mh.total_cost >= optimum - EPS
                      && fabs(mh.total_cost - evaluate(mh, inst)) < EPS,
                      name + ": matheuristic feasible and consistent");
            }
        }

        // rounding of the LP: basic solution with at most m split sellers
        mip::Rounding rounding = mip::lp_rounding(inst, 60);
        check(rounding.split <= m && feasible(rounding.solution, inst)
              && fabs(rounding.solution.total_cost - evaluate(rounding.solution, inst)) < EPS
              && fabs(rounding.lp_bound - lp.dual) < 1e-6 * max(1.0, fabs(lp.dual)),
              name + ": LP rounding feasible, at most m split sellers, same LP bound");

        // rounding of Shmoys and Tardos: cost at most the LP bound, load at most
        // b_i + max d_ij, and a feasible repair
        mip::ShmoysTardos st = mip::shmoys_tardos(inst, 60);
        check(st.cost <= lp.dual + 1e-6 && st.within_bound && fabs(st.lp_bound - lp.dual) < 1e-6
              && feasible(st.repaired, inst) && fabs(st.repaired.total_cost - evaluate(st.repaired, inst)) < EPS
              && fabs(st.cost - evaluate(st.assignment, inst)) < EPS,
              name + ": Shmoys-Tardos cost <= LP, load within the bound, feasible repair");

        // cover inequalities and the root of SCIP: valid bounds
        mip::CutResult covers = mip::cover_cuts(inst, 60);
        mip::Result root = mip::root_node(inst, 60);
        check(covers.bound >= lp.dual - EPS && covers.bound <= optimum + EPS && root.dual <= optimum + EPS,
              name + ": LP <= cover bound <= optimum, root bound <= optimum");

        // exact knapsack separation: the bound of the Dantzig-Wolfe master, which the
        // column generation with HiGHS also reaches (simplex and interior point)
        mip::CutResult exact = mip::exact_knapsack_cuts(inst, 60);
        vector<pair<int, vector<int>>> seed_columns;
        lagrangian::subgradient(inst, n * inst.penalty, 500, 0, 50, &seed_columns);
        column_generation::Result cg_simplex =
            column_generation::root(inst, 60, column_generation::Solver::SIMPLEX, seed_columns);
        column_generation::Result cg_interior =
            column_generation::root(inst, 60, column_generation::Solver::INTERIOR_POINT, seed_columns);
        check(exact.bound >= covers.bound - EPS && exact.bound <= optimum + EPS
              && cg_simplex.converged && cg_interior.converged
              && fabs(cg_simplex.bound - exact.bound) < 1e-4 * max(1.0, fabs(exact.bound))
              && fabs(cg_interior.bound - exact.bound) < 1e-4 * max(1.0, fabs(exact.bound)),
              name + ": covers <= exact separation = column generation (simplex, interior point) <= optimum");

        // branch-and-cut with exact separation
        mip::Result bc = mip::branch_and_cut(inst, 60);
        check(bc.optimal && fabs(bc.primal - optimum) < EPS && feasible(bc.solution, inst),
              name + ": branch-and-cut = brute force");

        // shadow prices of the capacities: the LP of HiGHS has the same optimum
        mip::CapacityDuals duals = mip::capacity_duals(inst, 60);
        bool prices_ok = duals.solved && fabs(duals.lp_bound - lp.dual) < 1e-6 * max(1.0, fabs(lp.dual));
        for (double price : duals.shadow_price) prices_ok = prices_ok && price >= 0;
        check(prices_ok, name + ": capacity duals of HiGHS consistent with the LP of SCIP");

        // Lagrangian heuristic
        lagrangian::Result heuristic = lagrangian::subgradient(inst, n * inst.penalty, 2000, 0, 50, nullptr, 100, 10);
        Solution from_relaxation = lagrangian::heuristic_solution(inst, 0, 0);
        check(heuristic.has_solution && feasible(heuristic.solution, inst)
              && fabs(heuristic.solution.total_cost - evaluate(heuristic.solution, inst)) < EPS
              && heuristic.solution.total_cost >= optimum - EPS && heuristic.bound <= optimum + EPS
              && feasible(from_relaxation, inst) && from_relaxation.total_cost >= optimum - EPS,
              name + ": Lagrangian heuristic feasible and consistent");
    }

    // medium instances with tight capacities: compact model against branch-and-price
    long long nodes = 0;
    int cover_cuts = 0, st_overloaded = 0;
    for (int t = 0; t < 40; t++) {
        int m = 3 + rng() % 6, n = 12 + rng() % 29;
        double tight = 0.5 + 0.1 * (rng() % 4);
        string name = "medium" + to_string(t);
        Instance inst(random_instance(dir + "/" + name, m, n, tight, rng));
        mip::Result r = mip::solve(inst, 120);
        branch_and_price::Result b = branch_and_price::solve(inst, 120);
        check(r.optimal && b.optimal && fabs(r.primal - b.primal) < EPS && feasible(b.solution, inst),
              name + ": compact model = branch-and-price");
        nodes += b.nodes;

        mip::Rounding rounding = mip::lp_rounding(inst, 60);
        mip::ShmoysTardos st = mip::shmoys_tardos(inst, 60);
        check(st.cost <= rounding.lp_bound + 1e-6 && st.within_bound && feasible(st.repaired, inst),
              name + ": Shmoys-Tardos cost <= LP, load within the bound, feasible repair");
        st_overloaded += st.overloaded;
        mip::CutResult covers = mip::cover_cuts(inst, 60);
        check(rounding.split <= m && feasible(rounding.solution, inst)
              && covers.bound >= rounding.lp_bound - EPS && covers.bound <= r.primal + EPS,
              name + ": at most m split sellers, LP <= cover bound <= optimum");
        cover_cuts += covers.cuts;

        Solution s = metaheuristics::initial_solution(inst);
        double before = s.total_cost;
        vector<int> freed;
        for (int j = 0; j < n; j += 2) freed.push_back(j);
        double delta = mip::reoptimize(inst, s, freed, 30);
        check(delta <= EPS && feasible(s, inst) && fabs(s.total_cost - evaluate(s, inst)) < EPS
              && fabs(before + delta - s.total_cost) < EPS, name + ": reoptimization never worsens");
    }
    string cleanup = "rm -rf " + dir;
    if (system(cleanup.c_str()) != 0) cout << "  could not remove " << dir << "\n";
    cout << "  branch-and-price nodes on the medium instances: " << nodes << "\n";
    cout << "  cover inequalities on the medium instances: " << cover_cuts << "\n";
    cout << "  warehouses overloaded by Shmoys-Tardos on the medium instances: " << st_overloaded << "\n";
    cout << "  " << g_pass << " PASS, " << g_fail << " FAIL\n";
}

// -----------------------------------------------------------------------------
//  E7 - Relaxations, relaxation-based heuristics and exact methods
// -----------------------------------------------------------------------------

static void exp_e7(int threads) {
    cout << "== E7: relaxations, relaxation-based heuristics and exact methods ==\n";
    vector<InstanceInfo> list = instances();
    vector<string> rows(list.size());
    Checkpoint checkpoint("e7_exact");
    parallel_for(list.size(), threads, [&](int k) {
        const InstanceInfo& I = list[k];
        if (checkpoint.get(I.name, rows[k])) { rows[k] += "\n"; return; }
        Instance inst(I.path);
        // the LNS of E5 with seed 1: the best method of the original project
        Solution lns = metaheuristics::lns(inst, 1000000000, lns_destroy(I.n), 1, time_budget(I.n));

        // linear relaxation and its rounding (+ VND)
        mip::Rounding rounding = mip::lp_rounding(inst, EXACT_TIME_LIMIT);
        Solution rounded = rounding.solution;
        double rounding_cost = evaluate(rounded, inst);
        metaheuristics::vnd(rounded, inst);
        check(feasible(rounded, inst), I.name + ": LP rounding feasible");
        check(rounding.split <= (int)inst.warehouses.size(), I.name + ": at most m split sellers");

        // Lagrangian relaxation with the Lagrangian heuristic (+ VND)
        lagrangian::Result lr;
        double lr_seconds = timed([&] {
            Solution constructive = metaheuristics::best_constructive(inst);
            lr = lagrangian::subgradient(inst, constructive.total_cost, 1000000, 60, 50, nullptr, 100, 10);
        });
        Solution repaired = lr.solution;
        double repaired_cost = evaluate(repaired, inst);
        metaheuristics::vnd(repaired, inst);
        check(lr.has_solution && feasible(repaired, inst), I.name + ": Lagrangian heuristic feasible");

        // the exact methods start from the best of the three solutions
        Solution start = lns;
        if (evaluate(rounded, inst) < evaluate(start, inst)) start = rounded;
        if (evaluate(repaired, inst) < evaluate(start, inst)) start = repaired;
        start.total_cost = evaluate(start, inst);
        mip::Result r = mip::solve(inst, EXACT_TIME_LIMIT, &start);
        fixing::Reduction reduction;
        mip::Result rf = mip::solve_with_fixing(inst, EXACT_TIME_LIMIT, start, &reduction);
        branch_and_price::Result b = branch_and_price::solve_with_fixing(inst, EXACT_TIME_LIMIT, start);
        check(!r.has_solution || feasible(r.solution, inst), I.name + ": compact model solution feasible");
        check(feasible(rf.solution, inst) && feasible(b.solution, inst), I.name + ": solutions with fixing feasible");

        double upper = min({start.total_cost, r.primal, rf.primal, b.primal});
        double lower = max({rounding.lp_bound, lr.bound, r.dual, rf.dual, b.dual});
        if (integral_costs(inst)) lower = ceil(lower - 1e-6);
        check(lower <= upper + 1e-6, I.name + ": lower bound <= upper bound");
        double gap = upper > 0 ? max(0.0, (upper - lower) / upper * 100.0) : 0.0;

        ostringstream row;
        row << fixed << setprecision(2)
            << I.name << "," << I.group << "," << I.n << "," << inst.warehouses.size() << ","
            << evaluate(lns, inst) << ","
            << rounding.lp_bound << "," << rounding.seconds << "," << rounding.split << ","
            << rounding_cost << "," << evaluate(rounded, inst) << ","
            << lr.bound << "," << lr.iterations << "," << lr_seconds << ","
            << repaired_cost << "," << evaluate(repaired, inst) << ","
            << start.total_cost << ","
            << r.primal << "," << r.dual << "," << r.optimal << "," << r.seconds << "," << r.nodes << ","
            << 100.0 * reduction.fixed_to_zero / max(1LL, reduction.pairs) << "," << reduction.fixed_to_one << ","
            << reduction.seconds << ","
            << rf.primal << "," << rf.dual << "," << rf.optimal << "," << rf.seconds << ","
            << b.primal << "," << b.dual << "," << b.root_bound << "," << b.optimal << ","
            << b.seconds << "," << b.nodes << "," << b.columns << ","
            << upper << "," << lower << "," << gap << "\n";
        rows[k] = row.str();
        checkpoint.put(I.name, rows[k].substr(0, rows[k].size() - 1));
        lock_guard<mutex> lock(g_print);
        cout << "  " << I.name << " lp=" << rounding.lp_bound << " lagrangian=" << lr.bound
             << " start=" << start.total_cost
             << " mip=" << r.primal << "/" << r.dual << (r.optimal ? " (optimal)" : "")
             << " mip+fixing=" << rf.primal << "/" << rf.dual << (rf.optimal ? " (optimal)" : "")
             << " bnp+fixing=" << b.primal << "/" << b.dual << (b.optimal ? " (optimal)" : "") << endl;
    });
    ofstream csv(output("e7_exact.csv"));
    csv << "instance,group,n,m,lns_cost,"
        << "lp_bound,lp_time_s,lp_split,lp_rounding_cost,lp_rounding_vnd_cost,"
        << "lagrangian_bound,lagrangian_iterations,lagrangian_time_s,"
        << "lagrangian_heuristic_cost,lagrangian_heuristic_vnd_cost,start_cost,"
        << "mip_best,mip_bound,mip_optimal,mip_time_s,mip_nodes,"
        << "fixed_pairs_pct,fixed_sellers,fixing_time_s,"
        << "mipfix_best,mipfix_bound,mipfix_optimal,mipfix_time_s,"
        << "bnp_best,bnp_bound,bnp_root_bound,bnp_optimal,bnp_time_s,bnp_nodes,bnp_columns,"
        << "best_upper,best_lower,gap_pct\n";
    for (const string& row : rows) csv << row;
    checkpoint.finish();
    cout << "  -> " << output("e7_exact.csv") << "\n";
}

// -----------------------------------------------------------------------------
//  E8 - Tuning of the matheuristic
// -----------------------------------------------------------------------------

static const char* start_name(matheuristic::Start s) {
    switch (s) {
        case matheuristic::Start::CONSTRUCTIVE: return "constructive";
        case matheuristic::Start::LP_ROUNDING: return "lp_rounding";
        default: return "lagrangian";
    }
}

static const char* destroy_name(matheuristic::Destroy d) {
    switch (d) {
        case matheuristic::Destroy::RANDOM: return "random";
        case matheuristic::Destroy::WAREHOUSES: return "warehouses";
        default: return "mixed";
    }
}

static void exp_e8(int threads) {
    cout << "== E8: tuning of the matheuristic (start, destruction operator and size) ==\n";
    vector<InstanceInfo> subset = {
        {"instances/gap/gap_b/b10200", "b10200", 'b', 200},
        {"instances/gap/gap_e/e20200", "e20200", 'e', 200},
        {"instances/gap/gap_e/e15900", "e15900", 'e', 900},
        {"instances/gap/gap_e/e801600", "e801600", 'e', 1600},
        {"instances/real/real_instance", "real", 'r', 1100},
    };
    vector<matheuristic::Start> starts = {matheuristic::Start::CONSTRUCTIVE, matheuristic::Start::LP_ROUNDING,
                                          matheuristic::Start::LAGRANGIAN};
    vector<matheuristic::Destroy> operators = {matheuristic::Destroy::RANDOM, matheuristic::Destroy::WAREHOUSES,
                                               matheuristic::Destroy::MIXED};
    vector<int> destroys = {50, 100, 200};
    struct Job { int instance; int start; int op; int destroy; unsigned seed; };
    vector<Job> jobs;
    for (int k = 0; k < (int)subset.size(); k++)
        for (int a = 0; a < (int)starts.size(); a++)
            for (int o = 0; o < (int)operators.size(); o++)
                for (int d : destroys)
                    for (unsigned s = 1; s <= SEEDS; s++) jobs.push_back({k, a, o, d, s});
    vector<double> costs(jobs.size());
    Checkpoint checkpoint("e8_tuning");
    parallel_for(jobs.size(), threads, [&](int k) {
        const Job& job = jobs[k];
        const InstanceInfo& I = subset[job.instance];
        string key = I.name + "|" + to_string(job.start) + "|" + to_string(job.op) + "|"
                   + to_string(job.destroy) + "|" + to_string(job.seed);
        string saved;
        if (checkpoint.get(key, saved)) { costs[k] = stod(saved); return; }
        Instance inst(I.path);
        Solution sol = matheuristic::mip_lns(inst, 1000000000, job.destroy, job.seed, time_budget(I.n), 5.0,
                                             operators[job.op], starts[job.start]);
        check(feasible(sol, inst), I.name + ": matheuristic solution feasible");
        costs[k] = evaluate(sol, inst);
        ostringstream value;
        value << setprecision(17) << costs[k];
        checkpoint.put(key, value.str());
    });
    ofstream csv(output("e8_tuning.csv"));
    csv << "instance,group,start,operator,destroy,seconds,seeds,best_cost,worst_cost,mean_cost,std_dev\n";
    csv << fixed << setprecision(2);
    for (int k = 0; k < (int)subset.size(); k++) {
        const InstanceInfo& I = subset[k];
        for (int a = 0; a < (int)starts.size(); a++) {
            for (int o = 0; o < (int)operators.size(); o++) {
                for (int d : destroys) {
                    vector<double> v;
                    for (int t = 0; t < (int)jobs.size(); t++) {
                        const Job& job = jobs[t];
                        if (job.instance == k && job.start == a && job.op == o && job.destroy == d) v.push_back(costs[t]);
                    }
                    Stats st = summarize(v);
                    csv << I.name << "," << I.group << "," << start_name(starts[a]) << ","
                        << destroy_name(operators[o]) << "," << d << "," << time_budget(I.n) << "," << SEEDS << ","
                        << st.best << "," << st.worst << "," << st.mean << "," << st.std_dev << "\n";
                }
            }
        }
        cout << "  " << I.name << " done\n";
    }
    checkpoint.finish();
    cout << "  -> " << output("e8_tuning.csv") << "\n";
}

// -----------------------------------------------------------------------------
//  E9 - LNS against the matheuristic
// -----------------------------------------------------------------------------

// The three methods of E9, with the same time budget: the LNS of the original
// project, the same LNS started from the Lagrangian heuristic (which separates the
// effect of the start from that of the repair) and the matheuristic.
enum class Method { LNS, LNS_LAGRANGIAN, MIP_LNS };

static Solution run_method(Method method, const Instance& inst, int n, unsigned seed) {
    double seconds = time_budget(n);
    if (method == Method::LNS) return metaheuristics::lns(inst, 1000000000, lns_destroy(n), seed, seconds);
    if (method == Method::MIP_LNS)
        return matheuristic::mip_lns(inst, 1000000000, mip_lns_destroy(n), seed, seconds, 5.0,
                                     MIP_LNS_OPERATOR, MIP_LNS_START);
    auto t0 = chrono::high_resolution_clock::now();
    Solution start = lagrangian::heuristic_solution(inst, 0.6 * seconds, seconds);
    double spent = chrono::duration_cast<chrono::milliseconds>(
        chrono::high_resolution_clock::now() - t0).count() / 1000.0;
    return metaheuristics::lns(inst, 1000000000, lns_destroy(n), seed, max(1e-3, seconds - spent), &start);
}

static void exp_e9(int threads) {
    cout << "== E9: LNS against the matheuristic ==\n";
    vector<InstanceInfo> list = instances();
    vector<Method> methods = {Method::LNS, Method::LNS_LAGRANGIAN, Method::MIP_LNS};
    struct Job { int instance; int method; unsigned seed; };
    vector<Job> jobs;
    for (int k = 0; k < (int)list.size(); k++)
        for (int q = 0; q < (int)methods.size(); q++)
            for (unsigned s = 1; s <= SEEDS; s++) jobs.push_back({k, q, s});
    vector<double> costs(jobs.size()), times(jobs.size());
    Checkpoint checkpoint("e9_matheuristic");
    parallel_for(jobs.size(), threads, [&](int k) {
        const Job& job = jobs[k];
        const InstanceInfo& I = list[job.instance];
        string key = I.name + "|" + to_string(job.method) + "|" + to_string(job.seed);
        string saved;
        if (checkpoint.get(key, saved)) {
            istringstream values(saved);
            values >> costs[k] >> times[k];
            return;
        }
        Instance inst(I.path);
        Solution sol;
        times[k] = timed([&] { sol = run_method(methods[job.method], inst, I.n, job.seed); });
        check(feasible(sol, inst), I.name + ": solution feasible");
        costs[k] = evaluate(sol, inst);
        ostringstream value;
        value << setprecision(17) << costs[k] << " " << times[k];
        checkpoint.put(key, value.str());
    });
    ofstream csv(output("e9_matheuristic.csv"));
    csv << "instance,group,n,seconds";
    for (string prefix : {"lns", "lns_lagrangian", "mip_lns"})
        csv << "," << prefix << "_best," << prefix << "_worst," << prefix << "_mean,"
            << prefix << "_std," << prefix << "_mean_time_s";
    csv << "\n" << fixed << setprecision(2);
    for (int k = 0; k < (int)list.size(); k++) {
        const InstanceInfo& I = list[k];
        csv << I.name << "," << I.group << "," << I.n << "," << time_budget(I.n);
        for (int q = 0; q < (int)methods.size(); q++) {
            vector<double> v, t;
            for (int j = 0; j < (int)jobs.size(); j++) {
                if (jobs[j].instance == k && jobs[j].method == q) {
                    v.push_back(costs[j]);
                    t.push_back(times[j]);
                }
            }
            Stats st = summarize(v);
            csv << "," << st.best << "," << st.worst << "," << st.mean << "," << st.std_dev << ","
                << accumulate(t.begin(), t.end(), 0.0) / t.size();
        }
        csv << "\n";
    }
    checkpoint.finish();
    cout << "  -> " << output("e9_matheuristic.csv") << "\n";
}

// -----------------------------------------------------------------------------
//  E11 - Rows against columns: cover inequalities, exact knapsack separation and
//  the Lagrangian bound
// -----------------------------------------------------------------------------

static vector<map<string, string>> read_csv(const string& file);

static void exp_e11(int threads) {
    cout << "== E11: cutting planes against the bound of the columns ==\n";
    map<string, map<string, string>> e7;
    for (const auto& row : read_csv(output("e7_exact.csv"))) e7[row.at("instance")] = row;
    vector<InstanceInfo> list = instances();
    vector<string> rows(list.size());
    Checkpoint checkpoint("e11_cuts");
    parallel_for(list.size(), threads, [&](int k) {
        const InstanceInfo& I = list[k];
        if (checkpoint.get(I.name, rows[k])) { rows[k] += "\n"; return; }
        Instance inst(I.path);
        mip::CutResult covers = mip::cover_cuts(inst, EXACT_TIME_LIMIT);
        mip::CutResult exact = mip::exact_knapsack_cuts(inst, EXACT_TIME_LIMIT);
        mip::Result root = mip::root_node(inst, EXACT_TIME_LIMIT);
        const auto& row7 = e7.at(I.name);
        double lp = stod(row7.at("lp_bound"));
        double lagrangian = stod(row7.at("lagrangian_bound"));
        double upper = stod(row7.at("best_upper"));
        // The bound of the Dantzig-Wolfe master (the Lagrangian dual): the exact
        // separation reaches it when it converges, and every Lagrangian bound is
        // below it, so the larger of both is the reference.
        double master = max(lagrangian, exact.bound);
        check(covers.bound >= lp - 1e-6 && covers.bound <= upper + 1e-6, I.name + ": LP <= covers <= upper bound");
        check(exact.bound >= lp - 1e-6 && exact.bound <= upper + 1e-6, I.name + ": LP <= exact cuts <= upper bound");
        check(root.dual <= upper + 1e-6, I.name + ": root bound of SCIP <= upper bound");
        auto closed = [&](double bound) {
            return master - lp > 1e-9 ? (bound - lp) / (master - lp) * 100.0 : 100.0;
        };
        ostringstream out;
        out << fixed << setprecision(2)
            << I.name << "," << I.group << "," << I.n << "," << inst.warehouses.size() << ","
            << lp << "," << covers.bound << "," << covers.cuts << "," << covers.rounds << "," << covers.seconds << ","
            << exact.bound << "," << exact.cuts << "," << exact.rounds << "," << exact.seconds << ","
            << root.dual << "," << root.seconds << "," << lagrangian << "," << master << "," << upper << ","
            << closed(covers.bound) << "," << closed(exact.bound) << "," << closed(root.dual) << ","
            << closed(lagrangian) << "\n";
        rows[k] = out.str();
        checkpoint.put(I.name, rows[k].substr(0, rows[k].size() - 1));
        lock_guard<mutex> lock(g_print);
        cout << "  " << I.name << " lp=" << lp << " covers=" << covers.bound << " exact=" << exact.bound
             << " (" << exact.cuts << " cuts, " << exact.seconds << " s) scip_root=" << root.dual
             << " lagrangian=" << lagrangian << endl;
    });
    ofstream csv(output("e11_cuts.csv"));
    csv << "instance,group,n,m,lp_bound,cover_bound,cover_cuts,cover_rounds,cover_time_s,"
        << "exact_bound,exact_cuts,exact_rounds,exact_time_s,"
        << "scip_root_bound,scip_root_time_s,lagrangian_bound,master_bound,best_upper,"
        << "closed_covers_pct,closed_exact_pct,closed_scip_root_pct,closed_lagrangian_pct\n";
    for (const string& row : rows) csv << row;
    checkpoint.finish();
    cout << "  -> " << output("e11_cuts.csv") << "\n";
}

// -----------------------------------------------------------------------------
//  E13 - What each technique of the branch-and-price contributes
// -----------------------------------------------------------------------------

static void exp_e13(int threads) {
    cout << "== E13: ablation of the branch-and-price ==\n";
    // instances that the branch-and-price closes within the limit, so that the
    // variants can be compared by their times (on the harder ones every variant
    // reaches the limit)
    vector<InstanceInfo> subset;
    for (const InstanceInfo& I : instances()) {
        for (string name : {"b05100", "b10200", "c05100", "c10200", "e05100", "e10100", "e20100", "e20200"})
            if (I.name == name) subset.push_back(I);
    }
    vector<string> variants = {"all", "no_early_termination", "no_extra_columns", "no_column_aging",
                               "no_automatic_smoothing", "no_strong_branching", "no_restricted_master",
                               "none"};
    struct Job { int instance; int variant; };
    vector<Job> jobs;
    for (int k = 0; k < (int)subset.size(); k++)
        for (int v = 0; v < (int)variants.size(); v++) jobs.push_back({k, v});
    vector<string> rows(jobs.size());
    Checkpoint checkpoint("e13_ablation");
    parallel_for(jobs.size(), threads, [&](int k) {
        const InstanceInfo& I = subset[jobs[k].instance];
        const string& variant = variants[jobs[k].variant];
        string key = I.name + "|" + variant;
        if (checkpoint.get(key, rows[k])) { rows[k] += "\n"; return; }
        Instance inst(I.path);
        branch_and_price::Options o;
        bool none = variant == "none";
        o.early_termination = !(none || variant == "no_early_termination");
        o.extra_columns = !(none || variant == "no_extra_columns");
        o.column_aging = !(none || variant == "no_column_aging");
        o.automatic_smoothing = !(none || variant == "no_automatic_smoothing");
        o.strong_branching = !(none || variant == "no_strong_branching");
        o.restricted_master_heuristic = !(none || variant == "no_restricted_master");
        // a common start: the best constructive solution + VND (no fixing, so that
        // the branch-and-price itself is measured)
        Solution start = metaheuristics::initial_solution(inst);
        branch_and_price::Result r = branch_and_price::solve(inst, EXACT_TIME_LIMIT, &start, o);
        check(!r.has_solution || feasible(r.solution, inst), I.name + ": branch-and-price solution feasible");
        ostringstream out;
        out << fixed << setprecision(2) << I.name << "," << I.group << "," << variant << ","
            << r.primal << "," << r.dual << "," << r.root_bound << "," << r.optimal << "," << r.seconds << ","
            << r.nodes << "," << r.columns << "," << r.lp_iterations << "," << r.early_stops << ","
            << r.mispricings << "," << r.heuristic_improvements << "\n";
        rows[k] = out.str();
        checkpoint.put(key, rows[k].substr(0, rows[k].size() - 1));
        lock_guard<mutex> lock(g_print);
        cout << "  " << I.name << " " << variant << ": " << r.primal << "/" << r.dual
             << (r.optimal ? " (optimal)" : "") << " in " << r.seconds << " s, " << r.nodes << " nodes" << endl;
    });
    ofstream csv(output("e13_ablation.csv"));
    csv << "instance,group,variant,best,bound,root_bound,optimal,time_s,nodes,columns,lp_iterations,"
        << "early_stops,mispricings,heuristic_improvements\n";
    for (const string& row : rows) csv << row;
    checkpoint.finish();
    cout << "  -> " << output("e13_ablation.csv") << "\n";
}

// -----------------------------------------------------------------------------
//  E14 - Solver of the master LP: simplex against interior point (HiGHS)
// -----------------------------------------------------------------------------

static void exp_e14(int threads) {
    cout << "== E14: master of the root: simplex, interior point method and bundle method ==\n";
    // the LP masters on 8 instances (they rarely converge on the large ones), the
    // bundle method on all of them
    vector<InstanceInfo> subset = instances();
    auto with_lp = [](const string& name) {
        for (string s : {"b05200", "c10200", "d10200", "e05100", "e10200", "e20400", "e15900", "real"})
            if (s == name) return true;
        return false;
    };
    const double LIMIT = 300;
    const char* names[] = {"simplex", "interior_point", "bundle"};
    const column_generation::Solver solvers[] = {column_generation::Solver::SIMPLEX,
                                                 column_generation::Solver::INTERIOR_POINT,
                                                 column_generation::Solver::BUNDLE};
    struct Job { int instance; int solver; };
    vector<Job> jobs;
    for (int k = 0; k < (int)subset.size(); k++)
        for (int v = 0; v < 3; v++)
            if (v == 2 || with_lp(subset[k].name)) jobs.push_back({k, v});
    vector<string> rows(jobs.size());
    Checkpoint checkpoint("e14_master");
    parallel_for(jobs.size(), threads, [&](int k) {
        const InstanceInfo& I = subset[jobs[k].instance];
        string solver = names[jobs[k].solver];
        string key = I.name + "|" + solver;
        if (checkpoint.get(key, rows[k])) { rows[k] += "\n"; return; }
        Instance inst(I.path);
        // the same initial columns for all: the knapsack solutions of the
        // subgradient method, whose multipliers start the bundle method
        Solution constructive = metaheuristics::best_constructive(inst);
        vector<pair<int, vector<int>>> columns;
        lagrangian::Result lr = lagrangian::subgradient(inst, constructive.total_cost, 5000, 60, 50, &columns, 100, 10);
        column_generation::Result r = column_generation::root(inst, LIMIT, solvers[jobs[k].solver], columns,
                                                              &lr.multipliers);
        ostringstream out;
        out << fixed << setprecision(3) << I.name << "," << I.group << "," << solver << "," << lr.bound << ","
            << r.bound << "," << r.lp_value << "," << r.converged << "," << r.rounds << "," << r.columns.size() << ","
            << r.seconds << "," << r.lp_seconds << "," << (r.rounds > 0 ? r.lp_seconds / r.rounds : 0.0) << ","
            << r.crossovers << "\n";
        rows[k] = out.str();
        checkpoint.put(key, rows[k].substr(0, rows[k].size() - 1));
        lock_guard<mutex> lock(g_print);
        cout << "  " << I.name << " " << solver << ": bound " << r.bound << ", LP " << r.lp_value
             << (r.converged ? " (converged)" : "") << ", " << r.rounds << " rounds in " << r.seconds << " s" << endl;
    });
    ofstream csv(output("e14_master.csv"));
    csv << "instance,group,solver,lagrangian_bound,bound,lp_value,converged,rounds,columns,time_s,lp_time_s,"
        << "lp_time_per_round_s,crossovers\n";
    for (const string& row : rows) csv << row;
    checkpoint.finish();
    cout << "  -> " << output("e14_master.csv") << "\n";
}

// -----------------------------------------------------------------------------
//  E15 - Sensitivity of the real instance: value of the capacity of each
//  warehouse and robustness to closures and to demand growth
// -----------------------------------------------------------------------------

static Instance with_capacity(const Instance& base, int warehouse, int capacity) {
    Instance inst = base;
    inst.warehouses[warehouse].capacity = capacity;
    return inst;
}

static Instance with_demand_factor(const Instance& base, double factor) {
    Instance inst = base;
    for (Seller& s : inst.sellers)
        for (int& d : s.demand) d = (int)ceil(d * factor - 1e-9);
    return inst;
}

static void exp_e15(int threads) {
    cout << "== E15: sensitivity of the real instance ==\n";
    Instance base("instances/real/real_instance");
    int m = base.warehouses.size();
    const double LIMIT = 300;
    // the optimum of the real instance, start of every re-solve
    Solution start = lagrangian::heuristic_solution(base, 60, 0);
    mip::Result optimum = mip::solve_with_fixing(base, LIMIT, start);
    check(optimum.optimal, "real: optimum proven");
    mip::CapacityDuals duals = mip::capacity_duals(base, LIMIT);
    vector<int> load(m, 0);
    for (int j = 0; j < (int)base.sellers.size(); j++) {
        int i = optimum.solution.assigned_warehouse[j];
        if (i >= 0) load[i] += base.demand(j, i);
    }
    // warehouses studied: the 15 largest shadow prices (capacity worth adding) and
    // the 15 most loaded ones (closures)
    vector<int> by_price(m), by_load(m);
    for (int i = 0; i < m; i++) by_price[i] = by_load[i] = i;
    sort(by_price.begin(), by_price.end(), [&](int a, int b) { return duals.shadow_price[a] > duals.shadow_price[b]; });
    sort(by_load.begin(), by_load.end(), [&](int a, int b) { return load[a] > load[b]; });
    const int EXTRA = 10;   // about the demand of one seller
    struct Job { int warehouse; string scenario; double factor; };
    vector<Job> jobs;
    for (int t = 0; t < 15; t++) jobs.push_back({by_price[t], "capacity_plus_10", 1.0});
    for (int t = 0; t < 15; t++) jobs.push_back({by_load[t], "closure", 1.0});
    for (double f : {1.05, 1.10, 1.20}) jobs.push_back({-1, "demand", f});
    vector<string> rows(jobs.size());
    Checkpoint checkpoint("e15_sensitivity");
    parallel_for(jobs.size(), threads, [&](int k) {
        const Job& job = jobs[k];
        ostringstream key_stream;
        key_stream << job.scenario << "|" << job.warehouse << "|" << job.factor;
        string key = key_stream.str();
        if (checkpoint.get(key, rows[k])) { rows[k] += "\n"; return; }
        Instance inst = base;
        if (job.scenario == "capacity_plus_10")
            inst = with_capacity(base, job.warehouse, base.warehouses[job.warehouse].capacity + EXTRA);
        else if (job.scenario == "closure")
            inst = with_capacity(base, job.warehouse, 0);
        else
            inst = with_demand_factor(base, job.factor);
        // start: the optimal assignment, repaired for the new capacities
        Solution s = optimum.solution;
        vector<int> capacity;
        for (const Warehouse& w : inst.warehouses) capacity.push_back(w.capacity);
        for (int j = 0; j < (int)inst.sellers.size(); j++) {
            int i = s.assigned_warehouse[j];
            if (i < 0) continue;
            if (capacity[i] >= inst.demand(j, i)) capacity[i] -= inst.demand(j, i);
            else s.remove(i, j);
        }
        constructive::complete(inst, s);
        metaheuristics::vnd(s, inst);
        s.total_cost = evaluate(s, inst);
        mip::Result r = mip::solve_with_fixing(inst, LIMIT, s);
        double price = job.warehouse >= 0 ? duals.shadow_price[job.warehouse] : 0.0;
        ostringstream out;
        out << fixed << setprecision(3) << job.scenario << "," << job.warehouse << "," << job.factor << ","
            << (job.warehouse >= 0 ? base.warehouses[job.warehouse].capacity : 0) << ","
            << (job.warehouse >= 0 ? load[job.warehouse] : 0) << "," << price << ","
            << (job.scenario == "capacity_plus_10" ? -EXTRA * price : 0.0) << ","
            << optimum.primal << "," << r.primal << "," << r.dual << "," << r.optimal << ","
            << r.primal - optimum.primal << "," << unassigned(r.solution) << "," << r.seconds << "\n";
        rows[k] = out.str();
        checkpoint.put(key, rows[k].substr(0, rows[k].size() - 1));
        lock_guard<mutex> lock(g_print);
        cout << "  " << job.scenario << " " << job.warehouse << " " << job.factor << ": " << r.primal
             << " (change " << r.primal - optimum.primal << ")" << (r.optimal ? " optimal" : "") << endl;
    });
    ofstream csv(output("e15_sensitivity.csv"));
    csv << "scenario,warehouse,demand_factor,capacity,load,lp_shadow_price,lp_estimated_change,"
        << "base_cost,cost,bound,optimal,change,unassigned,time_s\n";
    for (const string& row : rows) csv << row;
    checkpoint.finish();
    // shadow price of every warehouse, for the figure
    ofstream prices(output("e15_shadow_prices.csv"));
    prices << "warehouse,capacity,load,lp_shadow_price\n" << fixed << setprecision(4);
    for (int i = 0; i < m; i++)
        prices << i << "," << base.warehouses[i].capacity << "," << load[i] << "," << duals.shadow_price[i] << "\n";
    cout << "  -> " << output("e15_sensitivity.csv") << ", " << output("e15_shadow_prices.csv") << "\n";
}

// -----------------------------------------------------------------------------
//  E16 - Better start and re-fixing
// -----------------------------------------------------------------------------

// time limit of the exact part in E16 and time of the MIP-LNS start (a quick
// trial run, "./exact_tests e16-smoke", uses 10 and 5 seconds on two instances)
static double E16_TIME_LIMIT = 300;
static double E16_MIP_LNS_SECONDS = 60;

// The start of E7: the best of the LNS, the LP rounding + VND and the Lagrangian
// heuristic + VND.
static Solution e7_start(const Instance& inst, int n) {
    Solution best = metaheuristics::lns(inst, 1000000000, lns_destroy(n), 1, time_budget(n));
    mip::Rounding rounding = mip::lp_rounding(inst, EXACT_TIME_LIMIT);
    Solution rounded = rounding.solution;
    metaheuristics::vnd(rounded, inst);
    Solution constructive = metaheuristics::best_constructive(inst);
    lagrangian::Result lr = lagrangian::subgradient(inst, constructive.total_cost, 1000000, 60, 50, nullptr, 100, 10);
    if (evaluate(rounded, inst) < evaluate(best, inst)) best = rounded;
    if (lr.has_solution) {
        Solution repaired = lr.solution;
        metaheuristics::vnd(repaired, inst);
        if (evaluate(repaired, inst) < evaluate(best, inst)) best = repaired;
    }
    best.total_cost = evaluate(best, inst);
    return best;
}

// "cost seconds w_0 w_1 ..." (warehouse of each seller, -1 if unassigned) and back
static string encode_start(const Solution& sol, double seconds) {
    ostringstream out;
    out << fixed << setprecision(2) << sol.total_cost << " " << seconds;
    for (int w : sol.assigned_warehouse) out << " " << w;
    return out.str();
}

static Solution decode_start(const Instance& inst, const string& text, double& seconds) {
    istringstream in(text);
    double cost;
    in >> cost >> seconds;
    Solution sol;
    sol.sellers_per_warehouse.assign(inst.warehouses.size(), {});
    sol.assigned_warehouse.assign(inst.sellers.size(), -1);
    for (int j = 0; j < (int)inst.sellers.size(); j++) {
        int w;
        in >> w;
        if (w >= 0) sol.assign(w, j);
    }
    sol.total_cost = evaluate(sol, inst);
    return sol;
}

// The two ideas against the single fixing of E7, on the instances that it does
// not close within 100 s: a better upper bound for the fixing (the MIP-LNS, the
// best heuristic of E9) and the re-fixing during the search, for the compact
// model and for the branch-and-price. The starts are computed once per instance
// and shared by its variants; their time is reported apart (as in E7).
static void exp_e16(int threads, bool smoke = false) {
    cout << "== E16: better start and re-fixing ==\n";
    if (smoke) {
        E16_TIME_LIMIT = 10;
        E16_MIP_LNS_SECONDS = 5;
    }
    map<string, map<string, string>> e7;
    for (const auto& row : read_csv(output("e7_exact.csv"))) e7[row.at("instance")] = row;
    vector<InstanceInfo> subset;
    for (const InstanceInfo& I : instances()) {
        const auto& row = e7.at(I.name);
        if (stoi(row.at("mipfix_optimal")) == 0 || stod(row.at("mipfix_time_s")) > 100) subset.push_back(I);
    }
    if (smoke) subset.resize(2);
    const vector<string> variants = {"fix", "refix", "lns_fix", "lns_refix", "lns_bnp_fix", "lns_bnp_refix"};
    string name = smoke ? "e16_smoke" : "e16_refixing";
    Checkpoint checkpoint(name);

    // starts: that of E7 and the MIP-LNS (Lagrangian start, warehouse destroy)
    vector<Solution> start_e7(subset.size()), start_lns(subset.size());
    vector<double> time_e7(subset.size()), time_lns(subset.size());
    parallel_for(subset.size(), threads, [&](int k) {
        const InstanceInfo& I = subset[k];
        bool needed = false;
        string value;
        for (const string& v : variants) needed = needed || !checkpoint.get(I.name + "|" + v, value);
        if (!needed) return;
        Instance inst(I.path);
        if (checkpoint.get("start|" + I.name + "|e7", value)) {
            start_e7[k] = decode_start(inst, value, time_e7[k]);
        } else {
            time_e7[k] = timed([&] { start_e7[k] = e7_start(inst, I.n); });
            checkpoint.put("start|" + I.name + "|e7", encode_start(start_e7[k], time_e7[k]));
        }
        if (checkpoint.get("start|" + I.name + "|lns", value)) {
            start_lns[k] = decode_start(inst, value, time_lns[k]);
        } else {
            time_lns[k] = timed([&] {
                start_lns[k] = matheuristic::mip_lns(inst, 1000000000, mip_lns_destroy(I.n), 1, E16_MIP_LNS_SECONDS,
                                                     5.0, MIP_LNS_OPERATOR, MIP_LNS_START);
            });
            start_lns[k].total_cost = evaluate(start_lns[k], inst);
            checkpoint.put("start|" + I.name + "|lns", encode_start(start_lns[k], time_lns[k]));
        }
        lock_guard<mutex> lock(g_print);
        cout << "  " << I.name << " starts: E7 " << start_e7[k].total_cost << " (" << time_e7[k] << " s), MIP-LNS "
             << start_lns[k].total_cost << " (" << time_lns[k] << " s)" << endl;
    });

    struct Job { int instance; int variant; };
    vector<Job> jobs;
    for (int k = 0; k < (int)subset.size(); k++)
        for (int v = 0; v < (int)variants.size(); v++) jobs.push_back({k, v});
    vector<string> rows(jobs.size());
    parallel_for(jobs.size(), threads, [&](int t) {
        const InstanceInfo& I = subset[jobs[t].instance];
        const string& variant = variants[jobs[t].variant];
        string key = I.name + "|" + variant;
        if (checkpoint.get(key, rows[t])) { rows[t] += "\n"; return; }
        Instance inst(I.path);
        int k = jobs[t].instance;
        bool lns = variant.rfind("lns", 0) == 0;
        const Solution& start = lns ? start_lns[k] : start_e7[k];
        double start_time = lns ? time_lns[k] : time_e7[k];
        double best = 1e300, bound = -1e300, seconds = 0, first_pct = 0, final_pct = 0, fixing_time = 0;
        bool optimal = false;
        long long nodes = 0;
        int fixed_sellers = 0, incumbents = 0, lagrangian_solutions = 0;
        Solution solution;
        long long pairs = 1;
        if (variant == "fix" || variant == "lns_fix") {
            fixing::Reduction red;
            mip::Result r = mip::solve_with_fixing(inst, E16_TIME_LIMIT, start, &red);
            pairs = max(1LL, red.pairs);
            first_pct = final_pct = 100.0 * red.fixed_to_zero / pairs;
            fixed_sellers = red.fixed_to_one;
            fixing_time = red.seconds;
            best = r.primal; bound = r.dual; optimal = r.optimal; seconds = r.seconds; nodes = r.nodes;
            solution = r.solution;
        } else if (variant == "refix" || variant == "lns_refix") {
            mip::Refixing st;
            mip::Result r = mip::solve_with_refixing(inst, E16_TIME_LIMIT, start, &st);
            pairs = max(1LL, st.pairs);
            first_pct = st.first_fixed_pct;
            final_pct = 100.0 * st.fixed_to_zero / pairs;
            fixed_sellers = st.fixed_to_one;
            incumbents = st.incumbents;
            lagrangian_solutions = st.lagrangian_solutions;
            fixing_time = st.seconds;
            best = r.primal; bound = r.dual; optimal = r.optimal; seconds = r.seconds; nodes = r.nodes;
            solution = r.solution;
        } else if (variant == "lns_bnp_fix") {
            fixing::Reduction red;
            branch_and_price::Result r = branch_and_price::solve_with_fixing(inst, E16_TIME_LIMIT, start,
                                                                            branch_and_price::Options(), &red);
            pairs = max(1LL, red.pairs);
            first_pct = final_pct = 100.0 * red.fixed_to_zero / pairs;
            fixed_sellers = red.fixed_to_one;
            fixing_time = red.seconds;
            best = r.primal; bound = r.dual; optimal = r.optimal; seconds = r.seconds; nodes = r.nodes;
            solution = r.solution;
        } else {
            branch_and_price::Options o;
            o.refixing = true;
            branch_and_price::Result r = branch_and_price::solve(inst, E16_TIME_LIMIT, &start, o);
            pairs = max(1LL, r.pairs);
            first_pct = r.first_fixed_pct;
            final_pct = 100.0 * r.fixed_to_zero / pairs;
            fixed_sellers = r.fixed_to_one;
            incumbents = r.refixing_incumbents;
            lagrangian_solutions = r.lagrangian_solutions;
            fixing_time = r.fixing_seconds;
            best = r.primal; bound = r.dual; optimal = r.optimal; seconds = r.seconds; nodes = r.nodes;
            solution = r.solution;
        }
        const auto& row7 = e7.at(I.name);
        double upper7 = stod(row7.at("best_upper")), lower7 = stod(row7.at("best_lower"));
        check(feasible(solution, inst) && fabs(evaluate(solution, inst) - best) < 1e-6 && best <= start.total_cost + 1e-6,
              key + ": solution feasible, its cost, not worse than the start");
        check(bound <= min(best, upper7) + 1e-6 && best >= lower7 - 1e-6, key + ": bounds consistent with E7");
        ostringstream out;
        out << fixed << setprecision(2) << I.name << "," << I.group << "," << I.n << "," << inst.warehouses.size()
            << "," << variant << "," << start.total_cost << "," << start_time << "," << best << "," << bound << ","
            << optimal << "," << seconds << "," << nodes << "," << first_pct << "," << final_pct << ","
            << fixed_sellers << "," << incumbents << "," << lagrangian_solutions << "," << fixing_time << "\n";
        rows[t] = out.str();
        checkpoint.put(key, rows[t].substr(0, rows[t].size() - 1));
        lock_guard<mutex> lock(g_print);
        cout << "  " << I.name << " " << variant << ": " << best << "/" << bound << (optimal ? " (optimal)" : "")
             << " in " << seconds << " s, fixed " << first_pct << "% -> " << final_pct << "%" << endl;
    });
    ofstream csv(output(name + ".csv"));
    csv << "instance,group,n,m,variant,start_cost,start_time_s,best,bound,optimal,time_s,nodes,"
        << "first_fixed_pct,final_fixed_pct,fixed_sellers,incumbents,lagrangian_solutions,fixing_time_s\n";
    for (const string& row : rows) csv << row;
    checkpoint.finish();
    cout << "  -> " << output(name + ".csv") << "\n";
}

// -----------------------------------------------------------------------------
//  E12 - Rounding of the LP: simple rounding against Shmoys and Tardos
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
//  E17 - Stronger heuristics for type d and the bundle master, on the instances
//  still open
// -----------------------------------------------------------------------------

static const double E17_HEURISTIC_SECONDS = 120;
static const double E17_TIME_LIMIT = 600;

// Phase 1: four heuristics with the same time (seed 1): the MIP-LNS of E9 and
// E16 (Lagrangian start), the tabu search with ejection chains, the MIP-LNS from
// the rounding of the LP and the portfolio. Phase 2: from the solution of the
// portfolio, the compact model with re-fixing (with the multipliers of the
// subgradient method and refined by the bundle method) and the branch-and-price
// with re-fixing (with its usual root and with the bundle root).
static void exp_e17(int threads) {
    cout << "== E17: stronger heuristics and the bundle master on the open instances ==\n";
    vector<InstanceInfo> subset;
    map<string, map<string, string>> gaps;
    for (const auto& row : read_csv(output("e10_true_gaps.csv"))) gaps[row.at("instance")] = row;
    for (const InstanceInfo& I : instances())
        if (gaps.count(I.name) && gaps.at(I.name).at("proven_optimal") == "0") subset.push_back(I);
    const vector<string> heuristics = {"mip_lns", "tabu", "rounding_mip_lns", "portfolio"};
    const vector<string> exact = {"mip_refix", "mip_refix_bundle", "bnp_refix", "bnp_refix_bundle"};
    Checkpoint checkpoint("e17_open");

    // phase 1
    struct Job { int instance; int method; };
    vector<Job> jobs;
    for (int k = 0; k < (int)subset.size(); k++)
        for (int h = 0; h < (int)heuristics.size(); h++) jobs.push_back({k, h});
    vector<string> rows(jobs.size());
    vector<Solution> portfolio(subset.size());
    vector<double> portfolio_time(subset.size());
    parallel_for(jobs.size(), threads, [&](int t) {
        const InstanceInfo& I = subset[jobs[t].instance];
        const string& method = heuristics[jobs[t].method];
        string key = I.name + "|" + method;
        Instance inst(I.path);
        string value;
        if (checkpoint.get(key, value)) {
            rows[t] = value + "\n";
            if (method == "portfolio" && checkpoint.get("start|" + I.name, value))
                portfolio[jobs[t].instance] = decode_start(inst, value, portfolio_time[jobs[t].instance]);
            return;
        }
        int n = I.n;
        Solution sol;
        double seconds = timed([&] {
            if (method == "mip_lns") {
                sol = matheuristic::mip_lns(inst, 1000000000, mip_lns_destroy(n), 1, E17_HEURISTIC_SECONDS, 5.0,
                                            MIP_LNS_OPERATOR, MIP_LNS_START);
            } else if (method == "tabu") {
                sol = ejection::search(inst, E17_HEURISTIC_SECONDS, 1);
            } else if (method == "rounding_mip_lns") {
                Solution rounded = mip::lp_rounding(inst, 0.25 * E17_HEURISTIC_SECONDS).solution;
                metaheuristics::vnd(rounded, inst, 0.05 * E17_HEURISTIC_SECONDS);
                rounded.total_cost = evaluate(rounded, inst);
                sol = matheuristic::mip_lns(inst, 1000000000, mip_lns_destroy(n), 1, 0.7 * E17_HEURISTIC_SECONDS, 5.0,
                                            MIP_LNS_OPERATOR, MIP_LNS_START, &rounded);
            } else {
                sol = matheuristic::portfolio(inst, E17_HEURISTIC_SECONDS, 1);
            }
        });
        sol.total_cost = evaluate(sol, inst);
        check(feasible(sol, inst), key + ": feasible");
        double lower = stod(gaps.at(I.name).at("lower_bound"));
        ostringstream out;
        out << fixed << setprecision(2) << I.name << "," << I.group << "," << I.n << "," << inst.warehouses.size()
            << ",heuristic," << method << "," << sol.total_cost << "," << lower << ",0," << seconds
            << ",0,0,0,0,0";
        rows[t] = out.str() + "\n";
        if (method == "portfolio") {
            portfolio[jobs[t].instance] = sol;
            portfolio_time[jobs[t].instance] = seconds;
            checkpoint.put("start|" + I.name, encode_start(sol, seconds));
        }
        checkpoint.put(key, out.str());
        lock_guard<mutex> lock(g_print);
        cout << "  " << I.name << " " << method << ": " << sol.total_cost << " (lower bound " << lower << ")" << endl;
    });

    // phase 2
    vector<Job> exact_jobs;
    for (int k = 0; k < (int)subset.size(); k++)
        for (int v = 0; v < (int)exact.size(); v++) exact_jobs.push_back({k, v});
    vector<string> exact_rows(exact_jobs.size());
    parallel_for(exact_jobs.size(), threads, [&](int t) {
        const InstanceInfo& I = subset[exact_jobs[t].instance];
        const string& method = exact[exact_jobs[t].method];
        string key = I.name + "|" + method;
        if (checkpoint.get(key, exact_rows[t])) { exact_rows[t] += "\n"; return; }
        Instance inst(I.path);
        const Solution& start = portfolio[exact_jobs[t].instance];
        bool bundle = method.find("bundle") != string::npos;
        double best, bound, seconds, first, final_pct;
        bool optimal;
        long long nodes;
        Solution sol;
        if (method.rfind("mip", 0) == 0) {
            mip::Refixing st;
            mip::Result r = mip::solve_with_refixing(inst, E17_TIME_LIMIT, start, &st, bundle);
            best = r.primal; bound = r.dual; optimal = r.optimal; seconds = r.seconds; nodes = r.nodes; sol = r.solution;
            first = st.first_fixed_pct;
            final_pct = 100.0 * st.fixed_to_zero / max(1LL, st.pairs);
        } else {
            branch_and_price::Options o;
            o.refixing = true;
            o.bundle_root = bundle;
            branch_and_price::Result r = branch_and_price::solve(inst, E17_TIME_LIMIT, &start, o);
            best = r.primal; bound = r.dual; optimal = r.optimal; seconds = r.seconds; nodes = r.nodes; sol = r.solution;
            first = r.first_fixed_pct;
            final_pct = 100.0 * r.fixed_to_zero / max(1LL, r.pairs);
        }
        double lower = stod(gaps.at(I.name).at("lower_bound")), upper = stod(gaps.at(I.name).at("best_upper"));
        check(feasible(sol, inst) && fabs(evaluate(sol, inst) - best) < 1e-6 && best <= start.total_cost + 1e-6,
              key + ": solution feasible, its cost, not worse than the start");
        check(bound <= min(best, upper) + 1e-6 && best >= lower - 1e-6, key + ": bounds consistent with the table of gaps");
        ostringstream out;
        out << fixed << setprecision(2) << I.name << "," << I.group << "," << I.n << "," << inst.warehouses.size()
            << ",exact," << method << "," << best << "," << bound << "," << optimal << "," << seconds << ","
            << nodes << "," << first << "," << final_pct << "," << start.total_cost << "," << portfolio_time[exact_jobs[t].instance];
        exact_rows[t] = out.str() + "\n";
        checkpoint.put(key, out.str());
        lock_guard<mutex> lock(g_print);
        cout << "  " << I.name << " " << method << ": " << best << "/" << bound << (optimal ? " (optimal)" : "")
             << " in " << seconds << " s, fixed " << first << "% -> " << final_pct << "%" << endl;
    });
    ofstream csv(output("e17_open.csv"));
    csv << "instance,group,n,m,phase,method,best,bound,optimal,time_s,nodes,first_fixed_pct,final_fixed_pct,"
        << "start_cost,start_time_s\n";
    for (const string& row : rows) csv << row;
    for (const string& row : exact_rows) csv << row;
    checkpoint.finish();
    cout << "  -> " << output("e17_open.csv") << "\n";
}

static void exp_e12(int threads) {
    cout << "== E12: rounding of Shmoys and Tardos ==\n";
    vector<InstanceInfo> list = instances();
    vector<string> rows(list.size());
    Checkpoint checkpoint("e12_rounding");
    parallel_for(list.size(), threads, [&](int k) {
        const InstanceInfo& I = list[k];
        if (checkpoint.get(I.name, rows[k])) { rows[k] += "\n"; return; }
        Instance inst(I.path);
        mip::ShmoysTardos st = mip::shmoys_tardos(inst, EXACT_TIME_LIMIT);
        Solution repaired = st.repaired;
        double repaired_cost = evaluate(repaired, inst);
        metaheuristics::vnd(repaired, inst);
        mip::Rounding simple = mip::lp_rounding(inst, EXACT_TIME_LIMIT);
        Solution rounded = simple.solution;
        metaheuristics::vnd(rounded, inst);
        check(st.cost <= st.lp_bound + 1e-6 && st.within_bound, I.name + ": Shmoys-Tardos guarantee");
        check(feasible(repaired, inst) && feasible(rounded, inst), I.name + ": roundings feasible");
        ostringstream out;
        out << fixed << setprecision(2)
            << I.name << "," << I.group << "," << I.n << "," << inst.warehouses.size() << ","
            << st.lp_bound << "," << st.cost << "," << st.overloaded << "," << 100.0 * st.largest_overload << ","
            << st.within_bound << "," << repaired_cost << "," << evaluate(repaired, inst) << "," << st.seconds << ","
            << simple.split << "," << simple.solution.total_cost << "," << evaluate(rounded, inst) << "\n";
        rows[k] = out.str();
        checkpoint.put(I.name, rows[k].substr(0, rows[k].size() - 1));
        lock_guard<mutex> lock(g_print);
        cout << "  " << I.name << " lp=" << st.lp_bound << " st=" << st.cost << " overloaded=" << st.overloaded
             << " repaired+vnd=" << evaluate(repaired, inst) << " simple+vnd=" << evaluate(rounded, inst) << "\n";
    });
    ofstream csv(output("e12_rounding.csv"));
    csv << "instance,group,n,m,lp_bound,st_cost,st_overloaded,st_largest_overload_pct,st_within_bound,"
        << "st_repaired_cost,st_repaired_vnd_cost,st_time_s,simple_split,simple_cost,simple_vnd_cost\n";
    for (const string& row : rows) csv << row;
    checkpoint.finish();
    cout << "  -> " << output("e12_rounding.csv") << "\n";
}

// -----------------------------------------------------------------------------
//  True gaps: every method against the best lower bound
// -----------------------------------------------------------------------------

// rows of a CSV file as maps from the column name to the value
static vector<map<string, string>> read_csv(const string& file) {
    vector<map<string, string>> rows;
    ifstream input(file);
    if (!input.is_open()) {
        cerr << "missing " << file << "\n";
        exit(1);
    }
    auto split = [](const string& line) {
        vector<string> fields;
        string field;
        stringstream stream(line);
        while (getline(stream, field, ',')) fields.push_back(field);
        return fields;
    };
    string line;
    getline(input, line);
    vector<string> header = split(line);
    while (getline(input, line)) {
        vector<string> fields = split(line);
        map<string, string> row;
        for (int c = 0; c < (int)header.size() && c < (int)fields.size(); c++) row[header[c]] = fields[c];
        rows.push_back(row);
    }
    return rows;
}

static void exp_gaps() {
    cout << "== True gaps against the best lower bound ==\n";
    auto by_instance = [](const vector<map<string, string>>& rows) {
        map<string, map<string, string>> index;
        for (const auto& row : rows) index[row.at("instance")] = row;
        return index;
    };
    auto e5 = by_instance(read_csv(output("e5_final.csv")));
    auto e7 = by_instance(read_csv(output("e7_exact.csv")));
    auto e9 = by_instance(read_csv(output("e9_matheuristic.csv")));
    vector<map<string, string>> e16;
    if (ifstream(output("e16_refixing.csv")).good()) e16 = read_csv(output("e16_refixing.csv"));
    vector<map<string, string>> e17;
    if (ifstream(output("e17_open.csv")).good()) e17 = read_csv(output("e17_open.csv"));

    // (column of the output, table, column of the table)
    vector<tuple<string, map<string, map<string, string>>*, string>> methods = {
        {"constructive", &e5, "constructive_cost"}, {"vnd", &e5, "vnd_cost"},
        {"ils_best", &e5, "ils_best"}, {"ils_mean", &e5, "ils_mean"},
        {"lns_best", &e5, "lns_best"}, {"lns_mean", &e5, "lns_mean"},
        {"lp_rounding", &e7, "lp_rounding_vnd_cost"}, {"lagrangian_heuristic", &e7, "lagrangian_heuristic_vnd_cost"},
        {"lns_lagrangian_best", &e9, "lns_lagrangian_best"}, {"lns_lagrangian_mean", &e9, "lns_lagrangian_mean"},
        {"mip_lns_best", &e9, "mip_lns_best"}, {"mip_lns_mean", &e9, "mip_lns_mean"},
        {"mip", &e7, "mip_best"}, {"mip_fixing", &e7, "mipfix_best"}, {"bnp_fixing", &e7, "bnp_best"}};

    ofstream csv(output("e10_true_gaps.csv"));
    csv << "instance,group,lower_bound,best_upper,proven_optimal";
    for (const auto& method : methods) csv << ",gap_" << get<0>(method) << "_pct";
    csv << "\n" << fixed << setprecision(3);
    for (const InstanceInfo& I : instances()) {
        auto value = [&](map<string, map<string, string>>& table, const string& column) {
            return stod(table.at(I.name).at(column));
        };
        double lower = value(e7, "best_lower");
        double upper = value(e7, "best_upper");
        for (const auto& method : methods) upper = min(upper, value(*get<1>(method), get<2>(method)));
        // the bounds and solutions of E16 and E17, if they ran
        for (const auto& row : e16) {
            if (row.at("instance") != I.name) continue;
            lower = max(lower, stod(row.at("bound")));
            upper = min(upper, stod(row.at("best")));
        }
        for (const auto& row : e17) {
            if (row.at("instance") != I.name) continue;
            if (row.at("phase") == "exact") lower = max(lower, stod(row.at("bound")));
            upper = min(upper, stod(row.at("best")));
        }
        if (integral_costs(Instance(I.path))) lower = ceil(lower - 1e-6);
        // every cost is that of a feasible solution, so a negative gap is only the
        // tolerance of the solver on the bound
        auto gap = [&](double cost) { return lower > 0 ? max(0.0, (cost - lower) / lower * 100.0) : 0.0; };
        csv << I.name << "," << I.group << "," << lower << "," << upper << "," << (upper - lower < 1e-6);
        for (const auto& method : methods) csv << "," << gap(value(*get<1>(method), get<2>(method)));
        csv << "\n";
    }
    cout << "  -> " << output("e10_true_gaps.csv") << "\n";
}

// -----------------------------------------------------------------------------
//  main / dispatch
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    ensure_output_dir();
    string mode = (argc > 1) ? argv[1] : "check";
    int threads = (argc > 2) ? atoi(argv[2]) : 8;

    if      (mode == "check") { test_check(); }
    else if (mode == "e7")    { exp_e7(threads); }
    else if (mode == "e8")    { exp_e8(threads); }
    else if (mode == "e9")    { exp_e9(threads); }
    else if (mode == "e11")   { exp_e11(threads); }
    else if (mode == "e12")   { exp_e12(threads); }
    else if (mode == "e13")   { exp_e13(threads); }
    else if (mode == "e14")   { exp_e14(threads); }
    else if (mode == "e15")   { exp_e15(threads); }
    else if (mode == "e16")   { exp_e16(threads); }
    else if (mode == "e16-smoke") { exp_e16(threads, true); }
    else if (mode == "e17")   { exp_e17(threads); }
    else if (mode == "bundle") {
        char dir_template[] = "/tmp/gap_bundle_XXXXXX";
        mt19937 rng(2027);
        test_bundle(mkdtemp(dir_template), rng);
        cout << "  " << g_pass << " PASS, " << g_fail << " FAIL\n";
    }
    else if (mode == "refixing") {
        char dir_template[] = "/tmp/gap_refixing_XXXXXX";
        mt19937 rng(2026);
        test_refixing(mkdtemp(dir_template), rng);
        cout << "  " << g_pass << " PASS, " << g_fail << " FAIL\n";
    }
    else if (mode == "gaps")  { exp_gaps(); }
    else {
        cerr << "unknown mode: " << mode << "\n"
             << "usage: ./exact_tests [check|refixing|e7|e8|e9|e11|e12|e13|e14|e15|e16|e17|gaps] [threads]\n";
        return 1;
    }

    if (g_fail > 0) {
        cerr << "\n*** " << g_fail << " test(s) failed ***\n";
        return 1;
    }
    return 0;
}

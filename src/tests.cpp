// =============================================================================
//  tests.cpp - tests and experiments of the heuristics for the GAP (ThunderPack)
//
//  A single driver that:
//    1) checks correctness (feasibility + incremental cost == recomputed cost);
//    2) runs every method on the 58 instances and writes the CSV files of the
//       tables E1-E6 of the report (best/worst/mean = empirical best and worst
//       case);
//    3) builds synthetic instances that trigger the worst and best case of each
//       kind of method (constructive, local search, metaheuristic).
//
//  BUILD (from src/):  make tests
//  RUN:
//    ./tests                 everything (check + synthetic + e1..e6 + convergence)
//    ./tests check           correctness tests only
//    ./tests e1 .. e6        a single experiment (e5 [threads]: instances in parallel)
//    ./tests convergence     incumbent cost against time (ILS and LNS)
//    ./tests synthetic       worst and best case on synthetic instances
//
//  The CSV files are written to src/results/.
//  NOTE: run from src/ (the instance paths are relative).
// =============================================================================

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "ConstructiveHeuristics.h"
#include "Experiments.h"
#include "Instance.h"
#include "LocalSearch.h"
#include "Metaheuristics.h"
#include "Solution.h"

using namespace std;
using namespace experiments;

// -----------------------------------------------------------------------------
//  Correctness tests
// -----------------------------------------------------------------------------

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const string& desc) {
    if (ok) { g_pass++; }
    else { g_fail++; cout << "  [FAIL] " << desc << "\n"; }
}

static void test_check() {
    cout << "== Correctness tests ==\n";
    const double EPS = 1e-6;
    // small representative instances (one of each group)
    vector<string> cases = {
        "instances/gap/gap_a/a05100",
        "instances/gap/gap_b/b05100",
        "instances/gap/gap_e/e05100",
    };
    for (const string& path : cases) {
        Instance inst(path);

        // --- constructive heuristics: feasible and total_cost matches evaluate ---
        Solution nn = constructive::nearest_neighbor(inst);
        check(feasible(nn, inst), path + ": nearest_neighbor feasible");
        check(fabs(nn.total_cost - evaluate(nn, inst)) < EPS,
              path + ": nearest_neighbor total_cost == evaluate");

        Solution ri = constructive::regret_insertion(inst);
        check(feasible(ri, inst), path + ": regret_insertion feasible");
        check(fabs(ri.total_cost - evaluate(ri, inst)) < EPS,
              path + ": regret_insertion total_cost == evaluate");

        // --- regret heuristic of Martello and Toth: with the cost it is the regret
        //     insertion, and every measure gives a feasible, consistent solution ---
        Solution rc = constructive::regret_greedy(inst, constructive::Desirability::COST);
        check(rc.assigned_warehouse == ri.assigned_warehouse,
              path + ": regret_greedy(COST) == regret_insertion");
        for (auto d : {constructive::Desirability::DEMAND, constructive::Desirability::RELATIVE_DEMAND}) {
            Solution rg = constructive::regret_greedy(inst, d);
            check(feasible(rg, inst) && fabs(rg.total_cost - evaluate(rg, inst)) < EPS,
                  path + ": regret_greedy feasible and consistent");
        }

        // --- completion of a partial solution: feasible, consistent, and it keeps
        //     the sellers already assigned ---
        Solution partial = rc;
        for (int j = 0; j < (int)partial.assigned_warehouse.size(); j += 3) {
            if (partial.assigned_warehouse[j] >= 0) partial.remove(partial.assigned_warehouse[j], j);
        }
        Solution completed = partial;
        constructive::complete(inst, completed);
        bool kept = true;
        for (int j = 0; j < (int)partial.assigned_warehouse.size(); j++) {
            if (partial.assigned_warehouse[j] >= 0)
                kept = kept && completed.assigned_warehouse[j] == partial.assigned_warehouse[j];
        }
        check(kept && feasible(completed, inst) && fabs(completed.total_cost - evaluate(completed, inst)) < EPS
              && unassigned(completed) <= unassigned(partial),
              path + ": complete keeps the partial solution, feasible and consistent");

        // --- local search: the incremental cost (delta) still matches the
        //     recomputed one, and the solution stays feasible ---
        vector<pair<string, function<bool(Solution&, const Instance&)>>> ops = {
            {"relocate_first", local_search::relocate_first_improvement},
            {"relocate_best",  local_search::relocate_best_improvement},
            {"swap_first",     local_search::swap_first_improvement},
            {"swap_best",      local_search::swap_best_improvement},
        };
        for (auto& op : ops) {
            Solution s = nn;
            while (op.second(s, inst)) {}   // until a local optimum
            check(feasible(s, inst), path + ": " + op.first + " feasible");
            check(fabs(s.total_cost - evaluate(s, inst)) < EPS,
                  path + ": " + op.first + " incremental cost == evaluate");
            check(evaluate(s, inst) <= evaluate(nn, inst) + EPS,
                  path + ": " + op.first + " does not worsen");
        }

        // --- full VND: feasible and consistent cost ---
        Solution v = nn;
        metaheuristics::vnd(v, inst);
        check(feasible(v, inst), path + ": vnd feasible");
        check(fabs(v.total_cost - evaluate(v, inst)) < EPS, path + ": vnd consistent cost");
    }
    cout << "  " << g_pass << " PASS, " << g_fail << " FAIL\n";
}

// -----------------------------------------------------------------------------
//  E1 - Constructive heuristics (RQ1)
// -----------------------------------------------------------------------------

static void exp_e1() {
    cout << "== E1: constructive heuristics ==\n";
    ofstream csv(output("e1_constructive.csv"));
    csv << "instance,group,n,"
        << "nn_cost,nn_unassigned,nn_time_s,"
        << "ri_cost,ri_unassigned,ri_time_s,"
        << "rd_cost,rd_unassigned,rd_time_s,"
        << "rr_cost,rr_unassigned,rr_time_s\n";
    csv << fixed << setprecision(2);

    for (const auto& I : instances()) {
        Instance inst(I.path);
        Solution nn; double tn = timed([&] { nn = constructive::nearest_neighbor(inst); });
        Solution ri; double tr = timed([&] { ri = constructive::regret_insertion(inst); });
        Solution rd; double td = timed([&] {
            rd = constructive::regret_greedy(inst, constructive::Desirability::DEMAND); });
        Solution rr; double tq = timed([&] {
            rr = constructive::regret_greedy(inst, constructive::Desirability::RELATIVE_DEMAND); });
        csv << I.name << "," << I.group << "," << I.n << ","
            << evaluate(nn, inst) << "," << unassigned(nn) << "," << tn << ","
            << evaluate(ri, inst) << "," << unassigned(ri) << "," << tr << ","
            << evaluate(rd, inst) << "," << unassigned(rd) << "," << td << ","
            << evaluate(rr, inst) << "," << unassigned(rr) << "," << tq << "\n";
        cout << "  " << I.name << " nn=" << evaluate(nn, inst) << " ri=" << evaluate(ri, inst)
             << " rd=" << evaluate(rd, inst) << " rr=" << evaluate(rr, inst) << "\n";
    }
    cout << "  -> " << output("e1_constructive.csv") << "\n";
}

// -----------------------------------------------------------------------------
//  E2 - Local search operators (RQ2)
// -----------------------------------------------------------------------------

static void exp_e2() {
    cout << "== E2: local search operators ==\n";
    ofstream csv(output("e2_local_search.csv"));
    csv << "instance,group,constructive,operator,"
        << "initial_cost,final_cost,improvement_pct,time_s,passes\n";
    csv << fixed << setprecision(4);

    // generalized operators (relocate and swap multiple): they have no first and
    // best variants and apply a fixed strategy. To evaluate them alone, each "pass"
    // tries q = 1..QMAX and applies the first (q, pair) that improves, as in the
    // VND (QMAX = 10).
    const int QMAX = 10;
    auto relocate_mult = [QMAX](Solution& s, const Instance& inst) -> bool {
        for (int q = 1; q <= QMAX; q++)
            if (local_search::relocate_multiple(s, inst, q)) return true;
        return false;
    };
    auto swap_mult = [QMAX](Solution& s, const Instance& inst) -> bool {
        for (int q = 1; q <= QMAX; q++)
            if (local_search::swap_multiple(s, inst, q)) return true;
        return false;
    };

    vector<pair<string, function<bool(Solution&, const Instance&)>>> ops = {
        {"relocate_first",    local_search::relocate_first_improvement},
        {"relocate_best",     local_search::relocate_best_improvement},
        {"swap_first",        local_search::swap_first_improvement},
        {"swap_best",         local_search::swap_best_improvement},
        {"relocate_multiple", relocate_mult},
        {"swap_multiple",     swap_mult},
    };

    for (const auto& I : instances()) {
        Instance inst(I.path);
        Solution nn = constructive::nearest_neighbor(inst);
        Solution ri = constructive::regret_insertion(inst);
        Solution rd = constructive::regret_greedy(inst, constructive::Desirability::DEMAND);
        vector<pair<string, Solution*>> bases = {
            {"nearest_neighbor", &nn}, {"regret_insertion", &ri}, {"regret_demand", &rd}};

        for (auto& base : bases) {
            double initial = evaluate(*base.second, inst);
            for (auto& op : ops) {
                Solution s = *base.second;
                int passes = 0;
                double t = timed([&] { while (op.second(s, inst)) passes++; });
                double final_cost = evaluate(s, inst);
                double improvement = initial > 0 ? (initial - final_cost) / initial * 100.0 : 0.0;
                csv << I.name << "," << I.group << "," << base.first << ","
                    << op.first << "," << initial << "," << final_cost << ","
                    << improvement << "," << t << "," << passes << "\n";
            }
        }
        cout << "  " << I.name << " done\n";
    }
    cout << "  -> " << output("e2_local_search.csv") << "\n";
}

// -----------------------------------------------------------------------------
//  E3 - VND against single operators (RQ3)
// -----------------------------------------------------------------------------

// local search compositions to compare against the full VND
static void relocate_only(Solution& s, const Instance& inst) {
    while (local_search::relocate_first_improvement(s, inst)) {}
}
static void swap_only(Solution& s, const Instance& inst) {
    while (local_search::swap_first_improvement(s, inst)) {}
}
static void vnd_relocate_swap(Solution& s, const Instance& inst) {
    bool improved = true;
    while (improved) {
        improved = false;
        while (local_search::relocate_first_improvement(s, inst)) improved = true;
        while (local_search::swap_first_improvement(s, inst)) improved = true;
    }
}

static void exp_e3() {
    cout << "== E3: VND against single operators ==\n";
    ofstream csv(output("e3_vnd.csv"));
    csv << "instance,group,method,initial_cost,final_cost,improvement_pct,time_s\n";
    csv << fixed << setprecision(4);

    vector<pair<string, function<void(Solution&, const Instance&)>>> methods = {
        {"relocate_only",     relocate_only},
        {"swap_only",         swap_only},
        {"vnd_relocate_swap", vnd_relocate_swap},
        {"full_vnd",          [](Solution& s, const Instance& i) { metaheuristics::vnd(s, i); }},
    };

    for (const auto& I : instances()) {
        Instance inst(I.path);
        // best constructive heuristic as the starting point
        Solution base = metaheuristics::best_constructive(inst);
        double initial = evaluate(base, inst);

        for (auto& m : methods) {
            Solution s = base;
            double t = timed([&] { m.second(s, inst); });
            double final_cost = evaluate(s, inst);
            double improvement = initial > 0 ? (initial - final_cost) / initial * 100.0 : 0.0;
            csv << I.name << "," << I.group << "," << m.first << ","
                << initial << "," << final_cost << "," << improvement << "," << t << "\n";
        }
        cout << "  " << I.name << " done\n";
    }
    cout << "  -> " << output("e3_vnd.csv") << "\n";
}

// -----------------------------------------------------------------------------
//  E4 - Parameter tuning (RQ4): differences between parameters
// -----------------------------------------------------------------------------

static void exp_e4() {
    cout << "== E4: parameter tuning (ILS and LNS) ==\n";
    ofstream csv(output("e4_tuning.csv"));
    csv << "instance,group,metaheuristic,parameter,value,seconds,seeds,"
        << "best_cost,worst_cost,mean_cost,std_dev,mean_time_s\n";
    csv << fixed << setprecision(2);

    // representative subset: one per size tier, including a tight case (b and e)
    vector<InstanceInfo> subset = {
        {"instances/gap/gap_b/b10200", "b10200", 'b', 200},
        {"instances/gap/gap_e/e15900", "e15900", 'e', 900},
        {"instances/real/real_instance", "real", 'r', 1100},
    };
    vector<int> strengths = {5, 10, 20, 40};   // perturbation of the ILS
    vector<int> destroys = {20, 50, 100};      // destruction size of the LNS

    auto row = [&](const InstanceInfo& I, const string& mh, const string& param, int value,
                   double seconds, const function<Solution(unsigned)>& run, const Instance& inst) {
        vector<double> costs, times;
        for (unsigned s = 1; s <= SEEDS; s++) {
            Solution sol;
            double t = timed([&] { sol = run(s); });
            costs.push_back(evaluate(sol, inst));
            times.push_back(t);
        }
        Stats st = summarize(costs);
        double mean_time = accumulate(times.begin(), times.end(), 0.0) / times.size();
        csv << I.name << "," << I.group << "," << mh << "," << param << "," << value << ","
            << seconds << "," << SEEDS << "," << st.best << "," << st.worst << "," << st.mean << ","
            << st.std_dev << "," << mean_time << "\n";
        cout << "  " << I.name << " " << mh << " " << param << "=" << value << " mean=" << st.mean << "\n";
    };

    for (const auto& I : subset) {
        Instance inst(I.path);
        double seconds = time_budget(I.n);
        for (int f : strengths) {
            row(I, "ILS", "strength", f, seconds,
                [&](unsigned s) { return metaheuristics::ils(inst, 1000000000, f, s, seconds); }, inst);
        }
        for (int d : destroys) {
            row(I, "LNS", "destroy", d, seconds,
                [&](unsigned s) { return metaheuristics::lns(inst, 1000000000, d, s, seconds); }, inst);
        }
    }
    cout << "  -> " << output("e4_tuning.csv") << "\n";
}

// -----------------------------------------------------------------------------
//  E5 - Final comparison (RQ5): constructive -> VND -> ILS / LNS
// -----------------------------------------------------------------------------

// E5 runs the instances on 'threads' threads (each run of a metaheuristic is
// single-threaded, so its time budget means the same as in a sequential run).
static void exp_e5(int threads) {
    cout << "== E5: final comparison ==\n";
    vector<InstanceInfo> list = instances();
    vector<string> rows(list.size());
    mutex print;
    atomic<int> next(0);
    auto worker = [&] {
        for (int k = next++; k < (int)list.size(); k = next++) {
            const InstanceInfo& I = list[k];
            Instance inst(I.path);
            double seconds = time_budget(I.n);

            Solution base = metaheuristics::best_constructive(inst);
            double c_constructive = evaluate(base, inst);

            Solution vnd = base;
            metaheuristics::vnd(vnd, inst);
            double c_vnd = evaluate(vnd, inst);

            // a metaheuristic over SEEDS seeds: cost statistics and mean time
            auto seeds = [&](const function<Solution(unsigned)>& run, double& mean_time) {
                vector<double> costs, times;
                for (unsigned s = 1; s <= SEEDS; s++) {
                    Solution sol;
                    double t = timed([&] { sol = run(s); });
                    costs.push_back(evaluate(sol, inst));
                    times.push_back(t);
                }
                mean_time = accumulate(times.begin(), times.end(), 0.0) / times.size();
                return summarize(costs);
            };
            double ils_time, lns_time;
            Stats ils = seeds([&](unsigned s) {
                return metaheuristics::ils(inst, 1000000000, DEFAULT_STRENGTH, s, seconds);
            }, ils_time);
            int destroy = lns_destroy(I.n);
            Stats lns = seeds([&](unsigned s) {
                return metaheuristics::lns(inst, 1000000000, destroy, s, seconds);
            }, lns_time);

            double best = min({c_constructive, c_vnd, ils.best, lns.best});
            auto gap = [&](double c) { return best > 0 ? (c - best) / best * 100.0 : 0.0; };

            ostringstream row;
            row << fixed << setprecision(2)
                << I.name << "," << I.group << "," << I.n << ","
                << c_constructive << "," << c_vnd << ","
                << ils.best << "," << ils.worst << "," << ils.mean << "," << ils.std_dev << "," << ils_time << ","
                << lns.best << "," << lns.worst << "," << lns.mean << "," << lns.std_dev << "," << lns_time << ","
                << best << "," << gap(c_constructive) << "," << gap(c_vnd) << ","
                << gap(ils.best) << "," << gap(lns.best) << "\n";
            rows[k] = row.str();
            lock_guard<mutex> lock(print);
            cout << "  " << I.name << " constructive=" << c_constructive
                 << " vnd=" << c_vnd << " ils=" << ils.best << " lns=" << lns.best << "\n";
        }
    };
    vector<thread> pool;
    for (int t = 0; t < max(1, threads); t++) pool.emplace_back(worker);
    for (thread& th : pool) th.join();

    ofstream csv(output("e5_final.csv"));
    csv << "instance,group,n,"
        << "constructive_cost,vnd_cost,"
        << "ils_best,ils_worst,ils_mean,ils_std,ils_mean_time_s,"
        << "lns_best,lns_worst,lns_mean,lns_std,lns_mean_time_s,"
        << "best_found,gap_constructive_pct,gap_vnd_pct,gap_ils_pct,gap_lns_pct\n";
    for (const string& row : rows) csv << row;
    cout << "  -> " << output("e5_final.csv") << "\n";
}

// -----------------------------------------------------------------------------
//  E6 - Real case (RQ6)
// -----------------------------------------------------------------------------

static void exp_e6() {
    cout << "== E6: real case ==\n";
    ofstream csv(output("e6_real.csv"));
    csv << "method,seeds,best_cost,worst_cost,mean_cost,cost_std,"
        << "unassigned,mean_distance,mean_time_s\n";
    csv << fixed << setprecision(2);

    Instance inst("instances/real/real_instance");
    double seconds = time_budget(1100);

    // row of a deterministic method: one run, so the statistics collapse to the
    // single value (deviation 0). unassigned and mean_distance come from that solution.
    auto deterministic_row = [&](const string& name, const Solution& s, double t) {
        double c = evaluate(s, inst);
        csv << name << ",1," << c << "," << c << "," << c << ",0.00,"
            << unassigned(s) << "," << mean_distance(s, inst) << "," << t << "\n";
        cout << "  " << name << " cost=" << c << " unassigned=" << unassigned(s) << "\n";
    };

    // row of a metaheuristic: SEEDS runs, reporting best, worst, mean and deviation
    // and the mean time; unassigned and mean_distance are those of the best solution.
    auto metaheuristic_row = [&](const string& name, const function<Solution(unsigned)>& run) {
        vector<double> costs, times;
        Solution best_sol;
        double best_cost = 1e300;
        for (unsigned s = 1; s <= SEEDS; s++) {
            Solution sol;
            double t = timed([&] { sol = run(s); });
            double c = evaluate(sol, inst);
            costs.push_back(c);
            times.push_back(t);
            if (c < best_cost) { best_cost = c; best_sol = sol; }
        }
        Stats st = summarize(costs);
        double mean_time = accumulate(times.begin(), times.end(), 0.0) / times.size();
        csv << name << "," << SEEDS << ","
            << st.best << "," << st.worst << "," << st.mean << "," << st.std_dev << ","
            << unassigned(best_sol) << "," << mean_distance(best_sol, inst) << "," << mean_time << "\n";
        cout << "  " << name << " best=" << st.best << " mean=" << st.mean << " (" << SEEDS << " seeds)\n";
    };

    Solution base = metaheuristics::best_constructive(inst);
    deterministic_row("best_constructive", base, 0.0);

    Solution vnd = base;
    double tv = timed([&] { metaheuristics::vnd(vnd, inst); });
    deterministic_row("+VND", vnd, tv);

    metaheuristic_row("+ILS", [&](unsigned s) {
        return metaheuristics::ils(inst, 1000000000, DEFAULT_STRENGTH, s, seconds);
    });
    metaheuristic_row("+LNS", [&](unsigned s) {
        return metaheuristics::lns(inst, 1000000000, lns_destroy(1100), s, seconds);
    });

    cout << "  -> " << output("e6_real.csv") << "\n";
}

// -----------------------------------------------------------------------------
//  Convergence: incumbent cost against the time budget (for fig:conv)
// -----------------------------------------------------------------------------

static void exp_convergence() {
    cout << "== Convergence: cost against time ==\n";
    ofstream csv(output("convergence.csv"));
    csv << "instance,method,seconds,cost\n";
    csv << fixed << setprecision(2);

    struct Case { string path; string name; int n; vector<double> budgets; };
    vector<Case> cases = {
        {"instances/gap/gap_e/e20200", "e20200", 200, {0.25, 0.5, 1, 2, 3, 5, 8, 10}},
        {"instances/real/real_instance", "real", 1100, {1, 2, 5, 10, 20, 30, 45, 60}},
    };

    for (auto& c : cases) {
        Instance inst(c.path);
        int destroy = lns_destroy(c.n);
        for (double t : c.budgets) {
            // each budget restarts the metaheuristic (seed 1) and stops at 't' seconds
            Solution si = metaheuristics::ils(inst, 1000000000, DEFAULT_STRENGTH, 1, t);
            csv << c.name << ",ILS," << t << "," << evaluate(si, inst) << "\n";
            Solution sl = metaheuristics::lns(inst, 1000000000, destroy, 1, t);
            csv << c.name << ",LNS," << t << "," << evaluate(sl, inst) << "\n";
        }
        cout << "  " << c.name << " done\n";
    }
    cout << "  -> " << output("convergence.csv") << "\n";
}

// -----------------------------------------------------------------------------
//  Synthetic instances: worst and best case of each kind of method
// -----------------------------------------------------------------------------

// writes an instance in the OR-Library format (demand d_ij = d_j, equal in every warehouse)
static void write_instance(const string& path, int m, int n,
                           const vector<vector<double>>& cost_mn,
                           const vector<int>& demand,
                           const vector<int>& capacity) {
    ofstream f(path);
    f << m << " " << n << "\n";
    for (int i = 0; i < m; i++) {              // m x n cost matrix
        for (int j = 0; j < n; j++) f << cost_mn[i][j] << " ";
        f << "\n";
    }
    for (int i = 0; i < m; i++) {              // m x n demand matrix (d_ij = d_j)
        for (int j = 0; j < n; j++) f << demand[j] << " ";
        f << "\n";
    }
    for (int i = 0; i < m; i++) f << capacity[i] << " ";
    f << "\n";
}

static void test_synthetic() {
    cout << "== Synthetic instances: worst and best case ==\n";
    ofstream csv(output("synthetic.csv"));
    csv << "case,method,cost\n";
    csv << fixed << setprecision(2);
    string dir = string(OUTPUT_DIR) + "/synthetic_instances";
    mkdir(dir.c_str(), 0755);

    // ---------------------------------------------------------------------
    //  CASE 1 - constructive heuristics: myopic greedy (worst case of the
    //  nearest neighbor, best case of the regret insertion). m=2, n=2, demand 1.
    //  s0: w0=1, w1=2 ; s1: w0=1, w1=100. Capacities 1 and 1.
    //  Greedy: s0->w0 (1), s1->w0 full -> w1 (100) = 101.
    //  Regret: gives priority to s1 (regret 99) -> w0 (1), then s0->w1 (2) = 3.
    // ---------------------------------------------------------------------
    {
        string p = dir + "/greedy_myopic";
        // cost_mn[i][j]: warehouse i, seller j
        write_instance(p, 2, 2,
            {{1, 1}, {2, 100}},   // w0: s0=1, s1=1 ; w1: s0=2, s1=100
            {1, 1}, {1, 1});
        Instance inst(p);
        double cn = evaluate(constructive::nearest_neighbor(inst), inst);
        double cr = evaluate(constructive::regret_insertion(inst), inst);
        cout << "  [C1 myopic greedy] nearest_neighbor=" << cn << "  regret_insertion=" << cr << "\n";
        csv << "greedy_myopic,nearest_neighbor," << cn << "\n";
        csv << "greedy_myopic,regret_insertion," << cr << "\n";
        check(cn > cr, "C1: greedy worse than regret (myopic greedy)");
    }

    // ---------------------------------------------------------------------
    //  CASE 2 - constructive heuristics: both optimal (best case of both).
    //  Each seller has a clearly closest warehouse with room.
    // ---------------------------------------------------------------------
    {
        string p = dir + "/both_optimal";
        write_instance(p, 2, 2,
            {{1, 100}, {100, 1}},   // s0 prefers w0, s1 prefers w1
            {1, 1}, {1, 1});
        Instance inst(p);
        double cn = evaluate(constructive::nearest_neighbor(inst), inst);
        double cr = evaluate(constructive::regret_insertion(inst), inst);
        cout << "  [C2 both optimal] nearest_neighbor=" << cn << "  regret_insertion=" << cr << "\n";
        csv << "both_optimal,nearest_neighbor," << cn << "\n";
        csv << "both_optimal,regret_insertion," << cr << "\n";
        check(cn == cr && cn == 2, "C2: both constructive heuristics optimal (=2)");
    }

    // ---------------------------------------------------------------------
    //  CASE 3 - local search: a local optimum from which relocate+swap do NOT
    //  escape but the cyclic exchange (VND) does. m=3, n=3, all capacities 1
    //  (full warehouses). Diagonal assignment s_i->w_i (cost 10 each = 30). No
    //  swap improves and relocate is impossible (no room). The rotation
    //  s0->w1, s1->w2, s2->w0 costs 3. "Forward" edges = 1, "backward" = 100,
    //  diagonal = 10.
    // ---------------------------------------------------------------------
    {
        string p = dir + "/cycle3";
        // cost_mn[i][j]: warehouse i (row), seller j (column)
        // c[s][w]: diagonal=10; s0->w1=1, s1->w2=1, s2->w0=1; the rest=100
        // -> per warehouse: w0: s0=10, s1=100, s2=1
        //                   w1: s0=1,  s1=10,  s2=100
        //                   w2: s0=100, s1=1,  s2=10
        write_instance(p, 3, 3,
            {{10, 100, 1}, {1, 10, 100}, {100, 1, 10}},
            {1, 1, 1}, {1, 1, 1});
        Instance inst(p);

        // the diagonal assignment, built by hand (the trap local optimum)
        Solution s;
        s.sellers_per_warehouse.resize(3);
        s.assigned_warehouse.assign(3, -1);
        s.assign(0, 0); s.assign(1, 1); s.assign(2, 2);
        s.total_cost = evaluate(s, inst);
        double diagonal = s.total_cost;

        // relocate + swap: they cannot escape
        Solution ls = s;
        vnd_relocate_swap(ls, inst);
        double c_ls = evaluate(ls, inst);

        // full VND (with the cyclic exchange): it escapes
        Solution v = s;
        metaheuristics::vnd(v, inst);
        double c_vnd = evaluate(v, inst);

        cout << "  [C3 cycle3] diagonal=" << diagonal << "  relocate+swap=" << c_ls
             << "  full_vnd=" << c_vnd << "\n";
        csv << "cycle3,diagonal_start," << diagonal << "\n";
        csv << "cycle3,relocate+swap," << c_ls << "\n";
        csv << "cycle3,full_vnd," << c_vnd << "\n";
        check(c_ls == diagonal, "C3: relocate+swap stuck in the local optimum");
        check(c_vnd < c_ls, "C3: the VND (cyclic exchange) escapes the local optimum");
    }

    // ---------------------------------------------------------------------
    //  CASE 4 - metaheuristic: best case (the ILS recovers the optimum from a
    //  bad constructive solution) and worst case (an instance that is already
    //  optimal: overhead only).
    // ---------------------------------------------------------------------
    {
        // best case: greedy_myopic again (greedy = 101, optimum = 3)
        Instance inst(dir + "/greedy_myopic");
        double greedy = evaluate(constructive::nearest_neighbor(inst), inst);
        Solution ils = metaheuristics::ils(inst, 200, 5, 1, 0);
        double c_ils = evaluate(ils, inst);
        cout << "  [C4 MH best case] greedy=" << greedy << "  ILS=" << c_ils << "\n";
        csv << "mh_best_case,greedy," << greedy << "\n";
        csv << "mh_best_case,ILS," << c_ils << "\n";
        check(c_ils < greedy, "C4: the ILS improves a lot over the bad constructive solution");

        // worst case: an instance that is already optimal, the ILS cannot improve (pure overhead)
        Instance inst2(dir + "/both_optimal");
        double optimum = evaluate(constructive::nearest_neighbor(inst2), inst2);
        Solution ils2 = metaheuristics::ils(inst2, 200, 5, 1, 0);
        double c_ils2 = evaluate(ils2, inst2);
        cout << "  [C4 MH worst case] optimum=" << optimum << "  ILS=" << c_ils2 << "\n";
        csv << "mh_worst_case,optimum," << optimum << "\n";
        csv << "mh_worst_case,ILS," << c_ils2 << "\n";
        check(c_ils2 == optimum, "C4: the ILS does not improve an instance that is already optimal (overhead)");
    }

    cout << "  -> " << output("synthetic.csv") << "\n";
    cout << "  " << g_pass << " PASS, " << g_fail << " FAIL (cumulative)\n";
}

// -----------------------------------------------------------------------------
//  main / dispatch
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    ensure_output_dir();
    string mode = (argc > 1) ? argv[1] : "all";

    if      (mode == "check")       { test_check(); }
    else if (mode == "e1")          { exp_e1(); }
    else if (mode == "e2")          { exp_e2(); }
    else if (mode == "e3")          { exp_e3(); }
    else if (mode == "e4")          { exp_e4(); }
    else if (mode == "e5")          { exp_e5(argc > 2 ? atoi(argv[2]) : 1); }
    else if (mode == "e6")          { exp_e6(); }
    else if (mode == "convergence") { exp_convergence(); }
    else if (mode == "synthetic")   { test_synthetic(); }
    else if (mode == "all") {
        test_check();
        test_synthetic();
        exp_e1();
        exp_e2();
        exp_e3();
        exp_e4();
        exp_e5(1);
        exp_e6();
        exp_convergence();
    } else {
        cerr << "unknown mode: " << mode << "\n"
             << "usage: ./tests [check|e1|e2|e3|e4|e5 [threads]|e6|convergence|synthetic|all]\n";
        return 1;
    }

    if (g_fail > 0) {
        cerr << "\n*** " << g_fail << " test(s) failed ***\n";
        return 1;
    }
    return 0;
}

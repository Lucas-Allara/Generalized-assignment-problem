#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include "BranchAndPrice.h"
#include "ColumnGeneration.h"
#include "EjectionChains.h"
#include "Instance.h"
#include "LagrangianBound.h"
#include "Matheuristic.h"
#include "Metaheuristics.h"
#include "MipModel.h"
#include "SolutionIO.h"
#include "VariableFixing.h"

using namespace std;

// usage: ./gap_exact <method> <instance> [seconds] [output]
//   mip           compact integer model with SCIP
//   mipfix        the same after Lagrangian variable fixing
//   bc            branch-and-cut with exact knapsack separation
//   lp            its linear relaxation
//   rounding      rounding of the linear relaxation, improved with the VND
//   st            rounding of Shmoys and Tardos, repaired and improved with the VND
//   covers        linear relaxation with cover inequalities and with exact knapsack
//                 separation, and the root of SCIP
//   cg            master of the root: column generation with HiGHS (simplex and interior
//                 point) and the bundle method
//   duals         shadow prices of the capacities (LP relaxation)
//   fixing        Lagrangian variable fixing from the Lagrangian heuristic
//   lagrangian    Lagrangian bound (relaxing the assignment constraints) and the
//                 Lagrangian heuristic improved with the VND
//   bnp           branch-and-price
//   bnpfix        the same after Lagrangian variable fixing
//   refix         compact model with re-fixing (Lagrangian fixing after every new
//                 incumbent), started from the matheuristic
//   bnprefix      branch-and-price with re-fixing, started from the matheuristic
//   matheuristic  LNS with SCIP repair, started from the Lagrangian heuristic
//   tabu          tabu search with ejection chains through infeasible assignments
//   portfolio     rounding and Lagrangian heuristic, then tabu search and MIP-LNS
// The compact model, the branch-and-cut and the branch-and-price start from the
// best constructive solution + VND; their versions with fixing start from the
// Lagrangian heuristic; those with re-fixing, from the matheuristic run for
// min(60, seconds / 5) seconds (not counted in 'seconds'). If 'output' is given, the methods that build a solution write
// it in the format of gap_simulator.
int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "usage: " << argv[0]
             << " <mip|mipfix|bc|lp|rounding|st|covers|cg|duals|fixing|lagrangian|bnp|bnpfix|refix|bnprefix"
             << "|matheuristic|tabu|portfolio>"
             << " <instance> [seconds] [output]\n";
        return 1;
    }
    string method = argv[1];
    Instance inst(argv[2]);
    double seconds = (argc > 3) ? atof(argv[3]) : 60.0;
    string output = (argc > 4) ? argv[4] : "";

    cout.setf(ios::fixed);
    cout.precision(2);
    Solution best;
    if (method == "lp") {
        mip::Result r = mip::lp_relaxation(inst, seconds);
        cout << "LP bound " << r.dual << " (" << r.seconds << " s)\n";
        return 0;
    } else if (method == "covers") {
        mip::CutResult c = mip::cover_cuts(inst, seconds);
        cout << "LP bound with cover inequalities " << c.bound << " (" << c.cuts << " cuts, "
             << c.rounds << " rounds, " << c.seconds << " s)\n";
        mip::CutResult e = mip::exact_knapsack_cuts(inst, seconds);
        cout << "LP bound with exact knapsack separation " << e.bound << " (" << e.cuts << " cuts, "
             << e.rounds << " rounds, " << e.seconds << " s)\n";
        mip::Result r = mip::root_node(inst, seconds);
        cout << "root bound of SCIP " << r.dual << " (" << r.seconds << " s)\n";
        return 0;
    } else if (method == "cg") {
        vector<pair<int, vector<int>>> columns;
        lagrangian::Result lr = lagrangian::subgradient(inst, metaheuristics::best_constructive(inst).total_cost, 5000,
                                                        60, 50, &columns);
        for (auto solver : {column_generation::Solver::SIMPLEX, column_generation::Solver::INTERIOR_POINT,
                            column_generation::Solver::BUNDLE}) {
            column_generation::Result r = column_generation::root(inst, seconds, solver, columns, &lr.multipliers);
            cout << (solver == column_generation::Solver::SIMPLEX ? "simplex: "
                     : solver == column_generation::Solver::BUNDLE ? "bundle: " : "interior point: ")
                 << "bound " << r.bound << ", LP " << r.lp_value << (r.converged ? " (converged)" : "")
                 << ", " << r.rounds << " rounds, " << r.seconds << " s\n";
        }
        return 0;
    } else if (method == "duals") {
        mip::CapacityDuals d = mip::capacity_duals(inst, seconds);
        cout << "LP bound " << d.lp_bound << "\nwarehouse shadow_price\n";
        for (int i = 0; i < (int)d.shadow_price.size(); i++)
            if (d.shadow_price[i] > 1e-9) cout << i << " " << d.shadow_price[i] << "\n";
        return 0;
    } else if (method == "fixing") {
        Solution start = lagrangian::heuristic_solution(inst, 60, 0);
        fixing::Reduction r = fixing::reduce(inst, start.total_cost, seconds, 3);
        cout << "upper bound " << r.upper_bound << ", bound " << r.bound << ": " << r.fixed_to_zero << " of "
             << r.pairs << " pairs fixed to 0, " << r.fixed_to_one << " sellers fixed, " << r.passes
             << " passes, " << r.seconds << " s" << (r.no_improving_solution ? " (optimal)" : "") << "\n";
        return 0;
    } else if (method == "rounding") {
        mip::Rounding r = mip::lp_rounding(inst, seconds);
        best = r.solution;
        cout << "LP bound " << r.lp_bound << ", " << r.split << " split sellers, rounding "
             << best.total_cost;
        metaheuristics::vnd(best, inst);
        cout << ", after the VND " << metaheuristics::evaluate(best, inst) << "\n";
    } else if (method == "st") {
        mip::ShmoysTardos r = mip::shmoys_tardos(inst, seconds);
        cout << "LP bound " << r.lp_bound << ", Shmoys-Tardos " << r.cost << " with " << r.overloaded
             << " overloaded warehouses (largest overload " << 100 * r.largest_overload << "%)\n";
        best = r.repaired;
        cout << "repaired " << best.total_cost;
        metaheuristics::vnd(best, inst);
        cout << ", after the VND " << metaheuristics::evaluate(best, inst) << "\n";
    } else if (method == "lagrangian") {
        Solution constructive = metaheuristics::best_constructive(inst);
        lagrangian::Result r = lagrangian::subgradient(inst, constructive.total_cost, 100000, seconds,
                                                       50, nullptr, 100, 10);
        cout << "Lagrangian bound " << r.bound << " after " << r.iterations << " iterations ("
             << r.seconds << " s)\n";
        best = r.solution;
        cout << "Lagrangian heuristic " << best.total_cost;
        metaheuristics::vnd(best, inst);
        cout << ", after the VND " << metaheuristics::evaluate(best, inst) << "\n";
    } else if (method == "mip" || method == "mipfix" || method == "bc" || method == "bnp" || method == "bnpfix") {
        bool with_fixing = method == "mipfix" || method == "bnpfix";
        Solution start = with_fixing ? lagrangian::heuristic_solution(inst, 60, 0) : metaheuristics::initial_solution(inst);
        double primal, dual, s;
        bool optimal;
        long long nodes;
        if (method == "mip" || method == "mipfix" || method == "bc") {
            mip::Result r = method == "mip" ? mip::solve(inst, seconds, &start)
                          : method == "bc" ? mip::branch_and_cut(inst, seconds, &start)
                                           : mip::solve_with_fixing(inst, seconds, start);
            best = r.solution; primal = r.primal; dual = r.dual; optimal = r.optimal; nodes = r.nodes; s = r.seconds;
        } else if (method == "bnpfix") {
            branch_and_price::Result r = branch_and_price::solve_with_fixing(inst, seconds, start);
            best = r.solution; primal = r.primal; dual = r.dual; optimal = r.optimal; nodes = r.nodes; s = r.seconds;
        } else {
            branch_and_price::Result r = branch_and_price::solve(inst, seconds, &start);
            best = r.solution; primal = r.primal; dual = r.dual; optimal = r.optimal; nodes = r.nodes; s = r.seconds;
            cout << "root bound " << r.root_bound << ", " << r.columns << " columns, "
                 << r.heuristic_improvements << " improvements of the restricted master heuristic\n";
        }
        cout << "best " << primal << ", bound " << dual << (optimal ? " (optimal)" : "")
             << ", " << nodes << " nodes, " << s << " s\n";
    } else if (method == "refix" || method == "bnprefix") {
        int n = inst.sellers.size();
        double lns_seconds = min(60.0, seconds / 5);
        Solution start = matheuristic::mip_lns(inst, 1000000000, min(100, n / 2), 1, lns_seconds);
        start.total_cost = metaheuristics::evaluate(start, inst);
        cout << "start of the matheuristic " << start.total_cost << " (" << lns_seconds << " s)\n";
        double primal, dual, s, first, final_pct;
        bool optimal;
        int incumbents;
        if (method == "refix") {
            mip::Refixing st;
            mip::Result r = mip::solve_with_refixing(inst, seconds, start, &st);
            best = r.solution; primal = r.primal; dual = r.dual; optimal = r.optimal; s = r.seconds;
            first = st.first_fixed_pct;
            final_pct = 100.0 * st.fixed_to_zero / max(1LL, st.pairs);
            incumbents = st.incumbents;
        } else {
            branch_and_price::Options o;
            o.refixing = true;
            branch_and_price::Result r = branch_and_price::solve(inst, seconds, &start, o);
            best = r.solution; primal = r.primal; dual = r.dual; optimal = r.optimal; s = r.seconds;
            first = r.first_fixed_pct;
            final_pct = 100.0 * r.fixed_to_zero / max(1LL, r.pairs);
            incumbents = r.refixing_incumbents;
        }
        cout << "pairs fixed to 0: " << first << "% with the start, " << final_pct << "% at the end ("
             << incumbents << " better incumbents)\n";
        cout << "best " << primal << ", bound " << dual << (optimal ? " (optimal)" : "") << ", " << s << " s\n";
    } else if (method == "tabu" || method == "portfolio") {
        best = method == "tabu" ? ejection::search(inst, seconds, 1) : matheuristic::portfolio(inst, seconds, 1);
        cout << "Cost after the " << (method == "tabu" ? "tabu search" : "portfolio") << ": " << best.total_cost << "\n";
    } else if (method == "matheuristic") {
        int n = inst.sellers.size();
        best = matheuristic::mip_lns(inst, 1000000000, min(100, n / 2), 1, seconds);
        cout << "Cost after the matheuristic: " << best.total_cost << "\n";
    } else {
        cerr << "unknown method: " << method << "\n";
        return 1;
    }
    if (!output.empty()) write_solution(output, best);
    return 0;
}

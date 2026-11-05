#include "MipModel.h"
#include "ConstructiveHeuristics.h"
#include "Knapsack.h"
#include "ScipUtils.h"
#include <objscip/objscip.h>
#include <scip/heur_trysol.h>
#include <Highs.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <string>

using namespace std;

namespace mip {

namespace {

// Variables of a model over a subset of sellers: x[k][i] for the k-th seller of
// the subset (nullptr if its demand does not fit in warehouse i) and u[k].
struct Model {
    vector<int> sellers;
    vector<vector<SCIP_VAR*>> x;
    vector<SCIP_VAR*> u;
};

// Builds the model for 'sellers' with the given capacities in 'scip'.
void build(SCIP* scip, const Instance& inst, const vector<int>& sellers,
           const vector<int>& capacity, bool integer, Model& model) {
    int m = inst.warehouses.size();
    SCIP_VARTYPE type = integer ? SCIP_VARTYPE_BINARY : SCIP_VARTYPE_CONTINUOUS;
    SCIP_CHECK(SCIPcreateProbBasic(scip, "gap"));
    model.sellers = sellers;
    model.x.assign(sellers.size(), vector<SCIP_VAR*>(m, nullptr));
    model.u.assign(sellers.size(), nullptr);

    vector<vector<SCIP_VAR*>> in_warehouse(m);
    vector<vector<double>> weights(m);
    for (int k = 0; k < (int)sellers.size(); k++) {
        int j = sellers[k];
        vector<SCIP_VAR*> row;
        for (int i = 0; i < m; i++) {
            if (inst.demand(j, i) > capacity[i]) continue;
            string name = "x_" + to_string(i) + "_" + to_string(j);
            SCIP_CHECK(SCIPcreateVarBasic(scip, &model.x[k][i], name.c_str(), 0.0, 1.0,
                                          inst.cost[j][i], type));
            SCIP_CHECK(SCIPaddVar(scip, model.x[k][i]));
            row.push_back(model.x[k][i]);
            in_warehouse[i].push_back(model.x[k][i]);
            weights[i].push_back(inst.demand(j, i));
        }
        string name = "u_" + to_string(j);
        SCIP_CHECK(SCIPcreateVarBasic(scip, &model.u[k], name.c_str(), 0.0, 1.0, inst.penalty, type));
        SCIP_CHECK(SCIPaddVar(scip, model.u[k]));
        row.push_back(model.u[k]);

        SCIP_CONS* cons;
        vector<double> ones(row.size(), 1.0);
        string cname = "assign_" + to_string(j);
        SCIP_CHECK(SCIPcreateConsBasicLinear(scip, &cons, cname.c_str(), row.size(), row.data(),
                                             ones.data(), 1.0, 1.0));
        SCIP_CHECK(SCIPaddCons(scip, cons));
        SCIP_CHECK(SCIPreleaseCons(scip, &cons));
    }
    for (int i = 0; i < m; i++) {
        if (in_warehouse[i].empty()) continue;
        SCIP_CONS* cons;
        string cname = "capacity_" + to_string(i);
        SCIP_CHECK(SCIPcreateConsBasicLinear(scip, &cons, cname.c_str(), in_warehouse[i].size(),
                                             in_warehouse[i].data(), weights[i].data(),
                                             -SCIPinfinity(scip), capacity[i]));
        SCIP_CHECK(SCIPaddCons(scip, cons));
        SCIP_CHECK(SCIPreleaseCons(scip, &cons));
    }
}

// Hands the current warehouses of the model's sellers in 'sol' to SCIP.
void add_start(SCIP* scip, const Model& model, const Solution& sol) {
    SCIP_SOL* start;
    SCIP_CHECK(SCIPcreateSol(scip, &start, nullptr));
    for (int k = 0; k < (int)model.sellers.size(); k++) {
        int i = sol.assigned_warehouse[model.sellers[k]];
        if (i >= 0 && model.x[k][i] != nullptr) SCIP_CHECK(SCIPsetSolVal(scip, start, model.x[k][i], 1.0));
        else SCIP_CHECK(SCIPsetSolVal(scip, start, model.u[k], 1.0));
    }
    SCIP_Bool stored;
    SCIP_CHECK(SCIPaddSolFree(scip, &start, &stored));
}

// Warehouse of the k-th seller of the model in SCIP's best solution (-1 if unassigned).
int chosen_warehouse(SCIP* scip, SCIP_SOL* best, const Model& model, int k) {
    for (int i = 0; i < (int)model.x[k].size(); i++) {
        if (model.x[k][i] != nullptr && SCIPgetSolVal(scip, best, model.x[k][i]) > 0.5) return i;
    }
    return -1;
}

void release(SCIP* scip, Model& model) {
    for (auto& row : model.x)
        for (SCIP_VAR*& var : row)
            if (var != nullptr) SCIP_CHECK(SCIPreleaseVar(scip, &var));
    for (SCIP_VAR*& var : model.u) SCIP_CHECK(SCIPreleaseVar(scip, &var));
}

vector<int> capacities(const Instance& inst) {
    vector<int> capacity;
    for (const Warehouse& w : inst.warehouses) capacity.push_back(w.capacity);
    return capacity;
}

vector<int> all_sellers(const Instance& inst) {
    vector<int> sellers(inst.sellers.size());
    for (int j = 0; j < (int)sellers.size(); j++) sellers[j] = j;
    return sellers;
}

double seconds_since(chrono::high_resolution_clock::time_point t0) {
    return chrono::duration_cast<chrono::milliseconds>(
        chrono::high_resolution_clock::now() - t0).count() / 1000.0;
}

// For the bounds of the root node: the branching rule that runs at the end of the
// root must not raise its bound, so the most infeasible branching (no strong
// branching, which would also count in the dual bound) gets the highest priority.
void root_bound_only(SCIP* scip) {
    SCIP_CHECK(SCIPsetLongintParam(scip, "limits/nodes", 1));
    SCIP_CHECK(SCIPsetIntParam(scip, "branching/mostinf/priority", 10000000));
}

// Separator of cover inequalities (see cover_cuts in MipModel.h).
class CoverSeparator : public scip::ObjSepa {
public:
    CoverSeparator(SCIP* scip, const Instance* inst, const Model* model)
        : ObjSepa(scip, "gap_covers", "cover inequalities of the capacity constraints",
                  1000000, 0, 1.0, FALSE, FALSE),
          inst_(inst), model_(model) {}

    int cuts = 0;

    SCIP_DECL_SEPAEXECLP(scip_execlp) override {
        const Instance& inst = *inst_;
        const Model& model = *model_;
        int m = inst.warehouses.size();
        int k_sellers = model.sellers.size();
        *result = SCIP_DIDNOTFIND;
        for (int i = 0; i < m; i++) {
            // sellers with x*_ij > 0, the only ones that can be in a violated cover
            vector<int> support;
            vector<double> value;
            long long total = 0;
            for (int k = 0; k < k_sellers; k++) {
                SCIP_VAR* var = model.x[k][i];
                if (var == nullptr) continue;
                double v = SCIPgetSolVal(scip, nullptr, var);
                if (v <= 1e-6) continue;
                support.push_back(k);
                value.push_back(min(1.0, v));
                total += inst.demand(model.sellers[k], i);
            }
            int capacity = inst.warehouses[i].capacity;
            if (total <= capacity) continue;
            // The sellers left out of the cover form a knapsack of capacity
            // total - b_i - 1 that maximizes sum (1 - x*_ij).
            vector<double> profit;
            vector<int> weight;
            for (int t = 0; t < (int)support.size(); t++) {
                profit.push_back(1.0 - value[t]);
                weight.push_back(inst.demand(model.sellers[support[t]], i));
            }
            vector<int> left_out;
            knapsack::solve(profit, weight, (int)(total - capacity - 1), left_out);
            vector<bool> out(support.size(), false);
            for (int t : left_out) out[t] = true;
            vector<int> cover;
            long long weight_cover = 0;
            double slack = 0;   // sum over the cover of (1 - x*_ij)
            for (int t = 0; t < (int)support.size(); t++) {
                if (out[t]) continue;
                cover.push_back(t);
                weight_cover += weight[t];
                slack += 1.0 - value[t];
            }
            if (slack >= 1.0 - 1e-6) continue;   // not violated
            // Minimal cover: the sellers with x*_ij = 1 that are not needed are removed
            // (this keeps the violation and shortens the inequality).
            sort(cover.begin(), cover.end(), [&](int a, int b) { return weight[a] > weight[b]; });
            vector<int> minimal;
            for (int t : cover) {
                if (value[t] >= 1.0 - 1e-9 && weight_cover - weight[t] > capacity) {
                    weight_cover -= weight[t];
                    continue;
                }
                minimal.push_back(t);
            }
            int largest = 0;
            vector<bool> in_cover(k_sellers, false);
            for (int t : minimal) {
                in_cover[support[t]] = true;
                largest = max(largest, weight[t]);
            }
            SCIP_ROW* row;
            string name = "cover_" + to_string(i) + "_" + to_string(cuts);
            SCIP_CALL(SCIPcreateEmptyRowSepa(scip, &row, sepa, name.c_str(), -SCIPinfinity(scip),
                                             (double)minimal.size() - 1.0, FALSE, FALSE, TRUE));
            SCIP_CALL(SCIPcacheRowExtensions(scip, row));
            for (int k = 0; k < k_sellers; k++) {
                SCIP_VAR* var = model.x[k][i];
                if (var == nullptr) continue;
                // extension: sellers outside the cover at least as heavy as its heaviest
                if (in_cover[k] || inst.demand(model.sellers[k], i) >= largest)
                    SCIP_CALL(SCIPaddVarToRow(scip, row, var, 1.0));
            }
            SCIP_CALL(SCIPflushRowExtensions(scip, row));
            if (SCIPisCutEfficacious(scip, nullptr, row)) {
                SCIP_Bool infeasible;
                SCIP_CALL(SCIPaddRow(scip, row, FALSE, &infeasible));
                cuts++;
                *result = infeasible ? SCIP_CUTOFF : SCIP_SEPARATED;
            }
            SCIP_CALL(SCIPreleaseRow(scip, &row));
            if (*result == SCIP_CUTOFF) return SCIP_OKAY;
        }
        return SCIP_OKAY;
    }

private:
    const Instance* inst_;
    const Model* model_;
};

// Separator of the most violated inequality of each knapsack polytope (see
// exact_knapsack_cuts in MipModel.h).
class ExactKnapsackSeparator : public scip::ObjSepa {
public:
    ExactKnapsackSeparator(SCIP* scip, const Instance* inst, const Model* model)
        : ObjSepa(scip, "gap_exact_knapsack", "exact separation of the knapsack polytopes",
                  1000000, 0, 1.0, FALSE, FALSE),
          inst_(inst), model_(model) {}

    int cuts = 0;

    SCIP_DECL_SEPAEXECLP(scip_execlp) override {
        const Instance& inst = *inst_;
        const Model& model = *model_;
        int m = inst.warehouses.size();
        int k_sellers = model.sellers.size();
        *result = SCIP_DIDNOTFIND;
        for (int i = 0; i < m; i++) {
            vector<int> support;
            vector<double> value;
            bool fractional = false;
            for (int k = 0; k < k_sellers; k++) {
                SCIP_VAR* var = model.x[k][i];
                if (var == nullptr) continue;
                double v = SCIPgetSolVal(scip, nullptr, var);
                if (v <= 1e-6) continue;
                support.push_back(k);
                value.push_back(min(1.0, v));
                if (v < 1.0 - 1e-6) fractional = true;
            }
            if (!fractional) continue;   // an integral point of the knapsack is in its hull
            vector<double> pi;
            double pi0;
            if (!separate(inst, model, i, support, value, pi, pi0)) continue;
            SCIP_ROW* row;
            string name = "knapsack_" + to_string(i) + "_" + to_string(cuts);
            SCIP_CALL(SCIPcreateEmptyRowSepa(scip, &row, sepa, name.c_str(), -SCIPinfinity(scip), pi0,
                                             FALSE, FALSE, TRUE));
            SCIP_CALL(SCIPcacheRowExtensions(scip, row));
            for (int t = 0; t < (int)support.size(); t++) {
                if (pi[t] > 1e-9) SCIP_CALL(SCIPaddVarToRow(scip, row, model.x[support[t]][i], pi[t]));
            }
            SCIP_CALL(SCIPflushRowExtensions(scip, row));
            if (SCIPisCutEfficacious(scip, nullptr, row)) {
                SCIP_Bool infeasible;
                SCIP_CALL(SCIPaddRow(scip, row, FALSE, &infeasible));
                cuts++;
                *result = infeasible ? SCIP_CUTOFF : SCIP_SEPARATED;
            }
            SCIP_CALL(SCIPreleaseRow(scip, &row));
            if (*result == SCIP_CUTOFF) return SCIP_OKAY;
        }
        return SCIP_OKAY;
    }

private:
    const Instance* inst_;
    const Model* model_;

    // Separation LP of warehouse i over the support (see MipModel.h); returns
    // whether it found an inequality violated by more than 1e-6.
    static bool separate(const Instance& inst, const Model& model, int i, const vector<int>& support,
                         const vector<double>& value, vector<double>& pi, double& pi0) {
        int s = support.size();
        Highs highs;
        highs.setOptionValue("output_flag", false);
        highs.setOptionValue("threads", 1);
        highs.changeObjectiveSense(ObjSense::kMaximize);
        // columns pi_0 .. pi_{s-1} in [0, 1], then pi_0' (the right-hand side) >= 0
        for (int t = 0; t < s; t++) highs.addCol(value[t], 0.0, 1.0, 0, nullptr, nullptr);
        highs.addCol(-1.0, 0.0, kHighsInf, 0, nullptr, nullptr);
        auto add_set = [&](const vector<int>& positions) {
            vector<HighsInt> index;
            vector<double> coef;
            for (int t : positions) {
                index.push_back(t);
                coef.push_back(1.0);
            }
            index.push_back(s);
            coef.push_back(-1.0);
            highs.addRow(-kHighsInf, 0.0, index.size(), index.data(), coef.data());
        };
        for (int t = 0; t < s; t++) add_set({t});   // every single seller fits
        vector<double> profit(s);
        vector<int> weight(s), chosen;
        for (int t = 0; t < s; t++) weight[t] = inst.demand(model.sellers[support[t]], i);
        for (int iteration = 0; iteration < 1000; iteration++) {
            highs.run();
            if (highs.getModelStatus() != HighsModelStatus::kOptimal) return false;
            const vector<double>& x = highs.getSolution().col_value;
            for (int t = 0; t < s; t++) profit[t] = x[t];
            double best = knapsack::solve(profit, weight, inst.warehouses[i].capacity, chosen);
            if (best <= x[s] + 1e-9) {
                pi.assign(x.begin(), x.begin() + s);
                pi0 = best;   // the exact right-hand side for these coefficients
                double violation = -pi0;
                for (int t = 0; t < s; t++) violation += pi[t] * value[t];
                return violation > 1e-6;
            }
            add_set(chosen);
        }
        return false;
    }
};

// Propagator of the re-fixing (see solve_with_refixing in MipModel.h). It runs
// before the LP of every node, but only does something when the incumbent has
// improved since its last call: it tests the stored Lagrangian penalties against
// the new upper bound, runs the subgradient method again when the upper bound has
// closed at least a tenth of the gap left at its last run (and the time budget
// allows it), and fixes the new pairs globally in SCIP. The solutions of the
// Lagrangian heuristic that improve the incumbent are handed to SCIP through the
// trysol heuristic (a propagator cannot add solutions itself). Once the fixings
// prove that nothing is cheaper than the incumbent, every node is cut off.
class RefixingProp : public scip::ObjProp {
public:
    RefixingProp(SCIP* scip, const Instance& inst, const Model& model, double budget, Refixing& stats, bool bundle)
        : ObjProp(scip, "lagrangian_refixing", "Lagrangian variable fixing after each new incumbent",
                  2000000, 1, FALSE, SCIP_PROPTIMING_BEFORELP | SCIP_PROPTIMING_AFTERLPNODE, 0, 0,
                  SCIP_PRESOLTIMING_NONE),
          inst_(inst), model_(model), refixer_(inst, budget, bundle), stats_(stats) {
        int n = inst.sellers.size();
        int m = inst.warehouses.size();
        zero_.assign(n, vector<char>(m, 0));
        one_.assign(n, 0);
    }

    SCIP_DECL_PROPEXEC(scip_exec) override {
        *result = SCIP_DIDNOTRUN;
        if (done_) { *result = SCIP_CUTOFF; return SCIP_OKAY; }
        double ub = SCIPgetPrimalbound(scip);
        if (SCIPisInfinity(scip, ub)) return SCIP_OKAY;
        Solution best;
        if (!refixer_.update(ub, best) && best.total_cost >= ub) return SCIP_OKAY;
        *result = SCIP_DIDNOTFIND;
        const fixing::Reduction& r = refixer_.reduction();
        stats_.seconds = r.seconds;
        stats_.subgradient_runs = r.passes;
        stats_.bound = r.bound;
        stats_.pairs = r.pairs;
        stats_.fixed_to_zero = r.fixed_to_zero;
        stats_.fixed_to_one = r.fixed_to_one;
        stats_.first_fixed_pct = refixer_.first_fixed_pct;
        stats_.incumbents = refixer_.incumbents;
        if (best.total_cost < ub - 1e-9) {
            SCIP_CALL(pass_solution(scip, best));
            if (best.total_cost < stats_.lagrangian_best.total_cost) stats_.lagrangian_best = best;
        }
        if (refixer_.no_improving_solution()) {
            done_ = true;
            *result = SCIP_CUTOFF;
            return SCIP_OKAY;
        }
        return apply(scip, result);
    }

private:
    const Instance& inst_;
    const Model& model_;
    fixing::Refixer refixer_;
    Refixing& stats_;
    bool done_ = false;
    vector<vector<char>> zero_;   // x_ij already fixed to 0 in SCIP
    vector<char> one_;            // seller j already fixed to its forced warehouse

    SCIP_RETCODE pass_solution(SCIP* scip, const Solution& sol) {
        SCIP_HEUR* trysol = SCIPfindHeur(scip, "trysol");
        if (trysol == nullptr) return SCIP_OKAY;
        SCIP_SOL* candidate;
        SCIP_CALL(SCIPcreateOrigSol(scip, &candidate, nullptr));
        for (int j = 0; j < (int)inst_.sellers.size(); j++) {
            int i = sol.assigned_warehouse[j];
            if (i >= 0 && model_.x[j][i] != nullptr) SCIP_CALL(SCIPsetSolVal(scip, candidate, model_.x[j][i], 1.0));
            else SCIP_CALL(SCIPsetSolVal(scip, candidate, model_.u[j], 1.0));
        }
        SCIP_CALL(SCIPheurPassSolTrySol(scip, trysol, candidate));
        SCIP_CALL(SCIPfreeSol(scip, &candidate));
        stats_.lagrangian_solutions++;
        return SCIP_OKAY;
    }

    // Global bound of the transformed variable of 'var' to 'value' (0: upper bound,
    // 1: lower bound). Sets 'cutoff' if the current node contradicts it and 'done_'
    // if the whole problem does.
    SCIP_RETCODE fix(SCIP* scip, SCIP_VAR* var, double value, bool& reduced, bool& cutoff) {
        SCIP_VAR* transformed = nullptr;
        SCIP_CALL(SCIPgetTransformedVar(scip, var, &transformed));
        if (transformed == nullptr || SCIPvarGetStatus(transformed) == SCIP_VARSTATUS_MULTAGGR) return SCIP_OKAY;
        if (SCIPvarGetStatus(transformed) == SCIP_VARSTATUS_FIXED) {
            if (fabs(SCIPvarGetLbGlobal(transformed) - value) > 0.5) done_ = true;
            return SCIP_OKAY;
        }
        SCIP_Bool infeasible = FALSE, tightened = FALSE;
        if (value < 0.5) {
            SCIP_CALL(SCIPtightenVarUbGlobal(scip, transformed, 0.0, FALSE, &infeasible, &tightened));
            if (!infeasible && SCIPvarGetLbLocal(transformed) > 0.5) cutoff = true;
        } else {
            SCIP_CALL(SCIPtightenVarLbGlobal(scip, transformed, 1.0, FALSE, &infeasible, &tightened));
            if (!infeasible && SCIPvarGetUbLocal(transformed) < 0.5) cutoff = true;
        }
        if (infeasible) done_ = true;
        if (tightened) reduced = true;
        return SCIP_OKAY;
    }

    SCIP_RETCODE apply(SCIP* scip, SCIP_RESULT* result) {
        int n = inst_.sellers.size();
        int m = inst_.warehouses.size();
        bool reduced = false, cutoff = false;
        for (int j = 0; j < n && !done_; j++) {
            int forced = refixer_.forced(j);
            if (forced >= 0 && !one_[j] && model_.x[j][forced] != nullptr) {
                one_[j] = 1;
                SCIP_CALL(fix(scip, model_.x[j][forced], 1.0, reduced, cutoff));
                SCIP_CALL(fix(scip, model_.u[j], 0.0, reduced, cutoff));
            }
            for (int i = 0; i < m && !done_; i++) {
                if (zero_[j][i] || model_.x[j][i] == nullptr || !refixer_.ruled_out(j, i)) continue;
                zero_[j][i] = 1;
                SCIP_CALL(fix(scip, model_.x[j][i], 0.0, reduced, cutoff));
            }
        }
        if (done_ || cutoff) *result = SCIP_CUTOFF;
        else if (reduced) *result = SCIP_REDUCEDDOM;
        return SCIP_OKAY;
    }
};

Result run(const Instance& inst, double time_limit, bool integer, const Solution* start,
           double objective_limit = 1e300,
           const function<void(SCIP*, const Model&)>& setup = nullptr) {
    auto t0 = chrono::high_resolution_clock::now();
    SCIP* scip = create_scip(time_limit);
    Model model;
    build(scip, inst, all_sellers(inst), capacities(inst), integer, model);
    if (setup) setup(scip, model);
    if (objective_limit < 1e299) SCIP_CHECK(SCIPsetObjlimit(scip, objective_limit));
    if (start != nullptr) add_start(scip, model, *start);
    SCIP_CHECK(SCIPsolve(scip));

    Result result;
    result.dual = SCIPgetDualbound(scip);
    // with an objective limit, an infeasible status means that nothing is cheaper
    result.no_better_solution = SCIPgetStatus(scip) == SCIP_STATUS_INFEASIBLE;
    result.optimal = SCIPgetStatus(scip) == SCIP_STATUS_OPTIMAL || result.no_better_solution;
    result.nodes = SCIPgetNNodes(scip);
    SCIP_SOL* best = SCIPgetBestSol(scip);
    if (best != nullptr) {
        result.primal = SCIPgetSolOrigObj(scip, best);
        if (integer) {
            int n = inst.sellers.size();
            result.solution.sellers_per_warehouse.assign(inst.warehouses.size(), {});
            result.solution.assigned_warehouse.assign(n, -1);
            for (int j = 0; j < n; j++) {
                int i = chosen_warehouse(scip, best, model, j);
                if (i >= 0) {
                    result.solution.assign(i, j);
                    result.solution.total_cost += inst.cost[j][i];
                } else {
                    result.solution.total_cost += inst.penalty;
                }
            }
            result.has_solution = true;
        }
    }
    release(scip, model);
    SCIP_CHECK(SCIPfree(&scip));
    result.seconds = chrono::duration_cast<chrono::milliseconds>(
        chrono::high_resolution_clock::now() - t0).count() / 1000.0;
    return result;
}

}

Result solve(const Instance& inst, double time_limit, const Solution* start, double objective_limit) {
    return run(inst, time_limit, true, start, objective_limit);
}

Result solve_with_fixing(const Instance& inst, double time_limit, const Solution& start,
                         fixing::Reduction* reduction_out) {
    auto t0 = chrono::high_resolution_clock::now();
    Solution incumbent = start;
    double cost = 0;
    for (int j = 0; j < (int)inst.sellers.size(); j++) {
        int i = start.assigned_warehouse[j];
        cost += i >= 0 ? inst.cost[j][i] : inst.penalty;
    }
    incumbent.total_cost = cost;
    fixing::Reduction reduction = fixing::reduce(inst, incumbent.total_cost, 0.25 * time_limit, 3, &incumbent);

    Result result;
    bool solved = reduction.no_improving_solution || reduction.instance.sellers.empty();
    if (!reduction.no_improving_solution && reduction.instance.sellers.empty()) {
        // every seller is fixed: the forced assignment is the only cheaper candidate
        Solution forced = fixing::expand(inst, reduction, Solution());
        if (forced.total_cost < incumbent.total_cost) incumbent = forced;
    }
    if (!solved) {
        double left = time_limit > 0 ? max(1.0, time_limit - seconds_since(t0)) : 0;
        Result r = run(reduction.instance, left, true, nullptr, incumbent.total_cost - reduction.offset);
        if (r.has_solution) {
            Solution candidate = fixing::expand(inst, reduction, r.solution);
            if (candidate.total_cost < incumbent.total_cost) incumbent = candidate;
        }
        solved = r.optimal;
        result.dual = min(incumbent.total_cost, max(r.dual + reduction.offset, reduction.bound));
        result.nodes = r.nodes;
    }
    result.solution = incumbent;
    result.has_solution = true;
    result.primal = incumbent.total_cost;
    if (solved) result.dual = result.primal;
    result.optimal = solved;
    result.seconds = seconds_since(t0);
    if (reduction_out != nullptr) *reduction_out = reduction;
    return result;
}

Result solve_with_refixing(const Instance& inst, double time_limit, const Solution& start, Refixing* stats_out,
                           bool bundle) {
    Refixing stats;
    Solution incumbent = start;
    incumbent.total_cost = 0;
    for (int j = 0; j < (int)inst.sellers.size(); j++) {
        int i = start.assigned_warehouse[j];
        incumbent.total_cost += i >= 0 ? inst.cost[j][i] : inst.penalty;
    }
    Result result = run(inst, time_limit, true, &incumbent, 1e300, [&](SCIP* scip, const Model& model) {
        // the propagator fixes the transformed variables, which presolving must
        // not replace by sums of other variables
        SCIP_CHECK(SCIPsetBoolParam(scip, "presolving/donotmultaggr", TRUE));
        SCIP_CHECK(SCIPincludeObjProp(scip, new RefixingProp(scip, inst, model, 0.25 * time_limit, stats, bundle), TRUE));
    });
    for (const Solution* candidate : {&incumbent, &stats.lagrangian_best}) {
        if (!result.has_solution || candidate->total_cost < result.primal - 1e-9) {
            result.solution = *candidate;
            result.primal = candidate->total_cost;
            result.has_solution = true;
        }
    }
    if (!result.optimal) result.dual = max(result.dual, min(result.primal, stats.bound));
    if (fixing::integral_costs(inst)) result.dual = min(result.primal, ceil(result.dual - 1e-6));
    if (result.dual >= result.primal - 1e-9) {
        result.dual = result.primal;
        result.optimal = true;
    }
    if (stats_out != nullptr) *stats_out = stats;
    return result;
}

Result lp_relaxation(const Instance& inst, double time_limit) {
    return run(inst, time_limit, false, nullptr);
}

// Basic optimal solution of the LP relaxation (simplex method, without presolving
// or heuristics): x[j] lists the pairs (i, x*_ij) with x*_ij > 0 and u[j] is u*_j.
// Returns false if the LP was not solved within the time limit.
bool basic_lp_solution(const Instance& inst, double time_limit, vector<vector<pair<int, double>>>& x,
                       vector<double>& u, double& bound) {
    int n = inst.sellers.size();
    SCIP* scip = create_scip(time_limit);
    SCIP_CHECK(SCIPsetPresolving(scip, SCIP_PARAMSETTING_OFF, TRUE));
    SCIP_CHECK(SCIPsetHeuristics(scip, SCIP_PARAMSETTING_OFF, TRUE));
    Model model;
    build(scip, inst, all_sellers(inst), capacities(inst), false, model);
    SCIP_CHECK(SCIPsolve(scip));
    x.assign(n, {});
    u.assign(n, 0.0);
    SCIP_SOL* best = SCIPgetBestSol(scip);
    if (best != nullptr) {
        bound = SCIPgetSolOrigObj(scip, best);
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < (int)model.x[j].size(); i++) {
                if (model.x[j][i] == nullptr) continue;
                double v = SCIPgetSolVal(scip, best, model.x[j][i]);
                if (v > 1e-9) x[j].push_back({i, min(1.0, v)});
            }
            u[j] = max(0.0, SCIPgetSolVal(scip, best, model.u[j]));
        }
    }
    release(scip, model);
    SCIP_CHECK(SCIPfree(&scip));
    return best != nullptr;
}

CutResult exact_knapsack_cuts(const Instance& inst, double time_limit) {
    auto t0 = chrono::high_resolution_clock::now();
    SCIP* scip = create_scip(time_limit);
    SCIP_CHECK(SCIPsetPresolving(scip, SCIP_PARAMSETTING_OFF, TRUE));
    SCIP_CHECK(SCIPsetHeuristics(scip, SCIP_PARAMSETTING_OFF, TRUE));
    root_bound_only(scip);
    SCIP_CHECK(SCIPsetIntParam(scip, "propagating/maxroundsroot", 0));
    SCIP_CHECK(SCIPsetIntParam(scip, "separating/maxroundsroot", -1));
    SCIP_CHECK(SCIPsetIntParam(scip, "separating/maxstallroundsroot", -1));
    for (int k = 0; k < SCIPgetNSepas(scip); k++) {
        string param = string("separating/") + SCIPsepaGetName(SCIPgetSepas(scip)[k]) + "/freq";
        SCIP_CHECK(SCIPsetIntParam(scip, param.c_str(), -1));
    }
    Model model;
    ExactKnapsackSeparator* separator = new ExactKnapsackSeparator(scip, &inst, &model);
    SCIP_CHECK(SCIPincludeObjSepa(scip, separator, TRUE));
    build(scip, inst, all_sellers(inst), capacities(inst), true, model);
    SCIP_CHECK(SCIPsolve(scip));
    CutResult result;
    result.bound = SCIPgetDualbound(scip);
    result.cuts = separator->cuts;
    result.rounds = SCIPgetNSepaRounds(scip);
    release(scip, model);
    SCIP_CHECK(SCIPfree(&scip));
    result.seconds = seconds_since(t0);
    return result;
}

Result branch_and_cut(const Instance& inst, double time_limit, const Solution* start, double objective_limit) {
    auto t0 = chrono::high_resolution_clock::now();
    SCIP* scip = create_scip(time_limit);
    Model model;
    ExactKnapsackSeparator* separator = new ExactKnapsackSeparator(scip, &inst, &model);
    SCIP_CHECK(SCIPincludeObjSepa(scip, separator, TRUE));
    build(scip, inst, all_sellers(inst), capacities(inst), true, model);
    if (objective_limit < 1e299) SCIP_CHECK(SCIPsetObjlimit(scip, objective_limit));
    if (start != nullptr) add_start(scip, model, *start);
    SCIP_CHECK(SCIPsolve(scip));
    Result result;
    result.dual = SCIPgetDualbound(scip);
    result.no_better_solution = SCIPgetStatus(scip) == SCIP_STATUS_INFEASIBLE;
    result.optimal = SCIPgetStatus(scip) == SCIP_STATUS_OPTIMAL || result.no_better_solution;
    result.nodes = SCIPgetNNodes(scip);
    SCIP_SOL* best = SCIPgetBestSol(scip);
    if (best != nullptr) {
        result.primal = SCIPgetSolOrigObj(scip, best);
        int n = inst.sellers.size();
        result.solution.sellers_per_warehouse.assign(inst.warehouses.size(), {});
        result.solution.assigned_warehouse.assign(n, -1);
        for (int j = 0; j < n; j++) {
            int i = chosen_warehouse(scip, best, model, j);
            if (i >= 0) {
                result.solution.assign(i, j);
                result.solution.total_cost += inst.cost[j][i];
            } else {
                result.solution.total_cost += inst.penalty;
            }
        }
        result.has_solution = true;
    }
    release(scip, model);
    SCIP_CHECK(SCIPfree(&scip));
    result.seconds = seconds_since(t0);
    return result;
}

CapacityDuals capacity_duals(const Instance& inst, double time_limit) {
    int n = inst.sellers.size();
    int m = inst.warehouses.size();
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("threads", 1);
    highs.setOptionValue("solver", "simplex");
    if (time_limit > 0) highs.setOptionValue("time_limit", time_limit);
    // rows: assignment of each seller (= 1), then capacity of each warehouse (<= b_i)
    for (int j = 0; j < n; j++) highs.addRow(1.0, 1.0, 0, nullptr, nullptr);
    for (int i = 0; i < m; i++) highs.addRow(-kHighsInf, inst.warehouses[i].capacity, 0, nullptr, nullptr);
    for (int j = 0; j < n; j++) {
        HighsInt row = j;
        double one = 1.0;
        highs.addCol(inst.penalty, 0.0, 1.0, 1, &row, &one);   // u_j
        for (int i = 0; i < m; i++) {
            if (inst.demand(j, i) > inst.warehouses[i].capacity) continue;
            HighsInt rows[2] = {(HighsInt)j, (HighsInt)(n + i)};
            double values[2] = {1.0, (double)inst.demand(j, i)};
            highs.addCol(inst.cost[j][i], 0.0, 1.0, 2, rows, values);
        }
    }
    highs.run();
    CapacityDuals result;
    result.shadow_price.assign(m, 0.0);
    if (highs.getModelStatus() == HighsModelStatus::kOptimal) {
        result.solved = true;
        result.lp_bound = highs.getInfo().objective_function_value;
        const HighsSolution& solution = highs.getSolution();
        for (int i = 0; i < m; i++) result.shadow_price[i] = max(0.0, -solution.row_dual[n + i]);
    }
    return result;
}

Rounding lp_rounding(const Instance& inst, double time_limit) {
    auto t0 = chrono::high_resolution_clock::now();
    int n = inst.sellers.size();
    vector<vector<pair<int, double>>> x;
    vector<double> u;
    Rounding result;
    result.solution.sellers_per_warehouse.assign(inst.warehouses.size(), {});
    result.solution.assigned_warehouse.assign(n, -1);
    if (basic_lp_solution(inst, time_limit, x, u, result.lp_bound)) {
        for (int j = 0; j < n; j++) {
            int integral = -1;
            for (const auto& [i, v] : x[j]) {
                if (v > 1.0 - 1e-6) integral = i;
            }
            if (integral >= 0) result.solution.assign(integral, j);
            else if (u[j] < 1.0 - 1e-6) result.split++;
        }
    } else {
        result.split = n;
    }
    constructive::complete(inst, result.solution);
    result.seconds = seconds_since(t0);
    return result;
}

ShmoysTardos shmoys_tardos(const Instance& inst, double time_limit) {
    auto t0 = chrono::high_resolution_clock::now();
    int n = inst.sellers.size();
    int m = inst.warehouses.size();
    ShmoysTardos result;
    vector<vector<pair<int, double>>> x;
    vector<double> u;
    result.assignment.sellers_per_warehouse.assign(m, {});
    result.assignment.assigned_warehouse.assign(n, -1);
    if (!basic_lp_solution(inst, time_limit, x, u, result.lp_bound)) {
        result.repaired = result.assignment;
        constructive::complete(inst, result.repaired);
        result.cost = result.repaired.total_cost;
        result.seconds = seconds_since(t0);
        return result;
    }

    // Slots: warehouse m is the dummy one (unassigned sellers, demand 0).
    struct Edge { int seller; int slot; double cost; };
    vector<Edge> edges;
    vector<int> slot_warehouse;
    vector<vector<pair<int, double>>> by_warehouse(m + 1);
    for (int j = 0; j < n; j++) {
        for (const auto& [i, v] : x[j]) by_warehouse[i].push_back({j, v});
        if (u[j] > 1e-9) by_warehouse[m].push_back({j, u[j]});
    }
    for (int i = 0; i <= m; i++) {
        auto& list = by_warehouse[i];
        if (list.empty()) continue;
        if (i < m) {
            sort(list.begin(), list.end(), [&](const pair<int, double>& a, const pair<int, double>& b) {
                int da = inst.demand(a.first, i), db = inst.demand(b.first, i);
                return da != db ? da > db : a.first < b.first;
            });
        }
        int slot = slot_warehouse.size();
        slot_warehouse.push_back(i);
        double room = 1.0;   // room left in the current slot
        for (const auto& [j, v] : list) {
            double cost = i < m ? inst.cost[j][i] : inst.penalty;
            double amount = v;
            while (amount > 1e-9) {
                if (room <= 1e-9) {
                    slot = slot_warehouse.size();
                    slot_warehouse.push_back(i);
                    room = 1.0;
                }
                if (edges.empty() || edges.back().seller != j || edges.back().slot != slot)
                    edges.push_back({j, slot, cost});
                double poured = min(amount, room);
                amount -= poured;
                room -= poured;
            }
        }
    }

    // Minimum cost matching covering every seller, as a linear program.
    SCIP* scip = create_scip(time_limit > 0 ? max(1.0, time_limit - seconds_since(t0)) : 0);
    SCIP_CHECK(SCIPsetPresolving(scip, SCIP_PARAMSETTING_OFF, TRUE));
    SCIP_CHECK(SCIPsetHeuristics(scip, SCIP_PARAMSETTING_OFF, TRUE));
    SCIP_CHECK(SCIPcreateProbBasic(scip, "matching"));
    vector<SCIP_VAR*> y(edges.size());
    vector<vector<SCIP_VAR*>> of_seller(n), of_slot(slot_warehouse.size());
    for (int e = 0; e < (int)edges.size(); e++) {
        SCIP_CHECK(SCIPcreateVarBasic(scip, &y[e], nullptr, 0.0, 1.0, edges[e].cost, SCIP_VARTYPE_CONTINUOUS));
        SCIP_CHECK(SCIPaddVar(scip, y[e]));
        of_seller[edges[e].seller].push_back(y[e]);
        of_slot[edges[e].slot].push_back(y[e]);
    }
    auto add_row = [&](vector<SCIP_VAR*>& vars, double lhs) {
        if (vars.empty()) return;
        vector<double> ones(vars.size(), 1.0);
        SCIP_CONS* cons;
        SCIP_CHECK(SCIPcreateConsBasicLinear(scip, &cons, "row", vars.size(), vars.data(), ones.data(), lhs, 1.0));
        SCIP_CHECK(SCIPaddCons(scip, cons));
        SCIP_CHECK(SCIPreleaseCons(scip, &cons));
    };
    for (auto& vars : of_seller) add_row(vars, 1.0);
    for (auto& vars : of_slot) add_row(vars, -SCIPinfinity(scip));
    SCIP_CHECK(SCIPsolve(scip));
    SCIP_SOL* best = SCIPgetBestSol(scip);
    if (best != nullptr) {
        for (int e = 0; e < (int)edges.size(); e++) {
            int i = slot_warehouse[edges[e].slot];
            if (i < m && SCIPgetSolVal(scip, best, y[e]) > 0.5) result.assignment.assign(i, edges[e].seller);
        }
    }
    for (SCIP_VAR*& var : y) SCIP_CHECK(SCIPreleaseVar(scip, &var));
    SCIP_CHECK(SCIPfree(&scip));

    // cost, overloads and the bound of the theorem
    Solution& a = result.assignment;
    for (int j = 0; j < n; j++) {
        int i = a.assigned_warehouse[j];
        a.total_cost += i >= 0 ? inst.cost[j][i] : inst.penalty;
    }
    result.cost = a.total_cost;
    result.repaired = a;
    for (int i = 0; i < m; i++) {
        long long load = 0;
        int heaviest = 0;
        for (int j : a.sellers_per_warehouse[i]) load += inst.demand(j, i);
        for (const auto& [j, v] : by_warehouse[i]) heaviest = max(heaviest, inst.demand(j, i));
        int b = inst.warehouses[i].capacity;
        if (load > b + heaviest) result.within_bound = false;
        if (load <= b) continue;
        result.overloaded++;
        result.largest_overload = max(result.largest_overload, (double)(load - b) / b);
        // repair: the heaviest sellers leave until the warehouse fits
        vector<int> members = a.sellers_per_warehouse[i];
        sort(members.begin(), members.end(), [&](int p, int q) {
            return inst.demand(p, i) != inst.demand(q, i) ? inst.demand(p, i) > inst.demand(q, i) : p < q;
        });
        for (int j : members) {
            if (load <= b) break;
            result.repaired.remove(i, j);
            load -= inst.demand(j, i);
        }
    }
    constructive::complete(inst, result.repaired);
    result.seconds = seconds_since(t0);
    return result;
}

Result root_node(const Instance& inst, double time_limit) {
    auto t0 = chrono::high_resolution_clock::now();
    SCIP* scip = create_scip(time_limit);
    root_bound_only(scip);
    Model model;
    build(scip, inst, all_sellers(inst), capacities(inst), true, model);
    SCIP_CHECK(SCIPsolve(scip));
    Result result;
    result.dual = SCIPgetDualbound(scip);
    result.optimal = SCIPgetStatus(scip) == SCIP_STATUS_OPTIMAL;
    result.nodes = SCIPgetNNodes(scip);
    if (SCIPgetBestSol(scip) != nullptr) result.primal = SCIPgetPrimalbound(scip);
    release(scip, model);
    SCIP_CHECK(SCIPfree(&scip));
    result.seconds = seconds_since(t0);
    return result;
}

CutResult cover_cuts(const Instance& inst, double time_limit) {
    auto t0 = chrono::high_resolution_clock::now();
    SCIP* scip = create_scip(time_limit);
    SCIP_CHECK(SCIPsetPresolving(scip, SCIP_PARAMSETTING_OFF, TRUE));
    SCIP_CHECK(SCIPsetHeuristics(scip, SCIP_PARAMSETTING_OFF, TRUE));
    root_bound_only(scip);
    SCIP_CHECK(SCIPsetIntParam(scip, "propagating/maxroundsroot", 0));
    SCIP_CHECK(SCIPsetIntParam(scip, "separating/maxroundsroot", -1));
    SCIP_CHECK(SCIPsetIntParam(scip, "separating/maxstallroundsroot", -1));
    // only the cover separator: every separator of SCIP is switched off (the
    // constraint handlers do not separate covers without presolving, which is
    // where the linear constraints become knapsack constraints)
    for (int k = 0; k < SCIPgetNSepas(scip); k++) {
        string param = string("separating/") + SCIPsepaGetName(SCIPgetSepas(scip)[k]) + "/freq";
        SCIP_CHECK(SCIPsetIntParam(scip, param.c_str(), -1));
    }
    Model model;
    CoverSeparator* separator = new CoverSeparator(scip, &inst, &model);
    SCIP_CHECK(SCIPincludeObjSepa(scip, separator, TRUE));
    build(scip, inst, all_sellers(inst), capacities(inst), true, model);
    SCIP_CHECK(SCIPsolve(scip));
    CutResult result;
    result.bound = SCIPgetDualbound(scip);
    result.cuts = separator->cuts;
    result.rounds = SCIPgetNSepaRounds(scip);
    release(scip, model);
    SCIP_CHECK(SCIPfree(&scip));
    result.seconds = seconds_since(t0);
    return result;
}

double reoptimize(const Instance& inst, Solution& sol, const vector<int>& free_sellers,
                  double time_limit) {
    // Capacity left by the sellers that stay fixed.
    vector<int> capacity = capacities(inst);
    vector<bool> is_free(inst.sellers.size(), false);
    for (int j : free_sellers) is_free[j] = true;
    for (int j = 0; j < (int)inst.sellers.size(); j++) {
        int i = sol.assigned_warehouse[j];
        if (i >= 0 && !is_free[j]) capacity[i] -= inst.demand(j, i);
    }

    SCIP* scip = create_scip(time_limit);
    Model model;
    build(scip, inst, free_sellers, capacity, true, model);
    add_start(scip, model, sol);
    SCIP_CHECK(SCIPsolve(scip));

    double delta = 0;
    SCIP_SOL* best = SCIPgetBestSol(scip);
    if (best != nullptr) {
        vector<int> target(free_sellers.size());
        for (int k = 0; k < (int)free_sellers.size(); k++) {
            int j = free_sellers[k];
            int old_w = sol.assigned_warehouse[j];
            int new_w = chosen_warehouse(scip, best, model, k);
            target[k] = new_w;
            delta += (new_w >= 0 ? inst.cost[j][new_w] : inst.penalty)
                   - (old_w >= 0 ? inst.cost[j][old_w] : inst.penalty);
        }
        if (delta < -1e-9) {
            for (int k = 0; k < (int)free_sellers.size(); k++) {
                int j = free_sellers[k];
                if (sol.assigned_warehouse[j] >= 0) sol.remove(sol.assigned_warehouse[j], j);
            }
            for (int k = 0; k < (int)free_sellers.size(); k++) {
                if (target[k] >= 0) sol.assign(target[k], free_sellers[k]);
            }
            sol.total_cost += delta;
        } else {
            delta = 0;
        }
    }
    release(scip, model);
    SCIP_CHECK(SCIPfree(&scip));
    return delta;
}

}

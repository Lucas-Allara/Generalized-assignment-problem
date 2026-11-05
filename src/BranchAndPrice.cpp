#include "BranchAndPrice.h"
#include "ColumnGeneration.h"
#include "Knapsack.h"
#include "LagrangianBound.h"
#include "Metaheuristics.h"
#include "VariableFixing.h"
#include "ScipUtils.h"
#include <objscip/objscip.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

// Data of a branching constraint: seller j is forced into warehouse i (force) or
// forbidden from it. 'propagated_columns' counts the columns already checked
// against the decision.
struct SCIP_ConsData {
    int warehouse;
    int seller;
    bool force;
    SCIP_NODE* node;
    int propagated_columns;
    bool propagated;
};

namespace branch_and_price {

namespace {

const char* CONSHDLR_NAME = "assignment_branch";

// A column of the master problem: a set of sellers served by one warehouse.
struct Column {
    int warehouse;
    vector<int> sellers;
    SCIP_VAR* var;   // variable of the transformed problem
};

// Master problem shared by the pricer, the branching rule, the constraint handler
// of the branching decisions and the restricted master heuristic.
struct Master {
    const Instance* inst = nullptr;
    const Options* options = nullptr;
    bool integral = false;           // integral costs: bounds can be rounded up
    vector<SCIP_CONS*> assignment;   // transformed constraint of each seller
    vector<SCIP_CONS*> convexity;    // transformed constraint of each warehouse
    vector<SCIP_VAR*> unassigned;    // transformed u_j
    vector<Column> columns;
    map<pair<int, vector<int>>, int> known;   // index of each column in 'columns'
    int duplicates = 0;              // columns found again by the pricer (not added)
    int fallback_branchings = 0;     // LP solutions with fractional columns but integral x_ij
    int early_stops = 0;             // nodes whose column generation stopped early
    int mispricings = 0;             // smoothed pricing rounds that gave no column
    double root_bound = -1e300;      // best lower bound computed by the pricer at the root
    // Re-fixing (Options::refixing): the pairs ruled out for every solution cheaper
    // than the incumbent hold at every node, like branching decisions.
    fixing::Refixer* refixer = nullptr;
    // Solutions of the Lagrangian heuristic found by the re-fixing, as the columns
    // (already in the master) and the unassigned sellers, waiting to be handed to
    // SCIP by the heuristic PendingSolutions.
    struct Pending {
        vector<SCIP_VAR*> columns;
        vector<int> unassigned;
    };
    vector<Pending> pending;
    Solution lagrangian_best;           // best of them (total_cost +infinity if none): the
    Master() { lagrangian_best.total_cost = 1e300; }   // nodes may all be pruned first
};

// Whether column 'col' contradicts the branching decision 'd'.
bool violates(const Column& col, const SCIP_ConsData& d) {
    bool has_seller = false;
    for (int j : col.sellers) {
        if (j == d.seller) { has_seller = true; break; }
    }
    if (d.force) {
        // x_ij = 1: the columns of i must contain j, and the other warehouses cannot.
        return col.warehouse == d.warehouse ? !has_seller : has_seller;
    }
    // x_ij = 0: the columns of i cannot contain j.
    return col.warehouse == d.warehouse && has_seller;
}

// Constraint handler of the branching decisions. Each branching creates one
// constraint per child node; while a constraint is active, its propagation fixes
// to zero the columns that contradict the decision (and u_j if the seller is
// forced), including the columns generated later in other parts of the tree.
class BranchingConshdlr : public scip::ObjConshdlr {
public:
    BranchingConshdlr(SCIP* scip, Master* master)
        : ObjConshdlr(scip, CONSHDLR_NAME, "branching decisions on x_ij", 0, 0, 9999999, -1, 1, 1,
                      0, FALSE, FALSE, TRUE, SCIP_PROPTIMING_BEFORELP, SCIP_PRESOLTIMING_FAST),
          master_(master) {}

    SCIP_DECL_CONSDELETE(scip_delete) override {
        delete *consdata;
        *consdata = nullptr;
        return SCIP_OKAY;
    }

    SCIP_DECL_CONSTRANS(scip_trans) override {
        SCIP_ConsData* data = new SCIP_ConsData(*SCIPconsGetData(sourcecons));
        SCIP_CALL(SCIPcreateCons(scip, targetcons, SCIPconsGetName(sourcecons), conshdlr, data,
                                 SCIPconsIsInitial(sourcecons), SCIPconsIsSeparated(sourcecons),
                                 SCIPconsIsEnforced(sourcecons), SCIPconsIsChecked(sourcecons),
                                 SCIPconsIsPropagated(sourcecons), SCIPconsIsLocal(sourcecons),
                                 SCIPconsIsModifiable(sourcecons), SCIPconsIsDynamic(sourcecons),
                                 SCIPconsIsRemovable(sourcecons), SCIPconsIsStickingAtNode(sourcecons)));
        return SCIP_OKAY;
    }

    // The decisions hold in every solution of the subtree because propagation
    // fixes the contradicting columns, so enforcement and checking accept.
    SCIP_DECL_CONSENFOLP(scip_enfolp) override { *result = SCIP_FEASIBLE; return SCIP_OKAY; }
    SCIP_DECL_CONSENFOPS(scip_enfops) override { *result = SCIP_FEASIBLE; return SCIP_OKAY; }
    SCIP_DECL_CONSCHECK(scip_check) override { *result = SCIP_FEASIBLE; return SCIP_OKAY; }
    SCIP_DECL_CONSLOCK(scip_lock) override { return SCIP_OKAY; }

    SCIP_DECL_CONSACTIVE(scip_active) override {
        SCIP_ConsData* data = SCIPconsGetData(cons);
        if (data->propagated_columns < (int)master_->columns.size()) {
            data->propagated = false;
            SCIP_CALL(SCIPrepropagateNode(scip, data->node));
        }
        return SCIP_OKAY;
    }

    SCIP_DECL_CONSDEACTIVE(scip_deactive) override {
        SCIP_ConsData* data = SCIPconsGetData(cons);
        if (data->propagated) data->propagated_columns = master_->columns.size();
        return SCIP_OKAY;
    }

    SCIP_DECL_CONSPROP(scip_prop) override {
        *result = SCIP_DIDNOTFIND;
        for (int c = 0; c < nconss; c++) {
            SCIP_ConsData* data = SCIPconsGetData(conss[c]);
            if (data->propagated) continue;
            if (data->force) {
                SCIP_Bool infeasible, fixed;
                SCIP_CALL(SCIPfixVar(scip, master_->unassigned[data->seller], 0.0, &infeasible, &fixed));
                if (infeasible) { *result = SCIP_CUTOFF; return SCIP_OKAY; }
                if (fixed) *result = SCIP_REDUCEDDOM;
            }
            for (int k = data->propagated_columns; k < (int)master_->columns.size(); k++) {
                const Column& col = master_->columns[k];
                if (!violates(col, *data) || SCIPvarGetUbLocal(col.var) < 0.5) continue;
                SCIP_Bool infeasible, fixed;
                SCIP_CALL(SCIPfixVar(scip, col.var, 0.0, &infeasible, &fixed));
                if (infeasible) { *result = SCIP_CUTOFF; return SCIP_OKAY; }
                if (fixed) *result = SCIP_REDUCEDDOM;
            }
            data->propagated_columns = master_->columns.size();
            data->propagated = true;
        }
        return SCIP_OKAY;
    }

private:
    Master* master_;
};

// Decisions active at the current node.
vector<SCIP_ConsData> active_decisions(SCIP* scip) {
    vector<SCIP_ConsData> decisions;
    SCIP_CONSHDLR* conshdlr = SCIPfindConshdlr(scip, CONSHDLR_NAME);
    SCIP_CONS** conss = SCIPconshdlrGetConss(conshdlr);
    int nactive = SCIPconshdlrGetNActiveConss(conshdlr);
    for (int c = 0; c < nactive; c++) decisions.push_back(*SCIPconsGetData(conss[c]));
    return decisions;
}

// Best column of a warehouse for some dual values, with its reduced cost.
struct Candidate {
    int warehouse;
    vector<int> sellers;
    double reduced_cost;
};

// Branching decisions of the current node: sellers forbidden in each warehouse,
// sellers forced into it, and sellers forced somewhere (whose u_j is 0).
struct Decisions {
    vector<vector<bool>> forbidden;
    vector<vector<int>> forced;
    vector<bool> assigned;
    const fixing::Refixer* refixer = nullptr;   // fixings of the re-fixing, if any
    bool conflict = false;   // the branching decisions contradict the fixings
};

Decisions read_decisions(SCIP* scip, const Master& master) {
    const Instance& inst = *master.inst;
    int n = inst.sellers.size();
    int m = inst.warehouses.size();
    Decisions d{vector<vector<bool>>(m, vector<bool>(n, false)), vector<vector<int>>(m), vector<bool>(n, false)};
    vector<int> forced_into(n, -1);
    for (const SCIP_ConsData& c : active_decisions(scip)) {
        if (c.force) {
            d.forced[c.warehouse].push_back(c.seller);
            d.assigned[c.seller] = true;
            forced_into[c.seller] = c.warehouse;
            for (int i = 0; i < m; i++) {
                if (i != c.warehouse) d.forbidden[i][c.seller] = true;
            }
        } else {
            d.forbidden[c.warehouse][c.seller] = true;
        }
    }
    const fixing::Refixer* refixer = master.refixer;
    if (refixer != nullptr) {
        d.refixer = refixer;
        for (int j = 0; j < n; j++) {
            int f = refixer->forced(j);
            if (forced_into[j] >= 0 && refixer->ruled_out(j, forced_into[j])) d.conflict = true;
            if (f >= 0 && forced_into[j] < 0 && d.forbidden[f][j]) d.conflict = true;   // branching x_fj = 0
            if (f >= 0 && forced_into[j] < 0) {
                d.forced[f].push_back(j);
                d.assigned[j] = true;
            }
            for (int i = 0; i < m; i++) {
                if (refixer->ruled_out(j, i)) d.forbidden[i][j] = true;
            }
        }
    }
    return d;
}

// Column of minimum reduced cost of warehouse i under the decisions of the node,
// with one more seller forbidden ('also_forbidden') or forced ('also_forced') if
// given; the column may be empty. Returns false if the forced sellers do not fit.
// In Farkas pricing the costs are replaced by zero.
bool best_column(const Instance& inst, int i, const vector<double>& pi, const vector<double>& mu,
                 const Decisions& d, bool farkas, Candidate& out, int also_forbidden = -1,
                 int also_forced = -1) {
    int n = inst.sellers.size();
    int capacity = inst.warehouses[i].capacity;
    vector<int> forced = d.forced[i];
    if (also_forced >= 0) forced.push_back(also_forced);
    double forced_profit = 0;
    vector<bool> skip(n, false);
    for (int j : forced) {
        if (d.refixer != nullptr && d.refixer->ruled_out(j, i)) return false;
        capacity -= inst.demand(j, i);
        forced_profit += pi[j] - (farkas ? 0.0 : inst.cost[j][i]);
        skip[j] = true;
    }
    if (capacity < 0) return false;
    vector<double> profit(n);
    vector<int> weight(n);
    for (int j = 0; j < n; j++) {
        bool excluded = skip[j] || d.forbidden[i][j] || j == also_forbidden;
        profit[j] = excluded ? 0.0 : pi[j] - (farkas ? 0.0 : inst.cost[j][i]);
        weight[j] = inst.demand(j, i);
    }
    vector<int> chosen;
    double value = forced_profit + knapsack::solve(profit, weight, capacity, chosen);
    out.warehouse = i;
    out.sellers = forced;
    out.sellers.insert(out.sellers.end(), chosen.begin(), chosen.end());
    sort(out.sellers.begin(), out.sellers.end());
    out.reduced_cost = -value - mu[i];
    return true;
}

double reduced_cost(const Instance& inst, const Candidate& c, const vector<double>& pi, const vector<double>& mu) {
    double rc = -mu[c.warehouse];
    for (int j : c.sellers) rc += inst.cost[j][c.warehouse] - pi[j];
    return rc;
}

// Lagrangian bound of the node for duals (pi, mu <= 0), from the minimum reduced
// cost of each warehouse:
//   sum_j pi_j + sum_i mu_i + sum_i min(0, rc_i) + sum_{j not forced} min(0, penalty - pi_j).
double lagrangian_bound(const Instance& inst, const vector<double>& pi, const vector<double>& mu,
                        const vector<Candidate>& candidates, const Decisions& d) {
    double bound = 0;
    for (int j = 0; j < (int)pi.size(); j++) bound += pi[j] + (d.assigned[j] ? 0.0 : min(0.0, inst.penalty - pi[j]));
    for (double u : mu) bound += u;
    for (const Candidate& c : candidates) bound += min(0.0, c.reduced_cost);
    return bound;
}

// Whether some x_ij = sum of the columns of warehouse i that contain j is
// fractional in the current LP solution (and the values x_ij in 'x').
bool fractional_assignment(SCIP* scip, const Master& master, unordered_map<long long, double>& x) {
    int m = master.inst->warehouses.size();
    x.clear();
    for (const Column& col : master.columns) {
        double value = SCIPgetSolVal(scip, nullptr, col.var);
        if (value < 1e-9) continue;
        for (int j : col.sellers) x[(long long)j * m + col.warehouse] += value;
    }
    for (const auto& [key, value] : x) {
        if (!SCIPisFeasIntegral(scip, value)) return true;
    }
    return false;
}

// Pricer: for each warehouse i, the column of minimum reduced cost
//   sum_{j in p} (c_ij - pi_j) - mu_i
// is a 0-1 knapsack with profits pi_j - c_ij (pi: duals of the assignment
// constraints, mu: dual of the convexity constraint). The branching decisions of
// the node remove sellers from the knapsack or force them into it. In Farkas
// pricing (infeasible master LP) the costs are replaced by zero and the duals by
// the Farkas multipliers.
//
// The pricing is stabilized by dual smoothing (Wentges): it prices with
// alpha * center + (1 - alpha) * (LP duals), where the center is the dual vector
// with the best Lagrangian bound found at the node, and adds the columns with
// negative reduced cost for the LP duals. When the smoothed duals give none
// (mispricing), the k-th consecutive try uses alpha_k = max(0, 1 - k (1 - alpha)),
// down to the LP duals, as in the scheme of Pessoa, Sadykov, Uchoa and Vanderbeck
// (2018), so the column generation ends only when no column has negative reduced
// cost for the LP duals. With automatic smoothing, alpha also adapts after every
// round: the subgradient g of the Lagrangian function at the smoothed point tells
// whether moving towards the LP duals improves the bound; if g points towards them
// (g . (LP duals - center) > 0), alpha decreases by 0.1, otherwise it moves 10% of
// the way towards 1 (these constants are ours). Every pricing round also gives
// the Lagrangian bound of the node (see lagrangian_bound).
class KnapsackPricer : public scip::ObjPricer {
public:
    KnapsackPricer(SCIP* scip, Master* master)
        : ObjPricer(scip, "knapsack", "one 0-1 knapsack per warehouse", 0, TRUE), master_(master) {}

    // Initial center of the root: the multipliers of the subgradient method, with
    // the convexity duals that make every reduced cost nonnegative,
    // mu_i = -KP_i(pi). Their Lagrangian bound is the bound of the subgradient, so
    // the root bound starts there instead of at the poor bound of the first LP
    // duals, and the smoothing starts from duals close to the optimal ones.
    void set_root_center(const vector<double>& pi, const vector<double>& mu, double bound) {
        root_pi_ = pi;
        root_mu_ = mu;
        root_center_bound_ = bound;
    }

    double alpha() const { return alpha_; }

    SCIP_DECL_PRICERREDCOST(scip_redcost) override {
        const Instance& inst = *master_->inst;
        const Options& options = *master_->options;
        int n = inst.sellers.size();
        int m = inst.warehouses.size();
        vector<double> pi(n), mu(m);
        for (int j = 0; j < n; j++) pi[j] = SCIPgetDualsolLinear(scip, master_->assignment[j]);
        for (int i = 0; i < m; i++) mu[i] = min(0.0, SCIPgetDualsolLinear(scip, master_->convexity[i]));
        if (master_->refixer != nullptr) SCIP_CALL(refix(scip));
        Decisions decisions = read_decisions(scip, *master_);
        if (decisions.conflict || (master_->refixer != nullptr && master_->refixer->no_improving_solution())) {
            // no solution of this node is cheaper than the incumbent: the node is pruned
            *lowerbound = SCIPgetPrimalbound(scip);
            *result = SCIP_SUCCESS;
            return SCIP_OKAY;
        }

        long long node = SCIPnodeGetNumber(SCIPgetCurrentNode(scip));
        if (node != center_node_) {
            center_node_ = node;
            has_center_ = false;
            best_bound_ = -1e300;
            history_.clear();
            if (SCIPgetDepth(scip) == 0 && !root_pi_.empty()) {
                center_pi_ = root_pi_;
                center_mu_ = root_mu_;
                best_bound_ = root_center_bound_;
                has_center_ = true;
            }
        }
        int added = 0;
        if (has_center_ && alpha_ > 0) {
            for (int k = 0;; k++) {
                double a = k == 0 ? alpha_ : max(0.0, 1.0 - k * (1.0 - alpha_));
                if (a <= 0) break;
                vector<double> pi_s(n), mu_s(m);
                for (int j = 0; j < n; j++) pi_s[j] = a * center_pi_[j] + (1 - a) * pi[j];
                for (int i = 0; i < m; i++) mu_s[i] = min(0.0, a * center_mu_[i] + (1 - a) * mu[i]);
                vector<Candidate> candidates = all_columns(pi_s, mu_s, decisions);
                if (k == 0 && options.automatic_smoothing) adapt_alpha(pi, mu, pi_s, candidates, decisions);
                update_center(pi_s, mu_s, candidates, decisions);
                for (const Candidate& c : candidates) {
                    if (SCIPisDualfeasNegative(scip, reduced_cost(inst, c, pi, mu))) {
                        SCIP_CALL(add_if_new(scip, c));
                        added++;
                    }
                }
                if (added > 0) break;
                master_->mispricings++;
                if (!options.automatic_smoothing) break;   // fixed alpha: one try, then the LP duals
            }
        }
        if (added == 0 || options.extra_columns) {
            vector<Candidate> candidates = all_columns(pi, mu, decisions);
            update_center(pi, mu, candidates, decisions);
            for (const Candidate& c : candidates) {
                if (!SCIPisDualfeasNegative(scip, c.reduced_cost)) continue;
                SCIP_CALL(add_if_new(scip, c));
                added++;
                if (!options.extra_columns) continue;
                // second column: without the least profitable free seller of this one
                int weakest = -1;
                double weakest_profit = 1e300;
                for (int j : c.sellers) {
                    bool is_forced = find(decisions.forced[c.warehouse].begin(), decisions.forced[c.warehouse].end(), j)
                                   != decisions.forced[c.warehouse].end();
                    double profit = pi[j] - inst.cost[j][c.warehouse];
                    if (!is_forced && profit < weakest_profit) {
                        weakest_profit = profit;
                        weakest = j;
                    }
                }
                Candidate second;
                if (weakest >= 0 && best_column(inst, c.warehouse, pi, mu, decisions, false, second, weakest)
                    && SCIPisDualfeasNegative(scip, second.reduced_cost)) {
                    SCIP_CALL(add_if_new(scip, second));
                }
            }
        }
        *lowerbound = best_bound_;
        if (SCIPgetDepth(scip) == 0) master_->root_bound = max(master_->root_bound, best_bound_);
        history_.push_back(best_bound_);
        *stopearly = FALSE;
        if (options.early_termination && added > 0 && stop_early(scip)) {
            *stopearly = TRUE;
            master_->early_stops++;
        }
        *result = SCIP_SUCCESS;
        return SCIP_OKAY;
    }

    SCIP_DECL_PRICERFARKAS(scip_farkas) override {
        const Instance& inst = *master_->inst;
        int n = inst.sellers.size();
        int m = inst.warehouses.size();
        vector<double> pi(n), mu(m);
        for (int j = 0; j < n; j++) pi[j] = SCIPgetDualfarkasLinear(scip, master_->assignment[j]);
        for (int i = 0; i < m; i++) mu[i] = SCIPgetDualfarkasLinear(scip, master_->convexity[i]);
        if (master_->refixer != nullptr) SCIP_CALL(refix(scip));
        Decisions decisions = read_decisions(scip, *master_);
        if (decisions.conflict || (master_->refixer != nullptr && master_->refixer->no_improving_solution())) {
            *result = SCIP_SUCCESS;   // no columns: the infeasible node is pruned
            return SCIP_OKAY;
        }
        for (int i = 0; i < m; i++) {
            Candidate c;
            if (best_column(inst, i, pi, mu, decisions, true, c) && SCIPisDualfeasNegative(scip, c.reduced_cost))
                SCIP_CALL(add_if_new(scip, c));
        }
        *result = SCIP_SUCCESS;
        return SCIP_OKAY;
    }

private:
    Master* master_;
    double alpha_ = 0.8;
    long long center_node_ = -1;
    bool has_center_ = false;
    double best_bound_ = -1e300;
    vector<double> center_pi_, center_mu_;
    vector<double> root_pi_, root_mu_;
    double root_center_bound_ = -1e300;
    vector<double> history_;          // bound of the node after each round

    vector<Candidate> all_columns(const vector<double>& pi, const vector<double>& mu, const Decisions& d) const {
        vector<Candidate> candidates;
        for (int i = 0; i < (int)master_->inst->warehouses.size(); i++) {
            Candidate c;
            if (best_column(*master_->inst, i, pi, mu, d, false, c)) candidates.push_back(c);
        }
        return candidates;
    }

    // The duals become the new center if their Lagrangian bound improves.
    void update_center(const vector<double>& pi, const vector<double>& mu, const vector<Candidate>& candidates,
                       const Decisions& d) {
        double bound = lagrangian_bound(*master_->inst, pi, mu, candidates, d);
        if (bound > best_bound_) {
            best_bound_ = bound;
            center_pi_ = pi;
            center_mu_ = mu;
            has_center_ = true;
        }
    }

    // Subgradient of the Lagrangian function at the smoothed point, against the
    // direction from the center to the LP duals.
    void adapt_alpha(const vector<double>& pi, const vector<double>& mu, const vector<double>& pi_s,
                     const vector<Candidate>& candidates, const Decisions& d) {
        const Instance& inst = *master_->inst;
        int n = inst.sellers.size();
        vector<double> g(n, 1.0);
        for (int j = 0; j < n; j++) {
            if (!d.assigned[j] && inst.penalty - pi_s[j] < 0) g[j] -= 1.0;
        }
        double product = 0;
        for (const Candidate& c : candidates) {
            bool chosen = c.reduced_cost < 0;
            if (chosen) for (int j : c.sellers) g[j] -= 1.0;
            double g_mu = 1.0 - (chosen ? 1.0 : 0.0);
            product += g_mu * (mu[c.warehouse] - center_mu_[c.warehouse]);
        }
        for (int j = 0; j < n; j++) product += g[j] * (pi[j] - center_pi_[j]);
        if (product > 0) alpha_ = max(0.0, alpha_ - 0.1);
        else alpha_ = min(0.99, alpha_ + 0.1 * (1.0 - alpha_));
    }

    // Early branching: more columns cannot raise the integer bound of the node, or
    // the bound has not improved for a number of rounds (100 at the root, 20 below);
    // only when some x_ij is fractional, so that the node can be branched on.
    bool stop_early(SCIP* scip) const {
        double lp = SCIPgetLPObjval(scip);
        bool stop = master_->integral && ceil(best_bound_ - 1e-6) >= ceil(lp - 1e-6);
        int window = SCIPgetDepth(scip) == 0 ? 100 : 20;
        int rounds = history_.size();
        if (rounds > window && best_bound_ - history_[rounds - 1 - window] <= 1e-6 * max(1.0, fabs(best_bound_)))
            stop = true;
        if (!stop) return false;
        unordered_map<long long, double> x;
        return fractional_assignment(scip, *master_, x);
    }

    // Updates the re-fixing with the incumbent; an improving solution of the
    // Lagrangian heuristic gets its columns and waits for PendingSolutions.
    SCIP_RETCODE refix(SCIP* scip) {
        double ub = SCIPgetPrimalbound(scip);
        if (SCIPisInfinity(scip, ub)) return SCIP_OKAY;
        Solution best;
        master_->refixer->update(ub, best);
        if (best.total_cost >= ub - 1e-9) return SCIP_OKAY;
        const Instance& inst = *master_->inst;
        Master::Pending pending;
        for (int i = 0; i < (int)inst.warehouses.size(); i++) {
            vector<int> sellers = best.sellers_per_warehouse[i];
            if (sellers.empty()) continue;
            sort(sellers.begin(), sellers.end());
            auto it = master_->known.find({i, sellers});
            if (it == master_->known.end()) {
                master_->known.emplace(make_pair(i, sellers), (int)master_->columns.size());
                SCIP_CALL(add_column(scip, i, sellers));
                pending.columns.push_back(master_->columns.back().var);
            } else {
                pending.columns.push_back(master_->columns[it->second].var);
            }
        }
        for (int j = 0; j < (int)inst.sellers.size(); j++) {
            if (best.assigned_warehouse[j] < 0) pending.unassigned.push_back(j);
        }
        master_->pending.push_back(pending);
        if (best.total_cost < master_->lagrangian_best.total_cost) master_->lagrangian_best = best;
        return SCIP_OKAY;
    }

    SCIP_RETCODE add_if_new(SCIP* scip, const Candidate& c) {
        if (c.sellers.empty()) return SCIP_OKAY;
        if (!master_->known.emplace(make_pair(c.warehouse, c.sellers), (int)master_->columns.size()).second) {
            master_->duplicates++;   // already in the problem: SCIP prices it in if it left the LP
            return SCIP_OKAY;
        }
        return add_column(scip, c.warehouse, c.sellers);
    }

    SCIP_RETCODE add_column(SCIP* scip, int i, const vector<int>& sellers) {
        const Instance& inst = *master_->inst;
        double cost = 0;
        for (int j : sellers) cost += inst.cost[j][i];
        string name = "col_" + to_string(master_->columns.size());
        SCIP_VAR* var;
        // Removable only with column aging: SCIP keeps the variable in the problem
        // and its own pricing of the problem variables brings it back to the LP.
        SCIP_CALL(SCIPcreateVar(scip, &var, name.c_str(), 0.0, 1.0, cost, SCIP_VARTYPE_BINARY,
                                TRUE, master_->options->column_aging, nullptr, nullptr, nullptr, nullptr, nullptr));
        // The bound lambda <= 1 is implied by the convexity constraint. Making it lazy
        // keeps it out of the LP: otherwise a column at its upper bound could have a
        // negative reduced cost, the pricer would find it again and the column
        // generation could stop before the LP is optimal.
        SCIP_CALL(SCIPchgVarUbLazy(scip, var, 1.0));
        SCIP_CALL(SCIPaddPricedVar(scip, var, 1.0));
        for (int j : sellers) SCIP_CALL(SCIPaddCoefLinear(scip, master_->assignment[j], var, 1.0));
        SCIP_CALL(SCIPaddCoefLinear(scip, master_->convexity[i], var, 1.0));
        master_->columns.push_back({i, sellers, var});
        SCIP_CALL(SCIPreleaseVar(scip, &var));
        return SCIP_OKAY;
    }
};

// Branching on x_ij = sum of the columns of warehouse i that contain seller j,
// with the children x_ij = 0 and x_ij = 1. Without strong branching it picks the
// most fractional x_ij. With it, it evaluates the 10 most fractional ones by the
// Lagrangian bounds of their children for the current LP duals: the child
// x_ij = 0 changes the knapsack of warehouse i (j forbidden), the child x_ij = 1
// forces j into that knapsack, forbids it in the knapsacks that contained it and
// drops its u_j. It picks the largest product of the two bound improvements and
// hands the bounds to the children.
class AssignmentBranchrule : public scip::ObjBranchrule {
public:
    AssignmentBranchrule(SCIP* scip, Master* master)
        : ObjBranchrule(scip, "assignment", "branching on x_ij", 50000, -1, 1.0), master_(master) {}

    SCIP_DECL_BRANCHEXECLP(scip_execlp) override {
        const Instance& inst = *master_->inst;
        int n = inst.sellers.size();
        int m = inst.warehouses.size();
        unordered_map<long long, double> x;
        fractional_assignment(scip, *master_, x);
        vector<pair<double, long long>> fractional;
        for (const auto& [key, value] : x) {
            if (!SCIPisFeasIntegral(scip, value)) fractional.push_back({fabs(value - 0.5), key});
        }
        if (fractional.empty()) {
            master_->fallback_branchings++;
            *result = SCIP_DIDNOTRUN;
            return SCIP_OKAY;
        }
        sort(fractional.begin(), fractional.end());
        long long best_key = fractional[0].second;
        double bound0 = -1e300, bound1 = -1e300;
        if (master_->options->strong_branching && fractional.size() > 1) {
            vector<double> pi(n), mu(m);
            for (int j = 0; j < n; j++) pi[j] = SCIPgetDualsolLinear(scip, master_->assignment[j]);
            for (int i = 0; i < m; i++) mu[i] = min(0.0, SCIPgetDualsolLinear(scip, master_->convexity[i]));
            Decisions d = read_decisions(scip, *master_);
            vector<Candidate> base(m);
            vector<bool> feasible(m, false);
            vector<Candidate> present;
            for (int i = 0; i < m; i++) {
                feasible[i] = best_column(inst, i, pi, mu, d, false, base[i]);
                if (feasible[i]) present.push_back(base[i]);
            }
            double parent = lagrangian_bound(inst, pi, mu, present, d);
            double best_score = -1;
            int evaluated = min<int>(10, fractional.size());
            for (int t = 0; t < evaluated; t++) {
                long long key = fractional[t].second;
                int j = key / m, i = key % m;
                Candidate c;
                // child x_ij = 0
                double b0 = parent;
                if (feasible[i] && best_column(inst, i, pi, mu, d, false, c, j))
                    b0 += min(0.0, c.reduced_cost) - min(0.0, base[i].reduced_cost);
                // child x_ij = 1
                double b1 = parent;
                if (!feasible[i] || !best_column(inst, i, pi, mu, d, false, c, -1, j)) {
                    b1 = 1e300;   // the forced sellers do not fit
                } else {
                    b1 += min(0.0, c.reduced_cost) - min(0.0, base[i].reduced_cost);
                    for (int k = 0; k < m; k++) {
                        if (k == i || !feasible[k]) continue;
                        if (!binary_search(base[k].sellers.begin(), base[k].sellers.end(), j)) continue;
                        Candidate other;
                        if (best_column(inst, k, pi, mu, d, false, other, j))
                            b1 += min(0.0, other.reduced_cost) - min(0.0, base[k].reduced_cost);
                    }
                    if (!d.assigned[j]) b1 -= min(0.0, inst.penalty - pi[j]);
                }
                double score = max(b0 - parent, 1e-6) * max(min(b1, 1e12) - parent, 1e-6);
                if (score > best_score) {
                    best_score = score;
                    best_key = key;
                    bound0 = b0;
                    bound1 = b1;
                }
            }
        }
        int seller = best_key / m;
        int warehouse = best_key % m;
        for (bool force : {false, true}) {
            SCIP_NODE* child;
            SCIP_CALL(SCIPcreateChild(scip, &child, 0.0, SCIPgetLocalTransEstimate(scip)));
            double bound = force ? bound1 : bound0;
            if (bound > -1e299 && bound < 1e299) SCIP_CALL(SCIPupdateNodeLowerbound(scip, child, bound));
            else if (bound >= 1e299) SCIP_CALL(SCIPupdateNodeLowerbound(scip, child, SCIPinfinity(scip)));
            SCIP_ConsData* data = new SCIP_ConsData{warehouse, seller, force, child, 0, false};
            string name = string(force ? "force_" : "forbid_") + to_string(warehouse) + "_" + to_string(seller);
            SCIP_CONS* cons;
            SCIP_CALL(SCIPcreateCons(scip, &cons, name.c_str(), SCIPfindConshdlr(scip, CONSHDLR_NAME), data,
                                     FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE));
            SCIP_CALL(SCIPaddConsNode(scip, child, cons, nullptr));
            SCIP_CALL(SCIPreleaseCons(scip, &cons));
        }
        *result = SCIP_BRANCHED;
        return SCIP_OKAY;
    }

private:
    Master* master_;
};

// Restricted master heuristic (price-and-branch): a separate SCIP solves the
// integer master restricted to the columns generated so far, starting from the
// incumbent, and its solution is handed to the branch-and-price. It runs at the
// root: after the column generation of the root, and also during it at most once
// every 'interval' seconds, so that the large instances, whose root does not
// finish within the time limit, also get solutions built from the columns.
class RestrictedMasterHeuristic : public scip::ObjHeur {
public:
    RestrictedMasterHeuristic(SCIP* scip, Master* master, double sub_time_limit, double interval)
        : ObjHeur(scip, "restricted_master", "integer master over the generated columns", 'R',
                  1000, 0, 0, 0, SCIP_HEURTIMING_AFTERLPNODE | SCIP_HEURTIMING_DURINGPRICINGLOOP, TRUE),
          master_(master), sub_time_limit_(sub_time_limit), interval_(interval) {}

    int runs = 0;
    int improvements = 0;

    SCIP_DECL_HEUREXEC(scip_exec) override {
        *result = SCIP_DIDNOTRUN;
        int columns = master_->columns.size();
        double now = SCIPgetSolvingTime(scip);
        if (columns == columns_last_run_) return SCIP_OKAY;
        if (heurtiming == SCIP_HEURTIMING_DURINGPRICINGLOOP && now - time_last_run_ < interval_) return SCIP_OKAY;
        double limit;
        SCIP_CALL(SCIPgetRealParam(scip, "limits/time", &limit));
        double left = limit - now;
        if (left < 1.0) return SCIP_OKAY;
        columns_last_run_ = columns;
        time_last_run_ = now;
        runs++;
        *result = SCIP_DIDNOTFIND;

        const Instance& inst = *master_->inst;
        int n = inst.sellers.size();
        int m = inst.warehouses.size();
        SCIP* sub = nullptr;
        SCIP_CALL(SCIPcreate(&sub));
        SCIP_CALL(SCIPincludeDefaultPlugins(sub));
        SCIP_CALL(SCIPsetIntParam(sub, "display/verblevel", 0));
        SCIP_CALL(SCIPsetRealParam(sub, "limits/time", min(sub_time_limit_, left - 0.5)));
        SCIP_CALL(SCIPcreateProbBasic(sub, "restricted_master"));
        vector<SCIP_VAR*> lambda(columns), u(n);
        vector<vector<SCIP_VAR*>> in_seller(n);
        vector<vector<SCIP_VAR*>> in_warehouse(m);
        for (int k = 0; k < columns; k++) {
            const Column& col = master_->columns[k];
            double cost = 0;
            for (int j : col.sellers) cost += inst.cost[j][col.warehouse];
            SCIP_CALL(SCIPcreateVarBasic(sub, &lambda[k], nullptr, 0.0, 1.0, cost, SCIP_VARTYPE_BINARY));
            SCIP_CALL(SCIPaddVar(sub, lambda[k]));
            for (int j : col.sellers) in_seller[j].push_back(lambda[k]);
            in_warehouse[col.warehouse].push_back(lambda[k]);
        }
        for (int j = 0; j < n; j++) {
            SCIP_CALL(SCIPcreateVarBasic(sub, &u[j], nullptr, 0.0, 1.0, inst.penalty, SCIP_VARTYPE_BINARY));
            SCIP_CALL(SCIPaddVar(sub, u[j]));
            in_seller[j].push_back(u[j]);
            SCIP_CONS* cons;
            SCIP_CALL(SCIPcreateConsBasicSetpart(sub, &cons, "assign", in_seller[j].size(), in_seller[j].data()));
            SCIP_CALL(SCIPaddCons(sub, cons));
            SCIP_CALL(SCIPreleaseCons(sub, &cons));
        }
        for (int i = 0; i < m; i++) {
            if (in_warehouse[i].empty()) continue;
            SCIP_CONS* cons;
            SCIP_CALL(SCIPcreateConsBasicSetpack(sub, &cons, "convexity", in_warehouse[i].size(), in_warehouse[i].data()));
            SCIP_CALL(SCIPaddCons(sub, cons));
            SCIP_CALL(SCIPreleaseCons(sub, &cons));
        }
        // the incumbent of the branch-and-price as a starting solution
        SCIP_SOL* incumbent = SCIPgetBestSol(scip);
        if (incumbent != nullptr) {
            SCIP_SOL* start;
            SCIP_CALL(SCIPcreateSol(sub, &start, nullptr));
            for (int k = 0; k < columns; k++) {
                if (SCIPgetSolVal(scip, incumbent, master_->columns[k].var) > 0.5)
                    SCIP_CALL(SCIPsetSolVal(sub, start, lambda[k], 1.0));
            }
            for (int j = 0; j < n; j++) {
                if (SCIPgetSolVal(scip, incumbent, master_->unassigned[j]) > 0.5)
                    SCIP_CALL(SCIPsetSolVal(sub, start, u[j], 1.0));
            }
            SCIP_Bool stored;
            SCIP_CALL(SCIPaddSolFree(sub, &start, &stored));
        }
        SCIP_CALL(SCIPsolve(sub));

        SCIP_SOL* best = SCIPgetBestSol(sub);
        if (best != nullptr && SCIPisLT(scip, SCIPgetSolOrigObj(sub, best), SCIPgetPrimalbound(scip))) {
            SCIP_SOL* sol;
            SCIP_CALL(SCIPcreateSol(scip, &sol, heur));
            for (int k = 0; k < columns; k++) {
                if (SCIPgetSolVal(sub, best, lambda[k]) > 0.5)
                    SCIP_CALL(SCIPsetSolVal(scip, sol, master_->columns[k].var, 1.0));
            }
            for (int j = 0; j < n; j++) {
                if (SCIPgetSolVal(sub, best, u[j]) > 0.5)
                    SCIP_CALL(SCIPsetSolVal(scip, sol, master_->unassigned[j], 1.0));
            }
            SCIP_Bool stored;
            SCIP_CALL(SCIPtrySolFree(scip, &sol, FALSE, FALSE, TRUE, TRUE, TRUE, &stored));
            if (stored) {
                improvements++;
                *result = SCIP_FOUNDSOL;
            }
        }
        for (SCIP_VAR*& var : lambda) SCIP_CALL(SCIPreleaseVar(sub, &var));
        for (SCIP_VAR*& var : u) SCIP_CALL(SCIPreleaseVar(sub, &var));
        SCIP_CALL(SCIPfree(&sub));
        return SCIP_OKAY;
    }

private:
    Master* master_;
    double sub_time_limit_;
    double interval_;
    int columns_last_run_ = -1;
    double time_last_run_ = -1e300;
};

// Propagator of the re-fixing: fixes to zero, globally, the columns that contain a
// pair ruled out by the fixings or miss a seller fixed into their warehouse, and
// the u_j of the fixed sellers (the pricer already skips those pairs). It scans
// every column again when the fixings change and only the new ones otherwise.
class RefixingColumns : public scip::ObjProp {
public:
    RefixingColumns(SCIP* scip, Master* master)
        : ObjProp(scip, "refixing_columns", "columns ruled out by the re-fixing", 1000000, 1, FALSE,
                  SCIP_PROPTIMING_BEFORELP, 0, 0, SCIP_PRESOLTIMING_NONE),
          master_(master) {}

    SCIP_DECL_PROPEXEC(scip_exec) override {
        *result = SCIP_DIDNOTRUN;
        const fixing::Refixer* refixer = master_->refixer;
        if (refixer == nullptr || refixer->version() == 0) return SCIP_OKAY;
        if (refixer->no_improving_solution()) { *result = SCIP_CUTOFF; return SCIP_OKAY; }
        const Instance& inst = *master_->inst;
        int n = inst.sellers.size();
        int m = inst.warehouses.size();
        bool changed = refixer->version() != version_;
        int first = changed ? 0 : scanned_;
        if (!changed && first >= (int)master_->columns.size()) return SCIP_OKAY;
        *result = SCIP_DIDNOTFIND;
        bool reduced = false;
        if (changed) {
            version_ = refixer->version();
            forced_count_.assign(m, 0);
            if (u_fixed_.empty()) u_fixed_.assign(n, 0);
            for (int j = 0; j < n; j++) {
                int f = refixer->forced(j);
                if (f < 0) continue;
                forced_count_[f]++;
                if (u_fixed_[j]) continue;
                u_fixed_[j] = 1;
                SCIP_Bool infeasible, tightened;
                SCIP_CALL(SCIPtightenVarUbGlobal(scip, master_->unassigned[j], 0.0, FALSE, &infeasible, &tightened));
                if (tightened) reduced = true;
            }
        }
        for (int k = first; k < (int)master_->columns.size(); k++) {
            const Column& col = master_->columns[k];
            if (col.var == nullptr || SCIPvarGetUbGlobal(col.var) < 0.5) continue;
            int i = col.warehouse;
            int forced_here = 0;
            bool ruled = false;
            for (int j : col.sellers) {
                if (refixer->ruled_out(j, i)) { ruled = true; break; }
                if (refixer->forced(j) == i) forced_here++;
            }
            if (!ruled && forced_here < forced_count_[i]) ruled = true;   // misses a seller fixed into i
            if (!ruled) continue;
            SCIP_Bool infeasible, tightened;
            SCIP_CALL(SCIPtightenVarUbGlobal(scip, col.var, 0.0, FALSE, &infeasible, &tightened));
            if (tightened) reduced = true;
        }
        scanned_ = master_->columns.size();
        if (reduced) *result = SCIP_REDUCEDDOM;
        return SCIP_OKAY;
    }

private:
    Master* master_;
    long long version_ = 0;
    int scanned_ = 0;
    vector<int> forced_count_;    // sellers fixed into each warehouse
    vector<char> u_fixed_;
};

// Hands to SCIP the solutions of the Lagrangian heuristic found by the re-fixing
// in the pricer (whose columns are already in the master).
class PendingSolutions : public scip::ObjHeur {
public:
    PendingSolutions(SCIP* scip, Master* master)
        : ObjHeur(scip, "lagrangian_solutions", "solutions of the Lagrangian heuristic of the re-fixing", 'L',
                  2000, 1, 0, -1, SCIP_HEURTIMING_DURINGPRICINGLOOP | SCIP_HEURTIMING_AFTERNODE, FALSE),
          master_(master) {}

    int found = 0;

    SCIP_DECL_HEUREXEC(scip_exec) override {
        *result = SCIP_DIDNOTRUN;
        if (master_->pending.empty()) return SCIP_OKAY;
        *result = SCIP_DIDNOTFIND;
        for (const Master::Pending& p : master_->pending) {
            SCIP_SOL* sol;
            SCIP_CALL(SCIPcreateSol(scip, &sol, heur));
            for (SCIP_VAR* var : p.columns) SCIP_CALL(SCIPsetSolVal(scip, sol, var, 1.0));
            for (int j : p.unassigned) SCIP_CALL(SCIPsetSolVal(scip, sol, master_->unassigned[j], 1.0));
            SCIP_Bool stored;
            SCIP_CALL(SCIPtrySolFree(scip, &sol, FALSE, FALSE, TRUE, TRUE, TRUE, &stored));
            if (stored) {
                found++;
                *result = SCIP_FOUNDSOL;
            }
        }
        master_->pending.clear();
        return SCIP_OKAY;
    }

private:
    Master* master_;
};

bool integral_costs(const Instance& inst) {
    for (const auto& row : inst.cost)
        for (double c : row)
            if (c != floor(c)) return false;
    return inst.penalty == floor(inst.penalty);
}

}

Result solve(const Instance& inst, double time_limit, const Solution* start, const Options& options) {
    auto t0 = chrono::high_resolution_clock::now();
    Solution initial = start != nullptr ? *start : metaheuristics::initial_solution(inst);
    int n = inst.sellers.size();
    int m = inst.warehouses.size();

    Master master;
    master.inst = &inst;
    master.options = &options;
    master.integral = integral_costs(inst);
    SCIP* scip = create_scip(0);
    SCIP_CHECK(SCIPincludeObjConshdlr(scip, new BranchingConshdlr(scip, &master), TRUE));
    KnapsackPricer* pricer = new KnapsackPricer(scip, &master);
    SCIP_CHECK(SCIPincludeObjPricer(scip, pricer, TRUE));
    SCIP_CHECK(SCIPincludeObjBranchrule(scip, new AssignmentBranchrule(scip, &master), TRUE));
    // restricted master heuristic: up to 5% of the time limit per run, at most
    // once every 20% of it during the column generation of the root
    double budget = time_limit > 0 ? time_limit : 600;
    RestrictedMasterHeuristic* heuristic = nullptr;
    if (options.restricted_master_heuristic) {
        heuristic = new RestrictedMasterHeuristic(scip, &master, max(1.0, 0.05 * budget), 0.2 * budget);
        SCIP_CHECK(SCIPincludeObjHeur(scip, heuristic, TRUE));
    }
    // re-fixing: at most a quarter of the time limit in the subgradient method
    unique_ptr<fixing::Refixer> refixer;
    PendingSolutions* pending_solutions = nullptr;
    if (options.refixing) {
        refixer.reset(new fixing::Refixer(inst, 0.25 * budget, options.bundle_root));
        master.refixer = refixer.get();
        SCIP_CHECK(SCIPincludeObjProp(scip, new RefixingColumns(scip, &master), TRUE));
        pending_solutions = new PendingSolutions(scip, &master);
        SCIP_CHECK(SCIPincludeObjHeur(scip, pending_solutions, TRUE));
    }
    // The master must stay as it is: no presolving, no cutting planes (their duals
    // would have to enter the pricing), no restarts and no dual reductions.
    SCIP_CHECK(SCIPsetPresolving(scip, SCIP_PARAMSETTING_OFF, TRUE));
    SCIP_CHECK(SCIPsetSeparating(scip, SCIP_PARAMSETTING_OFF, TRUE));
    SCIP_CHECK(SCIPsetIntParam(scip, "presolving/maxrestarts", 0));
    SCIP_CHECK(SCIPsetBoolParam(scip, "misc/allowstrongdualreds", FALSE));
    SCIP_CHECK(SCIPsetBoolParam(scip, "misc/allowweakdualreds", FALSE));
    SCIP_CHECK(SCIPsetBoolParam(scip, "conflict/enable", FALSE));
    SCIP_CHECK(SCIPsetIntParam(scip, "misc/usesymmetry", 0));

    SCIP_CHECK(SCIPcreateProbBasic(scip, "gap_master"));
    SCIP_CHECK(SCIPsetObjsense(scip, SCIP_OBJSENSE_MINIMIZE));
    if (master.integral) SCIP_CHECK(SCIPsetObjIntegral(scip));
    if (options.objective_limit < 1e299) SCIP_CHECK(SCIPsetObjlimit(scip, options.objective_limit));

    // Constraints, modifiable so that the pricer can add columns, and not
    // propagated: only the branching decisions fix columns.
    vector<SCIP_CONS*> assignment(n), convexity(m);
    vector<SCIP_VAR*> unassigned(n);
    for (int j = 0; j < n; j++) {
        string name = "u_" + to_string(j);
        SCIP_CHECK(SCIPcreateVarBasic(scip, &unassigned[j], name.c_str(), 0.0, 1.0, inst.penalty,
                                      SCIP_VARTYPE_BINARY));
        SCIP_CHECK(SCIPchgVarUbLazy(scip, unassigned[j], 1.0));   // implied by the assignment constraint
        SCIP_CHECK(SCIPaddVar(scip, unassigned[j]));
        string cname = "assign_" + to_string(j);
        SCIP_CHECK(SCIPcreateConsBasicLinear(scip, &assignment[j], cname.c_str(), 0, nullptr, nullptr, 1.0, 1.0));
        SCIP_CHECK(SCIPaddCoefLinear(scip, assignment[j], unassigned[j], 1.0));
        SCIP_CHECK(SCIPsetConsModifiable(scip, assignment[j], TRUE));
        SCIP_CHECK(SCIPsetConsPropagated(scip, assignment[j], FALSE));
        SCIP_CHECK(SCIPaddCons(scip, assignment[j]));
    }
    for (int i = 0; i < m; i++) {
        string cname = "convexity_" + to_string(i);
        SCIP_CHECK(SCIPcreateConsBasicLinear(scip, &convexity[i], cname.c_str(), 0, nullptr, nullptr,
                                             -SCIPinfinity(scip), 1.0));
        SCIP_CHECK(SCIPsetConsModifiable(scip, convexity[i], TRUE));
        SCIP_CHECK(SCIPsetConsPropagated(scip, convexity[i], FALSE));
        SCIP_CHECK(SCIPaddCons(scip, convexity[i]));
    }
    // Subgradient method: its multipliers are the initial center of the
    // stabilization of the root, the knapsack solutions of its last iterations are
    // initial columns close to those of the optimal master LP, and its Lagrangian
    // heuristic may improve the initial solution.
    vector<pair<int, vector<int>>> lagrangian_columns;
    lagrangian::Result lr = lagrangian::subgradient(inst, initial.total_cost, 5000, max(1.0, 0.1 * time_limit),
                                                    50, &lagrangian_columns, 100, 10);
    if (lr.has_solution && lr.solution.total_cost < initial.total_cost) initial = lr.solution;
    // With bundle_root, the bundle method continues from the multipliers of the
    // subgradient method to the optimal duals of the master: they become the
    // center, and the columns with positive weight in its last quadratic program
    // (an approximate optimal master) initial columns.
    vector<pair<int, vector<int>>> bundle_columns;
    if (options.bundle_root) {
        column_generation::Result b = column_generation::bundle(
            inst, max(1.0, 0.2 * budget), lagrangian_columns, lr.multipliers.empty() ? nullptr : &lr.multipliers);
        if (!b.pi.empty() && b.bound > lr.bound) {
            lr.multipliers = b.pi;
            lr.bound = b.bound;
        }
        bundle_columns = b.active;
        master.root_bound = max(master.root_bound, b.bound);
    }
    if (!lr.multipliers.empty()) {
        vector<double> mu(m);
        vector<double> profit(n);
        vector<int> weight(n), chosen;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                profit[j] = lr.multipliers[j] - inst.cost[j][i];
                weight[j] = inst.demand(j, i);
            }
            mu[i] = -knapsack::solve(profit, weight, inst.warehouses[i].capacity, chosen);
        }
        pricer->set_root_center(lr.multipliers, mu, lr.bound);
    }

    // Initial columns: the warehouses of the initial solution, which must come
    // first (they make up the initial solution), and the knapsack solutions of the
    // subgradient method.
    vector<pair<int, vector<int>>> seeds;
    for (int i = 0; i < m; i++) {
        vector<int> sellers = initial.sellers_per_warehouse[i];
        if (sellers.empty()) continue;
        sort(sellers.begin(), sellers.end());
        seeds.push_back({i, sellers});
    }
    int n_initial_solution_columns = seeds.size();
    seeds.insert(seeds.end(), lagrangian_columns.begin(), lagrangian_columns.end());
    seeds.insert(seeds.end(), bundle_columns.begin(), bundle_columns.end());
    if (options.initial_columns != nullptr)
        seeds.insert(seeds.end(), options.initial_columns->begin(), options.initial_columns->end());

    vector<pair<int, SCIP_VAR*>> initial_columns;
    for (const auto& [i, sellers] : seeds) {
        if (master.known.count({i, sellers})) continue;
        double cost = 0;
        for (int j : sellers) cost += inst.cost[j][i];
        string name = "init_" + to_string(master.columns.size());
        SCIP_VAR* var;
        SCIP_CHECK(SCIPcreateVar(scip, &var, name.c_str(), 0.0, 1.0, cost, SCIP_VARTYPE_BINARY, TRUE,
                                 options.column_aging, nullptr, nullptr, nullptr, nullptr, nullptr));
        SCIP_CHECK(SCIPchgVarUbLazy(scip, var, 1.0));
        SCIP_CHECK(SCIPaddVar(scip, var));
        for (int j : sellers) SCIP_CHECK(SCIPaddCoefLinear(scip, assignment[j], var, 1.0));
        SCIP_CHECK(SCIPaddCoefLinear(scip, convexity[i], var, 1.0));
        master.known.emplace(make_pair(i, sellers), (int)master.columns.size());
        master.columns.push_back({i, sellers, nullptr});
        initial_columns.push_back({i, var});
    }
    SCIP_CHECK(SCIPactivatePricer(scip, SCIPfindPricer(scip, "knapsack")));

    // From here on everything works on the transformed problem.
    SCIP_CHECK(SCIPtransformProb(scip));
    master.assignment.resize(n);
    master.convexity.resize(m);
    master.unassigned.resize(n);
    for (int j = 0; j < n; j++) {
        SCIP_CHECK(SCIPgetTransformedCons(scip, assignment[j], &master.assignment[j]));
        SCIP_CHECK(SCIPgetTransformedVar(scip, unassigned[j], &master.unassigned[j]));
    }
    for (int i = 0; i < m; i++) SCIP_CHECK(SCIPgetTransformedCons(scip, convexity[i], &master.convexity[i]));
    for (int k = 0; k < (int)initial_columns.size(); k++) {
        SCIP_CHECK(SCIPgetTransformedVar(scip, initial_columns[k].second, &master.columns[k].var));
    }

    // Initial solution.
    SCIP_SOL* sol;
    SCIP_CHECK(SCIPcreateSol(scip, &sol, nullptr));
    for (int k = 0; k < n_initial_solution_columns; k++) {
        SCIP_CHECK(SCIPsetSolVal(scip, sol, master.columns[k].var, 1.0));
    }
    for (int j = 0; j < n; j++) {
        if (initial.assigned_warehouse[j] < 0) SCIP_CHECK(SCIPsetSolVal(scip, sol, master.unassigned[j], 1.0));
    }
    SCIP_Bool stored;
    SCIP_CHECK(SCIPaddSolFree(scip, &sol, &stored));

    // The time limit counts the time already spent building the master.
    double spent = chrono::duration_cast<chrono::milliseconds>(
        chrono::high_resolution_clock::now() - t0).count() / 1000.0;
    if (time_limit > 0) SCIP_CHECK(SCIPsetRealParam(scip, "limits/time", max(1.0, time_limit - spent)));
    SCIP_CHECK(SCIPsolve(scip));

    Result result;
    result.dual = SCIPgetDualbound(scip);
    // SCIP reports an infinite root bound when the root is pruned, so the bound
    // computed by the pricer at the root is reported instead.
    result.root_bound = master.root_bound > -1e299 ? master.root_bound : SCIPgetDualboundRoot(scip);
    // with an objective limit, an infeasible status means that nothing is cheaper
    result.no_better_solution = SCIPgetStatus(scip) == SCIP_STATUS_INFEASIBLE;
    result.optimal = SCIPgetStatus(scip) == SCIP_STATUS_OPTIMAL || result.no_better_solution;
    result.nodes = SCIPgetNNodes(scip);
    result.columns = master.columns.size();
    if (heuristic != nullptr) {
        result.heuristic_runs = heuristic->runs;
        result.heuristic_improvements = heuristic->improvements;
    }
    result.early_stops = master.early_stops;
    result.mispricings = master.mispricings;
    result.lp_iterations = SCIPgetNLPIterations(scip);
    SCIP_SOL* best = SCIPgetBestSol(scip);
    if (best != nullptr) {
        result.primal = SCIPgetSolOrigObj(scip, best);
        Solution& s = result.solution;
        s.sellers_per_warehouse.assign(m, {});
        s.assigned_warehouse.assign(n, -1);
        for (const Column& col : master.columns) {
            if (SCIPgetSolVal(scip, best, col.var) > 0.5) {
                for (int j : col.sellers) s.assign(col.warehouse, j);
            }
        }
        s.total_cost = metaheuristics::evaluate(s, inst);
        result.has_solution = true;
    }
    if (refixer != nullptr) {
        if (master.lagrangian_best.total_cost < result.primal - 1e-9) {
            result.solution = master.lagrangian_best;
            result.primal = master.lagrangian_best.total_cost;
            result.has_solution = true;
        }
        result.pairs = refixer->reduction().pairs;
        result.fixed_to_zero = refixer->reduction().fixed_to_zero;
        result.fixed_to_one = refixer->reduction().fixed_to_one;
        result.first_fixed_pct = refixer->first_fixed_pct;
        result.refixing_incumbents = refixer->incumbents;
        result.lagrangian_solutions = pending_solutions->found;
        result.fixing_seconds = refixer->reduction().seconds;
        if (result.has_solution) {
            if (refixer->no_improving_solution()) result.optimal = true;
            if (!result.optimal) result.dual = max(result.dual, min(result.primal, refixer->reduction().bound));
            if (master.integral) result.dual = min(result.primal, ceil(result.dual - 1e-6));
            if (result.dual >= result.primal - 1e-9) result.optimal = true;
            if (result.optimal) result.dual = result.primal;
        }
        master.refixer = nullptr;
    }
    if (master.duplicates > 0 || master.fallback_branchings > 0) {
        fprintf(stderr, "branch_and_price: %d duplicate columns, %d fallback branchings\n",
                master.duplicates, master.fallback_branchings);
    }

    for (auto& [i, var] : initial_columns) SCIP_CHECK(SCIPreleaseVar(scip, &var));
    for (int j = 0; j < n; j++) {
        SCIP_CHECK(SCIPreleaseVar(scip, &unassigned[j]));
        SCIP_CHECK(SCIPreleaseCons(scip, &assignment[j]));
    }
    for (int i = 0; i < m; i++) SCIP_CHECK(SCIPreleaseCons(scip, &convexity[i]));
    SCIP_CHECK(SCIPfree(&scip));
    result.seconds = chrono::duration_cast<chrono::milliseconds>(
        chrono::high_resolution_clock::now() - t0).count() / 1000.0;
    return result;
}

Result solve_with_fixing(const Instance& inst, double time_limit, const Solution& start, const Options& options,
                         fixing::Reduction* reduction_out) {
    auto t0 = chrono::high_resolution_clock::now();
    Solution incumbent = start;
    incumbent.total_cost = metaheuristics::evaluate(start, inst);
    fixing::Reduction reduction = fixing::reduce(inst, incumbent.total_cost, 0.25 * time_limit, 3, &incumbent, true,
                                                 nullptr, options.bundle_root);
    double spent = chrono::duration_cast<chrono::milliseconds>(
        chrono::high_resolution_clock::now() - t0).count() / 1000.0;

    Result result;
    bool solved = reduction.no_improving_solution || reduction.instance.sellers.empty();
    if (!reduction.no_improving_solution && reduction.instance.sellers.empty()) {
        // every seller is fixed: the forced assignment is the only cheaper candidate
        Solution forced = fixing::expand(inst, reduction, Solution());
        if (forced.total_cost < incumbent.total_cost) incumbent = forced;
    }
    if (!solved) {
        Options reduced_options = options;
        reduced_options.objective_limit = min(options.objective_limit, incumbent.total_cost - reduction.offset);
        Result r = solve(reduction.instance, time_limit > 0 ? max(1.0, time_limit - spent) : 0, nullptr,
                         reduced_options);
        if (r.has_solution) {
            Solution candidate = fixing::expand(inst, reduction, r.solution);
            if (candidate.total_cost < incumbent.total_cost) incumbent = candidate;
        }
        solved = r.optimal;
        result.dual = min(incumbent.total_cost, max(r.dual + reduction.offset, reduction.bound));
        result.root_bound = min(incumbent.total_cost, max(r.root_bound + reduction.offset, reduction.bound));
        result.nodes = r.nodes;
        result.columns = r.columns;
        result.heuristic_runs = r.heuristic_runs;
        result.heuristic_improvements = r.heuristic_improvements;
        result.early_stops = r.early_stops;
        result.mispricings = r.mispricings;
        result.lp_iterations = r.lp_iterations;
    }
    result.solution = incumbent;
    result.has_solution = true;
    result.primal = incumbent.total_cost;
    if (solved) {
        result.dual = result.primal;
        result.root_bound = max(result.root_bound, min(result.primal, reduction.bound));
    }
    result.optimal = solved;
    result.seconds = chrono::duration_cast<chrono::milliseconds>(
        chrono::high_resolution_clock::now() - t0).count() / 1000.0;
    if (reduction_out != nullptr) *reduction_out = reduction;
    return result;
}

}

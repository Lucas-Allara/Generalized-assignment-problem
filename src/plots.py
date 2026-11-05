#!/usr/bin/env python3
# =============================================================================
#  plots.py - draws the figures of the report from the CSV files in results/
#  and the JSON file of the real instance. Output: figs/*.png (included with
#  \includegraphics in the report).
#
#  USAGE:  python plots.py        (from src/ or anywhere else)
#  REQUIRES: matplotlib, numpy.
#
#  Each figure is drawn independently: if a CSV file is missing, that figure is
#  skipped with a warning and the others are still drawn.
# =============================================================================

import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.ticker
from matplotlib.collections import LineCollection
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "..", "figs")
JSON = os.path.join(HERE, "instances", "real", "real_instance.json")
SOLUTION = os.path.join(HERE, "competition_solution.txt")  # one line per warehouse
os.makedirs(FIGS, exist_ok=True)

# dimensions (m warehouses, n sellers) of an instance, from its name (group letter,
# two digits for m, the rest for n)
def dims(name):
    if name == "real":
        return 310, 1100
    return int(name[1:3]), int(name[3:])


GROUPS = ["a", "b", "c", "d", "e", "r"]
GROUP_LABELS = {"a": "Group a", "b": "Group b", "c": "Group c", "d": "Group d", "e": "Group e", "r": "Real"}

# One color per method in every figure (validated against color vision
# deficiencies for the combinations that appear together).
COLORS = {
    "Constructive": "#e87ba4",
    "VND": "#6250d6",
    "ILS": "#eb6834",
    "LNS": "#2a78d6",
    "LNS from the Lagrangian heuristic": "#eda100",
    "MIP-LNS": "#1baf7a",
}
EDGE = "#333333"


def read_csv(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name, dpi=150):
    out = os.path.join(FIGS, name)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  ok ->", os.path.relpath(out, HERE))


# ---------------------------------------------------------------------------
#  fig:gap  -  mean gap (%) of each method, by group a/b/e + real
# ---------------------------------------------------------------------------
def fig_gap():
    rows = read_csv("e5_final.csv")
    methods = [("Constructive", "gap_constructive_pct"), ("VND", "gap_vnd_pct"),
               ("ILS", "gap_ils_pct"), ("LNS", "gap_lns_pct")]
    methods = [(m, c) for (m, c) in methods if c in rows[0]]
    groups = [g for g in GROUPS if any(r["group"] == g for r in rows)]

    def mean(group, column):
        values = [float(r[column]) for r in rows if r["group"] == group]
        return sum(values) / len(values) if values else 0.0

    data = {m: [mean(g, c) for g in groups] for (m, c) in methods}

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(groups))
    w = 0.8 / len(methods)
    for i, (m, _) in enumerate(methods):
        ax.bar(x + i * w - 0.4 + w / 2, data[m], w, label=m, color=COLORS.get(m), edgecolor="white", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABELS[g] for g in groups])
    ax.set_ylabel("Mean gap (%) to the best found")
    ax.set_title("Quality of each method (gap to the best value of each instance)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_gap_method.png")


# ---------------------------------------------------------------------------
#  fig:time  -  time of the VND against the size n*m (log-log)
# ---------------------------------------------------------------------------
def fig_time():
    rows = [r for r in read_csv("e3_vnd.csv") if r["method"] == "full_vnd"]
    points = []
    for r in rows:
        m, n = dims(r["instance"])
        points.append((r["group"], m * n, max(float(r["time_s"]), 1e-4)))   # floor for the log scale
    groups = [g for g in GROUPS if any(p[0] == g for p in points)]
    # one panel per group: the group in color over all the instances in gray
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey=True)
    for ax, g in zip(axes.flat, groups):
        ax.scatter([p[1] for p in points], [p[2] for p in points], s=14, c="#c8c8c8", zorder=1)
        mine = [p for p in points if p[0] == g]
        ax.scatter([p[1] for p in mine], [p[2] for p in mine], s=36, c=COLORS["VND"], edgecolors=EDGE,
                   linewidths=0.4, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(GROUP_LABELS[g])
        ax.grid(True, which="both", alpha=0.3)
    for ax in axes.flat[len(groups):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(r"Size $n\cdot m$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Time of the VND (s)")
    fig.suptitle("Scalability of the local search (full VND from the best constructive solution)")
    save(fig, "fig_time_size.png")


# ---------------------------------------------------------------------------
#  fig:conv  -  incumbent cost against time (ILS and LNS) on e20200 and real
# ---------------------------------------------------------------------------
def fig_convergence():
    rows = read_csv("convergence.csv")
    names = []
    for r in rows:
        if r["instance"] not in names:
            names.append(r["instance"])

    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        for method in ("ILS", "LNS"):
            points = sorted((float(r["seconds"]), float(r["cost"])) for r in rows
                            if r["instance"] == name and r["method"] == method)
            if points:
                xs, ys = zip(*points)
                ax.plot(xs, ys, marker="o", label=method, color=COLORS[method])
        ax.set_title(f"Convergence — {name}")
        ax.set_xlabel("Time budget (s)")
        ax.set_ylabel("Incumbent cost")
        ax.legend()
        ax.grid(alpha=0.3)
    save(fig, "fig_convergence.png")


# ---------------------------------------------------------------------------
#  fig:map  -  geographic layout of the real instance (sellers and warehouses)
# ---------------------------------------------------------------------------
def fig_map():
    if not os.path.exists(JSON):
        raise FileNotFoundError(JSON)
    with open(JSON) as f:
        inst = json.load(f)["instance"]
    sellers = np.array(inst["client_position"])
    warehouses = np.array(inst["depot_position"])
    fig, ax = plt.subplots(figsize=(6.2, 6))

    # seller-warehouse lines of the competition solution: line i (0-indexed)
    # lists the ids of the sellers assigned to warehouse i. The indices match the
    # arrays of the JSON file. They are drawn below the points.
    segments = []
    if os.path.exists(SOLUTION):
        with open(SOLUTION) as f:
            for i, line in enumerate(f):
                if i >= len(warehouses):
                    break
                for token in line.split():
                    j = int(token)
                    if 0 <= j < len(sellers):
                        segments.append([sellers[j], warehouses[i]])
        if segments:
            ax.add_collection(LineCollection(
                segments, colors="black", linewidths=0.3, alpha=0.4, zorder=1))
            ax.plot([], [], color="black", lw=0.8, label=f"Assignments ({len(segments)})")
    else:
        print("  -- fig:map: competition_solution.txt is missing, map without assignment lines")

    ax.scatter(sellers[:, 0], sellers[:, 1], s=6, c="#d62728", alpha=0.5, zorder=2,
               label=f"Sellers (n={len(sellers)})")
    ax.scatter(warehouses[:, 0], warehouses[:, 1], s=26, c="#1f77b4", edgecolors="k",
               linewidths=0.3, zorder=3, label=f"Warehouses (m={len(warehouses)})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Solution of the real instance")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right")
    save(fig, "fig_map_real.png", dpi=600)


# ---------------------------------------------------------------------------
#  extra  -  improvement (%) of each local search operator (relocate/swap x first/best)
# ---------------------------------------------------------------------------
def fig_ls_operators():
    rows = read_csv("e2_local_search.csv")
    ops = ["relocate_first", "relocate_best", "swap_first", "swap_best"]
    improvements = {o: [] for o in ops}
    for r in rows:
        if r["operator"] in improvements:
            improvements[r["operator"]].append(float(r["improvement_pct"]))
    mean = [np.mean(improvements[o]) if improvements[o] else 0 for o in ops]
    largest = [np.max(improvements[o]) if improvements[o] else 0 for o in ops]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(ops))
    ax.bar(x - 0.2, mean, 0.4, label="Mean improvement", color=COLORS["LNS"], edgecolor="white", linewidth=1)
    ax.bar(x + 0.2, largest, 0.4, label="Largest improvement", color="#a9c9f0", edgecolor="white", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["relocate\nfirst", "relocate\nbest", "swap\nfirst", "swap\nbest"])
    ax.set_ylabel("Improvement over the constructive solution (%)")
    ax.set_title("Contribution of each local search operator")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_ls_operators.png")


# ---------------------------------------------------------------------------
#  extra  -  ILS against LNS on the tuning instances (mean cost, normalized)
# ---------------------------------------------------------------------------
def fig_tuning():
    rows = read_csv("e4_tuning.csv")
    names = []
    for r in rows:
        if r["instance"] not in names:
            names.append(r["instance"])
    # best mean cost of each (instance, metaheuristic)
    best = {}
    for r in rows:
        key = (r["instance"], r["metaheuristic"])
        best[key] = min(best.get(key, math.inf), float(r["mean_cost"]))

    ils = [best.get((i, "ILS"), math.nan) for i in names]
    lns = [best.get((i, "LNS"), math.nan) for i in names]
    # normalized to ILS = 1 to compare them on the same axis
    ils_norm = [1.0 for _ in names]
    lns_norm = [lns[k] / ils[k] if ils[k] else math.nan for k in range(len(names))]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(names))
    ax.bar(x - 0.2, ils_norm, 0.4, label="ILS (best configuration)", color=COLORS["ILS"])
    ax.bar(x + 0.2, lns_norm, 0.4, label="LNS (best configuration)", color=COLORS["LNS"])
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Mean cost relative to the ILS")
    ax.set_title("Tuning: ILS against LNS (lower is better)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_tuning.png")


# ---------------------------------------------------------------------------
#  fig:true_gaps  -  gap (%) of each metaheuristic against the best lower bound
#  (E10). The colors are a palette checked for color vision deficiencies, and
#  each method also has its own marker shape.
# ---------------------------------------------------------------------------
def fig_true_gaps():
    rows = read_csv("e10_true_gaps.csv")
    series = [("LNS", "gap_lns_mean_pct", COLORS["LNS"], "o"),
              ("LNS from the Lagrangian heuristic", "gap_lns_lagrangian_mean_pct",
               COLORS["LNS from the Lagrangian heuristic"], "s"),
              ("MIP-LNS", "gap_mip_lns_mean_pct", COLORS["MIP-LNS"], "D")]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    # each method is shifted horizontally so that equal values do not hide each other
    for k, (label, column, color, marker) in enumerate(series):
        ax.scatter(x + (k - 1) * 0.22, [float(r[column]) for r in rows], s=42, c=color, marker=marker,
                   edgecolors="#333333", linewidths=0.5, zorder=3, label=label)
    # separators between the groups a, b, e and the real instance
    for k in range(1, len(rows)):
        if rows[k]["group"] != rows[k - 1]["group"]:
            ax.axvline(k - 0.5, color="#999999", lw=0.8, zorder=1)
    ax.set_yscale("symlog", linthresh=0.1)
    largest = max(float(r[c]) for r in rows for (_, c, _, _) in series)
    ax.set_ylim(0, 10 ** math.ceil(math.log10(max(largest, 1.0)) + 0.3))
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_xticks(x)
    # an asterisk marks the instances whose optimum is not proven: their gap is
    # measured against a lower bound and overestimates the gap to the optimum
    ax.set_xticklabels([r["instance"] + ("" if r["proven_optimal"] == "1" else "*")
                        for r in rows], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Gap to the best lower bound (%)")
    ax.set_title("Mean gap over 5 seeds with the same time budget (* optimum not proven)")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="upper left")
    save(fig, "fig_true_gaps.png")


# ---------------------------------------------------------------------------
#  fig:cuts  -  share of the gap between the LP bound and the master LP bound
#  (column generation) closed by the cover inequalities and by the root of SCIP
# ---------------------------------------------------------------------------
def fig_cuts():
    # Reference: the bound of the bundle method (E14), the value of the master LP
    # where it converged (57 of the 58 instances); without E14, the reference of
    # E11 (the larger of the Lagrangian bound and the exact separation, which is
    # below the master where the separation does not converge).
    master = {}
    if os.path.exists(os.path.join(RESULTS, "e14_master.csv")):
        master = {r["instance"]: float(r["bound"]) for r in read_csv("e14_master.csv") if r["solver"] == "bundle"}
    rows = []
    for r in read_csv("e11_cuts.csv"):
        reference = master.get(r["instance"], float(r["master_bound"]))
        if reference - float(r["lp_bound"]) > 1e-6:
            rows.append((r, reference))

    def closed(r, reference, column):
        lp = float(r["lp_bound"])
        # the bounds of the CSV files have two decimals: cap the rounding at 100%
        return min(100.0, (float(r[column]) - lp) / (reference - lp) * 100.0)

    series = [("Cover inequalities (ours)", "cover_bound", "#2a78d6", "o"),
              ("Exact knapsack separation (ours)", "exact_bound", "#1baf7a", "D"),
              ("Root of SCIP (all its cutting planes)", "scip_root_bound", "#eb6834", "s")]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(14, 4.4))
    for k, (label, column, color, marker) in enumerate(series):
        ax.scatter(x + (k - 1) * 0.22, [closed(r, ref, column) for r, ref in rows], s=42, c=color, marker=marker,
                   edgecolors="#333333", linewidths=0.5, zorder=3, label=label)
    rows = [r for r, _ in rows]
    ax.axhline(100, color="#333333", lw=0.8, ls="--", zorder=2)
    for k in range(1, len(rows)):
        if rows[k]["group"] != rows[k - 1]["group"]:
            ax.axvline(k - 0.5, color="#999999", lw=0.8, zorder=1)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([r["instance"] for r in rows], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Gap from the LP to the master closed (%)")
    ax.set_title("Rows against columns: 100% is the bound of the column generation")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="lower right")
    save(fig, "fig_cuts.png")


# ---------------------------------------------------------------------------
#  fig:sensitivity  -  real instance: value of 10 more units of capacity in the
#  warehouses with the largest shadow prices (LP estimate against the exact
#  re-solve) and cost of closing each of the most loaded warehouses
# ---------------------------------------------------------------------------
def fig_sensitivity():
    rows = read_csv("e15_sensitivity.csv")
    plus = [r for r in rows if r["scenario"] == "capacity_plus_10"]
    closed = [r for r in rows if r["scenario"] == "closure"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    x = np.arange(len(plus))
    ax.bar(x - 0.2, [-float(r["lp_estimated_change"]) for r in plus], 0.4, label="LP estimate (10 x shadow price)",
           color="#2a78d6", edgecolor="white", linewidth=1)
    ax.bar(x + 0.2, [-float(r["change"]) for r in plus], 0.4, label="Exact (re-solved)",
           color="#eb6834", edgecolor="white", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([r["warehouse"] for r in plus], rotation=60, fontsize=8)
    ax.set_xlabel("Warehouse (largest shadow prices)")
    ax.set_ylabel("Cost saved")
    ax.set_title("10 more units of capacity")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    x = np.arange(len(closed))
    ax.bar(x, [float(r["change"]) for r in closed], 0.6, color="#2a78d6", edgecolor="white", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([r["warehouse"] for r in closed], rotation=60, fontsize=8)
    ax.set_xlabel("Warehouse (most loaded in the optimum)")
    ax.set_ylabel("Cost increase")
    ax.set_title("Closing the warehouse")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_sensitivity.png")


# ---------------------------------------------------------------------------
#  fig:ablation  -  branch-and-price with each technique switched off: instances
#  solved within the limit and shifted geometric mean of the time
# ---------------------------------------------------------------------------
def fig_ablation():
    rows = read_csv("e13_ablation.csv")
    variants = []
    for r in rows:
        if r["variant"] not in variants:
            variants.append(r["variant"])
    solved, mean_time = [], []
    for v in variants:
        rs = [r for r in rows if r["variant"] == v]
        solved.append(sum(r["optimal"] == "1" for r in rs))
        # shifted geometric mean (shift 10 s), unsolved runs counted at their time
        logs = [math.log(float(r["time_s"]) + 10) for r in rs]
        mean_time.append(math.exp(sum(logs) / len(logs)) - 10)
    labels = [v.replace("no_", "no ").replace("_", " ") for v in variants]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    y = np.arange(len(variants))
    axes[0].barh(y, solved, color="#2a78d6", edgecolor="white", linewidth=1)
    axes[0].set_xlabel(f"Instances solved within the limit (of {len(rows) // len(variants)})")
    axes[1].barh(y, mean_time, color="#2a78d6", edgecolor="white", linewidth=1)
    axes[1].set_xlabel("Shifted geometric mean of the time (s)")
    for ax in axes:
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Branch-and-price: what each technique contributes")
    save(fig, "fig_ablation.png")


# ---------------------------------------------------------------------------
#  fig:refixing  -  E16: the fixing of E7 against a better start (MIP-LNS) and
#  the re-fixing during the search, for the compact model and the
#  branch-and-price: instances solved over time and gap at the time limit
# ---------------------------------------------------------------------------
def fig_refixing():
    rows = read_csv("e16_refixing.csv")
    # best lower bound of each instance: E16 itself and, if present, the table of gaps
    lower = {}
    for r in rows:
        lower[r["instance"]] = max(lower.get(r["instance"], -1e300), float(r["bound"]))
    try:
        for r in read_csv("e10_true_gaps.csv"):
            if r["instance"] in lower:
                lower[r["instance"]] = max(lower[r["instance"]], float(r["lower_bound"]))
    except FileNotFoundError:
        pass
    instances = sorted(lower)
    one_shot, refixing = "#2a78d6", "#eb6834"
    variants = [  # (variant, label, color, line style)
        ("fix", "Fixing once, start of E7", one_shot, "--"),
        ("refix", "Re-fixing, start of E7", refixing, "--"),
        ("lns_fix", "Fixing once, MIP-LNS start", one_shot, "-"),
        ("lns_refix", "Re-fixing, MIP-LNS start", refixing, "-"),
        ("lns_bnp_fix", "Fixing once, MIP-LNS start", one_shot, "-"),
        ("lns_bnp_refix", "Re-fixing, MIP-LNS start", refixing, "-"),
    ]
    limit = max(float(r["time_s"]) + float(r["start_time_s"]) for r in rows)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    gaps = []
    most = 1
    for variant, label, color, style in variants:
        rs = [r for r in rows if r["variant"] == variant]
        if not rs:
            continue
        times = sorted(float(r["time_s"]) + float(r["start_time_s"]) for r in rs if r["optimal"] == "1")
        ax = axes[0] if not variant.startswith("lns_bnp") else axes[1]
        xs = [0] + [t for t in times for _ in (0, 1)] + [limit]
        ys = [0] + [k + d for k in range(len(times)) for d in (0, 1)] + [len(times)]
        ax.plot(xs, ys, color=color, ls=style, lw=2, label=f"{label} ({len(times)})")
        most = max(most, len(times))
        g = [max(0.0, (float(r["best"]) - lower[r["instance"]]) / lower[r["instance"]] * 100) for r in rs]
        gaps.append((("B&P: " if variant.startswith("lns_bnp") else "MIP: ") + label, sum(g) / len(g), color))
    for ax, title in zip(axes[:2], ["Compact model (SCIP)", "Branch-and-price"]):
        ax.set_title(title)
        ax.set_xlabel("Time, including the start (s)")
        ax.set_ylabel(f"Instances solved to optimality (of {len(instances)})")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, most + 1.5)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    y = np.arange(len(gaps))
    axes[2].barh(y, [g[1] for g in gaps], color=[g[2] for g in gaps], edgecolor="white", linewidth=2)
    for k, g in enumerate(gaps):
        axes[2].text(g[1], k, f" {g[1]:.3f}%", va="center", fontsize=8, color="#333333")
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([g[0] for g in gaps], fontsize=8)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Mean gap at the time limit against the best lower bound (%)")
    axes[2].grid(axis="x", alpha=0.3)
    axes[2].set_xlim(0, max(g[1] for g in gaps) * 1.25 + 1e-9)
    fig.suptitle(f"E16: better start and re-fixing ({len(instances)} instances not closed by the fixing of E7 "
                 "within 100 s)")
    fig.tight_layout()
    save(fig, "fig_refixing.png")


# ---------------------------------------------------------------------------
#  fig:master  -  E14: master of the root with the simplex method, the interior
#  point method (HiGHS) and the bundle method: time to converge on the 8
#  instances of the LP masters, and the time of the bundle method on all of them
# ---------------------------------------------------------------------------
def fig_master():
    rows = read_csv("e14_master.csv")
    lp_instances = []
    for r in rows:
        if r["solver"] == "simplex":
            lp_instances.append(r["instance"])
    solvers = [("simplex", "Simplex (HiGHS)", "#2a78d6"), ("interior_point", "Interior point (HiGHS)", "#eda100"),
               ("bundle", "Bundle method (ours)", "#1baf7a")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    width = 0.27
    x = np.arange(len(lp_instances))
    limit = 300
    for k, (solver, label, color) in enumerate(solvers):
        times, open_ = [], []
        for name in lp_instances:
            r = next(r for r in rows if r["instance"] == name and r["solver"] == solver)
            times.append(max(0.01, float(r["time_s"])))
            open_.append(r["converged"] != "1")
        bars = ax.bar(x + (k - 1) * width, times, width, color=color, edgecolor="white", linewidth=1, label=label)
        for bar, unconverged in zip(bars, open_):
            if unconverged:
                bar.set_hatch("///")
                bar.set_edgecolor("#333333")
    ax.axhline(limit, color="#333333", lw=0.8, ls="--")
    ax.text(len(lp_instances) - 0.5, limit * 1.15, "time limit (hatched: not converged)", ha="right", fontsize=8,
            color="#333333")
    ax.set_yscale("log")
    ax.set_ylim(0.01, limit * 40)
    ax.set_xticks(x)
    ax.set_xticklabels(lp_instances, rotation=30, ha="right")
    ax.set_ylabel("Time to converge (s)")
    ax.set_title("Root master: LP solvers against the bundle method")
    ax.grid(axis="y", alpha=0.3)
    # plain swatches: the first bar of a solver may be hatched
    handles = [matplotlib.patches.Patch(facecolor=color, label=label) for _, label, color in solvers]
    ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=3)
    ax = axes[1]
    bundle = [r for r in rows if r["solver"] == "bundle"]
    for g_index, g in enumerate(GROUPS):
        rs = [r for r in bundle if r["group"] == g]
        if not rs:
            continue
        sizes = [dims(r["instance"])[1] for r in rs]
        times = [max(0.001, float(r["time_s"])) for r in rs]
        jitter = [g_index + 0.12 * ((k % 5) - 2) for k in range(len(rs))]
        ax.scatter(jitter, times, s=[18 + s / 40 for s in sizes], c="#1baf7a", edgecolors="#333333", linewidths=0.5,
                   zorder=3)
        for xk, tk, r in zip(jitter, times, rs):
            if r["converged"] != "1":
                ax.annotate(f"{r['instance']}: stopped at the {tk:.0f} s limit", (xk, tk), xytext=(12, -3),
                            textcoords="offset points", fontsize=8, color="#333333", va="center")
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUPS])
    ax.set_xlim(-0.6, len(GROUPS) - 0.4)
    ax.set_yscale("log")
    ax.set_ylabel("Time of the bundle method (s; below 1 ms drawn at 1 ms)")
    ax.set_title(f"Bundle method on all {len(bundle)} instances (marker size: sellers); "
                 f"{sum(r['converged'] == '1' for r in bundle)} converged")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig_master.png")


# ---------------------------------------------------------------------------
#  fig:open  -  E17 on the instances still open: gap of four heuristics with the
#  same time, and the exact methods started from the portfolio (palette checked
#  with the validator: its closest pair under deuteranopia, green and pink, is
#  told apart by the marker shapes)
# ---------------------------------------------------------------------------
def fig_open():
    rows = read_csv("e17_open.csv")
    lower = {}
    for r in read_csv("e10_true_gaps.csv"):
        lower[r["instance"]] = float(r["lower_bound"])
    names = []
    for r in rows:
        if r["instance"] not in names:
            names.append(r["instance"])
    heuristics = [("mip_lns", "MIP-LNS (Lagrangian start)", "#2a78d6", "o"),
                  ("tabu", "Tabu search with ejection chains", "#eda100", "s"),
                  ("rounding_mip_lns", "MIP-LNS from the LP rounding", "#e87ba4", "D"),
                  ("portfolio", "Portfolio", "#1baf7a", "^")]
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    x = np.arange(len(names))
    for k, (method, label, color, marker) in enumerate(heuristics):
        ys = []
        for name in names:
            r = next(r for r in rows if r["instance"] == name and r["method"] == method)
            ys.append(max(1e-3, (float(r["best"]) - lower[name]) / lower[name] * 100))
        ax.scatter(x + (k - 1.5) * 0.18, ys, s=36, c=color, marker=marker, edgecolors="#333333", linewidths=0.5,
                   zorder=3, label=label)
    for k in range(1, len(names)):
        if names[k][0] != names[k - 1][0]:
            ax.axvline(k - 0.5, color="#999999", lw=0.8, zorder=1)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Gap to the best lower bound (%)")
    ax.set_title("Heuristics with 120 s on the instances still open")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="upper left", fontsize=8)
    ax = axes[1]
    exact = [("mip_refix", "MIP, re-fixing", "#2a78d6"), ("mip_refix_bundle", "MIP, re-fixing with bundle", "#1baf7a"),
             ("bnp_refix", "B&P, re-fixing", "#e87ba4"), ("bnp_refix_bundle", "B&P, re-fixing, bundle root", "#eda100")]
    labels, solved, gap = [], [], []
    for method, label, color in exact:
        rs = [r for r in rows if r["method"] == method]
        if not rs:
            continue
        labels.append((label, color))
        solved.append(sum(r["optimal"] == "1" for r in rs))
        gap.append(sum(max(0.0, (float(r["best"]) - lower[r["instance"]]) / lower[r["instance"]] * 100) for r in rs)
                   / len(rs))
    y = np.arange(len(labels))
    ax.barh(y, solved, color="#7d8b99", edgecolor="white", linewidth=2)
    for k in range(len(labels)):
        ax.text(solved[k], k, f"  {solved[k]} solved, mean gap {gap[k]:.3f}%", va="center", fontsize=8,
                color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels([l for l, _ in labels], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(solved + [1]) * 2.2)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel(f"Instances solved to optimality in 600 s (of {len(names)})")
    ax.set_title("Exact methods from the portfolio")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig_open.png")


FIGURES = [
    ("fig:gap", fig_gap),
    ("fig:time", fig_time),
    ("fig:conv", fig_convergence),
    ("fig:map", fig_map),
    ("fig_ls_operators", fig_ls_operators),
    ("fig_tuning", fig_tuning),
    ("fig:true_gaps", fig_true_gaps),
    ("fig:cuts", fig_cuts),
    ("fig:sensitivity", fig_sensitivity),
    ("fig:ablation", fig_ablation),
    ("fig:refixing", fig_refixing),
    ("fig:master", fig_master),
    ("fig:open", fig_open),
]

if __name__ == "__main__":
    print("Drawing the figures in", os.path.relpath(FIGS, HERE))
    for name, function in FIGURES:
        try:
            function()
        except FileNotFoundError as e:
            print(f"  -- {name}: missing file ({os.path.basename(str(e))}), skipped")
        except Exception as e:
            print(f"  !! {name}: error: {e}")

# Generalized assignment problem (GAP)

Assignment of sellers to warehouses under capacity constraints for the first-mile logistics of a startup: constructive heuristics, local search, ILS, LNS and a tabu search with ejection chains; Lagrangian and LP relaxations with their heuristics, Lagrangian variable fixing (once, or again after every new incumbent inside SCIP), cover and exact knapsack cuts, a matheuristic, a bundle method for the master, the compact integer model, a branch-and-cut and a branch-and-price in SCIP; on the real instance (with its shadow prices) and the benchmark groups a to e.

```
Generalized-assignment-problem/
├── Report.tex, Report.pdf       the report
├── figs/                        its figures
├── pixi.toml, pixi.lock         the SCIP and HiGHS environment
└── src/
    ├── include/                 the headers
    ├── Instance.cpp             reading of an instance (OR-Library format, demand d_ij)
    ├── ConstructiveHeuristics.cpp   nearest neighbor and the regret heuristic (cost, demand, relative demand)
    ├── LocalSearch.cpp          relocate, swap, their multiple versions, ejection and cyclic exchange
    ├── Metaheuristics.cpp       VND, ILS and LNS
    ├── EjectionChains.cpp       tabu search with ejection chains through infeasible assignments
    ├── Knapsack.cpp             0-1 knapsack by dynamic programming
    ├── LagrangianBound.cpp      Lagrangian bound by subgradient optimization and Lagrangian heuristic
    ├── Bundle.cpp               proximal bundle method for the dual of the master (no LP)
    ├── VariableFixing.cpp       Lagrangian variable fixing (forward and backward knapsack programs) and re-fixing
    ├── MipModel.cpp             compact model (with a re-fixing propagator), LP roundings, shadow prices, cuts
    ├── Matheuristic.cpp         LNS whose repair solves the subproblem with SCIP, and the portfolio start
    ├── BranchAndPrice.cpp       branch-and-price: knapsack pricing, stabilization, strong branching, re-fixing
    ├── ColumnGeneration.cpp     column generation of the root with HiGHS (simplex or interior point)
    ├── main.cpp                 gap_simulator: ILS on one instance
    ├── exact_main.cpp           gap_exact: any of the SCIP methods on one instance
    ├── tests.cpp                tests and experiments E1–E6 of the heuristics
    ├── exact_tests.cpp          tests and experiments E7–E17 of the relaxations and exact methods
    ├── plots.py                 figures of the report from results/
    ├── instances/               groups a to e (OR-Library and Yagiura) and the real instance
    ├── results/                 CSV files of the experiments
    └── competition_solution.txt     the solution delivered for the real instance
```

```bash
cd src
make && ./tests                  # heuristics; no SCIP needed
./gap_simulator instances/real/real_instance solution.txt 60
pixi install && make exact       # SCIP and HiGHS part
./exact_tests check
./gap_exact refix instances/gap/gap_d/d05100 600
pixi exec --spec matplotlib python plots.py
```

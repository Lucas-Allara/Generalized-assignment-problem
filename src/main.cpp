#include <cstdlib>
#include <iostream>
#include "ConstructiveHeuristics.h"
#include "Instance.h"
#include "Metaheuristics.h"
#include "Solution.h"
#include "SolutionIO.h"

using namespace std;

// usage: ./gap_simulator <instance> <output> [seconds]
int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "usage: " << argv[0] << " <instance> <output> [seconds]\n";
        return 1;
    }

    Instance inst(argv[1]);

    // optional: argv[3] = time budget in seconds (without it, 500 iterations are run)
    double seconds = (argc > 3) ? atof(argv[3]) : 0.0;
    Solution sol = metaheuristics::ils(inst, 500, 20, 1, seconds);

    cout << "Cost after ILS: " << sol.total_cost << endl;

    write_solution(argv[2], sol);

    return 0;
}

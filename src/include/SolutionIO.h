#ifndef SOLUTION_IO_H
#define SOLUTION_IO_H

#include <fstream>
#include <string>
#include "Solution.h"

// writes the solution in the format of the assignment: one line per warehouse
// with the indices of its sellers separated by spaces.
inline void write_solution(const std::string& file, const Solution& sol) {
    std::ofstream output(file);
    for (const auto& sellers : sol.sellers_per_warehouse) {
        for (int t = 0; t < (int)sellers.size(); t++) {
            if (t > 0) output << " ";
            output << sellers[t];
        }
        output << "\n";
    }
}

#endif

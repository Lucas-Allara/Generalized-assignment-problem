#ifndef SOLUTION_H
#define SOLUTION_H

#include <algorithm>
#include <vector>

// solution of the problem: the sellers of each warehouse, the warehouse of each
// seller (-1 if unassigned) and the total cost.
class Solution {
public:
    std::vector<std::vector<int>> sellers_per_warehouse;
    std::vector<int> assigned_warehouse;
    double total_cost = 0;

    // removes the seller from the warehouse and leaves it unassigned.
    void remove(int warehouse, int seller) {
        assigned_warehouse[seller] = -1;
        std::vector<int>& list = sellers_per_warehouse[warehouse];
        list.erase(std::find(list.begin(), list.end(), seller));
    }

    // assigns the seller to the warehouse.
    void assign(int warehouse, int seller) {
        assigned_warehouse[seller] = warehouse;
        sellers_per_warehouse[warehouse].push_back(seller);
    }
};

#endif

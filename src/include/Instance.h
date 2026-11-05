#ifndef INSTANCE_H
#define INSTANCE_H

#include <string>
#include <vector>
#include "Seller.h"
#include "Warehouse.h"

// data of an instance: warehouses, sellers, the cost of assigning each seller to
// each warehouse and the penalty of leaving a seller unassigned. It is loaded from
// a file in the OR-Library format: m n, the m x n cost matrix, the m x n demand
// matrix d_ij and the m capacities.
class Instance {
public:
    std::vector<Warehouse> warehouses;
    std::vector<Seller> sellers;
    std::vector<std::vector<double>> cost;    // cost[j][i]: seller j in warehouse i
    double penalty = 0;                       // cost of an unassigned seller: 3 * largest cost

    Instance() = default;                     // empty instance, filled by hand (reduced instances)
    explicit Instance(const std::string& file);

    int demand(int seller, int warehouse) const { return sellers[seller].demand[warehouse]; }
};

#endif

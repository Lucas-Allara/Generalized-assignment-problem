#ifndef SELLER_H
#define SELLER_H

#include <algorithm>
#include <utility>
#include <vector>

// seller to be assigned: an id, the demand it places on each warehouse (d_ij,
// which in the general model depends on the warehouse) and the list of distances
// to each warehouse, sorted in increasing order (with the warehouse id).
class Seller {
public:
    int id;
    std::vector<int> demand;                          // demand[i]: units used in warehouse i
    std::vector<std::pair<double, int>> distances;    // (distance, warehouse), increasing

    Seller(int id, std::vector<int> demand, const std::vector<double>& dist)
        : id(id), demand(std::move(demand)) {
        for (int i = 0; i < (int)dist.size(); i++) {
            distances.push_back(std::make_pair(dist[i], i));
        }
        std::sort(distances.begin(), distances.end());
    }
};

#endif

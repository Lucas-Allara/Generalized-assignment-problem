#include "Instance.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace std;

Instance::Instance(const string& file) {
    ifstream input(file);
    if (!input.is_open()) {
        cerr << "cannot open " << file << "\n";
        exit(1);
    }

    int m, n;
    input >> m >> n;

    vector<vector<double>> file_cost(m, vector<double>(n));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) input >> file_cost[i][j];
    }
    vector<vector<int>> file_demand(m, vector<int>(n));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) input >> file_demand[i][j];
    }
    vector<int> capacity(m);
    for (int i = 0; i < m; i++) input >> capacity[i];
    input.close();

    for (int i = 0; i < m; i++) warehouses.push_back(Warehouse(i, capacity[i]));
    double largest = 0;
    for (int j = 0; j < n; j++) {
        vector<double> dist(m);
        vector<int> dem(m);
        for (int i = 0; i < m; i++) {
            dist[i] = file_cost[i][j];
            dem[i] = file_demand[i][j];
            largest = max(largest, dist[i]);
        }
        cost.push_back(dist);
        sellers.push_back(Seller(j, dem, dist));
    }
    penalty = 3 * largest;
}

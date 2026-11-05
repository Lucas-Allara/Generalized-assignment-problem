#ifndef WAREHOUSE_H
#define WAREHOUSE_H

// warehouse to which sellers are assigned: an id and its maximum capacity.
class Warehouse {
public:
    int id;
    int capacity;

    Warehouse(int id, int capacity) : id(id), capacity(capacity) {}
};

#endif

#ifndef __34_2_SERIAL_H__
#define __34_2_SERIAL_H__
//#include "grid.h"
#include "automata.h"
void updateCells(grid_elem* curr_grid, grid_elem* next_grid, int width, int height);

class Automaton34_2_Serial: public Automaton{
    public:
        Grid* grid;
        int num_iters;
        grid_elem* curr_grid;
        grid_elem* next_grid;
        Rule* rule;
        Automaton34_2_Serial();
        virtual ~Automaton34_2_Serial();
        Grid* get_grid();
        void setup(int num_of_iters);
        void create_grid(char *filename, int pattern_x, int pattern_y, int zeroed);
        void update_cells();
        void run_automaton();
        void set_rule(Rule *rule);
};

#endif

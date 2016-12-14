#ifndef __SERIAL_AUTOMATON__
#define __SERIAL_AUTOMATON__
//#include "grid.h"
#include "automata.h"
void updateCells(grid_elem* curr_grid, grid_elem* next_grid, int width, int height);

class SerialAutomaton : public Automaton{
    public:
        Grid* grid;
        int num_iters;
        grid_elem* curr_grid;
        grid_elem* next_grid;
        Rule* rule;
        SerialAutomaton();
        virtual ~SerialAutomaton();
        Grid* get_grid();
        void setup(int num_of_iters);
        void create_grid(char *filename);
        void update_cells();
        void run_automaton();
        void set_rule(Rule *rule);
};

#endif

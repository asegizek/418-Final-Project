#include "grid.h"

void updateCells(grid_elem* curr_grid, grid_elem* next_grid, int width, int height);

class Automaton34_2_Serial{
    public:
        Grid* grid;
        int num_iters;
        grid_elem* curr_grid;
        grid_elem* next_grid;
        Automaton34_2_Serial();
        virtual ~Automaton34_2_Serial();
        Grid* get_grid();
        void setup(int num_of_iters);
        void create_grid(char *filename);
        void run_automaton();
};

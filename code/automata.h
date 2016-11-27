#ifndef __AUTOMATA__
#define __AUTOMATA__

#include "grid.h"


class Automaton {

public:

    virtual ~Automaton() { };

    virtual Grid* get_grid() = 0;

    virtual void setup(int num_of_iters) = 0;

    virtual void create_grid(char *filename) = 0;

    virtual void update_cells() = 0;

    virtual void run_automaton() = 0;

    //virtual void dumpParticles(const char* filename) {}

};


#endif

#ifndef __AUTOMATA__
#define __AUTOMATA__

#include "grid.h"

class Automaton {

public:

    ~Automaton() { };

    Grid* get_grid();

    void setup();

    void runAutomaton();

    //virtual void dumpParticles(const char* filename) {}

};


#endif

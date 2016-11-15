#ifndef __AUTOMATA__
#define __AUTOMATA__



class Automaton {

public:

    virtual ~Automaton() { };

    virtual const Grid* getGrid() = 0;

    virtual void setup() = 0;

    virtual void runAutomaton() = 0;

    //virtual void dumpParticles(const char* filename) {}

};


#endif

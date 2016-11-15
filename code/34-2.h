#ifndef __AUTOMATON_34-2__
#define __AUTOMATON_34-2__

#ifndef uint
#define uint unsigned int
#endif

#include "automata.h"


class Automaton34_2: public Automation {

private:

  Grid* grid;

public:

  Automaton34_2();
  virtual ~Automaton34_2();

  const Grid* getGrid();

  void setup();

  void runAutomaton();

};


#endif

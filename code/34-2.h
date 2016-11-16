#ifndef __AUTOMATON_34-2__
#define __AUTOMATON_34-2__

#ifndef uint
#define uint unsigned int
#endif

#include "automata.h"


class Automaton34_2: public Automation {

private:

  Grid* grid;
  int num_iters;

  grid_elem* cuda_device_grid_curr;
  grid_elem* cuda_device_grid_next;
public:

  Automaton34_2();
  virtual ~Automaton34_2();

  const Grid* getGrid(std::string filename);

  void setup(int num_of_iters);

  void create_grid(char *filename);

  void run_automaton();

};


#endif

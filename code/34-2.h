#ifndef __AUTOMATON_34_2__
#define __AUTOMATON_34_2__

#ifndef uint
#define uint unsigned int
#endif

#include "automata.h"


class Automaton34_2: public Automaton {


public:
  Grid* grid;
  int num_iters;

  grid_elem* cuda_device_grid_curr;
  grid_elem* cuda_device_grid_next;


  Automaton34_2();
  virtual ~Automaton34_2();

  Grid* get_grid();

  void setup(int num_of_iters);

  void create_grid(char *filename, int pattern_x, int pattern_y);

  void run_automaton();

};


#endif

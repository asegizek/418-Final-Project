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
  grid_elem* cuda_device_lookup_table;
  Rule* rule;


  Automaton34_2();
  virtual ~Automaton34_2();

  Grid* get_grid();

  void setup(int num_of_iters);

  void create_grid(char *filename);

  void update_cells();

  void run_automaton();

  void set_rule(Rule* rule);

};


#endif

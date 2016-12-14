#ifndef __PARALLEL_AUTOMATON__
#define __PARALLEL_AUTOMATON__

#ifndef uint
#define uint unsigned int
#endif

#include "automata.h"


class ParallelAutomaton: public Automaton {


public:
  Grid* grid;
  int num_iters;

  grid_elem* cuda_device_grid_curr;
  grid_elem* cuda_device_grid_next;
  grid_elem* cuda_device_lookup_table;
  Rule* rule;


  ParallelAutomaton();
  virtual ~ParallelAutomaton();

  Grid* get_grid();

  void setup(int num_of_iters);

  void create_grid(char *filename);

  void update_cells();

  void run_automaton();

  void set_rule(Rule* rule);

};


#endif

#ifndef __AUTOMATON_34_2__
#define __AUTOMATON_34_2__

#ifndef uint
#define uint unsigned int
#endif

#include "automata.h"

// active_list elements should be big enough to hold every cell number
typedef unsigned int active_list_t;
// value of blank space in the active_list
#define ALIST_BLANK UINT_MAX

class Automaton34_2: public Automaton {


public:
  Grid* grid;
  int num_iters;

  Automaton34_2();
  virtual ~Automaton34_2();

  Grid* get_grid();

  void setup(int num_of_iters);

  void create_grid(char *filename, int pattern_x, int pattern_y, int zeroed);

  void run_automaton();

};


#endif

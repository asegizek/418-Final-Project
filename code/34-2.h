#ifndef __AUTOMATON_34_2__
#define __AUTOMATON_34_2__

#ifndef uint
#define uint unsigned int
#endif

#include "automata.h"
#include <cuda.h>

// used as elements in the live_block grid
typedef uint8_t live_block_t;

class Automaton34_2: public Automaton {


public:
  Grid* grid;
  int num_iters;

  grid_elem* cuda_device_grid_curr;
  grid_elem* cuda_device_grid_next;
  grid_elem* cuda_device_live_blocks;

  Automaton34_2();
  virtual ~Automaton34_2();

  Grid* get_grid();

  void setup(int num_of_iters);

  void create_grid(char *filename, int pattern_x, int pattern_y, int zeroed);

  void run_automaton();

  dim3 get_grid_dim();

};


#endif

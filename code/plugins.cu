#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "plugins.h"

__device__ 
grid_elem update_cell(grid_elem cell, grid_elem *neighbors) {
  printf("calling update_cell!\n");
  int live_neighbors = 0;
  for (int i = 0; i < 6; i++) {
    live_neighbors += neighbors[i];
  }
  grid_elem new_cell;

  if (cell == 0) new_cell = (live_neighbors == 2);
  else new_cell = (live_neighbors == 3 || live_neighbors == 4);
  return new_cell;
}



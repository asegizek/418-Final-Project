#ifndef __PLUGINS__
#define __PLUGINS__
#include "grid.h"

//get new cell value based on neighboring cells
__device__ grid_elem update_cell(grid_elem cell, grid_elem neighbors[6]);

#endif

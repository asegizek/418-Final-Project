#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <thrust/scan.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "34-2.h"
#include "util.h"

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct global_constants {
  int grid_width;
  int grid_height;
  int num_cols;
  grid_elem* curr_grid;
  grid_elem* next_grid;
  Rule* rule;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ global_constants const_params;

// kernelClearGrid --  (CUDA device code)
//
// Clear the grid, setting all cells to 0
__global__ void kernel_clear_grid() {

  // cells at border are not modified
  int image_x = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int image_y = blockIdx.y * blockDim.y + threadIdx.y + 1;

  int width = const_params.grid_width;
  int height = const_params.grid_width;

  // cells at border are not modified
  if (image_x >= width - 1 || image_y >= height - 1)
      return;

  int offset = image_y*width + image_x;

  // write to global memory
  const_params.curr_grid[offset] = 0;
}


// kernel_single_iteration (CUDA device code)
//
// compute a single iteration on the grid, putting the results in next_grid
__global__ void kernel_single_iteration(grid_elem* curr_grid, grid_elem* next_grid) {
  // cells at border are not modified
  int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int y = blockIdx.y * blockDim.y + threadIdx.y + 1;

  int width = const_params.grid_width;
  int height = const_params.grid_width;
  int cols = const_params.num_cols;
  // index in the grid of this thread
  int grid_index = y*cols + x;

  // cells at border are not modified
  if (x >= width - 1 || y >= height - 1)
      return;
  int curr_col = x - 1;
  int y_above = y - 1 ;
  int y_below = y + 1;

  uint buffer_top = 0;
  uint buffer_mid = curr_grid[curr_col + cols*y] << 16;
  uint buffer_bot= curr_grid[curr_col + cols*y_below] << 16;
    //increment current col
  curr_col++;

  buffer_top  = curr_grid[curr_col + cols*y_above] << 8;
  buffer_mid |= curr_grid[curr_col + cols*y] << 8;
  buffer_bot |= curr_grid[curr_col + cols*y_below] << 8;

      //grab values from the right column
  curr_col++;

  buffer_top |= curr_grid[curr_col + cols*y_above];
  buffer_mid |= curr_grid[curr_col + cols*y];
      //buffer_bot |= curr_grid[curr_col + cols*y_below];

  grid_elem new_val = 0;
  uint live_cells;
  uint center_cell;
  for (int i = 0; i < 8; i++) {
    center_cell = (buffer_mid >> 15 & 1);
    live_cells = (buffer_top >> 15 & 1) + (buffer_top >> 14 & 1) + (buffer_mid >> 14 & 1) 
    + (buffer_mid >> 16 & 1) + (buffer_bot >> 15 & 1) + (buffer_bot >> 16 & 1);
    new_val = (new_val << 1 )| ( (center_cell) ? (live_cells == 3 || live_cells == 4) : live_cells == 2);

    buffer_top <<= 1;
    buffer_mid <<= 1;
    buffer_bot <<= 1;
  }
  next_grid[x + cols*y ] = new_val;

}
__global__ void old_kernel_single_iteration(grid_elem* curr_grid, grid_elem* next_grid) {
  // cells at border are not modified
  int image_x = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int image_y = blockIdx.y * blockDim.y + threadIdx.y + 1;

  int width = const_params.grid_width;
  int height = const_params.grid_width;
  // index in the grid of this thread
  int grid_index = image_y*width + image_x;

  // cells at border are not modified
  if (image_x >= width - 1 || image_y >= height - 1)
      return;

  uint8_t live_neighbors = 0;

  // compute the number of live_neighbors
  // neighbors = index of {up, up-right, right, down, down-left, left};
  int neighbors[] = {grid_index - width, grid_index - width + 1, grid_index + 1,
                     grid_index + width, grid_index + width - 1, grid_index - 1};

  //depending on which row the cell is at it has 2 different neighbors?
  // int neighbor_offset = 2 * (image_y % 2) - 1;
  // int neighbors[] = {grid_index - 1, grid_index + 1, grid_index - width, grid_index + width, 
  //                    grid_index - width + neighbor_offset, grid_index + width + neighbor_offset};
  for (int i = 0; i < 6; i++) {
    live_neighbors += const_params.curr_grid[neighbors[i]];
    // live_neighbors += curr_grid[neighbors[i]];
  }

  //grid_elem curr_value = const_params.curr_grid[grid_index];
  grid_elem curr_value = curr_grid[grid_index];
  // values for the next iteration
  grid_elem next_value;

  if (!curr_value) {
    next_value = (live_neighbors == 2);
  } else {
    next_value = (live_neighbors == 3 || live_neighbors == 4);
  }

  //const_params.next_grid[grid_index] = next_value;
  next_grid[grid_index] = next_value;

}


Automaton34_2::Automaton34_2() {
  num_iters = 0;
  grid = NULL;
  cuda_device_grid_curr = NULL;
  cuda_device_grid_next = NULL;
}

Automaton34_2::~Automaton34_2() {
  if (grid) {
    delete grid->data;
    delete grid;
  }
  if (cuda_device_grid_curr) {
    cudaFree(cuda_device_grid_curr);
    cudaFree(cuda_device_grid_next);
  }
}

Grid*
Automaton34_2::get_grid() {

  // need to copy contents of the final grid from device memory
  // before we expose it to the caller

  //printf("Copying grid data from device\n");

  cudaMemcpy(grid->data,
             cuda_device_grid_curr,
             sizeof(grid_elem) * grid->num_cols * grid->height,
             cudaMemcpyDeviceToHost);
  Grid *unpacked = new Grid(grid->width+2, grid->height);
  unpacked->blank();
  //grid_elem* unpacked_data =  new grid_elem[grid->width * (grid->height-2)]();
  //printf("get grid called!!\n");
  for (int y = 1; y < grid->height-1; y++) {
    for (int x = 0; x < grid->width; x++) {
      //magic number 8 (byte size)
      int grid_index = (x / 8) + y*grid->num_cols + 1;
      grid_elem block = grid->data[grid_index];
      // printf("block: %02X\n", block);
      grid_elem val = (block >> (7 - (x % 8))) & 1;
      unpacked->data[x + 1 + (y)*unpacked->width] = val;
      // unpacked_data[x + (y-1)*grid->width] = val;
      // printf("%d ", val);
    }
    // printf("\n");
  }

  return unpacked;
  //return grid;
}

void
Automaton34_2::setup(int num_of_iters) {

  int deviceCount = 0;
  bool isFastGPU = false;
  std::string name;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  printf("Number of iterations: %d\n", num_of_iters);
  num_iters = num_of_iters;

  printf("---------------------------------------------------------\n");
  printf("Initializing CUDA for CudaRenderer\n");
  printf("Found %d CUDA devices\n", deviceCount);



  // By this time the scene should be loaded.  Now copy all the key
  // data structures into device memory so they are accessible to
  // CUDA kernels

  cudaMalloc(&cuda_device_grid_curr, sizeof(grid_elem) * grid->num_cols * grid->height);
  cudaMalloc(&cuda_device_grid_next, sizeof(grid_elem) * grid->num_cols* grid->height);

  cudaMemcpy(cuda_device_grid_curr, grid->data,
              sizeof(grid_elem) * grid->num_cols * grid->height, cudaMemcpyHostToDevice);
  cudaMemset(cuda_device_grid_next, 0, sizeof(grid_elem) * grid->num_cols * grid->height);

  // Initialize parameters in constant memory.
  global_constants params;
  params.grid_height = grid->height;
  params.grid_width = grid->width;
  params.num_cols = grid->num_cols;
  params.curr_grid = cuda_device_grid_curr;
  params.next_grid = cuda_device_grid_next;
  params.rule = rule;

  cudaMemcpyToSymbol(const_params, &params, sizeof(global_constants));
}

void Automaton34_2::create_grid(char *filename) {
  FILE *input = NULL;
  int width, height;
  grid_elem *data;

  input = fopen(filename, "r");
  if (!input) {
    printf("Unable to open file: %s\n", filename);
    printf("\nTerminating program\n");
    exit(1);
  }

      //copy in width and height
  if (fscanf(input, "%d %d\n", &width, &height) != 2) {
    fclose(input);
    printf("Invalid input\n");
    printf("\nTerminating program\n");
    exit(1);
  }
  //note 8 is a magic number set it to the size of a byte
  int num_cols = (width + 7) / 8;
  //add border cells
  height += 2;
  num_cols += 2;
  data = new grid_elem [num_cols*height];
  // insert data from file into grid
  int grid_index = 0;
  int temp;
  for (int y = 1; y < height-1; y++) {
    grid_elem block = 0;
    for (int x = 0; x < width; x++) {
       if (fscanf(input, "%d", &temp) != 1) {
        fclose(input);
        printf("Invalid input\n");
        printf("\nTerminating program\n");
        exit(1);
      }
      block |= (temp & 0x1) << (7 - (x % 8));
      //printf("block is: %02X\n", block);
      //printf("x is: %d\n", x);
      if ((x+1) % 8 == 0 || (x+1) == width) {
        grid_index = (x / 8) + num_cols*y + 1;
        // printf("data[%d] is %02X\n", grid_index, block);
        data[grid_index] = block;
        grid_index++;
        block = 0;
      }
    }
  }


  fclose(input);

  grid = new Grid(width, height);
  grid->data = data;
  grid->num_cols = num_cols;
  printf("num_cols is %d\n", grid->num_cols);
}
// create the initial grid using the input file
// void
// Automaton34_2::create_grid(char *filename) {

//   FILE *input = NULL;
//   int width, height;
//   grid_elem *data;

//   input = fopen(filename, "r");
//   if (!input) {
//     printf("Unable to open file: %s\n", filename);
//     printf("\nTerminating program\n");
//     exit(1);
//   }

//   // copy in width and height from file
//   if (fscanf(input, "%d %d\n", &width, &height) != 2) {
//     fclose(input);
//     printf("Invalid input\n");
//     printf("\nTerminating program\n");
//     exit(1);
//   }

//   printf("Width: %d\nHeight: %d\n", width, height);

//   // increase grid size to account for border cells
//   width += 2;
//   height += 2;
//   data = new grid_elem [width*height];

//   // insert data from file into grid
//   for (int y = 1; y < height - 1; y++) {
//     for (int x = 1; x < width - 1; x++) {
//       int temp;
//       if (fscanf(input, "%d", &temp) != 1) {
//         fclose(input);
//         printf("Invalid input\n");
//         printf("\nTerminating program\n");
//         exit(1);
//       }

//       data[width*y + x] = (grid_elem)temp;
//     }
//   }

//   fclose(input);

//   grid = new Grid(width, height);
//   grid->data = data;
// }

#define THREAD_DIMX 32
#define THREAD_DIMY 32

//single update
void 
Automaton34_2::update_cells() {
  int width_cells = grid->width - 2;
  int height_cells = grid->height - 2;

  // block/grid size for the pixel kernal
  dim3 cell_block_dim(THREAD_DIMX, THREAD_DIMY);
  dim3 cell_grid_dim((width_cells + cell_block_dim.x - 1) / cell_block_dim.x,
              (height_cells + cell_block_dim.y - 1) / cell_block_dim.y);
  kernel_single_iteration<<<cell_grid_dim, cell_block_dim>>>( cuda_device_grid_curr, cuda_device_grid_next);
    cudaThreadSynchronize();
  grid_elem* temp = cuda_device_grid_curr;
  cuda_device_grid_curr = cuda_device_grid_next;
  cuda_device_grid_next = temp;
}

void
Automaton34_2::run_automaton() {

  // number of threads needed in the x and y directions
  // note that this is less than the width/height due to the border of unmodified cells
  int width_cells = grid->width - 2;
  int height_cells = grid->height - 2;

  // block/grid size for the pixel kernal
  dim3 cell_block_dim(THREAD_DIMX, THREAD_DIMY);
  dim3 cell_grid_dim((width_cells + cell_block_dim.x - 1) / cell_block_dim.x,
              (height_cells + cell_block_dim.y - 1) / cell_block_dim.y);

  for (int iter = 0; iter < num_iters; iter++) {
    kernel_single_iteration<<<cell_grid_dim, cell_block_dim>>>( cuda_device_grid_curr, cuda_device_grid_next);
    cudaThreadSynchronize();
    //cudaMemcpy(cuda_device_grid_curr, cuda_device_grid_next,
      //sizeof(grid_elem) * grid->width * grid->height, cudaMemcpyDeviceToDevice);
    grid_elem* temp = cuda_device_grid_curr;
    cuda_device_grid_curr = cuda_device_grid_next;
    cuda_device_grid_next = temp;
  }
}

void Automaton34_2_Serial::set_rule(Rule *_rule) {
  rule = _rule;
  printf("automaton serial called!: num_alive: %d, num_dead: %d\n", rule->num_alive, rule->num_dead);
  for (int i = 0; i < rule->num_alive; i++) {
        printf("alive[%d] = %d\n", i, rule->alive[i]);
    }

    for (int i = 0; i < rule->num_dead; i++) {
        printf("dead[%d] = %d\n", i, rule->dead[i]);
    }
  return;
}

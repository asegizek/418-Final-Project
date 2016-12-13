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
  grid_elem* lookup_table;
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
  int height = const_params.grid_height;
  int cols = const_params.num_cols;
  // index in the grid of this thread
  int grid_index = y*cols + x;

  // cells at border are not modified
  if (x >= cols - 1 || y >= height - 1)
      return;
  int curr_col = x - 1;
  int y_above = y - 1 ;
  int y_below = y + 1;

  int buffer_top = 0;
  int buffer_mid = curr_grid[curr_col + cols*y] << 16;
  int buffer_bot= curr_grid[curr_col + cols*y_below] << 16;
    //increment current col
  curr_col++;

  buffer_top  = curr_grid[curr_col + cols*y_above] << 8;
  buffer_mid |= curr_grid[curr_col + cols*y] << 8;
  buffer_bot |= curr_grid[curr_col + cols*y_below] << 8;

      //grab values from the right column
  curr_col++;

  buffer_top |= curr_grid[curr_col + cols*y_above];
  buffer_mid |= curr_grid[curr_col + cols*y];

  grid_elem new_val = 0;


  int left_half = (buffer_top & 0xf800) | ((buffer_mid & 0x1f800) >> 6) | ((buffer_bot & 0x1f000) >> 12);
  int right_half = ((buffer_top & 0xf80) << 4) | ((buffer_mid) & 0x1f80) >> 2 | ((buffer_bot & 0x1f00) >> 8);

  //left val and right val each contain 4 cells/bits
  grid_elem left_val = const_params.lookup_table[left_half];
  grid_elem right_val = const_params.lookup_table[right_half];
  new_val = (left_val << 4) | right_val;

  next_grid[grid_index] = new_val;
}


//computes and stores the lookup table given the next_state rule table
grid_elem* create_lookup_table(grid_elem* next_state) {
  grid_elem* table = (grid_elem*) malloc(sizeof(grid_elem) * (1<<16));
  int max = 1<<16;
  for (int num = 0; num < max; num++) {
    int i = num;
    grid_elem res = 0;
    for (int j = 0; j < 4; j++) {
        int center = ((i >> 6) & 1);
        int live_neighbors  = (i & 1) + ((i >> 1) & 1) + ((i >> 5) & 1) + ((i >> 7) & 1) + ((i >> 11) & 1) + ((i >> 12) & 1);
        int alive = next_state[live_neighbors + center*7];
        res |= alive << j;
        i >>= 1;
    }
    table[num] = res;
  }
  return table;
}


Automaton34_2::Automaton34_2() {
  num_iters = 0;
  grid = NULL;
  cuda_device_grid_curr = NULL;
  cuda_device_grid_next = NULL;
  cuda_device_lookup_table = NULL;
}

Automaton34_2::~Automaton34_2() {
  if (grid) {
    delete grid->data;
    delete grid;
  }
  if (cuda_device_grid_curr) {
    cudaFree(cuda_device_grid_curr);
    cudaFree(cuda_device_grid_next);
    cudaFree(cuda_device_lookup_table);
  }
}

Grid*
Automaton34_2::get_grid() {

  // need to copy contents of the final grid from device memory
  // before we expose it to the caller


  cudaMemcpy(grid->data,
             cuda_device_grid_curr,
             sizeof(grid_elem) * grid->num_cols * grid->height,
             cudaMemcpyDeviceToHost);
  Grid *unpacked = new Grid(grid->width+2, grid->height);
  unpacked->blank();
  for (int y = 1; y < grid->height-1; y++) {
    for (int x = 0; x < grid->width; x++) {
      //magic number 8 (byte size)
      int grid_index = (x / 8) + y*grid->num_cols + 1;
      grid_elem block = grid->data[grid_index];
      grid_elem val = (block >> (7 - (x % 8))) & 1;
      unpacked->data[x + 1 + (y)*unpacked->width] = val;
    }
  }
  return unpacked;
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

  for (int i=0; i<deviceCount; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);
    name = deviceProps.name;
    if (name.compare("GeForce GTX 480") == 0
        || name.compare("GeForce GTX 670") == 0
        || name.compare("GeForce GTX 780") == 0)
    {
      isFastGPU = true;
    }

    printf("Device %d: %s\n", i, deviceProps.name);
    printf("   Memory Clock Rate (KHz): %d\n", deviceProps.memoryClockRate);
    printf("   Memory Bus Width (bits): %d\n", deviceProps.memoryBusWidth);
    printf("   Peak Memory Bandwidth (GB/s): %f\n",
        2.0*deviceProps.memoryClockRate*(deviceProps.memoryBusWidth/8)/1.0e6);
    printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
    printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
    printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);

    printf("---------------------------------------------------------\n");
    if (!isFastGPU) {

      printf("WARNING: "
             "You're not running on a fast GPU, please consider using "
             "NVIDIA GTX 480, 670 or 780.\n");
      printf("---------------------------------------------------------\n");
    }
  }

  // By this time the scene should be loaded.  Now copy all the key
  // data structures into device memory so they are accessible to
  // CUDA kernels

  //create lookup table
  grid_elem* lookup_table = create_lookup_table(rule->next_state);

  cudaMalloc(&cuda_device_grid_curr, sizeof(grid_elem) * grid->num_cols * grid->height);
  cudaMalloc(&cuda_device_grid_next, sizeof(grid_elem) * grid->num_cols* grid->height);
  cudaMalloc(&cuda_device_lookup_table, sizeof(grid_elem) * (1 << 16));

  cudaMemcpy(cuda_device_grid_curr, grid->data,
              sizeof(grid_elem) * grid->num_cols * grid->height, cudaMemcpyHostToDevice);
  cudaMemset(cuda_device_grid_next, 0, sizeof(grid_elem) * grid->num_cols * grid->height);
  cudaMemcpy(cuda_device_lookup_table, lookup_table, sizeof(grid_elem) * (1 << 16), cudaMemcpyHostToDevice);

  // Initialize parameters in constant memory.
  global_constants params;
  params.grid_height = grid->height;
  params.grid_width = grid->width;
  params.num_cols = grid->num_cols;
  params.curr_grid = cuda_device_grid_curr;
  params.next_grid = cuda_device_grid_next;
  params.lookup_table = cuda_device_lookup_table;
  cudaMemcpyToSymbol(const_params, &params, sizeof(global_constants));
}


// create the initial grid using the input file
//

void
Automaton34_2::create_grid(char *filename) {
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
    //block contains 8 cells in a row
    grid_elem block = 0;
    for (int x = 0; x < width; x++) {
       if (fscanf(input, "%d", &temp) != 1) {
        fclose(input);
        printf("Invalid input\n");
        printf("\nTerminating program\n");
        exit(1);
      }
      block |= (temp & 0x1) << (7 - (x % 8));
      if ((x+1) % 8 == 0 || (x+1) == width) {
        grid_index = (x / 8) + num_cols*y + 1;
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
}

#define THREAD_DIMX 32
#define THREAD_DIMY 8

//single update
void Automaton34_2::update_cells() {
  int width_cells = grid->num_cols - 2;
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


void Automaton34_2::run_automaton() {
  // number of threads needed in the x and y directions
  // note that this is less than the width/height due to the border of unmodified cells
  int width_cells = grid->num_cols - 2;
  int height_cells = grid->height - 2;

   // block/grid size for the pixel kernal
  dim3 cell_block_dim(THREAD_DIMX, THREAD_DIMY);
  dim3 cell_grid_dim((width_cells + cell_block_dim.x - 1) / cell_block_dim.x,
              (height_cells + cell_block_dim.y - 1) / cell_block_dim.y);

  for (int iter = 0; iter < num_iters; iter++) {
    kernel_single_iteration<<<cell_grid_dim, cell_block_dim>>>( cuda_device_grid_curr, cuda_device_grid_next);
    grid_elem* temp = cuda_device_grid_curr;
    cuda_device_grid_curr = cuda_device_grid_next;
    cuda_device_grid_next = temp;
  }
}

void Automaton34_2::set_rule(Rule *_rule) {
  rule = _rule;
  return;
}


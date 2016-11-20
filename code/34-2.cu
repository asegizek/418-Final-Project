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
  grid_elem* curr_grid;
  grid_elem* next_grid;
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
__global__ void kernel_single_iteration() {

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
  // neighbors = index of {up, up-right, right, down, down-left, left}
  int neighbors[] = {grid_index - width, grid_index - width + 1, grid_index + 1,
                      grid_index + width, grid_index + width - 1, grid_index - 1};

  for (int i = 0; i < 6; i++) {
    live_neighbors += const_params.curr_grid[neighbors[i]];
  }

  grid_elem curr_value = const_params.curr_grid[grid_index];
  // values for the next iteration
  grid_elem next_value;

  if (!curr_value) {
    next_value = (live_neighbors == 2);
  } else {
    next_value = (live_neighbors == 3 || live_neighbors == 4);
  }

  const_params.next_grid[grid_index] = next_value;

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

  printf("Copying grid data from device\n");

  cudaMemcpy(grid->data,
             &cuda_device_grid_curr,
             sizeof(grid_elem) * grid->width * grid->height,
             cudaMemcpyDeviceToHost);

  return grid;
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

  cudaMalloc(&cuda_device_grid_curr, sizeof(grid_elem) * grid->width * grid->height);
  cudaMalloc(&cuda_device_grid_next, sizeof(grid_elem) * grid->width * grid->height);

  cudaMemcpy(&cuda_device_grid_curr, grid->data,
              sizeof(grid_elem) * grid->width * grid->height, cudaMemcpyHostToDevice);
  cudaMemset(&cuda_device_grid_next, 0, sizeof(grid_elem) * grid->width * grid->height);

  // Initialize parameters in constant memory.
  global_constants params;
  params.grid_height = grid->height;
  params.grid_width = grid->width;
  params.curr_grid = cuda_device_grid_curr;
  params.next_grid = cuda_device_grid_next;

  cudaMemcpyToSymbol(const_params, &params, sizeof(global_constants));
}


// create the initial grid using the input file
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

  // copy in width and height from file
  if (fscanf(input, "%d %d\n", &width, &height) != 2) {
    fclose(input);
    printf("Invalid input\n");
    printf("\nTerminating program\n");
    exit(1);
  }

  printf("Width: %d\nHeight: %d\n", width, height);

  // increase grid size to account for border cells
  width += 2;
  height += 2;
  data = new grid_elem [width*height];

  // insert data from file into grid
  for (int y = 1; y < height - 1; y++) {
    for (int x = 1; x < width - 1; x++) {
      int temp;
      if (fscanf(input, "%d", &temp) != 1) {
        fclose(input);
        printf("Invalid input\n");
        printf("\nTerminating program\n");
        exit(1);
      }

      data[width*y + x] = (grid_elem)temp;
    }
  }

  fclose(input);

  grid = new Grid(width, height);
  grid->data = data;
}

#define THREAD_DIMX 32
#define THREAD_DIMY 32

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
    kernel_single_iteration<<<cell_grid_dim, cell_block_dim>>>();
    cudaThreadSynchronize();
    cudaMemcpy(&cuda_device_grid_curr, &cuda_device_grid_next,
      sizeof(grid_elem) * grid->width * grid->height, cudaMemcpyDeviceToDevice);
  }
}

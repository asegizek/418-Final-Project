#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "34-2.h"
#include "util.h"

#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/scan.h>

#include <ctime>

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct global_constants {
  int grid_width;
  int grid_height;
  grid_elem *curr_grid;
  grid_elem *next_grid;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ global_constants const_params;

thrust::device_ptr<grid_elem> cuda_device_grid_curr;
thrust::device_ptr<grid_elem> cuda_device_grid_next;

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

#define NUM_NEIGHBORS 6

// kernel_single_iteration (CUDA device code)
//
// compute a single iteration on the grid, putting the results in next_grid
__global__ void kernel_single_iteration(grid_elem* curr_grid, grid_elem* next_grid,
                                        active_list_t* active_list,
                                        active_list_t* active_grid,
                                        size_t active_list_size){

  size_t active_list_index = blockIdx.x*blockDim.x + threadIdx.x;

  // only operate on cells within the active list
  if (active_list_index < active_list_size) {

    size_t grid_index = active_list[active_list_index];

    int width = const_params.grid_width;
    int height = const_params.grid_width;

    int pos_x = grid_index % width;
    int pos_y = grid_index / width;

    // cells at border are not modified
    if (0 < pos_x && pos_x < width - 1 && 0 < pos_y && pos_y < height - 1
        && 0 < grid_index && grid_index < width*height) {

      uint8_t live_neighbors = 0;

      // compute the number of live_neighbors
      // neighbors = index of {up, up-right, right, down, down-left, left}
      int neighbors[] = {grid_index - width, grid_index - width + 1, grid_index + 1,
                          grid_index + width, grid_index + width - 1, grid_index - 1};

      for (int i = 0; i < NUM_NEIGHBORS; i++) {
        //live_neighbors += const_params.curr_grid[neighbors[i]];
        live_neighbors += curr_grid[neighbors[i]];
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

      // if the new cell is alive, add it and its neighbors to the active_grid
      if (next_value) {
        // add neighbors
        for (int i = 0; i < NUM_NEIGHBORS; i++) {
          size_t neighbor_index = neighbors[i];
          active_grid[neighbor_index] = 1;
        }

        // add yourself
        active_grid[grid_index] = 1;
      }

    }
  }
}

// kernel_create_active_list (CUDA device code)
//
// creates the active list using the active grid
__global__ void kernel_create_active_list(active_list_t* active_grid,
                                          active_list_t* active_list){

  size_t grid_index = blockIdx.x*blockDim.x + threadIdx.x;

  int width = const_params.grid_width;
  int height = const_params.grid_width;

  // only operate on values inside the grid
  if (grid_index < width*height) {

    active_list_t grid_val = active_grid[grid_index];
    // if the value in the next index is one greater, the value at this index is active
    if (grid_val < active_grid[grid_index + 1]) {

      // the value of this index to the active list
      active_list[grid_val] = grid_index;

    }
  }
}

Automaton34_2::Automaton34_2() {
  num_iters = 0;
  grid = NULL;
}

Automaton34_2::~Automaton34_2() {
  if (grid) {
    delete grid->data;
    delete grid;
    thrust::device_free(cuda_device_grid_curr);
    thrust::device_free(cuda_device_grid_next);
  }
}

Grid*
Automaton34_2::get_grid() {

  // need to copy contents of the final grid from device memory
  // before we expose it to the caller

  printf("Copying grid data from device\n");

  cudaMemcpy(grid->data,
             cuda_device_grid_curr.get(),
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

  cuda_device_grid_curr = thrust::device_malloc<grid_elem>(grid->width * grid->height);
  cuda_device_grid_next = thrust::device_malloc<grid_elem>(grid->width * grid->height);

  cudaMemcpy(cuda_device_grid_curr.get(), grid->data,
              sizeof(grid_elem) * grid->width * grid->height, cudaMemcpyHostToDevice);
  thrust::fill(cuda_device_grid_next, cuda_device_grid_next + grid->width*grid->height,0);

  // Initialize parameters in constant memory.
  global_constants params;
  params.grid_height = grid->height;
  params.grid_width = grid->width;
  params.curr_grid = cuda_device_grid_curr.get();
  params.next_grid = cuda_device_grid_next.get();

  cudaMemcpyToSymbol(const_params, &params, sizeof(global_constants));
}


// create the initial grid using the input file
//
// pattern_x and pattern_y determine how many times the input grid is repeated in the
// x and y directions
//
// if zeroed is true than all the patterns are zeroed out
void
Automaton34_2::create_grid(char *filename, int pattern_x, int pattern_y, int zeroed) {

  FILE *input = NULL;
  int width, height; // width and height of entire image
  int section_width, section_height; // width and height of the input grid
  grid_elem *data;

  input = fopen(filename, "r");
  if (!input) {
    printf("Unable to open file: %s\n", filename);
    printf("\nTerminating program\n");
    exit(1);
  }

  // copy in width and height from file
  if (fscanf(input, "%d %d\n", &section_width, &section_height) != 2) {
    fclose(input);
    printf("Invalid input\n");
    printf("\nTerminating program\n");
    exit(1);
  }

  width = section_width*pattern_x;
  height = section_height*pattern_y;

  printf("Width: %d\nHeight: %d\n", width, height);

  // increase grid size to account for border cells
  width += 2;
  height += 2;
  data = new grid_elem [width*height]();

  // insert data from file into grid

  // section_y and section_x represent the position in one individual 'pattern'
  for (int section_y = 0; section_y < section_height; section_y++) {
    for (int section_x = 0; section_x < section_width; section_x++) {

      int temp;
      if (fscanf(input, "%d", &temp) != 1) {
        fclose(input);
        printf("Invalid input\n");
        printf("\nTerminating program\n");
        exit(1);
      }

      // write value for each pattern
      // py and px represent the current pattern we are in
      for (int py = 0; py < pattern_y; py++) {
        for (int px = 0; px < pattern_x; px++) {

          int y_index = py*section_height + section_y + 1;
          int x_index = px*section_width + section_x + 1;

          // zeroed means that all cells outside of the initial pattern are zeroed out
          if (zeroed && (py > 0 || px > 0)) {
            data[y_index*width + x_index] = 0;
          }

          else {
            data[y_index*width + x_index] = (grid_elem)temp;
          }
        }
      }
    }
  }

  fclose(input);

  grid = new Grid(width, height);
  grid->data = data;
}

#define THREAD_DIMX 32
#define THREAD_DIMY 8
#define ITER_DIM 256

#define THREAD_DIM 256
#define CRT_ALIST_DIM 256

void
Automaton34_2::run_automaton() {


  // allocate memory for the list of cells that need to be checked
  // set the space (total space the list takes up) to the maximum size needed
  size_t active_list_space = grid->width * grid->height;
  thrust::device_ptr<active_list_t> active_list
      = thrust::device_malloc<active_list_t>(active_list_space);

  // allocate space for the grid of active cells
  // 1 is added to the total grid space to facilitate with the exclusive scan
  size_t active_grid_space = grid->width*grid->height + 1;
  thrust::device_ptr<active_list_t> active_grid
      = thrust::device_malloc<active_list_t>(active_grid_space);

  // current (dynamic) size of the active list
  size_t active_list_size = grid->width*grid->height;

  // add all the cells in the grid to the active list
  thrust::sequence(active_list, active_list + active_list_size);

  // block/grid size for the pixel kernel
  dim3 iter_block_dim(ITER_DIM);
  dim3 iter_grid_dim;

  // used for timing
  //time_t a_start, a_end, b_start, b_end;
  //time_t a_tot = 0;
  //time_t b_tot = 0;

  for (int iter = 0; iter < num_iters; iter++) {

    // zero out the new grid
    thrust::fill(cuda_device_grid_next, cuda_device_grid_next
                  + grid->width*grid->height, 0);

    // zero out the active_grid
    thrust::fill(active_grid, active_grid + active_grid_space, 0);

    // reset the grid dimensions to reflect the new number of values which must be
    // computed on
    iter_grid_dim = dim3((active_list_size + iter_block_dim.x - 1) / iter_block_dim.x);

    kernel_single_iteration<<<iter_grid_dim, iter_block_dim>>>
                      (cuda_device_grid_curr.get(), cuda_device_grid_next.get(),
                       active_list.get(), active_grid.get(), active_list_size);
    cudaThreadSynchronize();

    thrust::exclusive_scan(active_grid, active_grid + active_grid_space, active_grid);

    dim3 alist_block_dim(CRT_ALIST_DIM);
    dim3 alist_grid_dim = dim3((grid->width*grid->height + alist_block_dim.x - 1)
                                / alist_block_dim.x);

    // create the active list using the active grid
    kernel_create_active_list<<<alist_grid_dim, alist_block_dim>>>
      (active_grid.get(), active_list.get());

    // copy in the new size of the active list from the last value in the active grid
    active_list_t new_size;
    cudaMemcpy(&new_size, (active_grid + grid->width*grid->height).get(),
              sizeof(active_list_t) * 1, cudaMemcpyDeviceToHost);
    active_list_size = new_size;

    // swap the current and next pointers for the next iteration
    // this gets rid of the need to copy values between the 2 grids
    thrust::device_ptr<grid_elem> temp1 = cuda_device_grid_curr;
    cuda_device_grid_curr = cuda_device_grid_next;
    cuda_device_grid_next = temp1;

    //b_start = clock();
    //b_end = clock();
    //a_end = clock();
    //a_tot += a_end - a_start;
    //b_tot += b_end - b_start;
  }

  // free allocated memory
  thrust::device_free(active_list);

  //double a_time = double(a_tot) / CLOCKS_PER_SEC;
  //double b_time = double(b_tot) / CLOCKS_PER_SEC;
  //printf("a time: %f s\n", a_time);
  //printf("b time: %f s\n", b_time);
}

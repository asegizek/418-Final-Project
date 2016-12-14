#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <vector>
#include "34-2-serial.h"


Automaton34_2_Serial::Automaton34_2_Serial() {
  num_iters = 0;
  grid = NULL;
  curr_grid = NULL;
  next_grid = NULL;
}

Automaton34_2_Serial::~Automaton34_2_Serial() {
  if (curr_grid) {
    delete curr_grid;
    delete next_grid;
  }
  if (grid) {
    if (grid->data != curr_grid) delete grid->data;
    delete grid;
  }
}

Grid* Automaton34_2_Serial::get_grid() {
  grid->data = curr_grid;
  return grid;
}


void Automaton34_2_Serial::setup(int num_of_iters) {
  printf("Number of iterations: %d\n", num_of_iters);
  num_iters = num_of_iters;
  curr_grid = new grid_elem[grid->width * grid->height]();
  next_grid = new grid_elem[grid->width * grid->height]();
  std::copy(grid->data, grid->data + (grid->width*grid->height), curr_grid);
}

void Automaton34_2_Serial::set_rule(Rule *_rule) {
  rule = _rule;

  return;
}




void Automaton34_2_Serial::create_grid(char *filename) {
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

void printGrid(grid_elem* grid, int width, int height) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      printf("%d ", grid[y*width+x]);
    }
    printf("\n");
  }
}

void Automaton34_2_Serial::run_automaton() {
  for (int iter = 0; iter < num_iters; iter++) {
    this->update_cells();

    // printf("updated board is: \n");
    // for (int y = 0; y < grid->height; y++) {
    //   for (int x = 0; x < grid->num_cols; x++) {
    //     grid_elem val = curr_grid[x+grid->num_cols*y];
    //     printf("%02X ", val);
    //   }
    //   printf("\n");
    // }
  }
}


void Automaton34_2_Serial::update_cells() {
  int grid_index;
  int width = grid->width;
  int height = grid->height;
  grid_elem curr_val;
  grid_elem next_val;
  int live_neighbors;
  for (int y = 1; y < height-1; y++) {
    for (int x = 1; x < width-1; x++) {
      grid_index = width*y + x;
      curr_val = curr_grid[grid_index];
      live_neighbors = 0;
      // int neighbor_offset = 2 * (y % 2) - 1;
      // int neighbors[] = {grid_index - 1, grid_index + 1, grid_index - width, grid_index + width, 
      //                grid_index - width + neighbor_offset, grid_index + width + neighbor_offset};
      int neighbors[] = {grid_index - width, grid_index - width + 1, grid_index + 1,
                         grid_index + width, grid_index + width - 1, grid_index - 1};
      for (int i = 0; i < 6; i++) {
        live_neighbors += curr_grid[neighbors[i]];
      }

      // next_val = 0;
      // if (!curr_val) {
      //   next_val = rule->dead[live_neighbors];
      // } else {
      //   next_val = rule->alive[live_neighbors];
      // }
      next_val = rule->next_state[live_neighbors + curr_val*7];
      // if (!curr_val) {  
      //   next_val = (live_neighbors == 2);
      // } else {
      //   next_val = (live_neighbors == 3 || live_neighbors == 4);
      // }
      next_grid[grid_index] = next_val;
    }
  }
  grid_elem* temp = curr_grid;
  curr_grid = next_grid;
  next_grid = temp;
}



void updateCells(grid_elem* curr_grid, grid_elem* next_grid, int width, int height) {

  int grid_index;
  grid_elem curr_val;
  grid_elem next_val;
  int live_neighbors;
  //printf("update!!\n");
  for (int y = 1; y < height-1; y++) {
    for (int x = 1; x < width-1; x++) {
      grid_index = width*y + x;
      curr_val = curr_grid[grid_index];
      live_neighbors = 0;
      int neighbors[] = {grid_index - width, grid_index - width + 1, grid_index + 1,
                         grid_index + width, grid_index + width - 1, grid_index - 1};
      for (int i = 0; i < 6; i++) {
        live_neighbors += curr_grid[neighbors[i]];
      }


      if (!curr_val) {
        next_val = (live_neighbors == 2);
      } else {
        next_val = (live_neighbors == 3 || live_neighbors == 4);
      }
      next_grid[grid_index] = next_val;
    }
  }
}







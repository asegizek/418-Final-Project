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

//returns unpacked grid
Grid* Automaton34_2_Serial::get_grid() {
  //grid->data = curr_grid;
  Grid *unpacked = new Grid(grid->width+2, grid->height);
  grid_elem* data = curr_grid;
  unpacked->blank();
  //grid_elem* unpacked_data =  new grid_elem[grid->width * (grid->height-2)]();
  //printf("get grid called!!\n");
  for (int y = 1; y < grid->height-1; y++) {
    for (int x = 0; x < grid->width; x++) {
      //magic number 8 (byte size)
      int grid_index = (x / 8) + y*grid->num_cols + 1;
      grid_elem block = data[grid_index];
      // printf("block: %02X\n", block);
      grid_elem val = (block >> (7 - (x % 8))) & 1;
      unpacked->data[x + 1 + (y)*unpacked->width] = val;
      // unpacked_data[x + (y-1)*grid->width] = val;
      // printf("%d ", val);
    }
    // printf("\n");
  }
  // printf("width: %d, height: %d\n", unpacked->width, unpacked->height);
  for (int y = 0; y < unpacked->height; y++) {
    for (int x = 0; x < unpacked->width; x++) {
      // printf("%d ", unpacked->data[x + y*(unpacked->width)]);
    }
    // printf("\n");
  }
  return unpacked;
  // return grid;
}

void Automaton34_2_Serial::setup(int num_of_iters) {
  printf("Number of iterations: %d\n", num_of_iters);
  num_iters = num_of_iters;
  curr_grid = new grid_elem[grid->num_cols * grid->height]();
  next_grid = new grid_elem[grid->num_cols * grid->height]();
  std::copy(grid->data, grid->data + (grid->num_cols*grid->height), curr_grid);
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


void old_create_grid(char *filename) {
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
  // width += 2;
  // height += 2;
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

  //grid = new Grid(width, height);
  //grid->data = data;
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
    //std::copy(next_grid, next_grid + (grid->width*grid->height), curr_grid);
  }
}

void Automaton34_2_Serial::update_cells() {
  int grid_index;
  int width = grid->width;
  int height = grid->height;
  int cols = grid->num_cols;
  grid_elem curr_val;
  grid_elem next_val;
  int live_neighbors;
  //printf("update!!!!!\n");


  //NOTE if board width not a multiple of 8 1's propagate to unused bigts and mess calculations up
  for (int y = 1; y < height-1; y++) {
    for (int x = 1; x < cols-1; x++) {

      //first grab blocks before current block
      int curr_col = x - 1;
      int y_above = y - 1 ;
      int y_below = y + 1;
      // printf("y_above: %d\n", y_above);
      // printf("y_below: %d\n", y_below);
      grid_elem cells_above = curr_grid[curr_col + cols*y_above];
      grid_elem cells_middle = curr_grid[curr_col + cols*y];
      grid_elem cells_below = curr_grid[curr_col + cols*y_below];

      //32 int buffer that holds 3 cell blocks at a time
      //we only need the previous block of the middle row
      uint buffer_top = 0;
      uint buffer_mid = cells_middle << 16;
      uint buffer_bot= cells_below << 16;

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
        // printf("live cells: %d\n", live_cells);
        // printf("next state: %d\n", new_val & 1);
      }
      // printf("new_val is %02X\n", new_val);
      next_grid[curr_col + cols*y - 1] = new_val;


      // printf("index above: %d\n", index_above);
      // printf("index middle: %d\n", index_middle);
      // printf("index below: %d\n", index_below);

      // printf("cells above: %02X\n", cells_above);
      // printf("cells middle: %02X\n", cells_middle);
      // printf("cells below: %02X\n", cells_below);

      

      // if (!curr_val) {
      //   next_val = (live_neighbors == 2);
      // } else {
      //   next_val = (live_neighbors == 3 || live_neighbors == 4);
      // }
      //next_grid[grid_index] = 0;
    }
  }
  // printf("printing after update cells\n");
  // for (int y = 1; y < grid->height-1; y++ ) {
  //   for (int x = 1; x < grid->num_cols-1; x++) {
  //     printf("%02X ", next_grid[x + y*grid->num_cols]);
  //   }
  //   printf("\n");
  // }
  std::copy(next_grid, next_grid + (grid->num_cols*grid->height), curr_grid);
}


// void old_update_cells() {
//   int grid_index;
//   int width = grid->width;
//   int height = grid->height;
//   grid_elem curr_val;
//   grid_elem next_val;
//   int live_neighbors;
//   printf("update!!!!!\n");
//   for (int y = 1; y < height-1; y++) {
//     for (int x = 1; x < width-1; x++) {
//       grid_index = width*y + x;
//       curr_val = curr_grid[grid_index];
//       live_neighbors = 0;
//       int neighbor_offset = 2 * (y % 2) - 1;
//       int neighbors[] = {grid_index - 1, grid_index + 1, grid_index - width, grid_index + width, 
//                      grid_index - width + neighbor_offset, grid_index + width + neighbor_offset};
//       // int neighbors[] = {grid_index - width, grid_index - width + 1, grid_index + 1,
//       //                    grid_index + width, grid_index + width - 1, grid_index - 1};
//       for (int i = 0; i < 6; i++) {
//         live_neighbors += curr_grid[neighbors[i]];
//       }

//       next_val = 0;
//       if (!curr_val) {
//         for (int i = 0; i < rule->num_dead; i++) {
//           if (next_val == 0 && live_neighbors == rule->dead[i]) {
//             next_val = 1;
//           }
//         }
//       } else {
//         for (int i = 0; i < rule->num_alive; i++) {
//           if (next_val == 0 && live_neighbors == rule->alive[i]) {
//             next_val = 1;
//           }
//         }
//       }

//       // if (!curr_val) {
//       //   next_val = (live_neighbors == 2);
//       // } else {
//       //   next_val = (live_neighbors == 3 || live_neighbors == 4);
//       // }
//       next_grid[grid_index] = next_val;
//     }
//   }
//   std::copy(next_grid, next_grid + (grid->width*grid->height), curr_grid);
// }



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







#include <stdint.h>
#ifndef  __GRID_H__
#define  __GRID_H__

//grid element is 1 byte (containing 8 cells)
typedef unsigned char grid_elem;

struct Grid {

  Grid(int w, int h) {
    width = w;
    height = h;
    data = new grid_elem[width * height];
  }

  void blank() {
    int num_pixels = width * height;
    for (int i=0; i<num_pixels; i++) {
      data[i] = 0;
    }
  }

  int width;
  int height;
  //number of columns in data array is less than the width since cells are packed into bytes
  int num_cols;
  grid_elem* data;
};


#endif

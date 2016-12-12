#include <stdint.h>
#ifndef  __GRID_H__
#define  __GRID_H__

typedef unsigned char grid_elem;

typedef unsigned int uint;

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
  //number of columns in data array is less than width since cells are packed into grid_elem (byte)
  int num_cols;
  grid_elem* data;
};


#endif

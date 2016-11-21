#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <algorithm>
#include <time.h>

#include "platformgl.h"
#include "34-2.h"
#include "34-2-serial.h"


void startRenderer(Automaton34_2_Serial* automaton, int rows , int cols);

void usage(const char* progname) {
  printf("Usage: %s [options]\n", progname);
  printf("Program Options:\n");
  printf("  -f  --file   <FILENAME>    Filename of the grid to be used\n");
  printf("  default: default.txt\n");
  printf("  -i  --iterations <INT>         Number of iterations in automotan\n");
  printf("  default: 1\n");
  //printf("  -s --serial                 Run serial implementation\n");
  //printf("  -d --display                Run in display mode\n");
  printf("  -?  --help                 This message\n");
}


int main(int argc, char** argv)
{

  char def[] = "tests/default.txt";
  char *filename = def;
  int num_of_iters = 1;
  int serial = 0;
  int display = 0;
  // parse commandline options ////////////////////////////////////////////
  int opt;
  static struct option long_options[] = {
    {"help",     0, 0,  '?'},
    {"file",     1, 0,  'f'},
    {"iterations",     1, 0,  'i'},
    {"serial",   0, 0, 's'},
    {"display",  0, 0, 'd'},
    {0 ,0, 0, 0}
  };

  while ((opt = getopt_long(argc, argv, "f:i:?:s", long_options, NULL)) != EOF) {

    switch (opt) {
    case 'f':
      filename = optarg;
      break;
    case 'i':
      num_of_iters = atoi(optarg);
      break;
    case 's':
      serial = 1;
      break;
    case '?':
    default:
      usage(argv[0]);
      return 1;
    }
  }
  // end parsing of commandline options //////////////////////////////////////

  clock_t t;

  Grid* output_grid;
  if (!serial) {
    printf("Running parallel version\n");
    Automaton34_2* automaton = new Automaton34_2();
    automaton->create_grid(filename);
    automaton->setup(num_of_iters);
    t = clock();
    automaton->run_automaton();
    t = clock() - t;
    output_grid = automaton->get_grid();
    int height = output_grid->height;
    int width = output_grid->width;

  // write to output file
    FILE *output = fopen("output.txt", "w");
    fprintf(output, "%d %d\n", width - 2, height - 2);
    for (int y = 1; y < height - 1; y++) {
      for (int x = 1; x < width - 1; x++) {
        grid_elem val = output_grid->data[y*width + x];
        fprintf(output, "%u ", val);
      }
      fprintf(output, "\n");
    }
    fclose(output);
    delete automaton;
  } 

  else {
    printf("Running serial version\n");
    Automaton34_2_Serial* a = new Automaton34_2_Serial();
    a->create_grid(filename);
    a->setup(num_of_iters);
    if (display) {
      glutInit(&argc, argv);
      startRenderer(a, a->grid->width, a->grid->height);
      return 0;
    }
    
    t = clock();
    a->run_automaton();
    t = clock() - t;
    output_grid = a->get_grid();
    int height = output_grid->height;
    int width = output_grid->width;

  // write to output file
    FILE *output = fopen("output-serial.txt", "w");
    fprintf(output, "%d %d\n", width - 2, height - 2);
    for (int y = 1; y < height - 1; y++) {
      for (int x = 1; x < width - 1; x++) {
        grid_elem val = output_grid->data[y*width + x];
        fprintf(output, "%u ", val);
      }
      fprintf(output, "\n");
    }
    fclose(output);
  }

  


  
  printf("computation time: %fs\n", ((float)t)/CLOCKS_PER_SEC);
  printf("done\n");
  return 0;
}

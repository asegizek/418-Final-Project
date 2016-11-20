#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

#include "34-2.h"

void usage(const char* progname) {
  printf("Usage: %s [options]\n", progname);
  printf("Program Options:\n");
  printf("  -f  --file   <FILENAME>    Filename of the grid to be used\n");
  printf("  default: default.txt\n");
  printf("  -i  --iterations <INT>         Number of iterations in automotan\n");
  printf("  default: 1\n");
  printf("  -?  --help                 This message\n");
}


int main(int argc, char** argv)
{

  char def[] = "tests/default.txt";
  char *filename = def;
  int num_of_iters = 1;

  // parse commandline options ////////////////////////////////////////////
  int opt;
  static struct option long_options[] = {
    {"help",     0, 0,  '?'},
    {"file",     1, 0,  'f'},
    {"iterations",     1, 0,  'i'},
    {0 ,0, 0, 0}
  };

  while ((opt = getopt_long(argc, argv, "f:i:?", long_options, NULL)) != EOF) {

    switch (opt) {
    case 'f':
      filename = optarg;
      break;
    case 'i':
      num_of_iters = atoi(optarg);
      break;
    case '?':
    default:
      usage(argv[0]);
      return 1;
    }
  }
  // end parsing of commandline options //////////////////////////////////////

  Automaton34_2* automaton = new Automaton34_2();

  automaton->create_grid(filename);
  automaton->setup(num_of_iters);
  automaton->run_automaton();

  Grid *output_grid = automaton->get_grid();

  int height = output_grid->height;
  int width = output_grid->width;

  // write to output file
  FILE *output = fopen("output.txt", "w");
  fprintf(output, "%d %d\n", width, height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      grid_elem val = output_grid->data[y*width + x];
      fprintf(output, "%u ", val);
    }
    fprintf(output, "\n");
  }
  fclose(output);

  delete automaton;

  return 0;
}

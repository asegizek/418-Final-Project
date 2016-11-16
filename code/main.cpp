#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

#include "34-2.h"

void usage(const char* progname) {
  printf("Usage: %s [options]\n", progname);
  printf("Program Options:\n");
  printf("  -f  --file   <FILENAME>    Filename of the grid to be used\n");
  printf("  default: default.txt"
  printf("  -c  --cycles <INT>         Number of cycles in automotan\n");
  printf("  default: 1"
  printf("  -?  --help                 This message\n");
}


int main(int argc, char** argv)
{

  char filename[] = "default.txt";
  int num_iters = 1;

  // parse commandline options ////////////////////////////////////////////
  int opt;
  static struct option long_options[] = {
    {"help",     0, 0,  '?'},
    {"file",     1, 0,  'f'},
    {"cycles",     1, 0,  'c'},
    {0 ,0, 0, 0}
  };

  while ((opt = getopt_long(argc, argv, "f:c?", long_options, NULL)) != EOF) {

    switch (opt) {
    case 'f':
      filename = optarg;
      break;
    case 'c':
      num_iters = atoi(optarg);
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
  automaton->setup(num_iters);
  automaton->run_automaton();

  Grid *output = get_gird();

  delete automation;

  return 0;
}

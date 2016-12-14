#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <algorithm>
#include <ctime>

#include "platformgl.h"
#include "34-2.h"
#include "34-2-serial.h"


void startRenderer(Automaton34_2_Serial* automaton, int rows , int cols);

Rule* getRule(char* rule_file) {
  Rule* rl = new Rule;

  FILE * input = NULL;
  input = fopen(rule_file, "r");
  if (!input) {
    printf("Unable to open file: %s\n", rule_file);
    printf("\nTerminating program\n");
    exit(1);
  }
  int temp;
  int n_states;
  //read number of states
  if (fscanf(input, "%d\n", &n_states) != 1) {
    fclose(input);
    printf("Invalid input\n");
    printf("\nTerminating program\n");
    exit(1);
  }
  if (n_states > 5) {
    fclose(input);
    printf("Rule file has too many states\n");
    printf("\nTerminating program\n");
    exit(1);
  }

  rl->num_states = n_states;
  rl->next_state = (grid_elem*)malloc(sizeof(grid_elem) * MAX_STATES * 7);

    //populating next state grid
  for (int i = 0; i < 7 * n_states; i++) {
    if (fscanf(input, "%d", &temp) != 1) {
      fclose(input);
      printf("Rule file %s in unreadable format", rule_file);
      printf("\nTerminating program\n");
      exit(1);
    }
    rl->next_state[i] = temp;
  }
  // fclose(input);

  return rl;
} 

void usage(const char* progname) {
  printf("Usage: %s [options]\n", progname);
  printf("Program Options:\n");
  printf("  -f  --file   <FILENAME>    Filename of the grid to be used\n");
  printf("  default: default.txt\n");
  printf("  -r --rule    <FILENAME>    Filename of the rule to be used\n");
  printf("  default: rules/rule342.txt\n");
  printf("  -i  --iterations <INT>     Number of iterations in automaton\n");
  printf("  default: 1\n");
  printf("  -i  --pattern x <INT>      Number of times grid is repeated in x direction\n");
  printf("  default: 1\n");
  printf("  -i  --pattern y <INT>      Number of times grid is repeated in y direction\n");
  printf("  default: 1\n");
  printf("  -z  --zeroed patterns      All the patterns are zeroed out\n");
  //printf("  -s --serial                 Run serial implementation\n");
  //printf("  -d --display                Run in display mode\n");
  printf("  -?  --help                 This message\n");
}


int main(int argc, char** argv)
{

  char def[] = "tests/default.txt";
  char *filename = def;
  char out[] = "output.txt";
  char *output_file = out;
  char rule_f[] = "rules/rule342.txt";
  char *rule_file = rule_f;
  int num_of_iters = 1;
  int serial = 0;
  int display = 0;
  int zeroed = 0;
  int pattern_x = 1;
  int pattern_y = 1;
  // parse commandline options ////////////////////////////////////////////
  int opt;

  // used to get time infromation
  time_t total_start, total_end;
  time_t compute_start, compute_end;

  total_start = clock();

  static struct option long_options[] = {
    {"help",     0, 0,  '?'},
    {"file",     1, 0,  'f'},
    {"rule",     1, 0, 'r'},
    {"output",     1, 0,  'o'},
    {"iterations",     1, 0,  'i'},
    {"pattern_x",  1, 0, 'x'},
    {"pattern_y",  1, 0, 'y'},
    {"serial",   0, 0, 's'},
    {"display",  0, 0, 'd'},
    {"zeroed",  0, 0, 'z'},
    {0 ,0, 0, 0}
  };

  while ((opt = getopt_long(argc, argv, "f:o:r:i:x:y:?:sz", long_options, NULL)) != EOF) {

    switch (opt) {
    case 'f':
      filename = optarg;
      break;
    case 'o':
      output_file = optarg;
      break;
    case 'r':
      rule_file = optarg;
      break;
    case 'i':
      num_of_iters = atoi(optarg);
      break;
    case 's':
      serial = 1;
      break;
    case 'x':
      pattern_x = atoi(optarg);
      break;
    case 'y':
      pattern_y = atoi(optarg);
      break;
    case 'z':
      zeroed = 1;
      break;
    case '?':
    default:
      usage(argv[0]);
      return 1;
    }
  }
  // end parsing of commandline options //////////////////////////////////////

  Grid* output_grid;
  Rule* rule = getRule(rule_file);
  if (!serial) {
    printf("Running parallel version\n");
    Automaton34_2* automaton = new Automaton34_2();
    automaton->set_rule(rule);
    automaton->create_grid(filename);
    automaton->setup(num_of_iters);

    compute_start = clock();
    automaton->run_automaton();
    compute_end = clock();

    output_grid = automaton->get_grid();
    int height = output_grid->height;
    int width = output_grid->width;

  // write to output file
    FILE *output = fopen(output_file, "w");
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
    a->set_rule(rule);
    a->create_grid(filename);
    a->setup(num_of_iters);
    if (display) {
      glutInit(&argc, argv);
      startRenderer(a, a->grid->width, a->grid->height);
      return 0;
    }

    compute_start = clock();
    a->run_automaton();
    compute_end = clock();

    output_grid = a->get_grid();
    int height = output_grid->height;
    int width = output_grid->width;

  // write to output file
    FILE *output = fopen(output_file, "w");
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


  total_end = clock();

  double total_time = double(total_end - total_start) / CLOCKS_PER_SEC;
  double compute_time = double(compute_end - compute_start) / CLOCKS_PER_SEC;
  printf("total time: %f s\n", total_time);
  printf("compute time: %f s\n", compute_time);
  printf("done\n");
  return 0;
}

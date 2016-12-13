#ifndef __RULE__
#define __RULE__
#include "grid.h"
#define MAX_STATES 5

struct Rule {
    int num_states;
    grid_elem* next_state;
};
#endif

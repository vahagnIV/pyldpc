//
// Created by vahagn on 10/25/19.
//

#ifndef PYLDPCC_MAKE_LDPC_H
#define PYLDPCC_MAKE_LDPC_H

#include "distrib.h"
#include "generator_matrix.h"

typedef enum
{ Evencol, 	/* Uniform number of bits per column, with number specified */
  Evenboth 	/* Uniform (as possible) over both columns and rows */
} make_method;

mod2sparse *make_ldpc
    ( int seed,		/* Random number seed */
      make_method method,	/* How to make it */
      int M, /* Rows */
      int N, /* Columns */
      distrib *d,		/* Distribution list specified */
      int no4cycle		/* Eliminate cycles of length four? */
    );

#endif //PYLDPCC_PY_MAKE_LDPC_H

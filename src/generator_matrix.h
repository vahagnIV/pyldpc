//
// Created by vahagn on 10/28/19.
//

#ifndef PYLDPCC_MATRIX_TUPLE_H
#define PYLDPCC_MATRIX_TUPLE_H
#include <stdio.h>
#include "matrix_union.h"


typedef enum { Sparse, Dense, Mixed } MatrixRepresentation;      /* Ways of making it */

typedef struct {
  MatrixRepresentation type;
  mod2dense * G;
  mod2sparse * L;
  mod2sparse * U;
  int * column_ordering;
  int * row_ordering;
  int M;
  int N;

} GeneratorMatrix;

GeneratorMatrix *create_empty_generator_matrix();
void deallocate_generator_matrix(GeneratorMatrix * matrix);

#endif //PYLDPCC_MATRIX_TUPLE_H

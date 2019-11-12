//
// Created by vahagn on 10/28/19.
//
#include <stdlib.h>
#include <stdio.h>
#include "generator_matrix.h"
#include "mod2sparse.h"
#include "mod2dense.h"

GeneratorMatrix *create_empty_generator_matrix() {
  return calloc(1, sizeof(GeneratorMatrix));
}

void deallocate_generator_matrix(GeneratorMatrix *matrix) {
  if (matrix == NULL)
    return;

  if (matrix->L)
    mod2sparse_free(matrix->L);
  if (matrix->U)
    mod2sparse_free(matrix->U);
  if (matrix->column_ordering)
    free(matrix->column_ordering);
  if (matrix->row_ordering)
    free(matrix->row_ordering);

  if (matrix->G)
    mod2dense_free(matrix->G);
  free(matrix);
}

//
// Created by vahagn on 10/25/19.
//

/* MAKE-GEN.C - Make generator matrix from parity-check matrix. */

/* Copyright (c) 1995-2012 by Radford M. Neal.
 *
 * Permission is granted for anyone to copy, use, modify, and distribute
 * these programs and accompanying documents for any purpose, provided
 * this copyright notice is retained and prominently displayed, and note
 * is made of any changes made to these programs.  These programs and
 * documents are distributed without any warranty, express or implied.
 * As the programs were written for research purposes only, they have not
 * been tested to the degree that would be advisable in any important
 * application.  All use of these programs is entirely at the user's own
 * risk.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "alloc.h"
#include "intio.h"

#include "mod2sparse.h"
#include "mod2dense.h"
#include "mod2convert.h"
#include "generator_matrix.h"
#include "make_gen.h"

GeneratorMatrix *make_dense_mixed
    (mod2sparse *H,
     MatrixRepresentation method
    );     /* Procs to make it */

GeneratorMatrix *make_sparse
    (mod2sparse *H,
     mod2sparse_strategy strategy,
     int abandon_number,
     int abandon_when
    );

GeneratorMatrix *make_generator(MatrixRepresentation method, mod2sparse *H,
                                mod2sparse_strategy strategy,
                                int abandon_number,
                                int abandon_when) {
  GeneratorMatrix *generatot_matrix;
  if (method == Dense || method == Mixed) {
    generatot_matrix = make_dense_mixed(H, method);
  } else if (method == Sparse) {
    return make_sparse(H, strategy, abandon_number, abandon_when);
  }
  else
    generatot_matrix = NULL;
  return generatot_matrix;
}

/* MAKE DENSE OR MIXED REPRESENTATION OF GENERATOR MATRIX. */

GeneratorMatrix *make_dense_mixed
    (mod2sparse *H,
     MatrixRepresentation method
    ) {
  mod2dense *DH, *A, *A2, *AI, *B;
  int i, j, c, c2, n;
  int *rows_inv;

  int M = H->n_rows;
  int N = H->n_cols;

  GeneratorMatrix *result = create_empty_generator_matrix();
  result->type = method;
  result->M = M;
  result->N = N;

  DH = mod2dense_allocate(M, N);
  AI = mod2dense_allocate(M, M);
  B = mod2dense_allocate(M, N - M);
  result->G = mod2dense_allocate(M, N - M);

  mod2sparse_to_dense(H, DH);

  /* If no other generator matrix was specified, invert using whatever
     selection of rows/columns is needed to get a non-singular sub-matrix. */

  result->column_ordering = chk_alloc(N, sizeof *result->row_ordering);
  result->row_ordering = chk_alloc(M, sizeof *result->column_ordering);

  A = mod2dense_allocate(M, N);
  A2 = mod2dense_allocate(M, N);

  n = mod2dense_invert_selected(DH, A2, result->row_ordering, result->column_ordering);
  mod2sparse_to_dense(H, DH);  /* DH was destroyed by invert_selected */

  if (n > 0) {
    fprintf(stderr, "Note: Parity check matrix has %d redundant checks\n", n);
  }

  rows_inv = chk_alloc(M, sizeof *rows_inv);

  for (i = 0; i < M; i++) {
    rows_inv[result->row_ordering[i]] = i;
  }

  mod2dense_copyrows(A2, A, result->row_ordering);
  mod2dense_copycols(A, A2, result->column_ordering);
  mod2dense_copycols(A2, AI, rows_inv);

  mod2dense_copycols(DH, B, result->column_ordering + M);


  /* Form final generator matrix. */

  if (method == Dense) {
    mod2dense_multiply(AI, B, result->G);
  } else if (method == Mixed) {
    result->G = AI;
  } else {
    return NULL;
  }

  /* Compute and print number of 1s. */

  if (method == Dense) {
    c = 0;
    for (i = 0; i < M; i++) {
      for (j = 0; j < N - M; j++) {
        c += mod2dense_get(result->G, i, j);
      }
    }
    fprintf(stderr,
            "Number of 1s per check in Inv(A) X B is %.1f\n", (double) c / M);
  }

  if (method == Mixed) {
    c = 0;
    for (i = 0; i < M; i++) {
      for (j = 0; j < M; j++) {
        c += mod2dense_get(result->G, i, j);
      }
    }
    c2 = 0;
    for (i = M; i < N; i++) {
      c2 += mod2sparse_count_col(H, result->column_ordering[i]);
    }
    fprintf(stderr,
            "Number of 1s per check in Inv(A) is %.1f, in B is %.1f, total is %.1f\n",
            (double) c / M, (double) c2 / M, (double) (c + c2) / M);
  }

  /* Write the represention of the generator matrix to the file. */
  //DH, *A, *A2, *AI, *B;
  mod2dense_free(DH);
  mod2dense_free(A);
  mod2dense_free(A2);
  if (method == Dense)
    mod2dense_free(AI);
  mod2dense_free(B);

  return result;
}

/* MAKE SPARSE REPRESENTATION OF GENERATOR MATRIX. */

GeneratorMatrix *make_sparse
    (mod2sparse *H,
     mod2sparse_strategy strategy,
     int abandon_number,
     int abandon_when
    ) {

  int n, cL, cU, cB;
  int i;

  /* Find LU decomposition. */
  int M = H->n_rows;
  int N = H->n_cols;

  GeneratorMatrix *result = create_empty_generator_matrix();
  result->type = Sparse;
  result->M = M;
  result->N = N;

  result->L = mod2sparse_allocate(M, M);

  result->U = mod2sparse_allocate(M, N);

  result->column_ordering = chk_alloc(N, sizeof *result->column_ordering);
  result->row_ordering = chk_alloc(M, sizeof *result->row_ordering);

  n = mod2sparse_decomp(H,
                        M,
                        result->L,
                        result->U,
                        result->row_ordering,
                        result->column_ordering,
                        strategy,
                        abandon_number,
                        abandon_when);

  if (n != 0 && abandon_number == 0) {
    fprintf(stderr, "Note: Parity check matrix has %d redundant checks\n", n);
  }
  if (n != 0 && abandon_number > 0) {
    fprintf(stderr,
            "Note: Have %d dependent columns, but this could be due to abandonment.\n", n);
    fprintf(stderr,
            "      Try again with lower abandonment number.\n");
    exit(1);
  }

  /* Compute and print number of 1s. */

  cL = cU = cB = 0;

  for (i = 0; i < M; i++) cL += mod2sparse_count_row(result->L, i);
  for (i = 0; i < M; i++) cU += mod2sparse_count_row(result->U, i);
  for (i = M; i < N; i++) cB += mod2sparse_count_col(H, result->column_ordering[i]);

  fprintf(stderr,
          "Number of 1s per check in L is %.1f, U is %.1f, B is %.1f, total is %.1f\n",
          (double) cU / M, (double) cL / M, (double) cB / M, (double) (cL + cU + cB) / M);

  return result;

}


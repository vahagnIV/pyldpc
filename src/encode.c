//
// Created by vahagn on 10/29/19.
//

#include <stdio.h>
#include <malloc.h>
#include "encode.h"
#include "generator_matrix.h"
#include "enc.h"
#include "alloc.h"

int encode(mod2sparse *H, GeneratorMatrix *G, int message_count, char *messages, char *out_buffer, int check_result) {
  char *chks, *message, *encoded_codeword;
  mod2dense *u = 0, *v = 0;
  int i, block_no, error_indicator = 0;

  int M, N;
  M = H->n_rows;
  N = H->n_cols;
  chks = chk_alloc(M, sizeof *chks);

  if (G->type == Dense) {
    u = mod2dense_allocate(N - M, 1);
    v = mod2dense_allocate(M, 1);
  }

  if (G->type == Mixed) {
    u = mod2dense_allocate(M, 1);
    v = mod2dense_allocate(M, 1);
  }
  for (block_no = 0; block_no < message_count; ++block_no) {
    message = messages + (N - M) * block_no;

    encoded_codeword = out_buffer + N * block_no;

    switch (G->type) {
      case Sparse: {
        sparse_encode(message, encoded_codeword, H, G->L, G->U, G->row_ordering, G->column_ordering);
        break;
      }
      case Dense: {
        dense_encode(message, encoded_codeword, G->G, G->row_ordering, G->column_ordering, u, v);
        break;
      }
      case Mixed: {
        mixed_encode(message, encoded_codeword, H, G->G, G->row_ordering, G->column_ordering, u, v);
        break;
      }
    }

    if (check_result) {
      error_indicator = 0;
      mod2sparse_mulvec(H, encoded_codeword, chks);
      for (i = 0; i < M; i++) {
        if (chks[i] == 1) {
          fprintf(stderr, "Output block is not a code word!  (Fails check %d)\n", i);
          error_indicator = 1;
        }
      }
    }
    if (check_result && error_indicator)
      break;
  }
  if (G->type == Dense || G->type == Mixed) {
    mod2dense_free(u);
    mod2dense_free(v);
  }
  free(chks);
  return block_no;

}

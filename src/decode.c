//
// Created by vahagn on 10/30/19.
//

#include "generator_matrix.h"
#include "decode.h"
#include "alloc.h"
#include "check.h"
#include <math.h>
#include <malloc.h>

int decode(mod2sparse *H,
           GeneratorMatrix *G,
           int encoded_message_count,
           double *encoded_messages,
           char *decoded_messages,
           channel_type channel,
           decoding_method dec_method,
           double bit_error_distribution_parameter,
           BlockDecodeResult *block_decode_result
) {



  double *lratio, *bitpr, *encoded_message;
  char *pchk, *decoded_message;
  int M = H->n_rows;
  int N = H->n_cols;
  int i;
  int block_no, iters, valid;

  pchk = chk_alloc(M, sizeof *pchk);
  bitpr = chk_alloc(N, sizeof *bitpr);
  lratio = chk_alloc(N, sizeof *lratio);

  for (block_no = 0; block_no < encoded_message_count; ++block_no) {
    iters = 0;
    encoded_message = encoded_messages + N * block_no;
    decoded_message = decoded_messages + N * block_no;
    switch (channel) {
      case BSC: {
        for (i = 0; i < N; i++) {
          lratio[i] =
              encoded_message[i] == 1 ? (1. - bit_error_distribution_parameter) / bit_error_distribution_parameter
                                      : bit_error_distribution_parameter / (1. - bit_error_distribution_parameter);
        }
        break;
      }
      case AWGN: {
        for (i = 0; i < N; i++) {
          lratio[i] =
              exp(2 * encoded_messages[i] / (bit_error_distribution_parameter * bit_error_distribution_parameter));
        }
        break;
      }
      case AWLN: {
        for (i = 0; i < N; i++) {
          double e, d1, d0;
          e = exp(-(encoded_messages[i] - 1) / bit_error_distribution_parameter);
          d1 = 1 / ((1 + e) * (1 + 1 / e));
          e = exp(-(encoded_messages[i] + 1) / bit_error_distribution_parameter);
          d0 = 1 / ((1 + e) * (1 + 1 / e));
          lratio[i] = d1 / d0;
        }
        break;
      }
      default:
        iters = -1;
    }


    if (iters == -1)
      break;

    switch (dec_method) {
      case Prprp:
        iters = prprp_decode(H, lratio, decoded_message, pchk, bitpr, 1);
        break;
      case Enum_block:
      case Enum_bit:
        iters = enum_decode(H, G, lratio, decoded_message, bitpr, 1);
        break;
      default:
        iters = -1;
        break;
    }

    if (iters == -1)
      break;

    valid = check(H, decoded_message, pchk) == 0;
    if (block_decode_result) {
      block_decode_result[block_no].is_valid = valid;
      block_decode_result[block_no].flipped_count = changed(lratio, decoded_message, N);
      block_decode_result[block_no].iter_count = iters;
    }
  }

  free(pchk);
  free(bitpr);
  free(lratio);

  return block_no;
}
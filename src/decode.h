//
// Created by vahagn on 10/30/19.
//

#ifndef PYLDPCC_DECODE_H
#define PYLDPCC_DECODE_H

typedef enum { BSC, AWGN, AWLN } channel_type;


typedef struct {
  int is_valid;
  int flipped_count;
  int iter_count;
} BlockDecodeResult;
#include "dec.h"
int decode(mod2sparse *H,
           GeneratorMatrix *G,
           int encoded_message_count,
           double *encoded_messages,
           char *decoded_messages,
           channel_type channel,
           decoding_method dec_method,
           double bit_error_distribution_parameter, /* For BSC it is the probability of error,
                                                    * for AWGN it is the standard deviation
                                                    * for AWLN it is the lwidth*/
           BlockDecodeResult * block_decode_result);
#endif //PYLDPCC_DECODE_H

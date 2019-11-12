//
// Created by vahagn on 10/29/19.
//

#ifndef PYLDPCC_ENCODE_H
#define PYLDPCC_ENCODE_H

#include "generator_matrix.h"
int encode(mod2sparse *H, GeneratorMatrix *G, int message_count, char *messages, char *out_buffer, int check_result);

#endif //PYLDPCC_ENCODE_H

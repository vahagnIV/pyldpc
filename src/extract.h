//
// Created by vahagn on 10/31/19.
//

#ifndef PYLDPCC_EXTRACT_H
#define PYLDPCC_EXTRACT_H
#include "generator_matrix.h"

int extract(GeneratorMatrix *G, int decoded_message_count, char *decoded_messages, char *extracted_messages);

#endif //PYLDPCC_EXTRACT_H

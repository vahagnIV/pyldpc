//
// Created by vahagn on 10/31/19.
//

#include "extract.h"

int extract(GeneratorMatrix *G,  int decoded_message_count, char *decoded_messages, char *extracted_messages) {
  char *decoded_message, *extracted_message;
  int block_no, i;
  // TODO: check this line
  static char zero_one[] = {0, 1};



  for (block_no = 0; block_no < decoded_message_count; ++block_no) {
    decoded_message = decoded_messages + G->N * block_no;
    extracted_message = extracted_messages + (G->N - G->M) * block_no;

    for (i = G->M; i < G->N; ++i) {
      extracted_message[i - G->M] = zero_one[decoded_message[G->column_ordering[i]]];
    }
  }
  return 1;

}
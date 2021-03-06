/* ENC.H - Interface to encoding procedures. */

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
#ifndef PYLDPCC_ENC_H
#define PYLDPCC_ENC_H

void sparse_encode (char *sblk,
                    char *cblk,
                    mod2sparse * H,
                    mod2sparse * L,
                    mod2sparse * U,
                    int * rows,
                    int * cols);
void dense_encode  (char *sblk,
                    char *cblk,
                    mod2dense * G,
                    int * rows,
                    int * cols,
                    mod2dense *u,
                    mod2dense *v);
void mixed_encode  (char *sblk,
                    char *cblk,
                    mod2sparse * H,
                    mod2dense * G,
                    int *rows,
                    int * cols,
                    mod2dense *u,
                    mod2dense *v);

#endif
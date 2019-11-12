/* DEC.H - Interface to decoding procedures. */

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


/* DECODING METHOD, ITS PARAMETERS, AND OTHER VARIABLES.  The global variables 
   declared here are located in dec.c. */
#ifndef PYLDPCC_DEC_H
#define PYLDPCC_DEC_H
#include "mod2sparse.h"
#include "generator_matrix.h"

typedef enum {
  Enum_block, Enum_bit, Prprp
} decoding_method;

/* PROCEDURES RELATING TO DECODING METHODS. */

void enum_decode_setup(int M, int N, int table);
unsigned enum_decode(mod2sparse *H,
                     GeneratorMatrix *G,
                     double *lratio,    /* Likelihood ratios for bits */
                     char *dblk,        /* Place to stored decoded message */
                     double *bitpr,    /* Place to store marginal bit probabilities */
                     int max_block);       /* Maximize probability of whole block being correct? */


void prprp_decode_setup(int table);
unsigned prprp_decode
    (mod2sparse *H,    /* Parity check matrix */
     double *lratio,    /* Likelihood ratios for bits */
     char *dblk,        /* Place to store decoding */
     char *pchk,        /* Place to store parity checks */
     double *bprb,        /* Place to store bit probabilities */
     int max_iter);

void initprp(mod2sparse *, double *, char *, double *);
void iterprp(mod2sparse *, double *, char *, double *);

#endif
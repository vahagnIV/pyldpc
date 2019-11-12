//
// Created by vahagn on 10/25/19.
//

#ifndef PYLDPCC_MAKE_GEN_H
#define PYLDPCC_MAKE_GEN_H

#include "generator_matrix.h"

GeneratorMatrix *make_generator(MatrixRepresentation method, mod2sparse * H,
                              mod2sparse_strategy strategy,
                              int abandon_number,
                              int abandon_when);

#endif //PYLDPCC_MAKE_GEN_H

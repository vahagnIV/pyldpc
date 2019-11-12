//
// Created by vahagn on 10/28/19.
//

#ifndef PYLDPCC_MATRIX_UNION_H
#define PYLDPCC_MATRIX_UNION_H
#include "mod2sparse.h"
#include "mod2dense.h"

union MatrixUnion {
  mod2sparse *sparse;
  mod2dense *dense;
};

#endif //PYLDPCC_MATRIX_UNION_H

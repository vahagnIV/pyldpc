//
// Created by vahagn on 10/29/19.
//

#ifndef PYLDPCC_PY_GEN_MATRIX_H
#define PYLDPCC_PY_GEN_MATRIX_H
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>
#include "generator_matrix.h"

typedef struct {
  PyObject_HEAD
  GeneratorMatrix *data;
} PyGeneratorMatrix;

int initialize_generator_matrix_constants(PyObject * m);

PyGeneratorMatrix *create_py_empty_generator_matrix();


#endif //PYLDPCC_PY_GEN_MATRIX_H

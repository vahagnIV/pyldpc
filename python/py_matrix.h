//
// Created by vahagn on 10/25/19.
//

#ifndef PYLDPCC_PY_MATRIX_H
#define PYLDPCC_PY_MATRIX_H
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "generator_matrix.h"


typedef struct {
  PyObject_HEAD
  mod2sparse * data;
} PyMatrix;




int initialize_matrix_constants(PyObject *module);
PyMatrix *create_py_empty_matrix();

#endif //PYLDPCC_PY_MATRIX_H

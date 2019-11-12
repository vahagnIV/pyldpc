//
// Created by vahagn on 11/1/19.
//

#ifndef PYLDPCC_UTILS_H
#define PYLDPCC_UTILS_H
#include <numpy/arrayobject.h>
#include <mod2sparse.h>
#include <mod2dense.h>

PyArrayObject *as_type(PyArrayObject *array, int ntype);

PyObject *sparse_matrix_to_string(mod2sparse *matrix);
PyObject *dense_matrix_to_string(mod2dense *matrix);
mod2sparse * sparse_matrix_from_py_stream(PyObject * stream);
mod2sparse * dense_matrix_from_py_stream(PyObject * stream);

PyObject * int_array_to_string(int * array, int length);
int int_array_from_py_stream(PyObject * stream, int **array);


PyArrayObject *sparse_matrix_to_numpy_array(mod2sparse *matrix);
PyArrayObject *dense_matrix_to_numpy_array(mod2dense *matrix);


#endif //PYLDPCC_UTILS_H

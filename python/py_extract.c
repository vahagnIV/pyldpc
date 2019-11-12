//
// Created by vahagn on 10/31/19.
//
#include "py_extract.h"
#include "extract.h"
#include "py_gen_matrix.h"
#include "utils.h"
#include "py_matrix.h"
#include <numpy/arrayobject.h>

extern MatrixType;
extern GeneratorMatrixType;

PyObject *py_extract(PyObject *self, PyObject *args,
                     PyObject *kwnames) {
  import_array();

  PyMatrix *py_H;
  PyGeneratorMatrix *py_G;
  PyArrayObject *py_message, *py_result;
  static char *kwlist[] = {"H", "G", "message", NULL};
  PyArrayObject *char_message;
  int *dims;
  int py_message_dim_count;
  npy_intp *py_message_dims;
  int M, N, message_count, result;

  if (!PyArg_ParseTupleAndKeywords(args, kwnames, "O!O!O!", kwlist,
                                   &MatrixType,
                                   &py_H,
                                   &GeneratorMatrixType,
                                   &py_G,
                                   &PyArray_Type,
                                   &py_message)) {
    return NULL;
  }

  N = py_H->data->n_cols;
  M = py_H->data->n_rows;

  char_message = as_type(py_message, NPY_UINT8);
  py_message_dim_count = PyArray_NDIM(char_message);
  py_message_dims = PyArray_DIMS(py_message);

  dims = malloc(py_message_dim_count * sizeof(*dims));
  message_count = 1;
  for (int i = 0; i < py_message_dim_count - 1; ++i) {
    dims[i] = py_message_dims[i];
    message_count *= dims[i];
  }
  dims[py_message_dim_count - 1] = N - M;
  py_result = PyArray_FromDims(py_message_dim_count, dims, NPY_UINT8);

  free(dims);

  result = extract(py_G->data,
                   message_count,
                   PyArray_DATA(char_message),
                   PyArray_DATA(py_result));

  if (char_message != py_message)
    Py_DECREF(char_message);

  if (result)
    return py_result;

  Py_DECREF(py_result);
  return NULL;

}
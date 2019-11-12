//
// Created by vahagn on 10/29/19.
//
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include "py_encode.h"
#include "encode.h"
#include "py_gen_matrix.h"
#include "py_matrix.h"

extern PyTypeObject MatrixType;
extern PyTypeObject GeneratorMatrixType;

PyObject *py_encode(PyObject *self, PyObject *args,
                    PyObject *kwnames) {
  import_array();
  static char *kwlist[] = {"H", "G", "payload", NULL};
  PyMatrix *py_H;
  PyGeneratorMatrix *py_G;
  PyArrayObject *py_payload;
  int payload_dim_count, message_count, M, N;
  int i;

  if (!PyArg_ParseTupleAndKeywords(args, kwnames, "O!O!O!", kwlist,
                                   &MatrixType,
                                   &py_H,
                                   &GeneratorMatrixType,
                                   &py_G,
                                   &PyArray_Type,
                                   &py_payload)) {

    PyErr_SetString(PyExc_ValueError, "One or more argument(s) to function py_encode is invalid.");
    return NULL;
  }
  /*if (PyArray_DTYPE(py_payload)->kind != NPY_UINT8) {
    PyErr_SetString(PyExc_ValueError, "Payload should be numpy.uint8");
    return NULL;
  }*/

  char *payload_data = PyArray_DATA(py_payload);
  N = py_H->data->n_cols;
  M = py_H->data->n_rows;

  payload_dim_count = PyArray_NDIM(py_payload);
  npy_intp *payload_dims = PyArray_DIMS(py_payload);

  if (payload_dims[payload_dim_count - 1] != N - M) {
    char error_string[256];
    sprintf(error_string, "The last dimesnion of the message array should be %d - %d = %d.", N, M, N - M);
    PyErr_SetString(PyExc_ValueError, error_string);
    return NULL;
  }

  int *dims = malloc(payload_dim_count);

  message_count = 1;
  for (i = 0; i < payload_dim_count - 1; ++i) {
    message_count *= payload_dims[i];
    dims[i] = payload_dims[i];
  }

  dims[payload_dim_count - 1] = N;

  PyArrayObject *result = PyArray_FromDims(payload_dim_count, dims, NPY_UINT8);
  free(dims);

  int encoded_block_count = encode(py_H->data, py_G->data, message_count, payload_data, PyArray_DATA(result), 0);

//  free(dims);

  if (encoded_block_count != message_count) {
    Py_DECREF(result);
    return NULL;
  } else {
    return result;
  }

}
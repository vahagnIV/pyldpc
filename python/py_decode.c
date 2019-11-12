//
// Created by vahagn on 10/31/19.
//
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "py_decode.h"
#include "py_matrix.h"
#include "py_gen_matrix.h"
#include "utils.h"

extern MatrixType;
extern GeneratorMatrixType;

int initialize_decode(PyObject *module) {
  PyModule_AddObject(module, "CHANNEL_BSC", PyLong_FromLong(BSC));
  PyModule_AddObject(module, "CHANNEL_AWGN", PyLong_FromLong(AWGN));
  PyModule_AddObject(module, "CHANNEL_AWLN", PyLong_FromLong(AWLN));

  PyModule_AddObject(module, "DECODE_METHOD_Enum_block", PyLong_FromLong(Enum_block));
  PyModule_AddObject(module, "DECODE_METHOD_Enum_bit", PyLong_FromLong(Enum_bit));
  PyModule_AddObject(module, "DECODE_METHOD_Prprp", PyLong_FromLong(Prprp));
  return 0;
}

PyObject *py_decode(PyObject *self, PyObject *args,
                    PyObject *kwnames) {
  import_array();

  // Argument parameters
  PyMatrix *py_H;
  PyGeneratorMatrix *py_G;
  PyArrayObject *py_payload;

  channel_type channel;
  decoding_method method;
  double param;

  // Internal variables
  int i, N;
  PyArrayObject *py_double_payload;
  int encoded_message_count;

  double *encoded_messages;
  char *decoded_messages;
  BlockDecodeResult *block_result = 0;

  int payload_dim_count;
  npy_intp *payload_dims;
  int *decoded_dims;

  PyArrayObject *result;

  static char *kwlist[] = {"H", "G", "message", "channel_type", "decode_method", "param", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwnames, "O!O!O!iid", kwlist,
                                   &MatrixType,
                                   &py_H,
                                   &GeneratorMatrixType,
                                   &py_G,
                                   &PyArray_Type,
                                   &py_payload,
                                   &channel,
                                   &method,
                                   &param)) {
    return NULL;
  }

  payload_dim_count = PyArray_NDIM(py_payload);
  payload_dims = PyArray_DIMS(py_payload);

  N = py_H->data->n_cols;
  //M = py_H->data->n_rows;

  if (payload_dims[payload_dim_count - 1] != N) {
    char error_string[256];
    sprintf(error_string,
            "The last dimesnion of the encoded message array should be %d. The last dimesnion is %ld", N,
            payload_dims[payload_dim_count - 1]);
    PyErr_SetString(PyExc_ValueError, error_string);
    return NULL;
  }

  if ((py_double_payload = as_type(py_payload, NPY_FLOAT64)) == NULL)
    return NULL;

  decoded_dims = malloc(payload_dim_count * sizeof(int));
  encoded_message_count = 1;
  for (i = 0; i < payload_dim_count; ++i) {
    decoded_dims[i] = payload_dims[i];
    if (i != payload_dim_count - 1)
      encoded_message_count *= payload_dims[i];
  }

  result = PyArray_FromDims(payload_dim_count, decoded_dims, NPY_UINT8);
  free(decoded_dims);
  encoded_messages = PyArray_DATA(py_double_payload);
  decoded_messages = PyArray_DATA(result);

  if (decode(py_H->data,
             py_G->data,
             encoded_message_count,
             encoded_messages,
             decoded_messages,
             channel,
             method,
             param,
             block_result) != encoded_message_count) {
    Py_DECREF(result);
    result = NULL;
  }

  if (py_double_payload != py_payload)
    Py_DECREF(py_double_payload);
  return result;
}


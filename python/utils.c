//
// Created by vahagn on 11/1/19.
//
#include <Python.h>
#include "utils.h"
#include <alloc.h>

PyArrayObject *as_type(PyArrayObject *array, int ntype) {
  import_array();
  int i, dim_count;
  npy_intp *array_dims;
  int *dims;
  PyArrayObject *result;

  dim_count = PyArray_NDIM(array);
  array_dims = PyArray_DIMS(array);

  if (PyArray_DESCR(array)->type_num != ntype) {
    dims = malloc(dim_count * sizeof(int));
    for (i = 0; i < dim_count; ++i) {
      dims[i] = array_dims[i];
    }
    result = PyArray_FromDims(dim_count, dims, ntype);
    free(dims);
    if (PyArray_CopyInto(result, array) == -1) {
      Py_DecRef(result);
      result = NULL;
    }
  } else {
    result = array;
  }
  return result;
}

PyObject *sparse_matrix_to_string(mod2sparse *matrix) {
  char row_number[5];
  char col_number[5];
  int row, col, offset = 0;
  PyObject *result;
  sprintf(row_number, "%d", matrix->n_rows);
  sprintf(col_number, "%d", matrix->n_cols);

  int total_length = strlen(row_number) + strlen(col_number) + 2 + 2 * matrix->n_cols * matrix->n_rows;
  char *str_repr = calloc(total_length, sizeof(char));
  strcat(str_repr, row_number);
  strcat(str_repr, "\n");
  strcat(str_repr, col_number);
  strcat(str_repr, "\n");
  offset = strlen(str_repr);
  for (row = 0; row < matrix->n_rows; ++row) {
    for (col = 0; col < matrix->n_cols; ++col) {
      str_repr[offset++] = mod2sparse_find(matrix, row, col) ? '1' : '0';
      if (matrix->n_cols - col - 1)
        str_repr[offset++] = '\t';
    }
    str_repr[offset++] = '\n';
  }
  result = PyUnicode_FromString(str_repr);
  free(str_repr);
  return result;
}

PyObject *dense_matrix_to_string(mod2dense *matrix) {
  char row_number[5];
  char col_number[5];
  int row, col, offset = 0;
  PyObject *result;
  sprintf(row_number, "%d", matrix->n_rows);
  sprintf(col_number, "%d", matrix->n_cols);

  int total_length = strlen(row_number) + strlen(col_number) + 2 + 2 * matrix->n_cols * matrix->n_rows;
  char *str_repr = calloc(total_length, sizeof(char));
  strcat(str_repr, row_number);
  strcat(str_repr, "\n");
  strcat(str_repr, col_number);
  strcat(str_repr, "\n");
  offset = strlen(str_repr);
  for (row = 0; row < matrix->n_rows; ++row) {
    for (col = 0; col < matrix->n_cols; ++col) {
      str_repr[offset++] = mod2dense_get(matrix, row, col) ? '1' : '0';
      if (matrix->n_cols - col - 1)
        str_repr[offset++] = '\t';
    }
    str_repr[offset++] = '\n';
  }
  result = PyUnicode_FromString(str_repr);
  free(str_repr);
  return result;
}

PyArrayObject *sparse_matrix_to_numpy_array(mod2sparse *matrix) {
  import_array();
  PyArrayObject *result;
  if (matrix->n_rows && matrix->n_cols) {
    int dims[2] = {matrix->n_rows, matrix->n_cols};

    result = PyArray_FromDims(2, dims, NPY_UINT8);
    u_char *array_data = PyArray_DATA(result);
    int row, col;
    for (row = 0; row < dims[0]; ++row) {
      for (col = 0; col < dims[1]; ++col) {
        int exists = mod2sparse_find(matrix, row, col) != 0;
        array_data[row * dims[1] + col] = exists;
      }
    }

  } else {
    int dims[2] = {0, 0};
    result = PyArray_FromDims(2, dims, NPY_UINT8);
  }

  return result;
}

PyArrayObject *dense_matrix_to_numpy_array(mod2dense *matrix) {
  import_array();
  PyArrayObject *result;
  if (matrix->n_rows && matrix->n_cols) {
    int dims[2] = {matrix->n_rows, matrix->n_cols};

    result = PyArray_FromDims(2, dims, NPY_UINT8);
    u_char *array_data = PyArray_DATA(result);
    int row, col;
    for (row = 0; row < dims[0]; ++row) {
      for (col = 0; col < dims[1]; ++col) {
        int exists = mod2dense_get(matrix, row, col) != 0;
        array_data[row * dims[1] + col] = exists;
      }
    }

  } else {
    int dims[2] = {0, 0};
    result = PyArray_FromDims(2, dims, NPY_UINT8);
  }

  return result;
}

PyObject *int_array_to_string(int *array, int length) {
  char buffer[10]; // max int has 10 digits
  int i, total_length = length, offset = 0;
  char *result_str;
  PyObject *result;

  sprintf(buffer, "%d", length);
  total_length += strlen(buffer) + 1;

  for (i = 0; i < length; ++i) {
    sprintf(buffer, "%d", array[i]);
    total_length += strlen(buffer) + 1;
  }
  result_str = calloc(total_length, sizeof(char));

  sprintf(buffer, "%d", length);
  strcat(result_str, buffer);
  strcat(result_str, "\n");
  for (i = 0; i < length; ++i) {
    sprintf(buffer, "%d", array[i]);
    strcat(result_str, buffer);
    if (i != length - 1)
      strcat(result_str, "\t");
  }
  strcat(result_str, "\n");
  result = PyUnicode_FromString(result_str);
  free(result_str);
  return result;

}

void *matrix_from_py_stream(PyObject *stream, int is_sparse) {
  PyObject *tmp;
  char *line;
  int M, N, row, col, row_length;
  void *result;
  tmp = PyFile_GetLine(stream, 0);
  M = atoi(PyUnicode_AsUTF8(tmp));
  Py_DECREF(tmp);

  tmp = PyFile_GetLine(stream, 0);
  N = atoi(PyUnicode_AsUTF8(tmp));
  Py_DECREF(tmp);

  result = is_sparse ? mod2sparse_allocate(M, N) : mod2dense_allocate(M, N);

  for (row = 0; row < M; ++row) {
    tmp = PyFile_GetLine(stream, 0);
    row_length = PyUnicode_GetLength(tmp);
    line = PyUnicode_AsUTF8(tmp);
    if ((line[row_length - 1] == '\n' && row_length != 2 * N)
        || (line[row_length - 1] != '\n' && row_length != 2 * N - 1)) {
      PyErr_SetString(PyExc_ValueError, "Invalid matrix format or currupted file");
      Py_DECREF(tmp);
      is_sparse ? mod2sparse_free(result) : mod2dense_free(result);
      return NULL;
    }

    for (col = 0; col < N; ++col) {
      if (line[2 * col] == '1')
        is_sparse ? mod2sparse_insert(result, row, col) : mod2dense_set(result, row, col, 1);
      else if (line[2 * col] != '0') {
        Py_DECREF(tmp);
        is_sparse ? mod2sparse_free(result) : mod2dense_free(result);
        PyErr_SetString(PyExc_ValueError, "Invalid matrix format or currupted file");
        return NULL;
      } else if (is_sparse == 0) {
        mod2dense_set(result, row, col, 0);
      }
    }
    Py_DECREF(tmp);
  }
  return result;

}

mod2sparse *sparse_matrix_from_py_stream(PyObject *stream) {
  return matrix_from_py_stream(stream, 1);
}

mod2sparse *dense_matrix_from_py_stream(PyObject *stream) {
  return matrix_from_py_stream(stream, 0);
}

int int_array_from_py_stream(PyObject *stream, int **array) {
  PyObject *tmp;
  char buffer[10];
  char *line;
  int count = 0, offset = 0, current_pos, index = 0;

  tmp = PyFile_GetLine(stream, 0);
  if (tmp == NULL) {
    PyErr_SetString(PyExc_ValueError, "Invalid input");
    return -1;
  }
  count = atoi(PyUnicode_AsUTF8(tmp));

  *array = malloc(count * sizeof(int));

  tmp = PyFile_GetLine(stream, 0);
  line = PyUnicode_AsUTF8(tmp);

  for (current_pos = 0; current_pos < strlen(line); ++current_pos) {
    if (line[current_pos] == '\t' || line[current_pos] == '\n' || current_pos == strlen(line) - 1) {
      memcpy(buffer, line + offset, current_pos - offset);
      buffer[current_pos - offset] = 0;
      (*array)[index++] = atoi(buffer);
      offset = current_pos;
    }
  }
  return count;
}
//
// Created by vahagn on 10/25/19.
//
#include <numpy/arrayobject.h>
#include <make_ldpc.h>
#include <py_matrix.h>
#include "py_make_ldpc.h"

PyObject *py_make_ldpc(PyObject *self, PyObject *args,
                       PyObject *kwnames) {
  import_array();

  static char *kwlist[] = {"method", "seed", "M", "N", "method_args", "no4cycle", NULL};

  make_method m_method;
  int M, N;
  int seed;
  int no4cycle = 0;
  distrib *d;
  char *c = "3x3";

  if (!PyArg_ParseTupleAndKeywords(args, kwnames, "iiii|si", kwlist,
                                   &m_method, &seed, &M, &N, &c, &no4cycle)) {
    PyErr_SetString(PyExc_ValueError, "One or more argument(s) to function make_ldpc is invalid.");
    return NULL;
  }

  if (m_method != Evencol && m_method != Evenboth) {
    PyErr_SetString(PyExc_ValueError, "The method argument of make_ldpc should be one of PCHK_METHOD_<> constants");
    return NULL;
  }

  d = distrib_create(c);
  if (d == 0) {
    PyErr_SetString(PyExc_ValueError, "Invalid method args");
    return NULL;
  }

  PyMatrix *new_matrix = create_py_empty_matrix();
  new_matrix->data = make_ldpc(seed, m_method, M, N, d, no4cycle);
  Py_INCREF(new_matrix);
  return new_matrix;
}

int initialize_make_ldpc(PyObject *module) {
  PyModule_AddObject(module, "PCHK_METHOD_EVENCOL", PyLong_FromLong(Evencol));
  PyModule_AddObject(module, "PCHK_METHOD_EVENBOTH", PyLong_FromLong(Evenboth));
}


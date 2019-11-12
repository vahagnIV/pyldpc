//
// Created by vahagn on 10/25/19.
//
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <numpy/arrayobject.h>
#include <mod2sparse.h>
#include "py_make_gen.h"
#include "make_gen.h"
#include "py_matrix.h"
#include "py_gen_matrix.h"

extern PyTypeObject MatrixType;

int initialize_make_gen(PyObject *module) {
  PyModule_AddObject(module, "GEN_STRATEGY_MINPROD", PyLong_FromLong(Mod2sparse_minprod));
  PyModule_AddObject(module, "GEN_STRATEGY_FIRST", PyLong_FromLong(Mod2sparse_first));
  PyModule_AddObject(module, "GEN_STRATEGY_MINCOL", PyLong_FromLong(Mod2sparse_mincol));

  PyModule_AddObject(module, "GEN_METHOD_SPARSE", PyLong_FromLong(Sparse));
  PyModule_AddObject(module, "GEN_METHOD_DENSE", PyLong_FromLong(Dense));
  PyModule_AddObject(module, "GEN_METHOD_MIXED", PyLong_FromLong(Mixed));
  return 0;
}

PyObject *py_make_gen(PyObject *self, PyObject *args,
                      PyObject *kwnames) {
  import_array();

  static char *kwlist[] = {"method", "H", "strategy", "abandon_number", "abandon_when", NULL};
  PyMatrix *py_H;
  MatrixRepresentation method;
  mod2sparse_strategy strategy;

  int abandon_number, abandon_when;

  printf("Name: %s\n", MatrixType.tp_name);

  if (!PyArg_ParseTupleAndKeywords(args, kwnames, "iO!iii", kwlist,
                                   &method,
                                   &MatrixType,
                                   &py_H, &strategy,
                                   &abandon_number,
                                   &abandon_when)) {

    return NULL;
  }

  //TODO: perfom checks on enums;

  PyGeneratorMatrix *m = create_py_empty_generator_matrix();

  m->data = make_generator(method, ((PyMatrix *) py_H)->data, strategy, abandon_number, abandon_when);

  return (PyObject *) m;
}





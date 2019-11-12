//
// Created by vahagn on 10/25/19.
//

#ifndef PYLDPCC_PY_MAKE_GEN_H
#define PYLDPCC_PY_MAKE_GEN_H

#include <Python.h>

PyObject * py_make_gen(PyObject *self, PyObject *args,
                       PyObject *kwnames);

int initialize_make_gen(PyObject *module);

#endif //PYLDPCC_PY_MAKE_GEN_H

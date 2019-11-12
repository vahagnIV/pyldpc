//
// Created by vahagn on 10/25/19.
//

#ifndef PYLDPCC_PY_MAKE_LDPC_H
#define PYLDPCC_PY_MAKE_LDPC_H
#include <Python.h>

PyObject * py_make_ldpc(PyObject *self, PyObject *args,
                        PyObject *kwnames);
int initialize_make_ldpc(PyObject *module);

#endif //PYLDPCC_PY_MAKE_LDPC_H

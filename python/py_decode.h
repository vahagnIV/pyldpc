//
// Created by vahagn on 10/31/19.
//

#ifndef PYLDPCC_PY_DECODE_H
#define PYLDPCC_PY_DECODE_H

#include <Python.h>
#include "decode.h"

PyObject *py_decode(PyObject *self, PyObject *args,
                    PyObject *kwnames);

int initialize_decode(PyObject *module);

#endif //PYLDPCC_PY_DECODE_H

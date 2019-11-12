//
// Created by vahagn on 10/25/19.
//
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "py_matrix.h"
#include <numpy/arrayobject.h>
#include "mod2sparse.h"
#include "utils.h"

void PyMatrixDeallocator(PyMatrix *self);
int PyMatrixConstructor(PyMatrix *self, PyObject *args, PyObject *kwds);
PyObject *MatrixAllocator(PyTypeObject *type, PyObject *args, PyObject *kwds);
PyArrayObject *ToNumpyArray(PyMatrix *self, PyObject *Py_Unused);
PyObject *PyMatrixSave(PyMatrix *self, PyObject *args);
PyObject *PyMatrix_to_string(PyMatrix *self);

static PyMethodDef Matrix_methods[] = {
    {"ToMat", (PyCFunction) ToNumpyArray, METH_NOARGS,
     "Convert to numpy array"
    },
    {"Save", (PyCFunction) PyMatrixSave, METH_VARARGS, ""},
    {NULL}  /* Sentinel */
};

PyObject *PyMatrixSave(PyMatrix *self, PyObject *args) {
  PyObject *stream;
  if (!PyArg_ParseTuple(args, "O", &stream)) {
    return NULL;
  }
  PyFile_WriteObject(self, stream, 0);

  Py_RETURN_NONE;
}

PyTypeObject MatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pyLdpc_internal.Matrix",
    .tp_doc = "Matrix",
    .tp_basicsize = sizeof(PyMatrix),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = MatrixAllocator,
    .tp_init = (initproc) PyMatrixConstructor,
    .tp_dealloc = (destructor) PyMatrixDeallocator,
    .tp_members = NULL,
    .tp_methods = Matrix_methods,
    .tp_repr = (reprfunc) PyMatrix_to_string
};

PyMatrix *create_py_empty_matrix() {
  // Use Python Api to create a new object
  PyObject *matrix_module = PyImport_ImportModule("pyLdpc_internal");
  if (matrix_module == NULL)
    return NULL;

  PyObject *matrix_type = PyObject_GetAttrString(matrix_module, "Matrix");

  PyMatrix *new_matrix = (PyMatrix *) PyObject_CallObject(matrix_type, NULL);
  Py_DECREF(matrix_module);
  Py_DECREF(matrix_type);
  return new_matrix;

}

void PyMatrixDeallocator(PyMatrix *self) {
  mod2sparse_free(self->data);
  Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject *MatrixAllocator(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  PyMatrix *self;
  self = (PyMatrix *) type->tp_alloc(type, 0);
  self->data = NULL;
  return (PyObject *) self;
}

int PyMatrixConstructor(PyMatrix *self, PyObject *args, PyObject *kwds) {
  static char *kwlist[] = {"stream", NULL};
  PyObject *stream = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist,
                                   &stream))
    return -1;
  if (stream)
    if ((self->data = sparse_matrix_from_py_stream(stream)) == NULL)
      return 0;
  return 0;
}

PyArrayObject *ToNumpyArray(PyMatrix *self, PyObject *Py_Unused) {
  return sparse_matrix_to_numpy_array(self->data);
}

PyObject *PyMatrix_to_string(PyMatrix *self) {
  return sparse_matrix_to_string(self->data);
}

int initialize_matrix_constants(PyObject *module) {
  if (PyType_Ready(&MatrixType) < 0)
    return -1;
  Py_INCREF(&MatrixType);
  if (PyModule_AddObject(module, "Matrix", (PyObject *) &MatrixType) < 0) {
    Py_DECREF(&MatrixType);
    Py_DECREF(module);
    return -1;
  }

  return 0;
}

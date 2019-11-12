//
// Created by vahagn on 10/29/19.
//
#include "py_gen_matrix.h"
#include "utils.h"

void PyGenMatrixDeallocator(PyGeneratorMatrix *self);
int PyGenMatrixConstructor(PyGeneratorMatrix *self, PyObject *args, PyObject *kwds);
PyObject *PyGenMatrixAllocator(PyTypeObject *type, PyObject *args, PyObject *kwds);
PyObject *PyGenToNumpyArray(PyGeneratorMatrix *self, PyObject *Py_Unused);
PyObject *PyGen_to_string(PyGeneratorMatrix *self);
PyObject *PyGenMatrixSave(PyGeneratorMatrix *self, PyObject *args);

static PyMethodDef PyGenMatrix_methods[] = {
    {"ToMat", (PyCFunction) PyGenToNumpyArray, METH_NOARGS,
     "Convert to numpy array"},
    {"Save", (PyCFunction) PyGenMatrixSave, METH_VARARGS,
     "Convert to numpy array"},
    {NULL}  /* Sentinel */
};

PyTypeObject GeneratorMatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pyLdpc_internal.GeneratorMatrix",
    .tp_doc = "Matrix",
    .tp_basicsize = sizeof(PyGeneratorMatrix),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyGenMatrixAllocator,
    .tp_init = (initproc) PyGenMatrixConstructor,
    .tp_dealloc = (destructor) PyGenMatrixDeallocator,
    .tp_members = NULL,
    .tp_methods = PyGenMatrix_methods,
    .tp_repr = (reprfunc) PyGen_to_string,
};

PyGeneratorMatrix *create_py_empty_generator_matrix() {
  // TODO: this is possible to implement purely on the C side of the code
  PyObject *matrix_module = PyImport_ImportModule("pyLdpc_internal");
  if (matrix_module == NULL)
    return NULL;

  PyObject *matrix_type = PyObject_GetAttrString(matrix_module, "GeneratorMatrix");

  PyGeneratorMatrix *new_matrix = (PyGeneratorMatrix *) PyObject_CallObject(matrix_type, NULL);

  Py_DECREF(matrix_module);
  Py_DECREF(matrix_type);
  return new_matrix;

}

MatrixRepresentation type_from_string(char *str) {
  if (0 == strcmp(str, "Sparse\n"))
    return Sparse;
  if (0 == strcmp(str, "Dense\n"))
    return Dense;
  if (0 == strcmp(str, "Mixed\n"))
    return Mixed;
  return -1;
}

void PyGenMatrixDeallocator(PyGeneratorMatrix *self) {

  deallocate_generator_matrix(self->data);

  Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject *PyGenMatrixAllocator(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  PyGeneratorMatrix *self;
  self = (PyGeneratorMatrix *) type->tp_alloc(type, 0);
  self->data = calloc(1, sizeof(*(self->data)));

  return (PyObject *) self;
}

int PyGenMatrixConstructor(PyGeneratorMatrix *self, PyObject *args, PyObject *kwds) {
  static char *kwlist[] = {"stream", NULL};
  PyObject *stream = 0;
  PyObject *tmp;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist,
                                   &stream))
    return -1;
  if (stream) {
    tmp = PyFile_GetLine(stream, 0);
    self->data->type = type_from_string(PyUnicode_AsUTF8(tmp));
    Py_DECREF(tmp);

    switch (self->data->type) {
      case Sparse:
        self->data->U = sparse_matrix_from_py_stream(stream);
        self->data->L = sparse_matrix_from_py_stream(stream);
        break;
      case Dense:
      case Mixed:
        self->data->G = dense_matrix_from_py_stream(stream);
        break;
      default:
        return -1;
    }

    self->data->N = int_array_from_py_stream(stream, &self->data->column_ordering);
    self->data->M = int_array_from_py_stream(stream, &self->data->row_ordering);

  }

  return 0;
}

PyObject *PyGenToNumpyArray(PyGeneratorMatrix *self, PyObject *Py_Unused) {
  import_array();
  // TODO: apply column and row ordering
  PyObject *result;
  if (self->data->type == Sparse && self->data->L && self->data->U) {
    result = PyTuple_New(2);
    PyTuple_SetItem(result, 0, sparse_matrix_to_numpy_array(self->data->L));
    PyTuple_SetItem(result, 1, sparse_matrix_to_numpy_array(self->data->U));
  } else {
    return dense_matrix_to_numpy_array(self->data->G);
  }
  return result;
}

int initialize_generator_matrix_constants(PyObject *module) {
  if (PyType_Ready(&GeneratorMatrixType) < 0)
    return -1;
  Py_INCREF(&GeneratorMatrixType);
  if (PyModule_AddObject(module, "GeneratorMatrix", (PyObject *) &GeneratorMatrixType) < 0) {
    Py_DECREF(&GeneratorMatrixType);
    Py_DECREF(module);
    return -1;
  }

  return 0;
}

PyObject *PyGenSparse_to_string(PyGeneratorMatrix *self) {
  PyObject *U_str = sparse_matrix_to_string(self->data->U);
  PyObject *L_str = sparse_matrix_to_string(self->data->L);
  PyObject *result = PyUnicode_Concat(L_str, U_str);
  Py_DECREF(U_str);
  Py_DECREF(L_str);
  return result;
}

PyObject *type_to_string(MatrixRepresentation type) {
  switch (type) {
    case Sparse:
      return PyUnicode_FromString("Sparse\n");
    case Mixed:
      return PyUnicode_FromString("Mixed\n");
    case Dense:
      return PyUnicode_FromString("Dense\n");
    default:
      return NULL;
  }
}

PyObject *PyGenMatrixSave(PyGeneratorMatrix *self, PyObject *args) {
  PyObject *stream;
  if (!PyArg_ParseTuple(args, "O", &stream)) {
    return NULL;
  }
  PyFile_WriteObject(self, stream, 0);

  Py_RETURN_NONE;
}

PyObject *PyGen_to_string(PyGeneratorMatrix *self) {
  PyObject *result;

  PyObject *py_type, *tmp, *py_cols, *py_rows;

  switch (self->data->type) {
    case Sparse:
      tmp = PyGenSparse_to_string(self);
      break;
    case Mixed:
    case Dense:
      tmp = dense_matrix_to_string(self->data->G);
      break;
    default:
      return NULL;
  }
  py_type = type_to_string(self->data->type);
  result = PyUnicode_Concat(py_type, tmp);
  Py_DECREF(tmp);

  tmp = result;
  py_cols = int_array_to_string(self->data->column_ordering, self->data->N);
  result = PyUnicode_Concat(tmp, py_cols);
  Py_DECREF(tmp);
  Py_DECREF(py_cols);

  tmp = result;
  py_rows = int_array_to_string(self->data->row_ordering, self->data->M);
  result = PyUnicode_Concat(tmp, py_rows);
  Py_DECREF(tmp);
  Py_DECREF(py_rows);
  return result;
}
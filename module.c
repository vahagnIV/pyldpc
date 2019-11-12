//
// Created by vahagn on 10/25/19.
//

#include <Python.h>
#include <stdio.h>
#include <py_gen_matrix.h>
#include "py_matrix.h"
#include "py_make_ldpc.h"
#include "py_make_gen.h"
#include "py_encode.h"
#include "py_decode.h"
#include "py_extract.h"

static PyMethodDef
    MakeLdpcMethods[] = {{"make_ldpc", py_make_ldpc, METH_VARARGS | METH_KEYWORDS, "Generate ldpc matrix."},
                         {"make_gen", py_make_gen, METH_VARARGS | METH_KEYWORDS, "Generate generator matrix. "},
                         {"encode", py_encode, METH_VARARGS | METH_KEYWORDS, "Encode data."},
                         {"decode", py_decode, METH_VARARGS | METH_KEYWORDS, "Decode data."},
                         {"extract", py_extract, METH_VARARGS | METH_KEYWORDS, "Extract data."},
                         {NULL, NULL, 0, NULL}};

static struct PyModuleDef pyLdpcModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "pyLdpc_internal",//MODULE_NAME,   /* name of module */
    .m_doc =NULL, /* module documentation, may be NULL */
    .m_size = -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    .m_methods = MakeLdpcMethods
};

PyMODINIT_FUNC PyInit_pyLdpc_internal(void) {

  PyObject *m;

  m = PyModule_Create(&pyLdpcModule);
  if (m == NULL)
    return NULL;
  if (initialize_matrix_constants(m))
    return NULL;
  if (initialize_generator_matrix_constants(m))
    return NULL;
  if (initialize_make_ldpc(m))
    return NULL;
  if (initialize_make_gen(m))
    return NULL;
  if (initialize_decode(m))
    return NULL;

  return m;

}
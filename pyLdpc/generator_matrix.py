import pyLdpc_internal

class GeneratorMatrix:
    def __init__(self, file=None):
        if (file):
            setattr(self, '__internal_matrix',  pyLdpc_internal.GeneratorMatrix(file))
        else:
            setattr(self, '__internal_matrix',  None)

    def Save(self, file):
        if file and hasattr(self, '__internal_matrix'):
            getattr(self, '__internal_matrix').Save(file)

    def __str__(self):
        return getattr(self,'__internal_matrix').__str__() if hasattr(self, '__internal_matrix') else ''

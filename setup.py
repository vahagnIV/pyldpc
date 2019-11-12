from setuptools import Extension
from distutils.core import setup, Extension

import os

src_files = [os.path.join(os.path.dirname(__file__), 'src', f) for f in os.listdir('src') if f.endswith('.c')]

src_files.extend(
    [os.path.join(os.path.dirname(__file__), 'python', f) for f in os.listdir('python') if f.endswith('.c')])

src_files.append('module.c')

module = Extension(name='pyLdpc_internal',
                   define_macros=[('MAJOR_VERSION', '1'),
                                  ('MINOR_VERSION', '0')],
                   include_dirs=['/usr/local/include', './src', './python'],
                   libraries=['m'],
                   library_dirs=['/usr/local/lib'],
                   sources=src_files)

setup(name='pyLdpc',
      version='1.0',
      description='This is a demo package',
      author='Vahagn',
      include_package_data=True,
      author_email='vahagn@nvision.am',
      packages=['pyLdpc'],
      long_description='''
A python wrapper for ldpc codes.
''',
      ext_modules=[module])

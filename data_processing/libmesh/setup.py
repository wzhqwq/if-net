from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

setup(name = 'libmesh',
      ext_modules = cythonize(
          Extension(
              "triangle_hash",
              sources=["triangle_hash.pyx"],
              include_dirs=[np.get_include()]
          )
      ),
      install_requires = ['numpy'])

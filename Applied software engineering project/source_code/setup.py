from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("dynamics_cython.pyx")
)

setup(
    ext_modules = cythonize("dynamics_async_cython.pyx")
)
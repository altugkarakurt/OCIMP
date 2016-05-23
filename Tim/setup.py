from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from distutils.extension import Extension

sources_list = ["timgraph.pyx", "Graph.cpp"]

setup(ext_modules=[Extension("pytim", sources=sources_list,language="c++")], cmdclass={'build_ext':build_ext})
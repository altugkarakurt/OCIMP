from libcpp.string cimport string

cdef extern from "Graph.h":
	cdef cppclass _CGraph "Graph":
		_CGraph(string, int, int, int, string) except +

cdef class PyGraph:
	cdef _CGraph *thisptr

	def __cinit__(self, string graph_file, int node_cnt, int edge_cnt, int seed_size, string model):
		self.thisptr = new _CGraph(graph_file, node_cnt, edge_cnt, seed_size, model)

	def __dealloc__(self):
		del self.thisptr

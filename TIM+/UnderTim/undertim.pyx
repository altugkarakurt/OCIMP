from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.set cimport set

cdef extern from "Graph.h":
	cdef cppclass Graph:
		Graph(string, int, int, int, string, vector[int]) except +

cdef extern from "InfGraph.h":
	cdef cppclass InfGraph(Graph):
		InfGraph(string, int, int, int, string, vector[int]) except +

cdef extern from "TimGraph.h":
	cdef cppclass TimGraph(InfGraph):
		TimGraph(string, int, int, int, string, vector[int]) except +
		void EstimateOPT(double)
		double InfluenceHyperGraph()
		set[int] seedSet

cdef class PyUnderGraph:
	cdef Graph *thisptr

	def __cinit__(self, string graph_file, int node_cnt, int edge_cnt, int seed_size, string model, vector[int] banned_nodes):
		self.thisptr = new Graph(graph_file, node_cnt, edge_cnt, seed_size, model, banned_nodes)

	def __dealloc__(self):
		del self.thisptr

cdef class PyUnderInfGraph(PyUnderGraph):
	cdef InfGraph *thisinfptr

	def __cinit__(self, string graph_file, int node_cnt, int edge_cnt, int seed_size, string model, vector[int] banned_nodes):
		self.thisinfptr = new InfGraph(graph_file, node_cnt, edge_cnt, seed_size, model, banned_nodes)

	def __dealloc__(self):
		del self.thisinfptr

cdef class PyUnderTimGraph(PyUnderInfGraph):
	cdef TimGraph *thistimptr

	def __cinit__(self, string graph_file, int node_cnt, int edge_cnt, int seed_size, string model, vector[int] banned_nodes):
		self.thistimptr = new TimGraph(graph_file, node_cnt, edge_cnt, seed_size, model, banned_nodes)

	def __dealloc__(self):
		del self.thistimptr

	def get_seed_set(self, double epsilon=0.1):
		self.thistimptr.EstimateOPT(epsilon)
		return self.thistimptr.seedSet

	def get_influence(self):
		return self.thistimptr.InfluenceHyperGraph()
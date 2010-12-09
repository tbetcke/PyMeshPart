%module mesh
%{
#define SWIG_FILE_WITH_INIT
#include "swig_headers.h"
%}
%include "numpy.i"
%init %{
import_array();
%}

%pythoncode %{
import numpy
%}

%apply (int* IN_ARRAY1, int DIM1) { (int *elmnts, int nelems ) };
%apply (int* ARGOUT_ARRAY1, int DIM1) { (int *epart, int neparts ), (int *npart, int nnparts ) };
%apply int *OUTPUT { int *edgecut };

%rename (part_mesh_dual) my_metis_partmeshdual;
%inline %{

void my_metis_partmeshdual(int nn, int ne, int *elmnts, int nelems, int etype, int numflag, int nparts, int *edgecut, int *epart, int neparts, int *npart, int nnparts){
	if ((ne!=neparts)||(nn!=nnparts)){
		PyErr_Format(PyExc_ValueError,
		"Out array lengts are not equal to input array lengths");
		}
	METIS_PartMeshDual(&ne,&nn,elmnts,&etype,&numflag,&nparts,edgecut,epart,npart);
	
}
%}

%pythoncode %{
def part_mesh_dual(mesh,nnodes,elemtype,nparts):
	"""Partition elements of a mesh with Metis via the dual graph
	
		(epart,npart,edgecut)=part_mesh_dual(mesh,nnodes,elemtype,nparts)
		
		INPUT:
		mesh     - List of lists of nodes describing the elements, e.g.
		           mesh=[[1,2,3],[2,3,4],[0,3,4]]. Nodes are counted from zero
		nnodes   - Number of nodes in the mesh
		elemtype - Type of the elements: 1-Triangles, 2-Tetraheda,
		                                 3-Hexahedra, 4-Quadrilaterals
		nparts   - Number of partitions		
	
		OUTPUT:
		epart    - List assigning each element a partition id
		nparts   - List assigning each node a partition id
		edgecut  - Number of edges cut by the partitioning
	
	"""
	
	nelems=len(mesh)
	m=numpy.array(mesh,dtype='i').flatten()
	(edgecut,epart,npart)=_mesh.part_mesh_dual(nnodes,nelems,m,elemtype,0,nparts,nelems,nnodes)
	return (epart,npart,edgecut)
%}	


%rename (part_mesh_nodal) my_metis_partmesh_nodal;
%inline %{

void my_metis_partmesh_nodal(int nn, int ne, int *elmnts, int nelems, int etype, int numflag, int nparts, int *edgecut, int *epart, int neparts, int *npart, int nnparts){
	if ((ne!=neparts)||(nn!=nnparts)){
		PyErr_Format(PyExc_ValueError,
		"Out array lengts are not equal to input array lengths");
		}
	METIS_PartMeshNodal(&ne,&nn,elmnts,&etype,&numflag,&nparts,edgecut,epart,npart);
	
}
%}

%pythoncode %{
def part_mesh_nodal(mesh,nnodes,elemtype,nparts):
	"""Partition elements of a mesh with Metis via nodal graph
	
		(epart,npart,edgecut)=part_mesh_nodal(mesh,nnodes,elemtype,nparts)
		
		INPUT:
		mesh     - List of lists of nodes describing the elements, e.g.
		           mesh=[[1,2,3],[2,3,4],[0,3,4]]. Nodes are counted from zero
		nnodes   - Number of nodes in the mesh
		elemtype - Type of the elements: 1-Triangles, 2-Tetraheda,
		                                 3-Hexahedra, 4-Quadrilaterals
		nparts   - Number of partitions		
	
		OUTPUT:
		epart    - List assigning each element a partition id
		nparts   - List assigning each node a partition id
		edgecut  - Number of edges cut by the partitioning
	
	"""
	
	nelems=len(mesh)
	m=numpy.array(mesh,dtype='i').flatten()
	(edgecut,epart,npart)=_mesh.part_mesh_nodal(nnodes,nelems,m,elemtype,0,nparts,nelems,nnodes)
	return (epart,npart,edgecut)
%}



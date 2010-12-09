%module graph
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

%apply (int* IN_ARRAY1, int DIM1) { (int *xadj, int nxadj ), (int *adjncy, int nadjncy) };
%apply (int* ARGOUT_ARRAY1, int DIM1) { (int *perm, int nperm ), (int *iperm, int niperm ) };
%apply (int IN_ARRAY1[ANY]) { int options[5], int options[8] };


%rename (metis_edge_nd) my_metis_edge_nd;
%inline %{

void my_metis_edge_nd(int n, int *xadj, int nxadj, int *adjncy, int nadjncy, int numflag, int options[5], int *perm, int nperm, int *iperm, int niperm){

        METIS_EdgeND(&n, xadj, adjncy, &numflag, options, perm, iperm);
}
%}
        
%pythoncode %{
def metis_edge_nd(xadj, adjncy, numflag=0, options=None):
        """Fill reducing ordering of a sparse matrix

           (perm, iperm)=metis_edge_nd(xadj, adjncy, numflag, options)

           For a description of the parameters see the METIS Manual

        """

        n=len(xadj)-1
        if options is None: options=[0,0,0,0,0]
        
        (perm,iperm)=_mesh.metis_edge_nd(n,xadj,adjncy,numflag,options,n,n)
        return (perm,iperm)
%}


%rename (metis_node_nd) my_metis_node_nd;
%inline %{

void my_metis_node_nd(int n, int *xadj, int nxadj, int *adjncy, int nadjncy, int numflag, int options[8], int *perm, int nperm, int *iperm, int niperm){

        METIS_NodeND(&n, xadj, adjncy, &numflag, options, perm, iperm);
}
%}
        
%pythoncode %{
def metis_node_nd(xadj, adjncy, numflag=0, options=None):
        """Fill reducing ordering of a sparse matrix

           (perm, iperm)=metis_node_nd(xadj, adjncy, numflag, options)

           For a description of the parameters see the METIS Manual

        """

        n=len(xadj)-1
        if options is None: options=[0,0,0,0,0,0,0,0]
        
        (perm,iperm)=_mesh.metis_node_nd(n,xadj,adjncy,numflag,options,n,n)
        return (perm,iperm)
%}

%pythoncode %{

if __name__ == "__main__":
        xadj=[0,2,5,8, 11, 13, 16, 20, 24, 28, 31, 33, 36, 39, 42, 44]
        adjncy=[1,5,0,2,6,1,3,7,2,4,8,3,9,0,6,10,1,5,7,11,2,6,8,12,3,7,9,13,4,8,14, 5, 11, 6, 10, 12, 7, 11, 13, 8, 12, 14, 9, 13]
        (perm,iperm)=metis_node_nd(xadj,adjncy)
        print perm, iperm

%}



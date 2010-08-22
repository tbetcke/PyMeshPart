if __name__ == "__main__":
    import core
    import numpy
    a=numpy.array(
       [[1,2,3],
       [2,4,6],
       [2,6,3],
       [4,5,6],
       [5,6,3]],dtype='i')-1
    (epart,npart,edgecut)=core.tri_mesh(a,6,2)
    print epart
    print npart
    print edgecut
    
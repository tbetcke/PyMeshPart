from distutils.core import setup, Extension
from numpy import get_include as np_include

src_dir='src/'
src_files=['mesh_swig.i','balance.c','bucketsort.c','ccgraph.c',
           'coarsen.c','compress.c','debug.c',
           'estmem.c','fm.c','fortran.c','frename.c',
           'graph.c','initpart.c','kmetis.c','kvmetis.c',
           'kwayfm.c','kwayrefine.c','kwayvolfm.c','kwayvolrefine.c',
           'match.c','mbalance.c','mbalance2.c','mcoarsen.c','memory.c',
           'mesh.c','meshpart.c','mfm.c','mfm2.c','mincover.c','minitpart.c',
           'minitpart2.c','mkmetis.c','mkwayfmh.c','mkwayrefine.c',
           'mmatch.c','mmd.c','mpmetis.c','mrefine.c','mrefine2.c',
           'mutil.c','myqsort.c','ometis.c','parmetis.c','pmetis.c',
           'pqueue.c','refine.c','separator.c','sfm.c','srefine.c',
           'stat.c','subdomains.c','timing.c','util.c']

src_files=[src_dir+object for object in src_files]

setup(name='PyMeshPart',
      version='0.1',
      description='Mesh Partitioner for Python based on Metis',
      author='Timo Betcke',
      author_email='timo.betcke@gmail.com',
      packages=['pymeshpart'],
      ext_package='pymeshpart',
      ext_modules=[Extension('_mesh',
                src_files,
                swig_opts=['-outdir','pymeshpart'],
                include_dirs=['src/',np_include()]
                )]
     )

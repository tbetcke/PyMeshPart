#include "Python.h"
#include "numpy/arrayobject.h"
#include "metis.h"

static PyObject *tri_mesh(PyObject *self, PyObject *args);
/*static PyObject *tet_mesh(PyObject *self, PyObject *args); */

static PyMethodDef coreMethods[] = {
		{"tri_mesh", tri_mesh, METH_VARARGS}
		/*{"tet_mesh", tet_mesh, METH_VARARGS}*/
};

void initcore() {
      (void) Py_InitModule("core", coreMethods);
      import_array(); // Must be present for NumPy. Called first after above line.
}

static PyObject *tri_mesh(PyObject *self, PyObject *args){

	PyArrayObject *mesh;
	PyArray_Descr *descr;
	int ne,nn;
	int *elmnts;
	int etype=1;
	int numflag=0;
	int nparts;
	int edgecut;
	int *epart;
	int *npart;
	PyArrayObject *epart_array;
	PyArrayObject *npart_array;
	npy_intp epart_length,npart_length;
	int i;

	if (!PyArg_ParseTuple(args, "O!ii",
	          &PyArray_Type,&mesh,&nn,&nparts)) return NULL;

	descr=mesh->descr;


	if (mesh->nd!=2) return NULL;
	if ((mesh->dimensions[1]%3!=0)||(mesh->dimensions[1]==0)) return NULL;
	if (descr->type_num!=NPY_INT) return NULL;

	ne=mesh->dimensions[0];
	elmnts= (int*) PyArray_DATA(mesh);

	epart_length=ne;
	npart_length=nn;
	epart_array= (PyArrayObject *) PyArray_SimpleNew(1,&epart_length,NPY_INT);
	npart_array= (PyArrayObject *) PyArray_SimpleNew(1,&npart_length,NPY_INT);
	epart= (int*) PyArray_DATA(epart_array);
	npart= (int*) PyArray_DATA(npart_array);

	METIS_PartMeshDual(&ne, &nn, elmnts, &etype, &numflag, &nparts, &edgecut, epart, npart);

	return Py_BuildValue("(OOi)", epart_array,npart_array,edgecut);
}

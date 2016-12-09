#include "PythonUtils.h"
#include <iostream>
using std::cout; using std::endl;


void AddPathToPythonSys(std::string path)
{
	// now insert the current working directory into the python path so module search can take advantage
	// this must happen after python has been initialised
	
	//cout << "adding to python path: \"" << path << "\"" << endl;
	PyObject* sysPath = PySys_GetObject("path");
	PyList_Insert( sysPath, 0, PyString_FromString(path.c_str()));
	
	//print python's search paths to confirm that it was added
	/*PyRun_SimpleString(	"import sys\n"
						"from pprint import pprint\n"
						"pprint(sys.path)\n");*/
}


bool GLOBAL_PYTHON_WAS_INITIALIZED = false;
/*#include <mutex>
std::mutex GLOBAL_PYTHON_INITIALIZING_MUTEX;*/
PyThreadState * state = NULL;

bool PrepareForPythonStuff()
{
	//GLOBAL_PYTHON_INITIALIZING_MUTEX.lock();
	
	if(GLOBAL_PYTHON_WAS_INITIALIZED == false)
	{
		if(Py_IsInitialized()) {
			cout<<"WARNING -- PrepareForPythonStuff() -- PYTHON INTERPETER ALREADY INITIALIZED?????"<<endl;
		}
		
		Py_Initialize();
		PyEval_InitThreads();
		state = PyEval_SaveThread();
		
		if(!Py_IsInitialized()) {
			cout<<"WARNING -- PrepareForPythonStuff() -- FAILED TO INITIALIZE PYTHON INTERPETER"<<endl;
		}
		
		GLOBAL_PYTHON_WAS_INITIALIZED = true;
	}
	
	//GLOBAL_PYTHON_INITIALIZING_MUTEX.unlock();
	
	return (Py_IsInitialized() != 0);
}




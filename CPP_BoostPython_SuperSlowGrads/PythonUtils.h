#ifndef __PYTHON_UTILS_SHARED_HPP___
#define __PYTHON_UTILS_SHARED_HPP___

#include <string>
#include <vector>
#include <boost/python.hpp>


void AddPathToPythonSys(std::string path);


template<class T>
boost::python::list std_vector_to_py_list(const std::vector<T>& v)
{
	boost::python::list l;
	typename std::vector<T>::const_iterator it;
	for (it = v.begin(); it != v.end(); ++it)
		l.append(*it);
	return l;
}

template<class T>
std::vector<T> py_list_to_std_vector(const boost::python::list& list)
{
	std::vector<T> vec;
	for(int ii=0; ii < boost::python::len(list); ++ii) {
		vec.push_back(boost::python::extract<T>(list[ii]));
	}
	return vec;
}


bool PrepareForPythonStuff();



/*
	Use when C++ might try to run multiple Python interpreters
*/
struct aquire_py_GIL
{
	PyGILState_STATE state;
	aquire_py_GIL() {
		state = PyGILState_Ensure();
	}

	~aquire_py_GIL() {
		PyGILState_Release(state);
	}
};

/*
	Use this when a Python script needs to call some external C++ utility; put it in the C++ utility function
*/
struct release_py_GIL
{
	PyThreadState *state;
	release_py_GIL() {
		state = PyEval_SaveThread();
	}
	~release_py_GIL() {
		PyEval_RestoreThread(state);
	}
};


#endif

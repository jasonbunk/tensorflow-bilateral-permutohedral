#ifndef __MISC_HEADER_HPP____
#define __MISC_HEADER_HPP____

#include <string>
#include <iomanip>
#include <sstream>


template <class T>
inline std::string to_istring(const T& t) {
	std::stringstream ss;
	ss << static_cast<int>(t);
	return ss.str();
}

template <class T>
inline std::string to_fstring(const T& t, int precision = 8) {
	std::stringstream ss;
	ss << std::setprecision(precision) << (t);
	return ss.str();
}

template <class T>
inline std::string to_sstring(const T& t) {
	std::stringstream ss;
	ss << (t);
	return ss.str();
}


#endif

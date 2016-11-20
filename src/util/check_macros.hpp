#ifndef CHECK_MACROS_HPP_
#define CHECK_MACROS_HPP_



#include <iostream>
extern std::ostream* unprinted_output_stops_here;
#define DONTPRINT (*unprinted_output_stops_here)


#ifndef LOG
#ifndef INFO
#ifndef ERROR
#ifndef FATAL

#define DEFAULTERRMSG "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"<<std::endl<<std::flush<<
#define INFO  0
#define ERROR 1
#define FATAL 2
#define LOG(x) x == INFO ? std::cout<<"INFO: " : (x == ERROR ? std::cout<<DEFAULTERRMSG "ERROR: " : (x == FATAL ? std::cout<<DEFAULTERRMSG "FATAL-ERROR: " : std::cout << #x ))

#endif
#endif
#endif
#endif



#ifndef CHECK_EQ

#define CHECK(a)             (a) ? DONTPRINT : LOG(ERROR)
#define CHECK_EQ(a,b) ((a)==(b)) ? DONTPRINT : LOG(ERROR)
#define CHECK_GT(a,b) ((a)> (b)) ? DONTPRINT : LOG(ERROR)
#define CHECK_GE(a,b) ((a)>=(b)) ? DONTPRINT : LOG(ERROR)
#define CHECK_LT(a,b) ((a)< (b)) ? DONTPRINT : LOG(ERROR)
#define CHECK_LE(a,b) ((a)<=(b)) ? DONTPRINT : LOG(ERROR)

#endif



#endif

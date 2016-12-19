#include <iostream>
#include <stdint.h>
#include <boost/python.hpp>
#include <thread>
#include "PythonCVMatConvert.h"
#include "PythonUtils.h"
#include "utils_RNG.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <omp.h> // OpenMP

namespace bp = boost::python;
using std::cout; using std::endl;

template <class T>
inline std::string to_istring(const T& t)
{
	std::stringstream ss;
	ss << static_cast<int>(t);
	return ss.str();
}
template <class T>
inline std::string to_sstring(const T& t) {
	std::stringstream ss;
	ss << (t);
	return ss.str();
}
#define CVMAP_MAKETYPE(thecvtype) cvdepthsmap[thecvtype] = #thecvtype
static std::map<int,std::string> cvdepthsmap;
void init_cvdepthsmap() {
	CVMAP_MAKETYPE(CV_8U);
	CVMAP_MAKETYPE(CV_16U);
	CVMAP_MAKETYPE(CV_8S);
	CVMAP_MAKETYPE(CV_16S);
	CVMAP_MAKETYPE(CV_32F);
	CVMAP_MAKETYPE(CV_64F);
}
std::string describemat(const cv::Mat & inmat) {
	std::string retstr = std::string("(rows,cols,channels,depth) == (")
					+to_istring(inmat.rows)+std::string(", ")
					+to_istring(inmat.cols)+std::string(", ")
					+to_istring(inmat.channels())+std::string(", ");
	if(cvdepthsmap.find((int)inmat.depth()) != cvdepthsmap.end()) {
		retstr += cvdepthsmap[inmat.depth()]+std::string(")");
	} else {
		retstr += to_istring(inmat.depth())+std::string(")");
	}
	double minval,maxval;
	cv::minMaxIdx(inmat, &minval, &maxval);
	retstr += std::string(", (min,max) = (")+to_sstring(minval)+std::string(", ")+to_sstring(maxval)+std::string(")");
	return retstr;
}
//==============================================================================

#define dSQR(a) ((a)*(a))

void bilateralf(cv::Mat const& input,
								cv::Mat const& featswrt,
						    cv::Mat & out_spatial,
								cv::Mat & out_bilateral,
							  double stdv_space,
							  double stdv_color,
								int grad_chan) {

	out_spatial   = cv::Mat::zeros(input.size(), input.type());
	out_bilateral = cv::Mat::zeros(input.size(), input.type());

	assert(input.rows == featswrt.rows && input.cols == featswrt.cols);
	assert(input.rows == out_spatial.rows && input.cols == out_spatial.cols);
	assert(input.rows == out_bilateral.rows && input.cols == out_bilateral.cols);
	assert(input.channels() == out_spatial.channels() && input.channels() == out_bilateral.channels());
	const int nchans_in  = input.channels();
	const int nchans_wrt = featswrt.channels();
	stdv_space *= stdv_space; // convert to variances
	stdv_color *= stdv_color;
	assert(grad_chan < (2+nchans_wrt));

	double const*const data_input    = (double*)input.data;
	double const*const data_featswrt = (double*)featswrt.data;
	double *const data_out_spatial    = (double*)out_spatial.data;
	double *const data_out_bilateral  = (double*)out_bilateral.data;
	const int step_in  = input.cols*nchans_in;
	const int step_wrt = featswrt.cols*nchans_wrt;

	#pragma omp parallel for
	for(int ii=0; ii<out_spatial.rows; ++ii) {
		// declare these inside the loop, so OpenMP creates one per thread
		double const* elin_oth = nullptr;
		double const* elft_oth = nullptr;
		int cc;
		double dists[2+nchans_wrt];
		double gauss_sp, gauss_bi;
		// each thread processes a row, and loops over all columns
		for(int jj=0; jj<out_spatial.cols; ++jj) {
			double* elou_sp = data_out_spatial   + ii*step_in  + jj*nchans_in;
			double* elou_bi = data_out_bilateral + ii*step_in  + jj*nchans_in;
			double const*const elft_ref = data_featswrt     + ii*step_wrt + jj*nchans_wrt;
			// iterate over all other pixels
			for(int mm=0; mm<input.rows; ++mm) {
				for(int nn=0; nn<input.cols; ++nn) {
					// distances
					dists[0] = (double)(ii-mm);
					dists[1] = (double)(jj-nn);
					elin_oth =    data_input + mm*step_in  + nn*nchans_in;
					elft_oth = data_featswrt + mm*step_wrt + nn*nchans_wrt;
					//vectype_wrt& elin =    input.at<vectype_wrt>(mm,nn);
					//vectype_wrt& elft = featswrt.at<vectype_wrt>(mm,nn);
					for(cc=0; cc<nchans_wrt; ++cc) {
						dists[cc+2] = (elft_ref[cc] - elft_oth[cc]);
					}
					// compute outputs
					gauss_sp = exp(-0.5* (dSQR(dists[0])+dSQR(dists[1]))/stdv_space);
					gauss_bi = exp(-0.5*((dSQR(dists[0])+dSQR(dists[1]))/stdv_color + dSQR(dists[2])+dSQR(dists[3])+dSQR(dists[4]))); //unit stdv for color dist

					if(grad_chan >= 0) {
						if(grad_chan < 2) {
							gauss_sp *= (0.5*dists[grad_chan]/stdv_space);
							gauss_bi *= (0.5*dists[grad_chan]/stdv_color);
						} else {
							gauss_bi *= (0.5*dists[grad_chan]);
						}
					}

					for(cc=0; cc<nchans_in; ++cc) {
						elou_sp[cc] += gauss_sp * elin_oth[cc];
						elou_bi[cc] += gauss_bi * elin_oth[cc];
					}
				}
			}
		}
	}
}

bp::object MyTestGradients(bp::list pyImagesBatch,
														double stdv_space,
														double stdv_color,
														int grad_chan)
{
	init_cvdepthsmap();
	int ii, jj;
	const int numimgs = bp::len(pyImagesBatch);
	if(numimgs <= 0) {
		cout<<"MyTestGradients: error: no images in batch"<<endl<<std::flush;
		return bp::object();
	}
	if(numimgs != 5) {
		cout<<"error: expect these images in batch:"<<endl;
		cout<<"0: input, 1: featswrt, 2: outspatial, 3: outbilat, 4: diffs_outbilat"<<endl<<std::flush;
		return bp::object();
	}

	NDArrayConverter cvt;
	bp::list returnedPyImages; //will be a list of cv2 numpy images
	std::vector<cv::Mat> expectedimages(numimgs);

	for(ii=0; ii < numimgs; ii++) {
		bp::object imgobject = bp::extract<bp::object>(pyImagesBatch[ii]);
		expectedimages[ii] = cvt.toMat(imgobject.ptr());

		cout << "img " << ii << ": " << describemat(expectedimages[ii]) << endl;
	}

	cv::Mat norm_input = cv::Mat::ones(expectedimages[0].size(), expectedimages[0].type());
	const int nchan = expectedimages[0].channels();
	const int npix = expectedimages[0].rows*expectedimages[0].cols;
	for(ii=0; ii<npix; ++ii) {
		for(jj=0; jj<nchan; ++jj) {
			((double*)norm_input.data)[ii*nchan + jj] = 1.0;
		}
	}
	cv::Mat norm_space(expectedimages[0].size(), expectedimages[0].type());
	cv::Mat norm_bilat(expectedimages[0].size(), expectedimages[0].type());

	cout<<"norm_input: "<<describemat(norm_input)<<endl;

	bilateralf(norm_input, expectedimages[1], norm_space, norm_bilat,
						 stdv_space, stdv_color, -1);

	bilateralf(expectedimages[0], expectedimages[1], expectedimages[2], expectedimages[3],
						 stdv_space, stdv_color, grad_chan);

  cout<<endl;
	cout<<"expectedimages[2]: "<<describemat(expectedimages[2])<<endl;
	cout<<"expectedimages[3]: "<<describemat(expectedimages[3])<<endl;
	cout<<"norm_space: "<<describemat(norm_space)<<endl;
	cout<<"norm_bilat: "<<describemat(norm_bilat)<<endl;
	cout<<endl;
	cv::divide(expectedimages[2], norm_space, expectedimages[2]);
	cv::divide(expectedimages[3], norm_bilat, expectedimages[3]);

	for(ii=0; ii < numimgs; ii++) {
		PyObject* thisImgCPP = cvt.toNDArray(expectedimages[ii]);
		//returnedPyImages.append(bp::object(bp::handle<>(bp::borrowed(thisImgCPP)))); // borrowed() increments the reference counter, causing a memory leak when we return to Python
		returnedPyImages.append(bp::object(bp::handle<>(thisImgCPP)));
	}

	return returnedPyImages;
}


static void init() {
	Py_Initialize();
	import_array();
	init_cvdepthsmap();
}
BOOST_PYTHON_MODULE(pymytestgradslib)
{
	init();
	bp::def("MyTestGradients", MyTestGradients);
}

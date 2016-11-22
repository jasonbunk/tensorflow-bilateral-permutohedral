import os,sys
# this file should be in the same path as the built .so
path2file = os.path.dirname(os.path.realpath(__file__))
builtlibpath = os.path.join(path2file, 'libtfgaussiancrf.so')

import tensorflow as tf
from tensorflow.python.framework import ops

libtfgaussiancrf = tf.load_op_library(builtlibpath)


@ops.RegisterGradient("BilateralFilters")
def _BilateralFiltersGrad(op, *grad):
  ret = libtfgaussiancrf.bilateral_filters_grad(
                          op.inputs[0],
                          op.inputs[1],
                          #op.outputs[0],
                          grad[0],
                          grad[1],
                          stdv_spatial_space=op.get_attr("stdv_spatial_space"),
                          stdv_bilater_space=op.get_attr("stdv_bilater_space"))
  ret = list(ret)
  assert len(ret) == 2
  ret[1] = None # no gradient for featswrt
  return ret


def bilateral_filters( input,
                      featswrt,
                      stdv_spatial_space=1.0,
                      stdv_bilater_space=1.0,
                      name=None):
  """
  interface to .so library function
  """
  with ops.name_scope(name, "BilateralFilters") as name:
    # process inputs
    #input    = ops.convert_to_tensor(input,    name="input")
    #featswrt = ops.convert_to_tensor(featswrt, name="featswrt")
    stdv_spatial_space = float(stdv_spatial_space)
    stdv_bilater_space = float(stdv_bilater_space)
    # call using loaded .so
    return libtfgaussiancrf.bilateral_filters(input,
                                              featswrt,
                                              stdv_spatial_space=stdv_spatial_space,
                                              stdv_bilater_space=stdv_bilater_space)

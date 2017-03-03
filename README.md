# tensorflow-bilateral-permutohedral
Port of permutohedral bilateral filtering to tensorflow as an op.

History of this permutohedral code:

- Adams, Andrew, Jongmin Baek, and Myers Abraham Davis.
"Fast High‐Dimensional Filtering Using the Permutohedral Lattice."
Computer Graphics Forum. Vol. 29. No. 2. Blackwell Publishing Ltd, 2010.
- Ported to mxnet https://github.com/piiswrong/permutohedral
- This port to Tensorflow

The mxnet port of the permutohedral lattice plays better with Tensorflow than
Caffe ports (e.g. https://github.com/torrvision/crfasrnn) because the mxnet
port was updated to CUDA streams.

# Installation /  Use

Currently only the GPU version is supported.

You will need to build Tensorflow from source;
will get linker errors if you try to use the pip version of Tensorflow.

To build, just type "make". It uses a similar Makefile as Caffe.

When you build it, try running some tests, like "test/test_segment.py" or "test/test_slider_window.py".

# Note:

This is ONLY supposed to do permutohedral lattice bilateral filtering, which is
useful as a main part of the CRF-RNN algorithm.

inputs:
- image_feats: shape [batch, channels, rows, cols] i.e. NCHW shape.
This is the image that is to be filtered.
- image_wrt: same shape as above, except possibly different number of channels.
When filtering, distances between points are computed with respect to this.
- standard deviation of spatial part of filters

The images will be filtered with a (2+channels)-dimensional filter.

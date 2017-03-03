
CRF-RNN:
- Given RGB image: at each pixel, construct 5d features: (x,y,R,G,B)
- Then, at each pixel, use 5d features to measure distances to nearest
neighbors, and use a Gaussian to get weights from distances
(higher weights when closer)
- Given unary label preds (L0,L1,...): Output label preds are weighted sum of
neighbors (weighted by distance in 5d space)
- Last step after the above bilateral filtering is a label compatibility
transform (a 1x1 convolution)

Multidimensional Bilateral Filtering (this repository):
- Given N-channel image: at each pixel, construct Nd features (n0,n1,...)
- Then, at each pixel, use Nd features to weigh distances to nearest neighbors
- Given M-channel image: output M-channel image is a weighted sum, weights
came from Nd feature distances

Example:
Input: 5-d features (x, y, R, G, B)...
Output: weighted, filtered (R, G, B) at each (x,y)

Example:
Input: 5-d features (x, y, R, G, B)..
Output: weighted, filtered (c0, c1, c2, ...) at each (x,y)

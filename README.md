# Probabilistic Tensor Decomposition Toolbox (prob-tensor-toolbox)
 
This repository contains functions for tensor decomposition using probabilistic/Bayesian methods. Currently supported decomposition structures are Candecomp/PARAFAC (CP), Tucker, and Tensor Train Decomposition. Model inference is done using either Variational Bayesian (VB) Inference or Markov Chain Monte Carlo Gibbs Sampling.

For CP and Tucker, it is possible to specify different distributions (constraining) on the factor matrices. For TT only orthogonal factors are supported. Furthermore, in the CP decomposition, it is possible to marginalize (ignore) missing values and/or model mode specific heteroscedastic noise.

An initial overview of the probabilistic tensor decomposition toolbox is provided in [1,2]. The toolbox, however, is still under active development with more features are planed, as well as improved demonstrations and documentation.

## Factor matrices
Let an N-way data array X be factorized into a set of latent matrices A^1, A^2, ... A^N using a sum of rank-1 components (CP decomposition) or into a core array G and latent matrices A^1, A^2, ... A^N (Tucker decomposition). Then presently, this toolbox allows A^n following either,

1) A follows a normal distribution (real valued).
2) A is non-negative and follow a truncated normal
3) A is non-negative and follow an exponential distribution.
4) A is orthogonal and follows a von Mises-Fisher distribution
5) A is infinity norm (box) constrained and follows a Uniform distribution.

Then for 1)-3) the precision (scale/rate) of the distribution can be either,

a) Fixed to or learned as a scalar value
b) Prune irrelevant components using a column-wise sparsity (i.e. automatic relevance determination) prior.
c) Enforce element-wise sparsity (i.e. discover an underlying sparse pattern)
d) Learn the precision matrix (inverse covariance) among the components using a Wishart distribution.

Note, d) is only available for 1) while presently c) is only available for 2)-3).

## Noise estimation
It is  possible to infer the noise precision (inverse variance) for either homo- or heteroscedastic noise in the CP model. However, for Tucker and TT decomposition only homoscedastic noise is inferred.

## Missing values
In CP decomposition, handling missing values is possible using either marginalization or Bayesian imputation. Note currently, the posterior distributions estimated via imputation is not correct.


## Getting started

1) For a simple demonstration, Run 'demo_CP.m' or 'demo_Tucker.m'
2) For a demonstration of heteroscedastic noise estimation, run 'demo_heteroscedastic_CP.m'.
3) For a demonstration on fluorescence spectroscopy data, run 'demo_AminoAcid.m'

### Dependencies
There is a few dependences on the N-way Toolbox, which is freely available at http://www.models.life.ku.dk/nwaytoolbox

The toolbox uses the in-build MATLAB testing framework to assess backward compatibility, this requires the 'runtests()' function from MATLAB2016b. This means the toolbox should work out of the box from MATLAB2016b and onwards. Earlier versions are not tested and might work, partially work, or fail completely.


# Notes regarding usage.
All code can be used freely in research and other non-profit applications (unless otherwise stated in the third party licenses). 

This tool was developed at:

The Technical University of Denmark, Department for Applied Mathematics and Computer Science, Section for Cognitive Systems.

# References
[1] Hinrich, J. L., Madsen, K. H., & Mørup, M. (2020). The probabilistic tensor decomposition toolbox. Machine Learning: Science and Technology.
[2] Hinrich, J. L., & Mørup, M. (2019, September). Probabilistic Tensor Train Decomposition. In 2019 27th European Signal Processing Conference (EUSIPCO) (pp. 1-5). IEEE.

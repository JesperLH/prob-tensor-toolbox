# Probabilistic Tensor Toolbox (prob-tensor-toolbox)
 
This repository contains functions for running Probabilistic Candecomp/PARAFAC (CP) using Bayesian Inference with a variety of constraints on the factor matrices and noise.

The toolbox is under development, but more features are planed, as well as improved demonstrations and documentation.

## Factor matrices
This toolbox factorizes a N-way data array X into a set of latent matrices A^1, A^2, ... A^N using a sum of rank-1 components. This tool facilitates the scenarios where,

1) A follows a normal distribution (real valued).
2) A is non-negative and follow a truncated normal or an exponential distribution.
3) A is orthogonal and follows a von Mises-Fisher distribution

In addition, for 1) and 2) the scale of the distribution can be either,

a) Fixed to a scalar value
b) Prune irrelevant components using a column-wise sparsity (i.e. automatic relevance determination)
c) Enforce element-wise sparsity (i.e. discover an underlying sparse pattern)

## Noise estimation
It is  possible to infer the noise precision (inverse variance) for either homo- or heteroscedastic noise.

## Missing values
Handling missing values is possible using either marginalization or Bayesian imputation. Note currently, the posterior distributions estimated via imputation is not correct.

## Getting started

1) For a simple demonstration, Run 'demo_CP.m'
2) For a demonstration of heteroscedastic noise estimation, run 'demo_heteroscedastic_CP.m'.
3) For a demonstration on fluorescence spectroscopy data, run 'demo_AminoAcid.m'


# Notes regarding usage.
All code can be used freely in research and other non-profit applications (unless otherwise stated in the third party licenses). 

This tool was developed at:

The Technical University of Denmark, Department for Applied Mathematics and Computer Science, Section for Cognitive Systems.

# References

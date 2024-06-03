> :warning: **Code will be available soon**

# VENI-VINDy-VICI
A variational reduced-order modeling framework with uncertainty quantification [1].

![overview](doc/figures/overview.png)

## Framework
The framework discovers probabilistic governing equations from high-dimensional data in a low-dimensional latent space. It consists of three steps:

#### VENI (Variational Encoding of Noisy Inputs)
A generative model utilizing variational autoencoders (VAEs) is applied to transform high-dimensional, noisy data into a low-dimensional latent space representation that is suitable to describe the dynamics of the system.

#### VINDy (Variational Identification of Nonlinear Dynamics)
On the time series data expressed in the new set of latent coordinates, a probabilistic dynamical model of the system is learned by a variational version of SINDy (Sparse Identification of Nonlinear Dynamics) [2].

#### VICI (Variational Inference with Certainty Intervals) 
The resulting ROM allows to evolve the temporal system solution by variational inference on both the latent variable distribution and the dynamic model, given new parameter/force values and initial conditions. This, naturally, provides an estimate of the reliability of the prediction through certainty intervals.

## Features
This repository implements the classic SINDy autoencoders [3] as well as its variational extension: the newly proposed VENI, VINDy, VICI framework [1].
* Autoencoders (AEs) for dimensionality reduction
* Variational autoencoders (VAEs) for probabilistic latent representations
* SINDy layer to identify interpretable governing equations from data using standard backpropagation algorithms
* VINDy layer to identify interpretable probabalistic governing equations, where coefficients are represented as distributions.
  * Direct uncertainty quantification on modeling terms
  * Sampling-based uncertainty quantification for time evolution of system states
* Infuse preknowledge
  * Select priors
  * Fix certain weights
  * Model your system as second order system dx/ddt = f(x, xdt, mu)
* Several callbacks
  * Update governing equation coefficients with separate SINDy optimization schemes (using pysindy)
  * Thresholding coefficents w.r.t their magnitude or their probability density function around zero
  * Log the coefficients during training to monitor convergence

The individual contributions can be used standalone (plain SINDy or VINDy) or arbitrarily be combined with dimensionality reducition schemes (e.g. VAEs with SINDy, AE with VINDy, VAE with VINDy, ...)

## Installation

Clone this repository and install it to your local environment as package using pip:

```bash
git clone git@github.com:jkneifl/VENI-VINDy-VICI.git
cd VENI-VINDy-VICI
pip install -e .
```

You can run the jupyter notebook for the Roessler system to check if the installation was successful. 
It is in the `examples` folder. Please note that you'll need to have jupyter installed in order to run the notebook.



## References

[1] Paolo Conti, Jonas Kneifl, Andrea Manzoni, Attilio Frangi, Jörg Fehr, Steven L. Brunton, J. Nathan Kutz. VENI, VINDy, VICI -- a variational reduced-order modeling framework with uncertainty quantification. Arxiv preprint:

[2] S. L. Brunton, J. L. Proctor, J. N. Kutz, Discovering governing equations from data by sparse identifi cation of nonlinear dynamical systems, Proceedings of the national academy of sciences 113 (15) (2016) 3932–3937. doi:10.1073/pnas.1517384113.

[3] Champion, K., Lusch, B., Kutz, J. N., & Brunton, S. L. (2019). Data-driven discovery of coordinates and governing equations. Proceedings of the National Academy of Sciences, 116(45), 22445-22451.

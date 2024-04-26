# ExponentialGM
R package for implementing full-likelihood based inference in pariwise exponential family graphical models. The models include fully observed (Ising model, Poisson graphical model) and partially observed (Boltzmann machine and Restricted Boltzmann machine) models. To install the package, first install the "devtools" package using install.packages("devtools"). Then use install_github("chenyujie1104/ExponentialGM").


## Description
Pairwise exponential family graphical models constitute a flexible class of models that allows modeling dependence in multivariate data through simple univariate exponential families. Moreover, when some variables are not observed, the resulting marginal distribution deviates from exponential family allowing further flexibility; these latent exponential family models are more popularly known as Boltzmann machines, which are fundamental building blocks of generative AI. However, a key roadblock in likelihood-based inference for these models is the intractable normalizing constant. We develop a Monte Carlo estimate of the normalizing constant, which enables us to peform full-likelihood and Bayes analysis on these models.

## Examples

### Ising model
The Ising model is a popular model for multivariate binary data. A typical example of modeling multivariate binary using the package is given below. 

```
# R code to generate multivariate binary data #
gen_theta0 = function(p, omega, eta, coupling = "positive")
{
  ## Each element is non-zero with probability eta ##
  ## omega is a positive number denoting edge strength ##
  theta0 = matrix(0, p, p)
  for(j in 1:p)
  {
    for(k in j:p)
    {
      z = rbinom(1, 1, eta)
      if(coupling == "mixed"){
        theta0[j,k] = z*(-1)^rbinom(1, 1, 0.5)*omega
        theta0[k,j] = theta0[j,k]
      }else if(coupling == "positive"){
        theta0[j,k] = z*omega
        theta0[k,j] = theta0[j,k]
      }else if(coupling == "negative"){
        theta0[j,k] = z*(-omega)
        theta0[k,j] = theta0[j,k]
      }
    }
  }
  return(theta0)
}

IsingSim = function(n, theta, max_iter)
{
  p = ncol(theta)
  X = matrix(rbinom(n*p, 1, exp(diag(theta))/(1+exp(diag(theta)))), n, p)
  
  for(k in 1:max_iter)
  {
    for(j in 1:p)
    {
      t = exp(theta[j,j] + 2*X[, -j]%*%matrix(theta[-j, j], p - 1, 1))
      X[,j] = rbinom(n, 1, t/(1+t))
    }
  }
  return(X)
}

# Generated data #
n = 100
p = 3
omega = 1  ## Edge strength of non-zero edges
eta = 0.05 ## Probability with which each edge is set to non-zero
theta0 = gen_theta0(p, omega, eta, coupling = "negative")
max_iter = 1000
X = IsingSim(n, theta0, max_iter)

step_size = 0.01
N = 500
epsilon = 0.0001
max_iter = 1000
```

\name{ExponentialGM-package}
\alias{ExponentialGM-package}
\alias{ExponentialGM}
\docType{package}
\title{
\packageTitle{ExponentialGM}
}
\description{
\packageDescription{ExponentialGM}
}
\details{
The DESCRIPTION file:
\packageDESCRIPTION{ExponentialGM}
\packageIndices{ExponentialGM}

Fully observed models are fitted with \code{\link{EGM.fit}} (\code{model = "ising"}
or \code{model = "pgm"}), with \code{\link{EGM.CV}} and \code{\link{EGM.SS}} for
cross-validation and stability selection, and \code{\link{EGM_Baysian}} for Bayesian
inference. Partially observed models are fitted with \code{\link{BM_fit}} and
\code{\link{RBM_fit}}, and images are reconstructed with
\code{\link{BM_reconstruct}} and \code{\link{RBM_reconstruct}}.
}
\author{
\packageAuthor{ExponentialGM}

Maintainer: \packageMaintainer{ExponentialGM}
}
\references{
Chen, Y., Bhadra, A. and Chakraborty, A. (2024). Likelihood Based Inference in
Fully and Partially Observed Exponential Family Graphical Models with Intractable
Normalizing Constants. \url{https://arxiv.org/abs/2404.17763}
}
\keyword{ package }
\seealso{
\code{\link{EGM.fit}}, \code{\link{RBM_fit}}
}
\examples{
theta0 <- matrix(-0.8, 3, 3)
X <- IsingSim(n = 100, theta = theta0, max_iter = 1000)
EGM.fit(X = X, model = "ising")
}

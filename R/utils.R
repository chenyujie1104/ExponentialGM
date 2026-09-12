`%dopar%` <- foreach::`%dopar%`
`%do%` <- foreach::`%do%`

#' Internal function
#' @keywords internal
#' @noRd 
pgm_network <- function(theta, step_size, X, N, epsilon, max_iter, lambda, penalty){

  p <- ncol(X)
  if (length(theta) == 1) {
    theta <- matrix(0, p, p)
  }

  if (penalty == "none") {
    convergence_check <- 10
    iter <- 0
    while (convergence_check > epsilon || iter <= max_iter) {
      theta_proximal <- theta + (1/(iter + 1)) * step_size * loglike_grad_cpp_pgm(theta, X, N)
      temp_diag <- diag(theta_proximal)
      theta_proximal[which(theta_proximal > 0)] <- 0
      theta_proximal <- (theta_proximal + t(theta_proximal)) / 2
      diag(theta_proximal) <- temp_diag
      convergence_check <- norm(theta_proximal - theta, "F")
      theta <- theta_proximal
      iter <- iter + 1
    }
    return(theta)
  } else if (penalty == "l1") {
    convergence_check <- 10
    iter <- 0
    while (convergence_check > epsilon || iter <= max_iter) {
      iter <- iter + 1
      theta_new <- theta + (1 / (iter * 0.5 + 1)) * step_size * loglike_grad_cpp_pgm(theta, X, N)
      theta_proximal <- sign(theta_new) * pmax(abs(theta_new) - (1/(iter*0.5+1)) * step_size * lambda, matrix(0, p, p))
      temp_diag <- diag(theta_proximal)
      theta_proximal[which(theta_proximal > 0)] <- 0
      theta_proximal <- (theta_proximal + t(theta_proximal)) / 2
      diag(theta_proximal) <- temp_diag
      convergence_check <- norm(theta_proximal - theta, "F")
      #likelihood = loglike_cpp_pgm(theta_proximal, X, N = 5000)
      theta <- theta_proximal
    }
    return(theta)
  }
}

#' Internal function
#' @keywords internal
#' @noRd 
ising_network <- function(theta, step_size, X, N, epsilon, max_iter, lambda, penalty) {
  p <- ncol(X)
  if (length(theta) == 1) {
    theta <- matrix(0, p, p)
  }

  if (penalty == "none") {
    convergence_check <- 10
    iter <- 0
    while (convergence_check > epsilon || iter <= max_iter) {
      theta_new = theta + (1 / (iter + 1) ) * step_size * loglike_grad_cpp_ising(theta, X, N)
      convergence_check <- norm(theta_new - theta, "F")
      theta <- theta_new
      iter <- iter + 1
    }
    return(theta)
  }else if (penalty == "l1") {
    convergence_check <- 1000
    iter <- 0
    while (convergence_check > epsilon || iter <= max_iter) {
      theta_new = theta + (1 / (iter + 1)) * step_size * loglike_grad_cpp_ising(theta, X, N)
      theta_proximal = sign(theta_new) * pmax(abs(theta_new) - (1 / (iter + 1)) * step_size * lambda, matrix(0, p, p))
      convergence_check <- norm(theta_proximal - theta, "F")
      theta <- theta_proximal
      iter <- iter + 1
    }
    return(theta)
  }
}

#' Internal function
#' @keywords internal
#' @noRd 
ising_bayes = function(X, nmcmc, burnin, tau, N, epsilon, L, a_lambda, b_lambda) {

  p = ncol(X)
  theta = matrix(rnorm(p^2, sd = 0.01), p, p)
  theta_vec = vech(theta)
  gamma_vec = rep(1, p*(p+1)/2)
  gamma_mat = invvech(gamma_vec)
  lambda = 1
  U = function(theta_vec)
  {
    theta = invvech(theta_vec)
    s = loglike_cpp_ising(theta, X, N)
    s1 = 0
    for(j in 1:p)
    {
      for(k in j:p)
      {
        if(j == k){
          s1 = s1 + 0.5*theta[j,k]^2/tau^2
        }else if(j != k){
          s1 = s1 + 0.5*theta[j,k]^2/gamma_mat[j,k]^2
        }
      }
    }
    return(-s + s1)
  }
  grad_U = function(theta_vec)
  {
    theta = invvech(theta_vec)
    S = loglike_grad_cpp_ising(theta, X, N)
    S1 = matrix(0, p, p)
    for(j in 1:p)
    {
      for(k in j:p)
      {
        if(j == k){
          S1[j,k] = theta[j,k]/tau^2
        }else if(j != k){
          S1[j,k] = theta[j,k]/gamma_mat[j,k]^2
          S1[k,j] = S1[j,k]
        }
      }
    }
    S2 = -S + S1
    return(vech(S2))
  }
  
  ## MCMC storage ##
  thetaout = matrix(0, p*(p+1)/2, nmcmc - burnin)
  lambdaout = numeric(nmcmc - burnin)
  MH_count = 0
  Uout = numeric(nmcmc - burnin)

  for(ii in 1:nmcmc)
  {
    ## Update gamma_mat ##
    for(j in 1:(p-1))
    {
      for(k in (j+1):p)
      {
        gamma_mat[j,k] = gamma_mat[k,j] = rinvgaussian(1, mu = lambda/abs(theta[j,k]), lambda^2)
      }
    }
    gamma_vec = vech(gamma_mat)
    
    ## Update theta ##
    hmc_res = HMC_cpp_ising(U, grad_U, epsilon, L, theta_vec)
    theta_vec = hmc_res$q
    MH_count = MH_count + hmc_res$MH_count
    
    ## Update lambda ##
    g_vec = gamma_vec[gamma_vec!=1]
    lambda_sq = rgamma(1, p*(p-1)/2 + a_lambda, 0.5*sum(1/g_vec) + b_lambda)
    lambda = sqrt(lambda_sq)
    
    if(ii%%1000 == 0){
      print(ii)

    } 
    if(ii > burnin) {
      thetaout[,ii-burnin] = theta_vec
      lambdaout[ii-burnin] = lambda
      Uout[ii-burnin] = U(theta_vec)
    }
    
  }

  bayes_theta = invvech(rowMeans(thetaout))

  result = list("thetaout" = thetaout, "lambdaout" = lambdaout, "MH_acceptance" = MH_count/nmcmc, "Uout" = Uout, "bayes_theta" = bayes_theta)
  return(result)
} 

#' Internal function
#' @keywords internal
#' @noRd 
PGM_bayes <- function(X, nmcmc, burnin, N, epsilon, L, a_lambda, b_lambda) {

  p <- ncol(X)

  # A matrix
  index_theta <- matrix(1:(p^2), p, p)
  vech_index_theta <- vech(index_theta)
  A_matrix <- diag(0, length(vech_index_theta), length(vech_index_theta))
  diag(A_matrix) <- 1 - vech(index_theta) %in% diag(index_theta)


  theta <- 0 - abs(matrix(rnorm(p^2, 0, sd = 0.01), p, p))
  theta <- (theta + t(theta)) / 2
  theta_vec <- vech(theta)
  gamma_mat <- rep(1, p)
  lambda <- sqrt(rgamma(1, a_lambda, rate = b_lambda))

  U <- function(theta_vec) {
    theta <- invvech(theta_vec)
    s <- loglike_cpp_pgm(theta, X, N)
    diag_s <- sum(0.5 * theta[diag(index_theta)]^2 / gamma_mat)
    off_diag <-  0.5 * sum(theta[-diag(index_theta)] * lambda)
    return(-s + diag_s - off_diag)
  }

  grad_U <- function(theta_vec) {
    theta <- invvech(theta_vec)
    S <- loglike_grad_cpp_pgm(theta, X, N)
    S1 <- matrix(-lambda, p, p)
    diag(S1) <- diag(theta) / gamma_mat

    S2 <- -S + S1
    return(vech(S2))
  }

  ## MCMC storage ##
  thetaout <- matrix(0, p * (p + 1) / 2, nmcmc - burnin)
  gammaout <- matrix(0, p, nmcmc - burnin)
  lambdaout <- numeric(nmcmc - burnin)
  MH_count <- 0
  Uout <- numeric(nmcmc - burnin)
  for (ii in 1:nmcmc){
    for (k in 1:p){
      gamma_temp <- max(rinvgaussian(1, mu = lambda / abs(theta[k,k]), lambda^2), 1e-08)
      gamma_mat[k] <- 1 / gamma_temp
    }

    ## Update theta ##
    hmc_res <- HMC_cpp_pgm(U, grad_U, epsilon, L, theta_vec, A_matrix)
    theta_vec <- hmc_res$q
    MH_count <- MH_count + hmc_res$MH_count

    ## Update lambda ##

    tempt1 <- exp(0.5 * sum(lambda * theta[-diag(index_theta)]))
    u1 <- stats::runif(1, 0, tempt1)
    tempt2 <- (sum(gamma_mat)/2) + b_lambda
    ubt <- log(u1) / (0.5 * sum(lambda * theta[-diag(index_theta)]))
    Fubt <- 1 - stats::pgamma(ubt^2, (p^2 + 3 * p + 4 * a_lambda) / 4, rate = tempt2)
    Fubt <- max(Fubt, 1e-08)
    ut <- stats::runif(1, 0, Fubt)
    lambda_2 <- stats::qgamma(ut, (p^2 + 3 * p + 4 * a_lambda) / 4, rate = tempt2)
    lambda <- sqrt(lambda_2)

    if (ii %% 1000 == 0) print(ii)

    if (ii > burnin) {
      thetaout[, ii - burnin] <- theta_vec
      lambdaout[ii - burnin] <- lambda
      Uout[ii - burnin] <- U(theta_vec)
      gammaout[, ii - burnin] <- gamma_mat
    }

  }

  bayes_theta = invvech(rowMeans(thetaout))

  result <- list("thetaout" = thetaout, "lambdaout" = lambdaout,
    "bayes_theta" = bayes_theta, "MH_acceptance" = MH_count / nmcmc,
    "Uout" = Uout
  )
  return(result)
}

#' Internal function
#' @keywords internal
#' @noRd 
gen_theta0 = function(p, omega, eta, coupling = "positive") {
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

#' Internal function
#' @keywords internal
#' @noRd 
IsingSim = function(n, theta, max_iter) {
  p = ncol(theta)
  X = matrix(rbinom(n*p, 1, exp(diag(theta))/(1+exp(diag(theta)))), n, p)

  for(k in 1:max_iter)
  {
    for(j in 1:p)
    {
      t = exp(theta[j,j] + 2*X[, -j]%*%matrix(theta[-j, j], p - 1, 1))
      #t = exp(theta[j,j] + X[, -j]%*%matrix(theta[-j, j], p - 1, 1))
      X[,j] = rbinom(n, 1, t/(1+t))
    }
  }
  return(X)
}

#' Internal function
#' @keywords internal
#' @noRd 
PGMsim <- function(n, theta, max_iter) {
  p = ncol(theta)
  X = matrix(rpois(n*p, exp(diag(theta))), n, p)
  for(k in 1:max_iter)
  {
    for(j in 1:p)
    {
      rate_j = exp(theta[j,j] + 2*X[, -j]%*%matrix(theta[-j, j], p - 1, 1))
      X[,j] = rpois(n, rate_j)
    }
  }
  return(X)
}


#' Internal function
#' @keywords internal
#' @noRd 
proximal_grad_descent_try_catch <- function(model,  X, step_size, theta, N, epsilon, max_iter, lambda, penalty){
    p = ncol(X)
    tryCatch( {
        if (model == "ising") {
          result <- ising_network(theta = theta, step_size = step_size, X = X, N = N, epsilon = epsilon, max_iter = max_iter, lambda = lambda, penalty = penalty)
          return(result)
        } else if (model == "pgm") {
          result <- pgm_network(theta = theta, step_size = step_size, X = X, N = N, epsilon = epsilon, max_iter = max_iter, lambda = lambda, penalty = penalty)
          return(result)
        }
        },
        error=function(e) {
            print(e)
            #print('If NA occur, try a different step size.')
            return(matrix(NA, p, p))
        },
        warning=function(w) {
            message('A Warning Occurred')
            print(w)
            return(NA)
        }
    )
}


#' Internal function
#' @keywords internal
#' @noRd 
parallel_descent_cross_validation <- function(model, train_set, test_set, step_size, lambda, data_set_index, theta = theta, 
                N = N, epsilon = epsilon, max_iter = max_iter, penalty = "l1"){

  theta_lasso = proximal_grad_descent_try_catch(model = model, X = train_set[[data_set_index]], theta = theta, 
                step_size = step_size,  N = N, epsilon = epsilon, max_iter = max_iter, lambda = lambda, penalty = "l1")
  
  if(is.na(theta_lasso[1,1])){
    return(list("log_l" = -Inf, "lambda" = lambda, "step_size" = step_size, "data_set_index" = data_set_index))
  }else{
    log_l_temp = numeric(10)
    for(k in 1:10){
    if (model == "ising") {
      log_l_temp[k] = loglike_cpp_ising(theta = theta_lasso, X = test_set[[data_set_index]], N = 10000)
    } else if (model == "pgm") {
      log_l_temp[k] = loglike_cpp_pgm(theta = theta_lasso, X = test_set[[data_set_index]], N = 10000)
    }
  }

    print(paste("cross_validation", data_set_index, "step_size", step_size, "lambda", lambda, "log_l", mean(log_l_temp)))
    return(list("log_l" = mean(log_l_temp), "lambda" = lambda, "step_size" = step_size, "data_set_index" = data_set_index))
  }
}

#' @noRd 
selection_set <- function(theta){
  p = ncol(theta)
  selection_matrix = matrix(0, p, p)
  selection_matrix[theta != 0] = 1
  return(vech(selection_matrix))
}


##################### BM
#' Internal function
#' @keywords internal
#' @noRd 
gen_bm_theta0 = function(p, m, eta){
  
  theta0 = matrix(0, p+m, p+m)
  for(j in 1:(p+m))
  {
    for(k in j:(p+m))
    {
      z = rbinom(1, 1, eta)
      theta0[j,k] = z*rnorm(1)
      theta0[k,j] = theta0[j,k]
    }
  }
  #theta0[1:p, 1:p] = diag(diag(theta0[1:p, 1:p]), p, p)
  return(theta0)
}

#' @noRd 
BMSim = function(n, theta, p, m, max_iter){
  p1 = ncol(theta)
  X = matrix(rbinom(n*p1, 1, exp(diag(theta))/(1+exp(diag(theta)))), n, p1)
  
  for(k in 1:max_iter)
  {
    for(j in 1:p1)
    {
      t = exp(theta[j,j] + 2*X[, -j]%*%matrix(theta[-j, j], p1 - 1, 1))
      X[,j] = rbinom(n, 1, t/(1+t))
    }
  }
  V = X[,1:p]
  H = X[,(p+1):(p+m)]
  return(list("V" = V, "H" = H))
}

#' Internal function
#' @keywords internal
#' @noRd 
Exp_g = function(v, theta, m, N){
  p = length(v)
  theta_v = theta[1:p, 1:p]
  W = theta[1:p, (p+1):(p+m)]
  theta_h = theta[(p+1):(p+m), (p+1):(p+m)]
  phi = rep(0, m)
  Y = matrix(0, N, m)
  for(k in 1:m)
  {
    phi[k] = theta_h[k,k] + sum(W[,k]*v)
    Y[,k] = rbinom(N, 1, exp(phi[k])/(1+exp(phi[k])))
  }
  S1 = 0
  #S2 = rep(0, m)
  S3 = matrix(0, m, m)
  for(i in 1:N)
  {
    d_ratio = exp(t(Y[i,])%*%theta_h%*%Y[i,] - sum(Y[i,]*diag(theta_h)))
    S1 = S1 + as.double(d_ratio)
    #S2 = S2 + as.double(d_ratio)*Y[i,]
    S3 = S3 + Y[i,]%*%t(Y[i,])*as.double(d_ratio)
  }
  result = list("z_theta_phi" = S1/N, "E_H" = S3/S1)
  return(result)
}

#' Internal function
#' @keywords internal
#' @noRd 
loglike_grad_BM = function(theta, V, N, p, m){
  n = nrow(V)
  p = ncol(V)
  S1 = matrix(0, m, m)
  S2 = matrix(0, p, m)
  for(i in 1:n)
  {
    E_H = Exp_g(V[i,], theta, m, N)$E_H
    E1_H = 2*E_H
    diag(E1_H) = diag(E_H)
    S1 = S1 + E1_H
    S2 = S2 + 2*V[i,]%*%t(diag(E_H))
  }
  G = matrix(0, p+m, p+m)
  G[1:p, 1:p] = 2*t(V)%*%V
  diag(G[1:p,1:p]) = colSums(V)
  G[1:p, (p+1):(p+m)] = S2
  G[(p+1):(p+m), 1:p] = t(S2)
  G[(p+1):(p+m), (p+1):(p+m)] = S1
  phi = matrix(diag(theta), p+m, p+m)
  res = z_functions_cpp_ising(theta, N, phi)
  Z_grad = res$Z_prime_theta
  Z = res$Z_theta
  G = G - n*Z_grad/Z
  #G = G + t(G) - diag(diag(G))
  return(G)
}

#' Internal function
#' @keywords internal
#' @noRd 
BM_marginal = function(v, theta, p, m, N){
    phi = matrix(diag(theta), p+m, p+m)
    Z = z_theta_cpp_ising(theta, N, phi)
    H_all = expand.grid(replicate(m, 0:1, simplify = F))
    H_all = matrix(unlist(H_all), 2^m, m)
    #H_all = matrix(unlist(expand.grid(0:1, 0:1, 0:1, 0:1)), 2^m, m)
    s = 0
    for(j in 1:nrow(H_all))
    {
      s = s + prob_kernel_cpp_ising(c(v, H_all[j,]), theta)
    }
    s = s/Z
    return(s)
}





########################## RBM
#' Internal function
#' @keywords internal
#' @noRd 
gen_rbm_theta1 = function(p, m, eta){
  b1 = rep(0, p)
  b2 = rep(0, m)
  W = matrix(0, m, p)
  for(i in 1:m)
  {
    b2[i] = rbinom(1, 1, eta)*(-1)^rbinom(1, 1, 0.5)*runif(1)
    for(j in 1:p)
    {
      b1[j] = rbinom(1, 1, eta)*(-1)^rbinom(1, 1, 0.5)*runif(1)
      W[i,j] = rbinom(1, 1, eta)*(-1)^rbinom(1, 1, 0.5)*runif(1)
    }
  }
  theta = matrix(0, p+m, p+m)
  diag(theta[1:p, 1:p]) = b1
  diag(theta[(p+1):(p+m), (p+1):(p+m)]) = b2
  theta[1:p, (p+1):(p+m)] = t(W)
  theta[(p+1):(p+m), 1:p] = W
  result = list("b" = b1, "c" = b2, "W" = W, "theta" = theta)
  return(result)
}

#' Internal function
#' @keywords internal
#' @noRd 
RBM_SIM1 = function(n, theta, p, m, max_iter){
  b1 = diag(theta[1:p, 1:p])
  b2 = diag(theta[(p+1):(p+m), (p+1):(p+m)])
  W = theta[(p+1):(p+m), 1:p]
  V = matrix(0, n, p)
  H = matrix(0, n, m)
  for(j in 1:p)
  {
    V[,j] = rbinom(n, 1, exp(b1[j])/(1+exp(b1[j])))
  }
  for(i in 1:m)
  {
    H[,i] = rbinom(n, 1, exp(b2[i])/(1+exp(b2[i])))
  }
  for(ii in 1:max_iter)
  {
    ## Sample H given V ##
    U = V%*%t(W)
    for(i in 1:m)
    {
      H[,i] = rbinom(n, 1, exp(b2[i] + 2*U[,i])/(1+exp(b2[i] + 2*U[,i])))
    }
    U = H%*%W
    for(j in 1:p)
    {
      V[,j] = rbinom(n, 1, exp(b1[j] + 2*U[,j])/(1+exp(b1[j] + 2*U[,j])))
    }
  }
  result = list("V" = V, "H" = H)
  return(result)
}

#' Internal function
#' @keywords internal
#' @noRd 
loglike_grad_rbm = function(theta, X, N, p, m){
  n = nrow(X)
  G = matrix(0, p+m, p+m)
  phi = matrix(0, p+m, p+m)
  diag(phi) = diag(theta)
  #res = z_functions(theta, N, phi)
  res = z_functions_cpp_ising(theta, N, phi)
  Z = res$Z_theta
  Z_grad = res$Z_prime_theta
  
  for(j in 1:(p+m))
  {
    G[j,j] = sum(X[,j])
  }
  
  for(j in 1:p)
  {
    for(k in (p+1):(p+m))
    {
      G[j,k] = G[k,j] = 2*sum(X[,j]*X[,k])
    }
  }
  Z_grad[1:p, 1:p] = diag(diag(Z_grad[1:p, 1:p]))
  Z_grad[(p+1):(p+m), (p+1):(p+m)] = diag(diag(Z_grad[(p+1):(p+m), (p+1):(p+m)]))
  G = G - n*Z_grad/Z
  return(G)
}

#' Internal function
#' @keywords internal
#' @noRd 
rbm_mle = function(theta, X, step_size, epsilon, max_iter, N, p, m){
  convergence_check = 10
  while(convergence_check > epsilon){
    theta_new = theta + (1/(iter+1))*step_size*loglike_grad_rbm(theta, X, N, p, m)
    convergence_check = norm(theta_new - theta, "F")
    theta = theta_new
    print(convergence_check)
    if(iter>max_iter) 
    {
      break
    }
    iter = iter + 1
  }
  return(theta)
}

#' Internal function
#' @keywords internal
#' @noRd 
loglike_grad_rbm1 = function(theta, V, N, p, m){
  n = nrow(V)
  G = matrix(0, p+m, p+m)
  phi = matrix(0, p+m, p+m)
  diag(phi) = diag(theta)
  #res = z_functions(theta, N, phi)
  res = z_functions_cpp_ising(theta, N, phi)
  Z = res$Z_theta
  Z_grad = res$Z_prime_theta
  
  b1 = diag(theta[1:p, 1:p])
  b2 = diag(theta[(p+1):(p+m), (p+1):(p+m)])
  W = theta[(p+1):(p+m), 1:p]
  
  U = V%*%t(W)
  E_H = matrix(0, n, m)
  H = matrix(0, n, m)
  for(i in 1:m)
  {
    E_H[,i] = exp(b2[i] + 2*U[,i])/(1+exp(b2[i] + 2*U[,i]))
    #H[,i] = rbinom(n, 1, E_H[,i])
  }
  
  X = cbind(V,E_H)
  #X = cbind(V,H)
  # for(j in 1:(p+m))
  # {
  #  G[j,j] = sum(X[,j])
  # }
  diag(G) = colSums(X)
  
  G[1:p, (p+1):(p+m)] = 2*t(V)%*%E_H
  G[(p+1):(p+m), 1:p] = t(G[1:p, (p+1):(p+m)])
  # for(j in 1:p)
  # {
  #  for(k in (p+1):(p+m))
  #  {
  #    G[j,k] = G[k,j] = 2*sum(X[,j]*X[,k])
  #  }
  # }
  Z_grad[1:p, 1:p] = diag(diag(Z_grad[1:p, 1:p])) ## Maintain bipartite structure by setting observed v's to be independent
  Z_grad[(p+1):(p+m), (p+1):(p+m)] = diag(diag(Z_grad[(p+1):(p+m), (p+1):(p+m)])) ## Same as above - this time for hidden variables
  G = G - n*Z_grad
  return(G)
}

#' Internal function
#' @keywords internal
#' @noRd 
loglike_grad_CD = function(theta, V, p, m, k){
  n = nrow(V)
  G = matrix(0, p+m, p+m)
  
  b1 = diag(theta[1:p, 1:p])
  b2 = diag(theta[(p+1):(p+m), (p+1):(p+m)])
  W = theta[(p+1):(p+m), 1:p]
  
  V_k = V
  for(it in 1:k)
  {
    ## Update H given V ##
    U = V_k%*%t(W)
    E_H = matrix(0, n, m)
    H = matrix(0, n, m)
    for(i in 1:m)
    {
      E_H[,i] = exp(b2[i] + 2*U[,i])/(1+exp(b2[i] + 2*U[,i]))
      H[,i] = rbinom(n, 1, E_H[,i])
    }
    
    ## Update V given H ##
    U1 = H%*%W
    E_V = matrix(0, n, p)
    for(j in 1:p)
    {
      E_V[,j] = exp(b1[j] + 2*U1[,j])/(1+exp(b1[j] + 2*U1[,j]))
      V_k[,j] = rbinom(n, 1, E_V[,j])
    }
    
  }
  
  diag(G[1:p, 1:p]) = colSums(V - V_k)
  U_data = V%*%t(W) + matrix(b2, n, m, byrow = T)
  U_model = V_k%*%t(W) + matrix(b2, n, m, byrow = T)
  p_data = exp(U_data)/(1+exp(U_data))
  p_model = exp(U_model)/(1+exp(U_model))
  
  diag(G[(p+1):(p+m), (p+1):(p+m)]) = colSums(p_data - p_model)
  
  for(l in 1:n)
  {
    G[1:p, (p+1):(p+m)] = G[1:p, (p+1):(p+m)] + V[l,]%*%t(p_data[l,]) - V_k[l,]%*%t(p_model[l,])
  }
  G[(p+1):(p+m), 1:p] = G[1:p, (p+1):(p+m)]
  
  return(G)
}

#' Internal function
#' @keywords internal
#' @noRd 
RBM_marginal = function(v, theta, p, m, N){
  b1 = diag(theta[1:p, 1:p])
  b2 = diag(theta[(p+1):(p+m), (p+1):(p+m)])
  W = theta[(p+1):(p+m), 1:p]
  phi = diag(diag(theta))
  Z = z_theta_cpp_ising(theta, N, phi)
  u = W%*%v
  log_marginal_prob = -log(Z) + sum(b1*v) + sum(log(1 + exp(b2+2*u)))
  return(log_marginal_prob)
}



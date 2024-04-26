#' Fit the likelihood-based Exponential Graphical Model via maximum likelihood or penalized maximum likelihood. 
#' 
#' @param X Input dataset of dimension n x p; each row is an observation vector. 
#' @param model The model to be fitted. The model must be either "pgm" or "ising".
#' @param theta The initial value of the parameter. The default is 0.
#' @param lambda The penalty parameter. The default is NULL.
#' @param penalty The penalty type. The penalty must be either "none" or "l1". The default is "none".
#' @param NMC The number of Monte Carlo samples for approximate the gradient of the log-likelihood. The default is 500.
#' @param step_size The step size for the gradient descent. The default is 0.003.
#' @param epsilon The convergence criterion. The default is 0.00001.
#' @param max_iter The maximum number of iterations. The default is 3000.
#' @return Estimate of parameter theta, a p x p matrix.
#' @examples
#' p = 3
#' n = 100
#' theta0 = matrix(-0.8, p, p)
#' X_3 = IsingSim(n, theta0, 1000)
#' EGM.fit(X = X_3, model = "ising")
#' @export
EGM.fit <- function(X, model, theta = 0, lambda = NULL, penalty = "none", NMC = 500, step_size = 0.003, epsilon = 0.00001, max_iter = 3000) {
  if (penalty == "none" && length(lambda) == 0 && length(step_size) == 1) {
    res <-  proximal_grad_descent_try_catch(model= model,  X = X, step_size = step_size, theta = theta, N = NMC, epsilon = epsilon, max_iter = max_iter, lambda = lambda, penalty = "none")
    return(res)
  } else if (penalty == "l1") {
    res <-  proximal_grad_descent_try_catch(model= model,  X = X, step_size = step_size, theta = theta, N = NMC, epsilon = epsilon, max_iter = max_iter, lambda = lambda, penalty = "l1")
    return(res)
  } else {
    stop('Check the penalty input. The penalty must be either "none" or "l1"')
  }
}

#' This function function implements the k-fold cross-validation for EGM.fit, produces the optimal lambda and step size, and returns the resulting theta.
#' 
#' @param X Input dataset, of dimension n x p; each row is an observation vector.
#' @param model The model to be fitted. The model must be either "pgm" or "ising".
#' @param theta The initial value of the parameter. The default is 0.
#' @param lambda A gird of lambda (penalty parameter) values used for the EGM.fit.
#' @param NMC The number of Monte Carlo samples to approximate the gradient of the log-likelihood. The default is 500.
#' @param step_size A grid of step size values for the gradient descent. The default is 0.003.
#' @param epsilon The convergence criterion. Default is 0.0001.
#' @param max_iter The maximum number of iterations. The default is 3000.
#' @param k The number of folds for cross-validation. The default is 10.
#' @param par Logical. If TRUE, the function will run in parallel. The default is FALSE.
#' @param max_cores The maximum number of cores to be used for parallel computation. The default is 4.
#' @return A list containing the following elements: 
#' \itemize{
#'   \item theta_lasso: The estimate of parameter theta, a p x p matrix.
#'   \item lambda: The optimal lambda value.
#'   \item step_size: The optimal step size value.
#' }#'
#' @examples 
#' 
#' theta0 = gen_theta0(p = 10, omega = 0.8, eta = 0.05, coupling = "negative")
#' X = IsingSim(n = 100, theta = theta0, max_iter = 1000)
#' res = EGM.CV(X = X, model = "ising", lambda = c(1:15), step_size = c(0.001, 0.0005, 0.0001), k = 10, par = TRUE, max_cores = 50)
#' @export
EGM.CV <-function(X, model, theta = 0, lambda = 0,  NMC = 500, step_size = 0.003, epsilon = 0.0001, max_iter = 3000, k = 10, par = FALSE, max_cores = 4) {

  n = nrow(X)
  if (length(lambda) > 1 || length(step_size) > 1) {
    train_set <- vector("list", k)
    test_set <- vector("list", k)
    for(h in 1:k){
      start_ind = (h-1)*round(n/k) + 1
      end_ind = h*round(n/k)
      test_set[[h]] <- X[start_ind:end_ind,]
      train_set[[h]] <- X[-c(start_ind:end_ind),]
    }

    long_lambda_seq = rep(lambda, each = k*length(step_size))
    long_step_seq = rep(rep(step_size, each = k), length(lambda))
    dataset_index <- rep(c(1:k),length(lambda)*length(step_size))
    i = NULL
    if(par == TRUE){
      n_cores <- min(max_cores, parallel::detectCores() - 1)
      doParallel::registerDoParallel(cores = n_cores)
      lasso_list <- foreach::foreach(i = 1:length(dataset_index)) %dopar% parallel_descent_cross_validation(model, train_set, test_set, long_step_seq[[i]], long_lambda_seq[[i]], dataset_index[[i]],
        theta = theta, N = NMC, epsilon = epsilon, max_iter = max_iter, penalty = "l1"
      )
    } else {
      lasso_list <- lapply(1:length(dataset_index), function(i){parallel_descent_cross_validation(model, train_set, test_set, long_step_seq[[i]], long_lambda_seq[[i]], dataset_index[[i]],
        theta = theta, N = NMC, epsilon = epsilon, max_iter = max_iter, penalty = "l1"
      )})
    }

    result_table <- data.frame(t(matrix(unlist(lasso_list), nrow = 4)))
    colnames(result_table) = c("log_l", "lambda", "step_size", "cv_index")

    if(all(result_table$log_l == -Inf)){
      stop('The step size or lambda is too large. The log likelihood is -Inf. Try smaller step size or lambda.')
    }else{
      result_table<- result_table[result_table$log_l != -Inf,]
      result_table$step_lambda <- paste(result_table$step_size, result_table$lambda, sep = "_")
      count_table <- data.frame(table(result_table$step_lambda))

      temp_table <-data.frame((sapply(split(result_table$log_l, result_table$step_lambda), mean))) 
      temp_table$step_lambda <- rownames(temp_table)
      colnames(temp_table) <- c("log_l","Var1")
      temp_table <- merge(temp_table, count_table, by = "Var1", all = FALSE)
      temp_table$step_size <- as.numeric(unlist(strsplit(temp_table$Var1, "_"))[seq(1, length(temp_table$Var1)*2, by = 2)])
      temp_table$lambda <- as.numeric(unlist(strsplit(temp_table$Var1, "_"))[seq(2, length(temp_table$Var1)*2, by = 2)])
      temp_table <- temp_table[temp_table$Freq == k,]
      temp_res <- temp_table[which.max(temp_table$log_l),]
      res_lasso = proximal_grad_descent_try_catch(model = model, X = X, theta = theta, step_size = temp_res$step_size,  N = NMC, epsilon = epsilon, max_iter = max_iter, lambda = temp_res$lambda, penalty = "l1")
      return(list(theta_lasso = res_lasso, lambda = temp_res$lambda, step_size = temp_res$step_size))
    }
  } else {
    stop('The length of lambda or step size must be at least 2 to perform cross validation.')
  }

}

#' Select non-zero edges via a stable selection method for the estimator of the likelihood-based Exponential Graphical Model. Produce the non-zero probability for each edge and the selected edges based on the chosen cutoff threshold.
#' 
#' @param X Input dataset, of dimension n x p; each row is an observation vector.
#' @param model The model to be fitted. The model must be either "pgm" or "ising".
#' @param theta The initial value of the parameter. Default is 0.
#' @param lambda A gird of lambda (penalty parameter) values used for the EGM.fit. Expecting at least 5 different lambda values to perform the stable selection.
#' @param NMC The number of Monte Carlo samples for approximate the gradient of log likelihood. Default is 500.
#' @param step_size The step size for the gradient descent. Default is 0.003.
#' @param epsilon The convergence criterion. Default is 0.0001.
#' @param max_iter The maximum number of iterations. Default is 3000.
#' @param stable_pi The cutoff threshold for selecting the edges. Default is 0.6.
#' @param par Logical. If TRUE, the function will run in parallel. Default is FALSE.
#' @param max_cores The maximum number of cores to be used for parallel computation. Default is 4.
#' @return A list containing the following elements: 
#' \itemize{
#'   \item selection_prob: The non-zero probability for each edge, a p xp matrix.
#'   \item selected_edges: The selected edges based on chosen cuutoff threshold, a p x p matrix.
#' }
#' @examples 
#' 
#' theta0 = gen_theta0(p = 10, omega = 0.8, eta = 0.05, coupling = "negative")
#' X = IsingSim(n = 100, theta = theta0, max_iter = 1000)
#' res = EGM.SS(X = X, model = "ising", lambda = seq(5, 70, 5), step_size = 0.0005, par = TRUE, max_cores = 50)
#' @export
EGM.SS <- function(X, model, theta = 0, lambda = NULL,NMC = 500, step_size = 0.003, epsilon = 0.0001, max_iter = 3000, stable_pi = 0.6,  par = FALSE, max_cores = 4) {
 if (length(lambda) > 4) {
  if(par == TRUE){
    n_cores <- min(max_cores, parallel::detectCores() - 1)
    doParallel::registerDoParallel(cores = n_cores)
    i = NULL
    stable_theta <- foreach::foreach(i = 1:length(lambda)) %dopar% proximal_grad_descent_try_catch(model = model, theta = theta, step_size = step_size, X = X, N = NMC,
      epsilon = epsilon, max_iter = max_iter, lambda =lambda[i] , penalty = "l1"
      )
  } else {
    stable_theta <- lapply(lambda, function(y){proximal_grad_descent_try_catch(model = model, theta = theta, step_size = step_size, X = X, N = NMC,
      epsilon = epsilon, max_iter = max_iter, lambda = y, penalty = "l1"
      )})
  }
  
  selected_vech_mle <- sapply(stable_theta, function(x){selection_set(x)})
  stable_prob_mle <- apply(selected_vech_mle, 1, mean)

  return(list(selection_prob = invvech(stable_prob_mle), selected_edges = invvech(as.numeric(stable_prob_mle > stable_pi))))
  } else {
    stop('The length of lambda must be at least 5 to perform stable selection')
  }
}

#' Fit the Bayesian Exponential Graphical Model via HMC algorithm. We assume parameter theta belongs to the class of global-local scale mixtures of normals.
#' 
#' @param model The model to be fitted. The model must be either "pgm" or "ising".
#' @param X Input dataset, of dimension nobs x nvars; each row is an observation vector.
#' @param NMCMC The length of the MCMC chain for HMC. Default is 2000.
#' @param burnin The number of burnin samples. The default is 1000.
#' @param N The number of MCMC samples for approximate the gradient of log likelihood and log likelihood. Default is 500.
#' @param L The number of leapfrog steps. The default is 15.
#' @param step_size The step size for HMC. The default is 0.01.
#' @param a_lambda The shape parameter of the Gamma prior for the global variance parameter. The default is 0.01.
#' @param b_lambda The rate parameter of the Gamma prior for the global variance parameter. The default is 0.01.
#' @param tau The standard deviation of the Gaussian prior for diagonals in theta in the Ising model. The default is 10.
#' @return A list containing the following elements:
#' \itemize{
#' \item thetaout: The posterior samples of the model parameter theta, a (p * (p + 1) / 2) x (NMCMC - burnin) matrix. Each column is a vectorized version of one posterior sample of the parameter theta.
#' \item lambdaout: The posterior samples of the local variance parameter lambda, a (NMCMC - burnin) x 1 vector.
#' \item bayesout_theta: The posterior mean of the parameter theta, a p x p matrix.
#' \item MH_acceptance: The acceptance rate of the Metropolis-Hastings step.
#' \item Uout: The posterior probabiliry, a (NMCMC - burnin) x 1 vector.
#' }
#' @examples 
#' theta0 = matrix(-0.8, 3, 3)
#' X_3 = IsingSim(100, theta0, 1000)
#' EGM_Baysian(X = X_3, model = "ising")
#' @export
EGM_Baysian <- function(model, X, NMCMC = 2000, burnin = 1000, N = 1000, L = 5, step_size = 0.0001, a_lambda = 0.01, b_lambda = 0.01, tau = 10) {
  if(model == "pgm"){
    res <- PGM_bayes(X = X, nmcmc = NMCMC, burnin = burnin, N = N, epsilon = step_size, L = L, a_lambda = a_lambda , b_lambda = b_lambda)
    return(res)
  } else if(model == "ising"){
    res <- ising_bayes(X =X, nmcmc = NMCMC, burnin = burnin, tau = tau, N = N, epsilon = step_size, L = L, a_lambda = a_lambda , b_lambda = b_lambda)
    return(res)
  } else {
    stop('Check the model input. The model must be either "pgm" or "ising"')
  }
}


#' Fit the Boltzmann Machine via maximum likelihood
#' 
#' @param V The input dataset on visible varibales, of dimension nobs x nvars; each row is an observation vector.
#' @param p The number of visible variables. Default is ncol(V).
#' @param m The number of hidden variables. Defaut value is 50.
#' @param N The number of Monte Carlo samples to estimate z(theta). Default is 1000
#' @param step_size The step size for the gradient descent. Defaut is 0.00001
#' @param epsilon The convergence criterion. Defaut is 0.1
#' @param max_iter The maximum number of iterations. Defaut is 100
#' @return Estimate of the parameter theta, a (p+m) x (p+m) matrix.
#' @examples
#' n\ = 1000
#' p = 3
#' rho = 0.5
#' Sigma = (1-rho)*diag(p) + rho*matrix(1, p, p)
#' z = rmvnorm(n, sigma = Sigma)
#' y = matrix(0, n, p)
#' y[z>0] = 1
#' step_size = 0.001
#' k = 1
#' epsilon = 0.001
#' N = 1000
#' max_iter = 1000
#' res_rbm = RBM_fit(y, p, m = 2*p, N, step_size, epsilon, k, method = "likelihood", max_iter)
#' res_bm = BM_fit(y, p, m = 2*p, N, step_size = 0.0001, epsilon = 0.03, max_iter)
#' @export
BM_fit = function(V, p = ncol(V), m = 50, N = 1000, step_size =  0.00001, epsilon = 0.1, max_iter = 100)
{
  theta = matrix(0, p+m, p+m)
  #epsilon = 0.03
  #step_size = 0.001
  convergence_check = 10
  iter = 0 
  while(convergence_check > epsilon){
    theta_new = theta + (1/(iter+1))*step_size*loglike_grad_BM(theta, V, N, p, m)
    convergence_check = norm(theta_new - theta, "F")
    theta = theta_new
    if(iter>max_iter) 
    {
      break
    }
    iter = iter + 1
    #print(iter)
    print(convergence_check)
  }
  return(theta)
}

#' Reconstruct the data from the Boltzmann Machine
#' 
#' @param v The input data on visible variables.
#' @param theta The parameter of the Boltzmann Machine.
#' @param p The number of visible variables. The default is ncol(v).
#' @param m The number of hidden variables. The default is 50.
#' @param max_iter The maximum number of iterations. The default is 100.
#' @param L The number of samples to be generated. The default is 200.
#' @return The reconstructed data.
#' @examples
#' V_recon = BM_reconstruct(v, theta_BM, p, m, max_iter, L)
#' @export
BM_reconstruct = function(v, theta, p = ncol(v), m = 50, max_iter = 100, L = 200)
{
  p = length(v)
  theta_v = theta[1:p, 1:p]
  W = theta[1:p, (p+1):(p+m)]
  theta_h = theta[(p+1):(p+m), (p+1):(p+m)]
  theta_h_given_v = matrix(0, m, m)
  for(k in 1:m)
  {
    theta_h_given_v[k,k] = theta_h[k,k] + sum(W[,k]*v)
  }
  
  h = IsingSim(1, theta_h_given_v, max_iter)
  theta_v_given_h = matrix(0, p, p)
  for(j in 1:p)
  {
    theta_v_given_h[j,j] = theta_v[j,j] + sum(W[j,]*h)
  }
  V = IsingSim(L, theta_v_given_h, max_iter)
  return(V)
}

#' Fit the Restricted Boltzmann Machine via maximum likelihood method.
#' 
#' @param V The input dataset on visible varibales, of dimension n x p; each row is an observation vector.
#' @param p The number of visible variables.
#' @param m The number of hidden variables.
#' @param N The number of Monte Carlo samples to estimate z(theta).
#' @param step_size The step size for the gradient descent.
#' @param epsilon The convergence criterion.
#' @param k The number of Gibbs samples in CD.
#' @param method The method to be used for fitting the RBM. The method must be either "likelihood" or "CD". Default is "likelihood".
#' @param max_iter The maximum number of iterations.
#' @return Estimate of the parameter theta, a (p+m) x (p+m) matrix.
#' @examples 
#' n = 1000
#' p = 3
#' rho = 0.5
#' Sigma = (1-rho)*diag(p) + rho*matrix(1, p, p)
#' z = rmvnorm(n, sigma = Sigma)
#' y = matrix(0, n, p)
#' y[z>0] = 1
#' step_size = 0.001
#' k = 1
#' epsilon = 0.001
#' N = 1000
#' max_iter = 1000
#' res_rbm = RBM_fit(y, p, m = 2*p, N, step_size, epsilon, k, method = "likelihood", max_iter)
#' res_bm = BM_fit(y, p, m = 2*p, N, step_size = 0.0001, epsilon = 0.03, max_iter)
#' @export
RBM_fit = function(V, p = ncol(V), m = 30, N = 1000, step_size = 0.00001, epsilon = 0.06, k = 1, method = "likelihood", max_iter = 1000)
{
  if(method == "likelihood")
  {
    theta = matrix(0, p+m, p+m)
    diag(theta[1:p, 1:p]) = colMeans(V)
    convergence_check = 10
    iter = 0
    while(convergence_check > epsilon){
      theta_new = theta + (1/(iter+1))*step_size*loglike_grad_rbm1(theta, V, N, p, m)
      convergence_check = norm(theta_new - theta, "F")
      theta = theta_new
      #E_H = Exp_H(V, theta, p, m)
      #X = cbind(V, E_H)
      print(convergence_check)
      if(iter>max_iter) 
      {
        break
      }
      iter = iter + 1
    }
  }else if(method == "CD")
  {
    theta = matrix(0, p+m, p+m)
    diag(theta[1:p, 1:p]) = colMeans(V)
    convergence_check = 10
    iter = 0
    while(convergence_check > epsilon){
      theta_new = theta + (1/(iter+1))*step_size*loglike_grad_CD(theta, V, p, m, k)
      convergence_check = norm(theta_new - theta, "F")
      theta = theta_new
      #E_H = Exp_H(V, theta, p, m)
      #X = cbind(V, E_H)
      print(convergence_check)
      if(iter>max_iter) 
      {
        break
      }
      iter = iter + 1
    }
  }
  return(theta)
}

#' Reconstruct the data from the Restricted Boltzmann Machine
#' 
#' @param v The input data on visible variables.
#' @param theta The parameter of the Restricted Boltzmann Machine.
#' @param p The number of visible variables.
#' @param m The number of hidden variables.
#' @return The reconstructed data.
#' @export
RBM_reconstruct = function(v, theta, p = ncol(v), m = 30)
{
  b1 = diag(theta[1:p, 1:p])
  b2 = diag(theta[(p+1):(p+m), (p+1):(p+m)])
  W = theta[(p+1):(p+m), 1:p]
  h = rep(0, m)
  v1 = rep(0, p)
  v2 = rep(0, p)
  u = v%*%t(W)
  for(i in 1:m)
  {
    h[i] = rbinom(1, 1, exp(b2[i] + 2*u[i])/(1+exp(b2[i] + 2*u[i])))
  }
  
  u = h%*%W
  for(j in 1:p)
  {
    v1[j] = exp(b1[j] + 2*u[j])/(1+exp(b1[j] + 2*u[j]))
    v2[j] = rbinom(1, 1, v1[j])
  }
  result = list("v1" = v1, "v2" = v2)
  return(result)
}

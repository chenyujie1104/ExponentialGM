#include <RcppArmadillo.h>
#include <Rcpp.h>


// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

double sum_lfac(NumericVector y){
  return (sum(lfactorial(y)));
}

arma::vec rp(int n, double lambda) {
  return (rpois(n, lambda));
}

arma::vec fac(NumericVector y){
  NumericVector temp_v = factorial(y);
  arma::vec v(temp_v);
  return v;
}

// [[Rcpp::export]]
double prob_kernel_cpp_pgm(arma::vec y, arma::mat theta){
  double sum_lfac_y = sum_lfac(wrap(y));
  arma::mat temp_y = y * y.t();
  temp_y.diag() = y;

  arma::mat resultMatrix = temp_y % theta;
  double s = arma::accu(resultMatrix) - sum_lfac_y;

  return exp(s);
}

arma::mat d_kernel_matrix(arma::vec y){
  arma::mat temp_y = y * y.t();
  temp_y.diag() = y;
  return temp_y;
}


// [[Rcpp::export]]
List z_functions_cpp_pgm(arma::mat theta, int N, arma::mat phi){
  int p = theta.n_cols;
  arma::vec phi_vec = arma::diagvec(phi);
  arma::mat Y = arma::zeros(N, p);
  arma::mat q_phi = arma::zeros(N, p);

  for(int j=0; j<p; j++){
    double t = exp(phi_vec(j));
    Y(arma::span(0, N-1), j) = rp(N, t);
  }

  double s = 0;
  arma::mat S = arma::zeros(p, p);
  Y = Y.t();
  for(int i =0; i<N; i++){
    arma::mat d_ratio = exp(Y(arma::span(0, p-1), i).t()*theta*Y(arma::span(0, p-1), i) - Y(arma::span(0, p-1), i).t()*arma::diagmat(theta)*Y(arma::span(0, p-1), i));
    s = s + arma::as_scalar(d_ratio);
    S = S + arma::as_scalar(d_ratio)*(d_kernel_matrix(Y(arma::span(0, p-1), i)));
  }
  double Z_phi = 1;
  double Z_theta = Z_phi*(s/N);
  arma::mat Z_prime_theta = S/N;
  List result;
  result["Z_theta"] = Z_theta;
  result["Z_prime_theta"] = Z_prime_theta;
  return result;
}

// [[Rcpp::export]]
arma::mat loglike_grad_cpp_pgm(arma::mat theta, arma::mat X, int N){
  int n = X.n_rows;
  int p = X.n_cols;
  arma::mat phi = arma::diagmat(arma::diagvec(theta));
  arma::mat G = arma::zeros(p, p);
  List res = z_functions_cpp_pgm(theta, N, phi);
  double Z = res[0];
  arma::mat Z_grad = res[1];
  G = X.t()*X;
  G.diag() = sum(X, 0);
  G = G  - n*Z_grad/Z;
  return G;
}

// [[Rcpp::export]]
double z_theta_cpp_pgm(arma::mat theta, int N, arma::mat phi){
  int p = theta.n_cols;
  arma::vec phi_vec = arma::diagvec(phi);
  arma::mat Y = arma::zeros(N,p);
  arma::mat q_phi = arma::zeros(N,p);

  for(int j = 0; j<p; j++){
    double t = exp(phi_vec(j));
    Y(arma::span(0, N-1), j) = rp(N, t);
    arma::vec f = fac(wrap(Y(arma::span(0, N-1), j)));
    q_phi(arma::span(0, N-1), j) = Y(arma::span(0, N-1), j);
    q_phi(arma::span(0, N-1), j).for_each([t](arma::vec::elem_type& val){ val = pow(t, val); });
    q_phi(arma::span(0, N-1), j) =  q_phi(arma::span(0, N-1), j)/f;
  }

  double s = 0;
  Y = Y.t();
  q_phi = q_phi.t();
  for(int i =0; i<N; i++){
    s = s + prob_kernel_cpp_pgm(Y(arma::span(0, p-1), i),theta)/arma::prod(q_phi(arma::span(0, p-1), i));
  }
  double Z_phi = exp(sum(exp(phi_vec)));
  double Z_theta = Z_phi*(s/N);
  return Z_theta;
}

// [[Rcpp::export]]
double loglike_cpp_pgm(arma::mat theta, arma::mat X, int N){
  int n = X.n_rows;
  int p = X.n_cols;
  arma::mat phi = arma::diagmat(arma::diagvec(theta));
  double Z = z_theta_cpp_pgm(theta, N, phi);
  double s = 0;
  X = X.t();
  for(int i=0; i<n; i++){
    s = s + log(prob_kernel_cpp_pgm(X(arma::span(0, p-1), i), theta));
  }
  s = s - n*log(Z);
  return s;

}

// [[Rcpp::export]]
List HMC_cpp_pgm(Function U, Function grad_U, double epsilon, int L, arma::vec current_q, arma::mat matrix_A){
  arma::vec q = current_q;
  int m = q.size();
  arma::vec p = arma::randn(m, 1); // independent standard normal variates
  arma::vec current_p = p;
  arma::vec A_theta = matrix_A*q;
  bool anygreaterThanZero = any(A_theta > 0);

  // Make a half step for momentum at the beginning
  arma::vec g_vec = as<arma::vec> (grad_U(q));
  p = p - epsilon*g_vec/2;

  // Alternate full steps for position and momentum

  for(int i=1; i<=L; i++)
  {
    // Make a full step for the position

    q = q + epsilon * p;

    // Make a full step for the momentum, except at end of trajectory
    A_theta = matrix_A*q;
    anygreaterThanZero = any(A_theta > 0);
    if(i!=L){

      if(anygreaterThanZero){
        p = p - 2*matrix_A*p;
      }else{
        p = p - epsilon * g_vec;
      }
    }
  }
  // Make a half step for momentum at the end.
  q = q + epsilon * p;

  A_theta = matrix_A*q;
  anygreaterThanZero = any(A_theta>0);

  int MH_count = 0;
  if(anygreaterThanZero){
    current_q = current_q;
  }else{
    p = p - epsilon * g_vec/2;
    p = -p;

    double current_U = as<double>(U(current_q));
    double current_K = arma::sum(current_p%current_p)/2;
    double proposed_U = as<double>(U(q));
    double proposed_K = arma::sum(p%p)/2;
    if (arma::randu() < exp(current_U-proposed_U+current_K-proposed_K))
    {
      MH_count = MH_count + 1;
      // print(MH_count)
      current_q = q;
    }else{
      current_q = current_q;
    }
  }

  List result;
  result["MH_count"] = MH_count;
  result["q"] = current_q;
  return result;
}

arma::vec rb(int n, int m, double p) {
  return(rbinom(n,m,p));
}
arma::vec db(arma::vec y, double theta){
  int n = y.size();
  arma::vec r = arma::zeros(n);
  for(int i = 0; i<n; i++){
    r(i) = pow(theta, y(i))*pow(1 - theta, 1 - y(i));
  }
  return r;
}


// [[Rcpp::export]]

List z_functions_cpp_ising(arma::mat theta, int N, arma::mat phi){
  int p = theta.n_cols;
  arma::vec phi_vec = arma::diagvec(phi);
  arma::mat Y = arma::zeros(N, p);
  arma::mat q_phi = arma::zeros(N, p);
  for(int j=0; j<p; j++){
    double t = exp(phi_vec(j))/(1+exp(phi_vec(j)));
    arma::vec b_samples = rb(N, 1, t);
    Y(arma::span(0, N-1), j) = rb(N, 1, t);
  }

  double s= 0;
  arma::mat S = arma::zeros(p, p);
  Y = Y.t();
  for(int i =0; i<N; i++){
    arma::mat d_ratio = exp(Y(arma::span(0, p-1), i).t()*theta*Y(arma::span(0, p-1), i) - Y(arma::span(0, p-1), i).t()*arma::diagmat(theta)*Y(arma::span(0, p-1), i));
    s = s + arma::as_scalar(d_ratio);
    S = S + arma::as_scalar(d_ratio)*(2*Y(arma::span(0, p-1), i)*Y(arma::span(0, p-1), i).t() - arma::diagmat(Y(arma::span(0, p-1),i)));
  }
  double Z_phi = 1;
  double Z_theta = Z_phi*(s/N);
  arma::mat Z_prime_theta = S/N;
  List result;
  result["Z_theta"] = Z_theta;
  result["Z_prime_theta"] = Z_prime_theta;
  return result;
}

// [[Rcpp::export]]

arma::mat loglike_grad_cpp_ising(arma::mat theta, arma::mat X, int N){
  int n = X.n_rows;
  int p = X.n_cols;
  arma::mat phi = arma::diagmat(arma::diagvec(theta));
  arma::mat G = arma::zeros(p, p);
  List res = z_functions_cpp_ising(theta, N, phi);
  double Z = res[0];
  arma::mat Z_grad = res[1];
  G = 2*X.t()*X;
  G.diag() = sum(X, 0);
  G = G  - n*Z_grad/Z;
  return G;
}




// [[Rcpp::export]]
double prob_kernel_cpp_ising(arma::vec y, arma::mat theta){

  arma::mat temp_y = y * y.t();
  arma::mat resultMatrix = temp_y % theta;
  double s = arma::accu(resultMatrix);

  return exp(s);
}

// [[Rcpp::export]]
double z_theta_cpp_ising(arma::mat theta, int N, arma::mat phi){
  int p = theta.n_cols;
  arma::vec phi_vec = arma::diagvec(phi);
  arma::mat Y = arma::zeros(N, p);
  arma::mat q_phi = arma::zeros(N, p);
  for(int j=0; j<p; j++){
    double t = exp(phi_vec(j))/(1+exp(phi_vec(j)));
    arma::vec b_samples = rb(N, 1, t);
    Y(arma::span(0, N-1), j) = rb(N, 1, t); //b_samples; //rb(N, 1, t);
    q_phi(arma::span(0, N-1), j) = db(Y(arma::span(0, N-1), j), t)*(1+exp(phi_vec(j)));
  }
  double s= 0;
  Y = Y.t();
  q_phi = q_phi.t();
  for(int i =0; i<N; i++){
    s = s+ prob_kernel_cpp_ising(Y(arma::span(0, p-1), i), theta)/arma::prod(q_phi(arma::span(0, p-1), i));
  }
  double Z_phi = prod(1 + exp(phi_vec));
  double Z_theta = Z_phi*(s/N);
  return Z_theta;
}

// [[Rcpp::export]]

double loglike_cpp_ising(arma::mat theta, arma::mat X, int N){
  int n = X.n_rows;
  int p = X.n_cols;
  arma::mat phi = arma::diagmat(arma::diagvec(theta));
  double Z = z_theta_cpp_ising(theta, N, phi);
  double s = 0;
  X = X.t();
  for(int i=0; i<n; i++){
    s = s + log(prob_kernel_cpp_ising(X(arma::span(0, p-1), i), theta));
  }
  s = s - n*log(Z);
  return s;
}

// [[Rcpp::export]]
List HMC_cpp_ising(Function U, Function grad_U, double epsilon, int L, arma::vec current_q){
  arma::vec q = current_q;
  int m = q.size();
  arma::vec p = arma::randn(m, 1); // independent standard normal variates
  arma::vec current_p = p;

  // Make a half step for momentum at the beginning
  arma::vec g_vec = as<arma::vec> (grad_U(q));
  p = p - epsilon*g_vec/2;

  // Alternate full steps for position and momentum

  for(int i=1; i<=L; i++)
  {
    // Make a full step for the position

    q = q + epsilon * p;

    // Make a full step for the momentum, except at end of trajectory

    if(i!=L){
      p = p - epsilon * g_vec;
    }
  }
  // Make a half step for momentum at the end.

  p = p - epsilon * g_vec/2;

  // Negate momentum at end of trajectory to make the proposal symmetric

  p = -p;

  // Evaluate potential and kinetic energies at start and end of trajectory

  double current_U = as<double>(U(current_q));
  double current_K = arma::sum(current_p%current_p)/2;
  double proposed_U = as<double>(U(q));
  double proposed_K = arma::sum(p%p)/2;

  // Accept or reject the state at end of trajectory, returning either
  // the position at the end of the trajectory or the initial position
  int MH_count = 0;
  if (arma::randu() < exp(current_U-proposed_U+current_K-proposed_K))
  {
    MH_count = MH_count + 1;
    // print(MH_count)
    current_q = q;
  }else{
    current_q = current_q;
  }
  List result;
  result["MH_count"] = MH_count;
  result["q"] = current_q;
  return result;
}

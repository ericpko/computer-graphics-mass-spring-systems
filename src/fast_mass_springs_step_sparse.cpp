#include "fast_mass_springs_step_sparse.h"
#include <igl/matlab_format.h>

typedef Eigen::Triplet<double> T;


void fast_mass_springs_step_sparse(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXi & b,
  const double delta_t,
  const Eigen::MatrixXd & fext,
  const Eigen::VectorXd & r,
  const Eigen::SparseMatrix<double>  & M,
  const Eigen::SparseMatrix<double>  & A,
  const Eigen::SparseMatrix<double>  & C,
  const Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > & prefactorization,
  const Eigen::MatrixXd & Uprev,
  const Eigen::MatrixXd & Ucur,
  Eigen::MatrixXd & Unext)
{
  //////////////////////////////////////////////////////////////////////////////
  // Matrix y (eq. 16) doesn't change inside the loop, so we can precompute it
  Eigen::MatrixXd y = (1.0 / pow(delta_t, 2.0)) * M * (2 * Ucur - Uprev) + fext;

  // Compute Wp = w * C^T * C * p^{rest}. See iPad for calc
  Eigen::MatrixXd Wp = (1.0e10 * C.transpose() * C) * V;
  y += Wp;


  // Conduct a single step of the "Fast Simulation of Mass-Spring Systems" method
  Unext = Ucur;
  for (int iter = 0; iter < 50; iter++) {

    // Step 1 (local): Given current values of 𝐩 determine 𝐝𝑖𝑗 for each spring.
    Eigen::MatrixXd d(E.rows(), 3);
    for (int e = 0; e < E.rows(); e++) {
      int i = E(e, 0);
      int j = E(e, 1);

      Eigen::Vector3d p_i = Unext.row(i);
      Eigen::Vector3d p_j = Unext.row(j);

      // See SIGGRAPH paper “Fast Simulation of Mass-Spring Systems”
      Eigen::Vector3d d_ij = r(e) * (p_i - p_j).normalized();

      d.row(e) = d_ij;
    }


    // Step 2 (global): Given all 𝐝𝑖𝑗 vectors, find positions 𝐩 that minimize quadratic energy E.
    Eigen::MatrixXd b_ = k * A.transpose() * d + y;


    // Compute p^{t + Δt} = argmin p
    Unext = prefactorization.solve(b_);
  }
  //////////////////////////////////////////////////////////////////////////////
}

#include "fast_mass_springs_precomputation_dense.h"
#include "signed_incidence_matrix_dense.h"
#include <Eigen/Dense>

bool fast_mass_springs_precomputation_dense(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXd & m,
  const Eigen::VectorXi & b,
  const double delta_t,
  Eigen::VectorXd & r,
  Eigen::MatrixXd & M,
  Eigen::MatrixXd & A,
  Eigen::MatrixXd & C,
  Eigen::LLT<Eigen::MatrixXd> & prefactorization)
{
  /////////////////////////////////////////////////////////////////////////////
  // Set up the list of edge lengths (rest spring length)
  r.resize(E.rows());
  for (int e = 0; e < E.rows(); e++) {
    int i = E(e, 0);
    int j = E(e, 1);

    // Eigen::Vector3d p0(V(i, 0), V(i, 1), V(i, 2));
    Eigen::Vector3d p_i = V.row(i);
    Eigen::Vector3d p_j = V.row(j);

    // Set r
    r(e) = (p_i - p_j).norm();
  }

  // Get the mass matrix
  M = m.asDiagonal();

  // Get the signed incidence matrix A
  signed_incidence_matrix_dense(V.rows(), E, A);

  // Set the selection matrix
  C = Eigen::MatrixXd::Zero(b.size(), V.rows());
  for (int i = 0; i < b.size(); i++) {
    for (int j = 0; j < V.rows(); j++) {
      if (b(i) == j) {
        C(i, j) = 1;
      } else {
        C(i, j) = 0;
      }
    }
  }

  // Compute W = w C^T * C                         See iPad
  Eigen::MatrixXd W = 1.0e10 * C.transpose() * C;

  // Compute Q --- NOTE: We add W to Q for the pinned vertices
  Eigen::MatrixXd Q = k * A.transpose() * A + (1.0 / pow(delta_t, 2.0)) * M + W;
  /////////////////////////////////////////////////////////////////////////////
  prefactorization.compute(Q);
  return prefactorization.info() != Eigen::NumericalIssue;
}

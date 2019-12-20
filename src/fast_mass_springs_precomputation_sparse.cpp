#include "fast_mass_springs_precomputation_sparse.h"
#include "signed_incidence_matrix_sparse.h"
#include <vector>

typedef Eigen::Triplet<double> T;


bool fast_mass_springs_precomputation_sparse(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXd & m,
  const Eigen::VectorXi & b,
  const double delta_t,
  Eigen::VectorXd & r,
  Eigen::SparseMatrix<double>  & M,
  Eigen::SparseMatrix<double>  & A,
  Eigen::SparseMatrix<double>  & C,
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > & prefactorization)
{
  /////////////////////////////////////////////////////////////////////////////
  // Number of point masses
  const int n = V.rows();

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

  // Get the sparse mass matrix
  std::vector<T> ijv;
  for (int i = 0; i < m.size(); i++) {
    ijv.emplace_back(T(i, i, m(i)));
  }
  M.resize(n, n);
  M.setFromTriplets(ijv.begin(), ijv.end());

  // Get the signed incidence matrix A
  signed_incidence_matrix_sparse(n, E, A);

  // Set the selection matrix C
  ijv.clear();
  // std::vector<T> triplet_list;
  for (int i = 0; i < b.size(); i++) {
    for (int j = 0; j < n; j++) {
      if (b(i) == j) {
        ijv.emplace_back(T(i, j, 1));
      }
    }
  }
  C.resize(b.size(), n);
  C.setFromTriplets(ijv.begin(), ijv.end());

  // Compute sparse matrix W
  Eigen::SparseMatrix<double> W(n, n);
  W = 1.0e10 * C.transpose() * C;


  // Compute Q -> nxn
  Eigen::SparseMatrix<double> Q(n, n);
  Q = k * A.transpose() * A + (1.0 / pow(delta_t, 2.0)) * M + W;
  /////////////////////////////////////////////////////////////////////////////
  prefactorization.compute(Q);
  return prefactorization.info() != Eigen::NumericalIssue;
}

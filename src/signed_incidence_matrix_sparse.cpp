#include "signed_incidence_matrix_sparse.h"
#include <vector>


typedef Eigen::Triplet<double> T;


void signed_incidence_matrix_sparse(
  const int n,
  const Eigen::MatrixXi & E,
  Eigen::SparseMatrix<double>  & A)
{
  //////////////////////////////////////////////////////////////////////////////
  // For each spring
  std::vector<T> triplet_list;
  for (int e = 0; e < E.rows(); e++) {
    // Get the edge/spring ij
    int i = E(e, 0);
    int j = E(e, 1);

    // For each vertex/point mass
    for (int k = 0; k < n; k++) {
      if (k == i) {
        triplet_list.emplace_back(T(e, k, 1));
      } else if (k == j) {
        triplet_list.emplace_back(T(e, k, -1));
      }
    }
  }

  A.resize(E.rows(), n);
  A.setFromTriplets(triplet_list.begin(), triplet_list.end());
  //////////////////////////////////////////////////////////////////////////////
}

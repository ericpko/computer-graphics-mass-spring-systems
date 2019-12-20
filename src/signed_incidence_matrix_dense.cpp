#include "signed_incidence_matrix_dense.h"

void signed_incidence_matrix_dense(
  const int n,
  const Eigen::MatrixXi & E,
  Eigen::MatrixXd & A)
{
  //////////////////////////////////////////////////////////////////////////////
  // A is in R^ #springs x #point masses == E.rows x n
  A = Eigen::MatrixXd::Zero(E.rows(), n);

  // For each spring
  for (int e = 0; e < E.rows(); e++) {
    // Get the edge/spring ij
    int i = E(e, 0);
    int j = E(e, 1);

    // For each vertex/point mass
    for (int k = 0; k < n; k++) {
      if (k == i) {
        A(e, k) = 1;
      } else if (k == j) {
        A(e, k) = -1;
      } else {
        A(e, k) = 0;
      }
    }
  }
  //////////////////////////////////////////////////////////////////////////////
}

/****************************************************************************
 * Copyright (c) 2024 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include <ArborX_DetailsSymmetricEigenProblem.hpp>
#include <ArborX_DetailsVector.hpp>

#include <Kokkos_Core.hpp>

#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE EigenProblem

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE(eigen_problem_2D)
{
  constexpr int DIM = 2;
  using Coordinate = float;

  using Vector = ArborX::Details::Vector<DIM, Coordinate>;
  using Matrix = Kokkos::Array<Kokkos::Array<Coordinate, DIM>, DIM>;

  using ArborX::Details::symmetricEigenProblem;

  Coordinate eigs[DIM];
  Vector eigv[DIM];

  // Two vectors are proportional if they line on the same line, meaning the
  // angle between them is 0 or 180.
  auto eigv_compare = [](Vector const &v1, Vector const &v2) {
    return std::abs(std::abs(v1.dot(v2)) - v1.norm() * v2.norm()) <
           std::numeric_limits<Coordinate>::epsilon();
  };

  {
    // zero matrix
    Matrix A = {{{0, 0}, {0, 0}}};
    symmetricEigenProblem(A, eigs, eigv);
    BOOST_TEST(eigs[0] == 0);
    BOOST_TEST(eigs[1] == 0);
    BOOST_TEST(eigv_compare(eigv[0], Vector{1, 0}));
    BOOST_TEST(eigv_compare(eigv[1], Vector{0, 1}));
  }

  {
    // One non-zero element on diagonal
    Matrix A = {{{2, 0}, {0, 0}}};
    symmetricEigenProblem(A, eigs, eigv);
    BOOST_TEST(eigs[0] == 2);
    BOOST_TEST(eigs[1] == 0);
    BOOST_TEST(eigv_compare(eigv[0], Vector{1, 0}));
    BOOST_TEST(eigv_compare(eigv[1], Vector{0, 1}));
  }
  {
    // Diagonal matrix
    Matrix A = {{{2, 0}, {0, 3}}};
    symmetricEigenProblem(A, eigs, eigv);
    BOOST_TEST(eigs[0] == 2);
    BOOST_TEST(eigs[1] == 3);
    BOOST_TEST(eigv_compare(eigv[0], Vector{1, 0}));
    BOOST_TEST(eigv_compare(eigv[1], Vector{0, 1}));
  }
  {
    // Matrix with a non-trivial null space
    Matrix A = {{{1, -1}, {-1, 1}}};
    symmetricEigenProblem(A, eigs, eigv);
    BOOST_TEST(eigs[0] == 0);
    BOOST_TEST(eigs[1] == 2);
    BOOST_TEST(eigv_compare(eigv[0], Vector{1, -1}));
    BOOST_TEST(eigv_compare(eigv[1], Vector{1, 1}));
  }
  {
    // Matrix with duplicate eigenvalues
    // The only 2x2 matrices with repeated eigenvalues are
    // | a 0 |
    // | 0 a |
    // Any orthogonal pair of vectors would be correct. But we treat it as a
    // special case of offdiagonal 0, and get axis aligned vectors.
    Matrix A = {{{2, 0}, {0, 2}}};
    symmetricEigenProblem(A, eigs, eigv);
    BOOST_TEST(eigs[0] == 2);
    BOOST_TEST(eigs[1] == 2);
    BOOST_TEST(eigv_compare(eigv[0], Vector{1, 0}));
    BOOST_TEST(eigv_compare(eigv[1], Vector{0, 1}));
  }
  {
    // General matrix
    Matrix A = {{{3, 1}, {1, 3}}};
    symmetricEigenProblem(A, eigs, eigv);
    BOOST_TEST(eigs[0] == 2);
    BOOST_TEST(eigs[1] == 4);
    BOOST_TEST(eigv_compare(eigv[0], Vector{1, 1}));
    BOOST_TEST(eigv_compare(eigv[1], Vector{1, -1}));
  }
}

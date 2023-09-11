/****************************************************************************
 * Copyright (c) 2023 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "ArborX_EnableDeviceTypes.hpp"
#include <interpolation/details/ArborX_InterpDetailsSymmetricPseudoInverseSVD.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;
namespace axid = ArborX::Interpolation::Details;

// Checks for equality betweem 2D or 3D views
template <typename U, typename V>
void equalityCheck(U const &u, V const &v,
                   double tol = Kokkos::Experimental::epsilon_v<float>)
{
  if constexpr (U::rank == 2)
  {
    for (std::size_t i = 0; i < u.extent(0); i++)
      for (std::size_t j = 0; j < u.extent(1); j++)
        BOOST_TEST(u(i, j) == v(i, j), tt::tolerance(tol));
  }
  else if constexpr (U::rank == 3)
  {
    for (std::size_t i = 0; i < u.extent(0); i++)
      for (std::size_t j = 0; j < u.extent(1); j++)
        for (std::size_t k = 0; k < u.extent(2); k++)
          BOOST_TEST(u(i, j, k) == v(i, j, k), tt::tolerance(tol));
  }
}

// Makes a unmanaged view from a 2D C array
template <typename U, std::size_t N>
auto viewHostRef2(double (&a)[N][N])
{
  return typename U::HostMirror(&a[0][0], N, N);
}

// Makes a unmanaged view from a 3D C array
template <typename U, std::size_t M, std::size_t N>
auto viewHostRef3(double (&a)[M][N][N])
{
  return typename U::HostMirror(&a[0][0][0], M, N, N);
}

// Computes the pseudo-inverse from a single matrix
template <typename ES, typename U, std::size_t N>
auto makePseudoInverse(ES const &es, typename U::HostMirror const &a)
{
  U inv("inv", N, N);
  Kokkos::deep_copy(es, inv, a);
  axid::symmetricPseudoInverseSVD(es, inv);
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, inv);
}

// Computes the pseudo-inverse from a list of matrices
template <typename ES, typename U, std::size_t M, std::size_t N>
auto makePseudoInverseList(ES const &es, typename U::HostMirror const &a)
{
  U inv("inv", M, N, N);
  Kokkos::deep_copy(es, inv, a);
  axid::symmetricPseudoInverseSVD(es, inv);
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, inv);
}

// Makes the case for a single matrix
template <typename ES, typename U, std::size_t N>
void makeCase(ES const &es, double (&src)[N][N], double (&ref)[N][N])
{
  auto rsrc = viewHostRef2<U, N>(src);
  auto rref = viewHostRef2<U, N>(ref);
  auto inv = makePseudoInverse<ES, U, N>(es, rsrc);
  equalityCheck(rref, inv);
}

// Makes the case for a list of matrices
template <typename ES, typename U, std::size_t M, std::size_t N>
void makeCaseList(ES const &es, double (&src)[M][N][N], double (&ref)[M][N][N])
{
  auto rsrc = viewHostRef3<U, M, N>(src);
  auto rref = viewHostRef3<U, M, N>(ref);
  auto inv = makePseudoInverseList<ES, U, M, N>(es, rsrc);
  equalityCheck(rref, inv);
}

// Pseudo-inverses are computed using numpy's "linalg.pinv" solver
// If possible, results will be written with their exact values (as ratios)
// Layout is forced to be LayoutRight so that matrices are represented in the
// same way under host and device code

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_symm2, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  using view_t = Kokkos::View<double **, Kokkos::LayoutRight, MemorySpace>;
  ExecutionSpace space{};

  double mat0[2][2] = {{1, 2}, {2, 3}};
  double inv0[2][2] = {{-3, 2}, {2, -1}};
  makeCase<ExecutionSpace, view_t, 2>(space, mat0, inv0);

  double mat1[2][2] = {{4, 0}, {0, 4}};
  double inv1[2][2] = {{1 / 4., 0}, {0, 1 / 4.}};
  makeCase<ExecutionSpace, view_t, 2>(space, mat1, inv1);

  double mat2[2][2] = {{0, 5}, {5, 0}};
  double inv2[2][2] = {{0, 1 / 5.}, {1 / 5., 0}};
  makeCase<ExecutionSpace, view_t, 2>(space, mat2, inv2);

  double mat3[2][2] = {{2, 2}, {2, 2}};
  double inv3[2][2] = {{1 / 8., 1 / 8.}, {1 / 8., 1 / 8.}};
  makeCase<ExecutionSpace, view_t, 2>(space, mat3, inv3);

  double mat4[2][2] = {{1, -3}, {-3, 1}};
  double inv4[2][2] = {{-1 / 8., -3 / 8.}, {-3 / 8., -1 / 8.}};
  makeCase<ExecutionSpace, view_t, 2>(space, mat4, inv4);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_symm3, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  using view_t = Kokkos::View<double **, Kokkos::LayoutRight, MemorySpace>;
  ExecutionSpace space{};

  double mat0[3][3] = {{2, 2, 3}, {2, 0, 1}, {3, 1, -2}};
  double inv0[3][3] = {{-1 / 18., 7 / 18., 2 / 18.},
                       {7 / 18., -13 / 18., 4 / 18.},
                       {2 / 18., 4 / 18., -4 / 18.}};
  makeCase<ExecutionSpace, view_t, 3>(space, mat0, inv0);

  double mat1[3][3] = {{0, 1, 2}, {1, 2, 3}, {2, 3, 4}};
  double inv1[3][3] = {{-5 / 6., -1 / 6., 3 / 6.},
                       {-1 / 6., 0, 1 / 6.},
                       {3 / 6., 1 / 6., -1 / 6.}};
  makeCase<ExecutionSpace, view_t, 3>(space, mat1, inv1);

  double mat2[3][3] = {{5, 5, 5}, {5, 5, 5}, {5, 5, 5}};
  double inv2[3][3] = {{1 / 45., 1 / 45., 1 / 45.},
                       {1 / 45., 1 / 45., 1 / 45.},
                       {1 / 45., 1 / 45., 1 / 45.}};
  makeCase<ExecutionSpace, view_t, 3>(space, mat2, inv2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_symm128, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  using view_t = Kokkos::View<double **, Kokkos::LayoutRight, MemorySpace>;
  ExecutionSpace space{};

  // 128x128 matrix full of -2
  // eigenvalues are -2*128 and 127 0s
  // pseudo-inverse is a 128x128 matrix of 1 / (128 * 128 * -2)
  double mat[128][128] = {};
  double inv[128][128] = {};
  for (std::size_t i = 0; i < 128; i++)
    for (std::size_t j = 0; j < 128; j++)
    {
      mat[i][j] = -2;
      inv[i][j] = 1 / (128 * 128 * -2.);
    }
  makeCase<ExecutionSpace, view_t, 128>(space, mat, inv);

  // Case for invertible 128x128 matrix
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_scalar_like, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  using view_t = Kokkos::View<double **, Kokkos::LayoutRight, MemorySpace>;
  ExecutionSpace space{};

  double mat0[1][1] = {{2}};
  double inv0[1][1] = {{1 / 2.}};
  makeCase<ExecutionSpace, view_t, 1>(space, mat0, inv0);

  double mat1[1][1] = {{0}};
  double inv1[1][1] = {{0}};
  makeCase<ExecutionSpace, view_t, 1>(space, mat1, inv1);
}

#if 0
BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_empty, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  using view_t = Kokkos::View<double **, Kokkos::LayoutRight, MemorySpace>;
  ExecutionSpace space{};

  view_t mat(0, 0);
  axid::symmetricPseudoInverseSVD(space, mat);
  BOOST_TEST(mat.size() == 0);
}
#endif

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_symm2_list, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  using view_t = Kokkos::View<double ***, Kokkos::LayoutRight, MemorySpace>;
  ExecutionSpace space{};

  double mat[5][2][2] = {{{1, 2}, {2, 3}},
                         {{4, 0}, {0, 4}},
                         {{0, 5}, {5, 0}},
                         {{2, 2}, {2, 2}},
                         {{1, -3}, {-3, 1}}};
  double inv[5][2][2] = {{{-3, 2}, {2, -1}},
                         {{1 / 4., 0}, {0, 1 / 4.}},
                         {{0, 1 / 5.}, {1 / 5., 0}},
                         {{1 / 8., 1 / 8.}, {1 / 8., 1 / 8.}},
                         {{-1 / 8., -3 / 8.}, {-3 / 8., -1 / 8.}}};
  makeCaseList<ExecutionSpace, view_t, 5, 2>(space, mat, inv);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_symm3_list, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  using view_t = Kokkos::View<double ***, Kokkos::LayoutRight, MemorySpace>;
  ExecutionSpace space{};

  double mat[3][3][3] = {{{2, 2, 3}, {2, 0, 1}, {3, 1, -2}},
                         {{0, 1, 2}, {1, 2, 3}, {2, 3, 4}},
                         {{5, 5, 5}, {5, 5, 5}, {5, 5, 5}}};
  double inv[3][3][3] = {{{-1 / 18., 7 / 18., 2 / 18.},
                          {7 / 18., -13 / 18., 4 / 18.},
                          {2 / 18., 4 / 18., -4 / 18.}},
                         {{-5 / 6., -1 / 6., 3 / 6.},
                          {-1 / 6., 0, 1 / 6.},
                          {3 / 6., 1 / 6., -1 / 6.}},
                         {{1 / 45., 1 / 45., 1 / 45.},
                          {1 / 45., 1 / 45., 1 / 45.},
                          {1 / 45., 1 / 45., 1 / 45.}}};
  makeCaseList<ExecutionSpace, view_t, 3, 3>(space, mat, inv);
}
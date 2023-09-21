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
#include "ArborX_EnableViewComparison.hpp"
#include <interpolation/details/ArborX_InterpDetailsSymmetricPseudoInverseSVD.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

template <typename ES, typename U, typename V>
void makeCase(ES const &es, V &src_arr, V &ref_arr, int n, int m)
{
  using host_view = typename U::HostMirror;

  host_view src("Testing::src", m, n, n);
  host_view ref("Testing::ref", m, n, n);
  U inv("Testing::inv", m, n, n);

  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
      {
        src(i, j, k) = src_arr[i][j][k];
        ref(i, j, k) = ref_arr[i][j][k];
      }

  Kokkos::deep_copy(es, inv, src);
  ArborX::Interpolation::Details::symmetricPseudoInverseSVD(es, inv);
  ARBORX_MDVIEW_TEST_TOL(ref, inv, Kokkos::Experimental::epsilon_v<float>);
}

// Pseudo-inverses were computed using numpy's "linalg.pinv" solver and
// simplified to be ratios

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_symm2, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  using view_t = Kokkos::View<double ***, MemorySpace>;
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
  makeCase<ExecutionSpace, view_t, double[5][2][2]>(space, mat, inv, 2, 5);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_symm3, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  using view_t = Kokkos::View<double ***, MemorySpace>;
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
  makeCase<ExecutionSpace, view_t, double[3][3][3]>(space, mat, inv, 3, 3);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_symm128, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  using view_t = Kokkos::View<double ***, MemorySpace>;
  ExecutionSpace space{};

  // 128x128 matrix full of -2
  // eigenvalues are -2*128 and 127 0s
  // pseudo-inverse is a 128x128 matrix of 1 / (128 * 128 * -2)
  double mat[1][128][128] = {};
  double inv[1][128][128] = {};
  for (int i = 0; i < 128; i++)
    for (int j = 0; j < 128; j++)
    {
      mat[0][i][j] = -2;
      inv[0][i][j] = 1 / (128 * 128 * -2.);
    }
  makeCase<ExecutionSpace, view_t, double[1][128][128]>(space, mat, inv, 128,
                                                        1);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_scalar_like, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  using view_t = Kokkos::View<double ***, MemorySpace>;
  ExecutionSpace space{};

  double mat[2][1][1] = {{{2}}, {{0}}};
  double inv[2][1][1] = {{{1 / 2.}}, {{0}}};
  makeCase<ExecutionSpace, view_t, double[2][1][1]>(space, mat, inv, 1, 2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_empty, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  using view_t = Kokkos::View<double ***, MemorySpace>;
  ExecutionSpace space{};

  view_t mat("mat", 0, 0, 0);
  ArborX::Interpolation::Details::symmetricPseudoInverseSVD(space, mat);
  BOOST_TEST(mat.size() == 0);
}
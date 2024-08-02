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
#include <ArborX_DetailsSymmetricSVD.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

template <typename MemorySpace, typename ExecutionSpace, typename Value, int M,
          int N>
void makeCase(ExecutionSpace const &exec, Value const (&src_arr)[M][N][N],
              Value const (&ref_arr)[M][N][N])
{
  using DeviceView = Kokkos::View<Value[N][N], MemorySpace>;
  using HostView = typename DeviceView::HostMirror;

  HostView src("Testing::src");
  HostView ref("Testing::ref");

  DeviceView inv("Testing::inv");
  DeviceView unit("Testing::unit");
  Kokkos::View<Value[N], MemorySpace> diag("Testing::diag");

  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < N; j++)
      for (int k = 0; k < N; k++)
      {
        src(j, k) = src_arr[i][j][k];
        ref(j, k) = ref_arr[i][j][k];
      }

    // Test SVD
    Kokkos::deep_copy(exec, inv, src);
    Kokkos::parallel_for(
        "Testing::run_svd", Kokkos::RangePolicy<ExecutionSpace>(exec, 0, 1),
        KOKKOS_LAMBDA(int) {
          ArborX::Interpolation::Details::symmetricSVDKernel(inv, diag, unit);
          for (int p = 0; p < N; ++p)
            for (int q = 0; q < N; ++q)
            {
              inv(p, q) = 0;
              for (int s = 0; s < N; ++s)
                inv(p, q) += unit(p, s) * diag(s) * unit(q, s);
            }
        });
    ARBORX_MDVIEW_TEST_TOL(src, inv, Kokkos::Experimental::epsilon_v<float>);

    // Test pseudo-inverse through SVD
    Kokkos::deep_copy(exec, inv, src);
    Kokkos::parallel_for(
        "Testing::run_inverse", Kokkos::RangePolicy<ExecutionSpace>(exec, 0, 1),
        KOKKOS_LAMBDA(int) {
          ArborX::Interpolation::Details::symmetricPseudoInverseSVDKernel(
              inv, diag, unit);
        });
    ARBORX_MDVIEW_TEST_TOL(ref, inv, Kokkos::Experimental::epsilon_v<float>);
  }
}

// Pseudo-inverses were computed using numpy's "linalg.pinv" solver and
// simplified to be ratios

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_symm2, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
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
  makeCase<MemorySpace>(space, mat, inv);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_symm3, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
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
  makeCase<MemorySpace>(space, mat, inv);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_symm128, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
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
  makeCase<MemorySpace>(space, mat, inv);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_scalar_like, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  ExecutionSpace space{};

  double mat[2][1][1] = {{{2}}, {{0}}};
  double inv[2][1][1] = {{{1 / 2.}}, {{0}}};
  makeCase<MemorySpace>(space, mat, inv);
}

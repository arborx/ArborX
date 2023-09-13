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

namespace tt = boost::test_tools;
namespace axid = ArborX::Interpolation::Details;

template <typename ES, typename U, typename V>
void makeCase(ES const &es, std::size_t id, V &src_arr, V &ref_arr,
              std::size_t n, std::size_t m = 0)
{
  std::string id_string = std::to_string(id);
  using host_view = typename U::HostMirror;

  host_view src;
  host_view ref;
  U inv;

  if constexpr (U::rank == 2)
  {
    src = host_view("src" + id_string, n, n);
    ref = host_view("ref" + id_string, n, n);
    inv = U("inv" + id_string, n, n);

    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = 0; j < n; j++)
      {
        src(i, j) = src_arr[i][j];
        ref(i, j) = ref_arr[i][j];
      }
  }
  else if constexpr (U::rank == 3)
  {
    src = host_view("src" + id_string, m, n, n);
    ref = host_view("ref" + id_string, m, n, n);
    inv = U("inv" + id_string, m, n, n);

    for (std::size_t i = 0; i < m; i++)
      for (std::size_t j = 0; j < n; j++)
        for (std::size_t k = 0; k < n; k++)
        {
          src(i, j, k) = src_arr[i][j][k];
          ref(i, j, k) = ref_arr[i][j][k];
        }
  }

  Kokkos::deep_copy(es, inv, src);
  axid::symmetricPseudoInverseSVD(es, inv);
  ARBORX_MDVIEW_TEST(ref, inv, Kokkos::Experimental::epsilon_v<float>);
}

// Pseudo-inverses were computed using numpy's "linalg.pinv" solver and
// simplified to be ratios

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_symm2, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  using view_t = Kokkos::View<double **, MemorySpace>;
  ExecutionSpace space{};

  double mat0[2][2] = {{1, 2}, {2, 3}};
  double inv0[2][2] = {{-3, 2}, {2, -1}};
  makeCase<ExecutionSpace, view_t, double[2][2]>(space, 0, mat0, inv0, 2);

  double mat1[2][2] = {{4, 0}, {0, 4}};
  double inv1[2][2] = {{1 / 4., 0}, {0, 1 / 4.}};
  makeCase<ExecutionSpace, view_t, double[2][2]>(space, 1, mat1, inv1, 2);

  double mat2[2][2] = {{0, 5}, {5, 0}};
  double inv2[2][2] = {{0, 1 / 5.}, {1 / 5., 0}};
  makeCase<ExecutionSpace, view_t, double[2][2]>(space, 2, mat2, inv2, 2);

  double mat3[2][2] = {{2, 2}, {2, 2}};
  double inv3[2][2] = {{1 / 8., 1 / 8.}, {1 / 8., 1 / 8.}};
  makeCase<ExecutionSpace, view_t, double[2][2]>(space, 3, mat3, inv3, 2);

  double mat4[2][2] = {{1, -3}, {-3, 1}};
  double inv4[2][2] = {{-1 / 8., -3 / 8.}, {-3 / 8., -1 / 8.}};
  makeCase<ExecutionSpace, view_t, double[2][2]>(space, 4, mat4, inv4, 2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_symm3, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  using view_t = Kokkos::View<double **, MemorySpace>;
  ExecutionSpace space{};

  double mat0[3][3] = {{2, 2, 3}, {2, 0, 1}, {3, 1, -2}};
  double inv0[3][3] = {{-1 / 18., 7 / 18., 2 / 18.},
                       {7 / 18., -13 / 18., 4 / 18.},
                       {2 / 18., 4 / 18., -4 / 18.}};
  makeCase<ExecutionSpace, view_t, double[3][3]>(space, 0, mat0, inv0, 3);

  double mat1[3][3] = {{0, 1, 2}, {1, 2, 3}, {2, 3, 4}};
  double inv1[3][3] = {{-5 / 6., -1 / 6., 3 / 6.},
                       {-1 / 6., 0, 1 / 6.},
                       {3 / 6., 1 / 6., -1 / 6.}};
  makeCase<ExecutionSpace, view_t, double[3][3]>(space, 1, mat1, inv1, 3);

  double mat2[3][3] = {{5, 5, 5}, {5, 5, 5}, {5, 5, 5}};
  double inv2[3][3] = {{1 / 45., 1 / 45., 1 / 45.},
                       {1 / 45., 1 / 45., 1 / 45.},
                       {1 / 45., 1 / 45., 1 / 45.}};
  makeCase<ExecutionSpace, view_t, double[3][3]>(space, 2, mat2, inv2, 3);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_symm128, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  using view_t = Kokkos::View<double **, MemorySpace>;
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
  makeCase<ExecutionSpace, view_t, double[128][128]>(space, 0, mat, inv, 128);

  // Case for invertible 128x128 matrix
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_scalar_like, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  using view_t = Kokkos::View<double **, MemorySpace>;
  ExecutionSpace space{};

  double mat0[1][1] = {{2}};
  double inv0[1][1] = {{1 / 2.}};
  makeCase<ExecutionSpace, view_t, double[1][1]>(space, 0, mat0, inv0, 1);

  double mat1[1][1] = {{0}};
  double inv1[1][1] = {{0}};
  makeCase<ExecutionSpace, view_t, double[1][1]>(space, 1, mat1, inv1, 1);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_empty, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  using view_t = Kokkos::View<double **, MemorySpace>;
  ExecutionSpace space{};

  view_t mat("mat", 0, 0);
  axid::symmetricPseudoInverseSVD(space, mat);
  BOOST_TEST(mat.size() == 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_symm2_list, DeviceType,
                              ARBORX_DEVICE_TYPES)
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
  makeCase<ExecutionSpace, view_t, double[5][2][2]>(space, 0, mat, inv, 2, 5);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pseudo_inv_symm3_list, DeviceType,
                              ARBORX_DEVICE_TYPES)
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
  makeCase<ExecutionSpace, view_t, double[3][3][3]>(space, 0, mat, inv, 3, 3);
}
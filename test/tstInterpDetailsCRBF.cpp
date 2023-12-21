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
#include <ArborX_InterpDetailsCompactRadialBasisFunction.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/math/tools/polynomial.hpp>
#include <boost/test/unit_test.hpp>

template <typename T, typename CRBF, typename ES>
void makeCase(ES const &es, std::function<T(T const)> const &tf, T tol = 1e-5)
{
  using View = Kokkos::View<T *, typename ES::memory_space>;
  using HostView = typename View::HostMirror;
  static constexpr int range = 15;

  HostView input("Testing::input", 4 * range);
  for (int i = 0; i < range; i++)
  {
    input(4 * i + 0) = -i - 1;
    input(4 * i + 1) = -T(i) / range;
    input(4 * i + 2) = T(i) / range;
    input(4 * i + 3) = i + 1;
  }

  View eval("Testing::eval", 4 * range);
  Kokkos::deep_copy(es, eval, input);
  Kokkos::parallel_for(
      "Testing::eval_crbf", Kokkos::RangePolicy<ES>(es, 0, 4 * range),
      KOKKOS_LAMBDA(int const i) { eval(i) = CRBF::evaluate(eval(i)); });

  if (bool(tf))
  {
    HostView reference("Testing::reference", 4 * range);
    for (int i = 0; i < range; i++)
    {
      reference(4 * i + 0) = 0;
      reference(4 * i + 1) = tf(T(i) / range);
      reference(4 * i + 2) = tf(T(i) / range);
      reference(4 * i + 3) = 0;
    }

    ARBORX_MDVIEW_TEST_TOL(eval, reference, tol);
  }

  auto heval = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, eval);
  for (int i = 0; i < 4 * range; i++)
    BOOST_TEST(heval(i) >= T(0));
}

#define MAKE_TEST(F, I, TF)                                                    \
  BOOST_AUTO_TEST_CASE_TEMPLATE(crbf_##F##_##I, DeviceType,                    \
                                ARBORX_DEVICE_TYPES)                           \
  {                                                                            \
    makeCase<double, ArborX::Interpolation::CRBF::F<I>>(                       \
        typename DeviceType::execution_space{}, TF<double>);                   \
  }

#define MAKE_TEST_POLY(F, I, POLY)                                             \
  template <typename T>                                                        \
  T func##F##I(T const x)                                                      \
  {                                                                            \
    using boost::math::tools::pow;                                             \
    using poly = boost::math::tools::polynomial<T>;                            \
    return (POLY)(x);                                                          \
  }                                                                            \
                                                                               \
  MAKE_TEST(F, I, func##F##I)

template <typename T>
static std::function<T(T const)> emptyFunc = {};

#define MAKE_TEST_NONE(F, I) MAKE_TEST(F, I, emptyFunc)

MAKE_TEST_POLY(Wendland, 0, (pow(poly{1, -1}, 2)))
MAKE_TEST_POLY(Wendland, 2, (pow(poly{1, -1}, 4) * poly{1, 4}))
MAKE_TEST_POLY(Wendland, 4, (pow(poly{1, -1}, 6) * poly{3, 18, 35}))
MAKE_TEST_POLY(Wendland, 6, (pow(poly{1, -1}, 8) * poly{1, 8, 25, 32}))
MAKE_TEST_POLY(Wu, 2, (pow(poly{1, -1}, 4) * poly{4, 16, 12, 3}))
MAKE_TEST_POLY(Wu, 4, (pow(poly{1, -1}, 6) * poly{6, 36, 82, 72, 30, 5}))
MAKE_TEST_NONE(Buhmann, 2)
MAKE_TEST_NONE(Buhmann, 3)
MAKE_TEST_NONE(Buhmann, 4)

#undef MAKE_TEST_NONE
#undef MAKE_TEST_POLY
#undef MAKE_TEST
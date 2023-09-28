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

#include <interpolation/details/ArborX_InterpDetailsCompactRadialBasisFunction.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

#include <tuple>

using Functions = std::tuple<ArborX::Interpolation::CRBF::Wendland<0>,
                             ArborX::Interpolation::CRBF::Wendland<2>,
                             ArborX::Interpolation::CRBF::Wendland<4>,
                             ArborX::Interpolation::CRBF::Wendland<6>,
                             ArborX::Interpolation::CRBF::Wu<2>,
                             ArborX::Interpolation::CRBF::Wu<4>,
                             ArborX::Interpolation::CRBF::Buhmann<2>,
                             ArborX::Interpolation::CRBF::Buhmann<3>,
                             ArborX::Interpolation::CRBF::Buhmann<4>>;

template <typename CRBF, typename T>
void makeCase(T tol = 1e-5)
{
  static constexpr int range = 15;

  for (int i = 0; i < range; i += 1)
  {
    T in_unit = T(i) / range;
    T out_unit = i + 1;
    T zero = 0;

    // ]-inf; -1]
    BOOST_TEST(CRBF::evaluate(-out_unit) >= zero);
    BOOST_TEST(CRBF::evaluate(-out_unit) == zero,
               boost::test_tools::tolerance(tol));

    // ]-1; 1[
    BOOST_TEST(CRBF::evaluate(-in_unit) >= zero);
    BOOST_TEST(CRBF::evaluate(in_unit) >= zero);
    BOOST_TEST(CRBF::evaluate(in_unit) == CRBF::evaluate(-in_unit),
               boost::test_tools::tolerance(tol));

    // [1; +inf[
    BOOST_TEST(CRBF::evaluate(out_unit) >= zero);
    BOOST_TEST(CRBF::evaluate(out_unit) == zero,
               boost::test_tools::tolerance(tol));
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(crbf, Func, Functions)
{
  makeCase<Func, float>();
  makeCase<Func, double>();
  makeCase<Func, long double>();
}
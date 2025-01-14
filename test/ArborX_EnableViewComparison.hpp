/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_ENABLE_VIEW_COMPARISON_HPP
#define ARBORX_ENABLE_VIEW_COMPARISON_HPP

#include <kokkos_ext/ArborX_KokkosExtAccessibilityTraits.hpp> // is_accessible_from_host

#include <Kokkos_Core.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>
#include <boost/test/utils/is_forward_iterable.hpp>

template <typename U, typename V>
using CommonValueType =
    typename boost::common_type<typename U::non_const_value_type,
                                typename V::non_const_value_type>::type;

template <typename U, typename V>
void arborxViewCheck(U const &u, V const &v, std::string const &u_name,
                     std::string const &v_name, CommonValueType<U, V> tol = 0)
{
  static constexpr int rank = U::rank();

  bool same_dim_size = true;
  for (int i = 0; i < rank; i++)
  {
    int ui = u.extent_int(i);
    int vi = v.extent_int(i);
    BOOST_TEST(ui == vi, "check " << u_name << " == " << v_name
                                  << " failed at dimension " << i << " size ["
                                  << ui << " != " << vi << "]");
    same_dim_size = (ui == vi) && same_dim_size;
  }

  if (!same_dim_size)
    return;

  int index[8]{0, 0, 0, 0, 0, 0, 0, 0};
  auto make_index = [&]() {
    std::stringstream sstr;
    sstr << "(";
    for (int i = 0; i < rank - 1; i++)
      sstr << index[i] << ", ";
    sstr << index[rank - 1] << ")";
    return sstr.str();
  };

  while (index[0] != u.extent_int(0))
  {
    auto uval = u.access(index[0], index[1], index[2], index[3], index[4],
                         index[5], index[6], index[7]);
    auto vval = v.access(index[0], index[1], index[2], index[3], index[4],
                         index[5], index[6], index[7]);
    std::string index_str = make_index();

    // Can "tol" be used as a tolerance value? If not, go back to regular
    // comparison
    if constexpr (boost::math::fpc::tolerance_based<decltype(tol)>::value)
      BOOST_TEST(uval == vval, u_name << " == " << v_name << " at " << index_str
                                      << boost::test_tools::tolerance(tol));
    else
      BOOST_TEST(uval == vval, "check " << u_name << " == " << v_name
                                        << " failed at " << index_str << " ["
                                        << uval << " != " << vval << "]");

    index[rank - 1]++;
    for (int i = rank - 1; i > 0; i--)
      if (index[i] == u.extent_int(i))
      {
        index[i] = 0;
        index[i - 1]++;
      }
  }
}

#define ARBORX_MDVIEW_TEST_TOL(VIEWA, VIEWB, TOL)                              \
  [&](decltype(VIEWA) const &u, decltype(VIEWB) const &v) {                    \
    auto view_a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u); \
    auto view_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v); \
                                                                               \
    static_assert(unsigned(std::decay_t<decltype(u)>::rank()) ==               \
                      unsigned(std::decay_t<decltype(v)>::rank()),             \
                  "'" #VIEWA "' and '" #VIEWB "' must have the same rank");    \
                                                                               \
    std::string view_a_name(#VIEWA);                                           \
    view_a_name += " (" + view_a.label() + ")";                                \
                                                                               \
    std::string view_b_name(#VIEWB);                                           \
    view_b_name += " (" + view_b.label() + ")";                                \
                                                                               \
    arborxViewCheck(view_a, view_b, view_a_name, view_b_name, TOL);            \
  }(VIEWA, VIEWB)

#define ARBORX_MDVIEW_TEST(VIEWA, VIEWB) ARBORX_MDVIEW_TEST_TOL(VIEWA, VIEWB, 0)

// Enable element-wise comparison for views that are accessible from the host
namespace boost
{
namespace unit_test
{

template <typename T, typename... P>
struct is_forward_iterable<Kokkos::View<T, P...>> : public boost::mpl::true_
{
  // NOTE Prefer static assertion to SFINAE because error message about no
  // operator== for the operands is not as clear.
  static_assert(
      Kokkos::View<T, P...>::rank() == 1 &&
          !std::is_same_v<typename Kokkos::View<T, P...>::array_layout,
                          Kokkos::LayoutStride> &&
          ArborX::Details::KokkosExt::is_accessible_from_host<
              Kokkos::View<T, P...>>::value,
      "Restricted to contiguous rank-one host-accessible views");
};

template <typename T, typename... P>
struct bt_iterator_traits<Kokkos::View<T, P...>, true>
{
  using view_type = Kokkos::View<T, P...>;
  using value_type = typename view_type::value_type;
  using const_iterator =
      std::add_pointer_t<typename view_type::const_value_type>;
  static const_iterator begin(view_type const &v) { return v.data(); }
  static const_iterator end(view_type const &v) { return v.data() + v.size(); }
  static std::size_t size(view_type const &v) { return v.size(); }
};

template <typename T, size_t N>
struct is_forward_iterable<Kokkos::Array<T, N>> : public boost::mpl::true_
{};

template <typename T, size_t N>
struct bt_iterator_traits<Kokkos::Array<T, N>, true>
{
  using array_type = Kokkos::Array<T, N>;
  using value_type = typename array_type::value_type;
  using const_iterator = typename array_type::const_pointer;
  static const_iterator begin(array_type const &v) { return v.data(); }
  static const_iterator end(array_type const &v) { return v.data() + v.size(); }
  static std::size_t size(array_type const &v) { return v.size(); }
};

} // namespace unit_test
} // namespace boost

#endif

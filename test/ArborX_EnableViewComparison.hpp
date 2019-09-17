#ifndef ARBORX_ENABLE_VIEW_COMPARISON_HPP
#define ARBORX_ENABLE_VIEW_COMPARISON_HPP

#include <ArborX_DetailsKokkosExt.hpp> // is_accessible_from_host

#include <Kokkos_Core.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/utils/is_forward_iterable.hpp>

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
      Kokkos::View<T, P...>::rank == 1 &&
          KokkosExt::is_accessible_from_host<Kokkos::View<T, P...>>::value,
      "Restricted to rank-one host-accessible views");
};

template <typename T, typename... P>
struct bt_iterator_traits<Kokkos::View<T, P...>, true>
{
  using view_type = Kokkos::View<T, P...>;
  using value_type = typename view_type::value_type;
  using const_iterator =
      typename std::add_pointer<typename view_type::const_value_type>::type;
  static const_iterator begin(view_type const &v) { return v.data(); }
  static const_iterator end(view_type const &v) { return v.data() + v.size(); }
  static std::size_t size(view_type const &v) { return v.size(); }
};

template <typename T, size_t N, typename Proxy>
struct is_forward_iterable<Kokkos::Array<T, N, Proxy>>
    : public boost::mpl::true_
{
};

template <typename T, size_t N, typename Proxy>
struct bt_iterator_traits<Kokkos::Array<T, N, Proxy>, true>
{
  using array_type = Kokkos::Array<T, N, Proxy>;
  using value_type = typename array_type::value_type;
  using const_iterator = typename array_type::const_pointer;
  static const_iterator begin(array_type const &v) { return v.data(); }
  static const_iterator end(array_type const &v) { return v.data() + v.size(); }
  static std::size_t size(array_type const &v) { return v.size(); }
};

} // namespace unit_test
} // namespace boost

#endif

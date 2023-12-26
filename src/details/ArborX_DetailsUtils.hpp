/****************************************************************************
 * Copyright (c) 2017-2023 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DETAILS_UTILS_HPP
#define ARBORX_DETAILS_UTILS_HPP

#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>
#include <ArborX_DetailsKokkosExtMinMaxOperations.hpp>
#include <ArborX_DetailsKokkosExtStdAlgorithms.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{

namespace Details
{

namespace internal
{
template <typename PointerType>
struct PointerDepth
{
  static constexpr int value = 0;
};

template <typename PointerType>
struct PointerDepth<PointerType *>
{
  static constexpr int value = PointerDepth<PointerType>::value + 1;
};

template <typename PointerType, std::size_t N>
struct PointerDepth<PointerType[N]>
{
  static constexpr int value = PointerDepth<PointerType>::value;
};
} // namespace internal

template <typename View, typename ExecutionSpace, typename MemorySpace>
inline Kokkos::View<typename View::traits::data_type, Kokkos::LayoutRight,
                    typename ExecutionSpace::memory_space>
create_layout_right_mirror_view_no_init(ExecutionSpace const &execution_space,
                                        MemorySpace const &memory_space,
                                        View const &src)
{
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value);
  static_assert(Kokkos::is_memory_space<MemorySpace>::value);

  constexpr bool has_compatible_layout =
      (std::is_same_v<typename View::array_layout, Kokkos::LayoutRight> ||
       (View::rank == 1 &&
        (std::is_same_v<typename View::array_layout, Kokkos::LayoutLeft> ||
         std::is_same_v<typename View::array_layout, Kokkos::LayoutRight>)));
  constexpr bool has_compatible_memory_space =
      std::is_same_v<typename View::memory_space, MemorySpace>;

  if constexpr (has_compatible_layout && has_compatible_memory_space)
  {
    return src;
  }
  else
  {
    constexpr int pointer_depth =
        internal::PointerDepth<typename View::traits::data_type>::value;
    return Kokkos::View<typename View::traits::data_type, Kokkos::LayoutRight,
                        MemorySpace>(
        Kokkos::view_alloc(
            execution_space, memory_space, Kokkos::WithoutInitializing,
            std::string(src.label()).append("_layout_right_mirror")),
        src.extent(0), pointer_depth > 1 ? src.extent(1) : KOKKOS_INVALID_INDEX,
        pointer_depth > 2 ? src.extent(2) : KOKKOS_INVALID_INDEX,
        pointer_depth > 3 ? src.extent(3) : KOKKOS_INVALID_INDEX,
        pointer_depth > 4 ? src.extent(4) : KOKKOS_INVALID_INDEX,
        pointer_depth > 5 ? src.extent(5) : KOKKOS_INVALID_INDEX,
        pointer_depth > 6 ? src.extent(6) : KOKKOS_INVALID_INDEX,
        pointer_depth > 7 ? src.extent(7) : KOKKOS_INVALID_INDEX);
  }
}

template <typename View>
inline auto create_layout_right_mirror_view_no_init(View const &src)
{
  typename View::traits::host_mirror_space::execution_space exec;
  auto mirror_view = create_layout_right_mirror_view_no_init(
      exec, typename View::traits::host_mirror_space{}, src);
  exec.fence();
  return mirror_view;
}

template <typename View, typename ExecutionSpace, typename MemorySpace>
inline auto
create_layout_right_mirror_view_and_copy(ExecutionSpace const &execution_space,
                                         MemorySpace const &memory_space,
                                         View const &src)
{
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value);
  static_assert(Kokkos::is_memory_space<MemorySpace>::value);

  constexpr bool has_compatible_layout =
      (std::is_same_v<typename View::array_layout, Kokkos::LayoutRight> ||
       (View::rank == 1 &&
        (std::is_same_v<typename View::array_layout, Kokkos::LayoutLeft> ||
         std::is_same_v<typename View::array_layout, Kokkos::LayoutRight>)));
  constexpr bool has_compatible_memory_space =
      std::is_same_v<typename View::memory_space, MemorySpace>;

  if constexpr (has_compatible_layout && has_compatible_memory_space)
  {
    return src;
  }
  else
  {
    constexpr int pointer_depth =
        internal::PointerDepth<typename View::traits::data_type>::value;

    auto exec = [execution_space]() {
      if constexpr (Kokkos::SpaceAccessibility<ExecutionSpace,
                                               MemorySpace>::accessible)
        return execution_space;
      else
        return typename MemorySpace::execution_space{};
    }();

    Kokkos::View<typename View::traits::data_type, Kokkos::LayoutRight,
                 MemorySpace>
        layout_right_view(
            Kokkos::view_alloc(
                exec, Kokkos::WithoutInitializing,
                std::string(src.label()).append("_layout_right_mirror")),
            src.extent(0),
            pointer_depth > 1 ? src.extent(1) : KOKKOS_INVALID_INDEX,
            pointer_depth > 2 ? src.extent(2) : KOKKOS_INVALID_INDEX,
            pointer_depth > 3 ? src.extent(3) : KOKKOS_INVALID_INDEX,
            pointer_depth > 4 ? src.extent(4) : KOKKOS_INVALID_INDEX,
            pointer_depth > 5 ? src.extent(5) : KOKKOS_INVALID_INDEX,
            pointer_depth > 6 ? src.extent(6) : KOKKOS_INVALID_INDEX,
            pointer_depth > 7 ? src.extent(7) : KOKKOS_INVALID_INDEX);
    auto tmp_view = Kokkos::create_mirror_view(
        Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, memory_space),
        src);
    if constexpr (!Kokkos::SpaceAccessibility<ExecutionSpace,
                                              MemorySpace>::accessible)
      exec.fence();
    Kokkos::deep_copy(execution_space, tmp_view, src);
    Kokkos::deep_copy(execution_space, layout_right_view, tmp_view);
    return layout_right_view;
  }
}

} // namespace Details

template <typename ExecutionSpace, typename ST, typename... SP, typename DT,
          typename... DP>
[[deprecated]] void exclusivePrefixSum(ExecutionSpace &&space,
                                       Kokkos::View<ST, SP...> const &src,
                                       Kokkos::View<DT, DP...> const &dst)
{
  Details::KokkosExt::exclusive_scan(std::forward<ExecutionSpace>(space), src,
                                     dst);
}

template <typename ExecutionSpace, typename T, typename... P>
[[deprecated]] inline std::enable_if_t<
    Kokkos::is_execution_space<std::remove_reference_t<ExecutionSpace>>::value>
exclusivePrefixSum(ExecutionSpace &&space, Kokkos::View<T, P...> const &v)
{
  exclusivePrefixSum(std::forward<ExecutionSpace>(space), v, v);
}

template <typename ST, typename... SP, typename DT, typename... DP>
[[deprecated]] inline void
exclusivePrefixSum(Kokkos::View<ST, SP...> const &src,
                   Kokkos::View<DT, DP...> const &dst)
{
  using ExecutionSpace = typename Kokkos::View<DT, DP...>::execution_space;
  exclusivePrefixSum(ExecutionSpace{}, src, dst);
}

template <typename T, typename... P>
[[deprecated]] inline void exclusivePrefixSum(Kokkos::View<T, P...> const &v)
{
  using ExecutionSpace = typename Kokkos::View<T, P...>::execution_space;
  exclusivePrefixSum(ExecutionSpace{}, v);
}

/** \brief Get a copy of the last element.
 *
 *  Returns a copy of the last element in the view on the host.  Note that it
 *  may require communication between host and device (e.g. if the view passed
 *  as an argument lives on the device).
 *
 *  \pre \c v is of rank 1 and not empty.
 */
template <typename T, typename... P>
[[deprecated]] typename Kokkos::ViewTraits<T, P...>::non_const_value_type
lastElement(Kokkos::View<T, P...> const &v)
{
  using ExecutionSpace = typename Kokkos::View<T, P...>::execution_space;
  return Details::KokkosExt::lastElement(ExecutionSpace{}, v);
}

/** \brief Fills the view with a sequence of numbers
 *
 *  \param[in] space Execution space
 *  \param[out] v Output view
 *  \param[in] value (optional) Initial value
 *
 *  \note Similar to \c std::iota() but differs in that it directly assigns
 *  <code>v(i) = value + i</code> instead of repetitively evaluating
 *  <code>++value</code> which would be difficult to achieve in a performant
 *  manner while still guaranteeing the order of execution.
 */
template <typename ExecutionSpace, typename T, typename... P>
void iota(ExecutionSpace &&space, Kokkos::View<T, P...> const &v,
          typename Kokkos::ViewTraits<T, P...>::value_type value = 0)
{
  using ValueType = typename Kokkos::ViewTraits<T, P...>::value_type;
  static_assert(unsigned(Kokkos::ViewTraits<T, P...>::rank) == unsigned(1),
                "iota requires a View of rank 1");
  static_assert(std::is_arithmetic<ValueType>::value,
                "iota requires a View with an arithmetic value type");
  static_assert(
      std::is_same<ValueType, typename Kokkos::ViewTraits<
                                  T, P...>::non_const_value_type>::value,
      "iota requires a View with non-const value type");
  auto const n = v.extent(0);
  Kokkos::RangePolicy<std::decay_t<ExecutionSpace>> policy(
      std::forward<ExecutionSpace>(space), 0, n);
  Kokkos::parallel_for(
      "ArborX::Algorithms::iota", policy,
      KOKKOS_LAMBDA(int i) { v(i) = value + (ValueType)i; });
}

template <typename T, typename... P>
[[deprecated]] inline void
iota(Kokkos::View<T, P...> const &v,
     typename Kokkos::ViewTraits<T, P...>::value_type value = 0)
{
  using ExecutionSpace = typename Kokkos::ViewTraits<T, P...>::execution_space;
  iota(ExecutionSpace{}, v, value);
}

/** \brief Returns the smallest and the greatest element in the view
 *
 *  \param[in] space Execution space
 *  \param[in] v Input view
 *
 *  Returns a pair on the host with the smallest value in the view as the first
 *  element and the greatest as the second.
 */
template <typename ExecutionSpace, typename ViewType>
std::pair<typename ViewType::non_const_value_type,
          typename ViewType::non_const_value_type>
minMax(ExecutionSpace &&space, ViewType const &v)
{
  static_assert(ViewType::rank == 1, "minMax requires a View of rank 1");
  auto const n = v.extent(0);
  ARBORX_ASSERT(n > 0);
  using ValueType = typename ViewType::non_const_value_type;
  ValueType min_val;
  ValueType max_val;
  Kokkos::RangePolicy<std::decay_t<ExecutionSpace>> policy(
      std::forward<ExecutionSpace>(space), 0, n);
  Kokkos::parallel_reduce(
      "ArborX::Algorithms::minmax", policy,
      KOKKOS_LAMBDA(int i, ValueType &local_min, ValueType &local_max) {
        auto const &val = v(i);
        if (val < local_min)
        {
          local_min = val;
        }
        if (local_max < val)
        {
          local_max = val;
        }
      },
      Kokkos::Min<ValueType>(min_val), Kokkos::Max<ValueType>(max_val));
  return std::make_pair(min_val, max_val);
}

template <typename ViewType>
[[deprecated]] inline std::pair<typename ViewType::non_const_value_type,
                                typename ViewType::non_const_value_type>
minMax(ViewType const &v)
{
  using ExecutionSpace = typename ViewType::execution_space;
  return minMax(ExecutionSpace{}, v);
}

/** \brief Returns the smallest element in the view
 *
 *  \param[in] space Execution space
 *  \param[in] v Input view
 */
template <typename ExecutionSpace, typename ViewType>
typename ViewType::non_const_value_type min(ExecutionSpace &&space,
                                            ViewType const &v)
{
  static_assert(ViewType::rank == 1, "min requires a View of rank 1");
  auto const n = v.extent(0);
  ARBORX_ASSERT(n > 0);
  using ValueType = typename ViewType::non_const_value_type;
  ValueType result;
  Kokkos::Min<typename ViewType::non_const_value_type> reducer(result);
  Kokkos::RangePolicy<std::decay_t<ExecutionSpace>> policy(
      std::forward<ExecutionSpace>(space), 0, n);
  Kokkos::parallel_reduce(
      "ArborX::Algorithms::min", policy,
      KOKKOS_LAMBDA(int i, ValueType &update) {
        if (v(i) < update)
          update = v(i);
      },
      reducer);
  return result;
}

template <typename ViewType>
[[deprecated]] inline typename ViewType::non_const_value_type
min(ViewType const &v)
{
  using ExecutionSpace = typename ViewType::execution_space;
  return min(ExecutionSpace{}, v);
}

/** \brief Returns the greatest element in the view
 *
 *  \param[in] space Execution space
 *  \param[in] v Input view
 */
template <typename ExecutionSpace, typename ViewType>
typename ViewType::non_const_value_type max(ExecutionSpace &&space,
                                            ViewType const &v)
{
  static_assert(ViewType::rank == 1, "max requires a View of rank 1");
  auto const n = v.extent(0);
  ARBORX_ASSERT(n > 0);
  using ValueType = typename ViewType::non_const_value_type;
  ValueType result;
  Kokkos::Max<typename ViewType::non_const_value_type> reducer(result);
  Kokkos::RangePolicy<std::decay_t<ExecutionSpace>> policy(
      std::forward<ExecutionSpace>(space), 0, n);
  Kokkos::parallel_reduce(
      "ArborX::Algorithms::max", policy,
      KOKKOS_LAMBDA(int i, ValueType &update) {
        if (v(i) > update)
          update = v(i);
      },
      reducer);
  return result;
}

template <typename ViewType>
[[deprecated]] inline typename ViewType::non_const_value_type
max(ViewType const &v)
{
  using ExecutionSpace = typename ViewType::execution_space;
  return max(ExecutionSpace{}, v);
}

template <typename ExecutionSpace, typename ViewType>
[[deprecated]] typename ViewType::non_const_value_type
accumulate(ExecutionSpace &&space, ViewType const &v,
           typename ViewType::non_const_value_type init)
{
  Details::KokkosExt::reduce(std::forward<ExecutionSpace>(space), v, init);
}

template <typename ViewType>
[[deprecated]] inline typename ViewType::non_const_value_type
accumulate(ViewType const &v, typename ViewType::non_const_value_type init)
{
  using ExecutionSpace = typename ViewType::execution_space;
  return accumulate(ExecutionSpace{}, v, init);
}

// FIXME shameless forward declaration
template <typename View>
typename View::non_const_type clone(View &v);

template <typename ExecutionSpace, typename SrcViewType, typename DstViewType>
[[deprecated]] void adjacentDifference(ExecutionSpace &&space,
                                       SrcViewType const &src,
                                       DstViewType const &dst)
{
  Details::KokkosExt::adjacent_difference(std::forward<ExecutionSpace>(space),
                                          src, dst);
}

template <typename SrcViewType, typename DstViewType>
[[deprecated]] inline void adjacentDifference(SrcViewType const &src,
                                              DstViewType const &dst)
{
  using ExecutionSpace = typename DstViewType::execution_space;
  adjacentDifference(ExecutionSpace{}, src, dst);
}

// FIXME split this into one for STL-like algorithms and another one for view
// utility helpers

// NOTE: not possible to avoid initialization with Kokkos::realloc()
template <typename View>
[[deprecated]] void
reallocWithoutInitializing(View &v, size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                           size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                           size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                           size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                           size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                           size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                           size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                           size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG)
{
  using ExecutionSpace = typename View::execution_space;
  Details::KokkosExt::reallocWithoutInitializing(ExecutionSpace{}, v, n0, n1,
                                                 n2, n3, n4, n5, n6, n7);
}

template <typename View>
[[deprecated]] void
reallocWithoutInitializing(View &v, const typename View::array_layout &layout)
{
  using ExecutionSpace = typename View::execution_space;
  Details::KokkosExt::reallocWithoutInitializing(ExecutionSpace{}, v, layout);
}

template <typename View>
[[deprecated]] typename View::non_const_type
cloneWithoutInitializingNorCopying(View &v)
{
  using ExecutionSpace = typename View::execution_space;
  return Details::KokkosExt::cloneWithoutInitializingNorCopying(
      ExecutionSpace{}, v);
}

template <typename ExecutionSpace, typename View>
[[deprecated]] typename View::non_const_type clone(ExecutionSpace const &space,
                                                   View &v)
{
  return Details::KokkosExt::clone(space, v);
}

template <typename View>
[[deprecated]] inline typename View::non_const_type clone(View &v)
{
  using ExecutionSpace = typename View::execution_space;
  return Details::KokkosExt::clone(ExecutionSpace{}, v);
}

namespace Details
{
template <typename ExecutionSpace, typename View, typename Offset>
void computeOffsetsInOrderedView(ExecutionSpace const &exec_space, View view,
                                 Offset &offsets)
{
  static_assert(KokkosExt::is_accessible_from<typename View::memory_space,
                                              ExecutionSpace>::value);
  static_assert(KokkosExt::is_accessible_from<typename Offset::memory_space,
                                              ExecutionSpace>::value);

  auto const n = view.extent_int(0);

  int num_offsets;
  KokkosExt::reallocWithoutInitializing(exec_space, offsets, n + 1);
  Kokkos::parallel_scan(
      "ArborX::Algorithms::compute_offsets_in_sorted_view",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n + 1),
      KOKKOS_LAMBDA(int i, int &update, bool final_pass) {
        bool const is_cell_first_index =
            (i == 0 || i == n || view(i) != view(i - 1));
        if (is_cell_first_index)
        {
          if (final_pass)
            offsets(update) = i;
          ++update;
        }
      },
      num_offsets);
  Kokkos::resize(Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing),
                 offsets, num_offsets);
}

} // namespace Details

} // namespace ArborX

#endif

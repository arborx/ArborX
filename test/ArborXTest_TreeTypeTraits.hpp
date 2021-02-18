/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_TEST_TREE_TYPE_TRAITS_HPP
#define ARBORX_TEST_TREE_TYPE_TRAITS_HPP

#include <ArborX.hpp>

#include <tuple>

// NOTE Because std::tuple does not take template template parameters
template <template <class> class...>
struct Tuple
{
};

#ifndef ARBORX_TEST_TREE_TYPES
// NOTE Emulate resulting name from using ArborX::BoundingVolumeHierarchy as
// template parameter in Boost.Test
template <class MemorySpace>
using ArborX__BoundingVolumeHierarchy =
    ArborX::BoundingVolumeHierarchy<MemorySpace>;
#define ARBORX_TEST_TREE_TYPES Tuple<ArborX__BoundingVolumeHierarchy>
#endif

#ifndef ARBORX_TEST_DEVICE_TYPES
#define ARBORX_TEST_DEVICE_TYPES                                               \
  std::tuple<Kokkos::DefaultExecutionSpace::device_type>
#endif

template <template <class> class Tree, class DeviceType>
struct TreeExecutionAndMemorySpaces
// NOTE The name of this class will be part of the resulting name of the unit
// test produced by Boost.Test, such as
//
// clang-format off
// test_case_name<this_class_name<ArborX__BVH_ Kokkos__Device<Kokkos__Cuda_ Kokkos__CudaSpace>>>*
// clang-format on
{
  static_assert(Kokkos::is_device<DeviceType>{}, "");
  using device_type = DeviceType;
  using execution_space = typename DeviceType::execution_space;
  using memory_space = typename DeviceType::memory_space;
  using type = Tree<memory_space>;
};

template <class...>
struct Concatenate;

template <class... Ts, class... Us>
struct Concatenate<std::tuple<Ts...>, std::tuple<Us...>>
{
  using type = std::tuple<Ts..., Us...>;
};

template <class T, class U>
struct CartesianProduct;

template <class... Us>
struct CartesianProduct<Tuple<>, std::tuple<Us...>>
{
  using type = std::tuple<>;
};

template <template <class> class T, template <class> class... Ts, class... Us>
struct CartesianProduct<Tuple<T, Ts...>, std::tuple<Us...>>
{
  using type = typename Concatenate<
      std::tuple<TreeExecutionAndMemorySpaces<T, Us>...>,
      typename CartesianProduct<Tuple<Ts...>, std::tuple<Us...>>::type>::type;
};

using TreeTypeTraitsList =
    typename CartesianProduct<ARBORX_TEST_TREE_TYPES,
                              ARBORX_TEST_DEVICE_TYPES>::type;

#endif

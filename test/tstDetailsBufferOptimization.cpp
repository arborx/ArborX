/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_DetailsBufferOptimization.hpp>
//#include <ArborX_Callbacks.hpp>
#include <ArborX_DetailsBoundingVolumeHierarchyImpl.hpp> // FIXME
#include <ArborX_Predicates.hpp>

#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE DetailsBufferOptiization

namespace tt = boost::test_tools;

struct Test1
{
  template <typename ExecutionSpace, typename Predicates, typename Callbacks>
  void operator()(ExecutionSpace const &space, Predicates const &predicates,
                  Callbacks const &callbacks) const
  {
    using Access =
        ArborX::Traits::Access<Predicates, ArborX::Traits::PredicatesTag>;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, Access::size(predicates)),
        KOKKOS_LAMBDA(int i) {
          for (int j = 0; j < i; ++j)
            callbacks(i)(j);
        });
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(query_impl, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  Kokkos::View<int *, DeviceType> offset("offset", 0);
  Kokkos::View<int *, DeviceType> indices("indices", 0);

  int const n = 10;
  // Build a view of predicates.  Won't actually call any of them.  All is
  // required is a valid access traits assocated to it. 'get()' nevers get
  // called, only 'size()'.
  using Predicate = decltype(ArborX::nearest(std::declval<ArborX::Point>()));
  Kokkos::View<Predicate *, DeviceType> predicates(
      Kokkos::view_alloc("predicates", Kokkos::WithoutInitializing), n);

  ArborX::Details::queryImpl(ExecutionSpace{}, Test1{}, predicates,
                             ArborX::Details::CallbackDefaultSpatialPredicate{},
                             indices, offset, 0);

  auto indices_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices);
  auto offset_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offset);

  std::vector<int> indices_ref;
  std::vector<int> offset_ref;
  offset_ref.push_back(0);
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < i; ++j)
      indices_ref.push_back(j);
    offset_ref.push_back(indices_ref.size());
  }

  BOOST_TEST(offset_host == offset_ref, tt::per_element());
  BOOST_TEST(indices_host == indices_ref, tt::per_element());
}

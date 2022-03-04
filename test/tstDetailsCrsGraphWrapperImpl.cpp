/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
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
#include <ArborX_DetailsCrsGraphWrapperImpl.hpp>
#include <ArborX_Predicates.hpp>
#include <ArborX_TraversalPolicy.hpp>

#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE DetailsCrsGraphWrapperImpl

namespace tt = boost::test_tools;

struct Test1
{
  template <typename ExecutionSpace, typename Predicates,
            typename InsertGenerator>
  void query(ExecutionSpace const &space, Predicates const &predicates,
             InsertGenerator const &insert_generator,
             ArborX::Experimental::TraversalPolicy const & =
                 ArborX::Experimental::TraversalPolicy()) const
  {
    using Access = ArborX::AccessTraits<Predicates, ArborX::PredicatesTag>;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, Access::size(predicates)),
        KOKKOS_LAMBDA(int predicate_index) {
          for (int primitive_index = 0; primitive_index < predicate_index;
               ++primitive_index)
            insert_generator(attach(Access::get(predicates, predicate_index),
                                    predicate_index),
                             primitive_index);
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
  // called, only 'size()'.  FIXME
  using Predicate = decltype(ArborX::intersects(std::declval<ArborX::Point>()));
  Kokkos::View<Predicate *, DeviceType> predicates(
      Kokkos::view_alloc("predicates", Kokkos::WithoutInitializing), n);

  int const buffer_size = 2 * (n + 1);

  Kokkos::realloc(offset, n + 1);
  Kokkos::deep_copy(offset, buffer_size);

  Kokkos::View<unsigned int *, DeviceType> permute(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Testing::permute"), n);
  ExecutionSpace space;
  ArborX::iota(space, permute);

  ArborX::exclusivePrefixSum(space, offset);
  Kokkos::realloc(indices, KokkosExt::lastElement(space, offset));
  ArborX::Details::CrsGraphWrapperImpl::queryImpl(
      space, Test1{}, predicates, ArborX::Details::DefaultCallback{}, indices,
      offset, permute, ArborX::Details::BufferStatus::PreallocationHard);

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

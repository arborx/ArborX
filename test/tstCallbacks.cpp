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

#include <ArborX_AccessTraits.hpp>
#include <ArborX_Callbacks.hpp>
#include <ArborX_Predicates.hpp>

struct NearestPredicates
{
};

struct SpatialPredicates
{
};

namespace ArborX
{
template <>
struct AccessTraits<NearestPredicates, PredicatesTag>
{
  using memory_space = Kokkos::HostSpace;
  static int size(NearestPredicates const &) { return 1; }
  static auto get(NearestPredicates const &, int) { return nearest(Point{}); }
};
template <>
struct AccessTraits<SpatialPredicates, PredicatesTag>
{
  using memory_space = Kokkos::HostSpace;
  static int size(SpatialPredicates const &) { return 1; }
  static auto get(SpatialPredicates const &, int)
  {
    return intersects(Point{});
  }
};
} // namespace ArborX

// Custom callbacks
struct CallbackMissingTag
{
  template <typename Predicate, typename OutputFunctor>
  void operator()(Predicate const &, int, OutputFunctor const &) const
  {
  }
};

struct Wrong
{
};

struct CallbackDoesNotTakeCorrectArgument
{
  template <typename OutputFunctor>
  void operator()(Wrong, int, OutputFunctor const &) const
  {
  }
};

struct CustomCallback
{
  template <class Predicate>
  KOKKOS_FUNCTION void operator()(Predicate const &, int) const
  {
  }
};

struct CustomCallbackMissingConstQualifier
{
  template <class Predicate>
  KOKKOS_FUNCTION void operator()(Predicate const &, int)
  {
  }
};

struct CustomCallbackNonVoidReturnType
{
  template <class Predicate>
  KOKKOS_FUNCTION auto operator()(Predicate const &, int) const
  {
    return Wrong{};
  }
};

struct LegacyNearestPredicateCallback
{
  template <class Predicate, class OutputFunctor>
  void operator()(Predicate const &, int, float, OutputFunctor const &) const
  {
  }
};

int main()
{
  using ArborX::Details::check_valid_callback;

  // view type does not matter as long as we do not call the output functor
  Kokkos::View<float *> v;

  check_valid_callback(ArborX::Details::DefaultCallback{}, SpatialPredicates{},
                       v);
  check_valid_callback(ArborX::Details::DefaultCallback{}, NearestPredicates{},
                       v);

  // not required to tag inline callbacks any more
  check_valid_callback(CallbackMissingTag{}, SpatialPredicates{}, v);
  check_valid_callback(CallbackMissingTag{}, NearestPredicates{}, v);

  check_valid_callback(CustomCallback{}, SpatialPredicates{});
  check_valid_callback(CustomCallback{}, NearestPredicates{});

  // generic lambdas are supported if not using NVCC
#ifndef __NVCC__
  check_valid_callback([](auto const & /*predicate*/, int /*primitive*/,
                          auto const & /*out*/) {},
                       SpatialPredicates{}, v);

  check_valid_callback([](auto const & /*predicate*/, int /*primitive*/,
                          auto const & /*out*/) {},
                       NearestPredicates{}, v);

  check_valid_callback([](auto const & /*predicate*/, int /*primitive*/) {},
                       SpatialPredicates{});

  check_valid_callback([](auto const & /*predicate*/, int /*primitive*/) {},
                       NearestPredicates{});
#endif

  // Uncomment to see error messages

  // check_valid_callback(LegacyNearestPredicateCallback{}, NearestPredicates{},
  //                     v);

  // check_valid_callback(CallbackDoesNotTakeCorrectArgument{},
  //                     SpatialPredicates{}, v);

  // check_valid_callback(CustomCallbackNonVoidReturnType{},
  //                     SpatialPredicates{});

  // check_valid_callback(CustomCallbackMissingConstQualifier{},
  //                     SpatialPredicates{});

#ifndef __NVCC__
  // check_valid_callback([](Wrong, int /*primitive*/) {}, SpatialPredicates{});
#endif
}

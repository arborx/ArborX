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

#ifndef ARBORX_BOOST_RTREE_HELPERS_HPP
#define ARBORX_BOOST_RTREE_HELPERS_HPP

#include <ArborX_Config.hpp>

#include "ArborX_BoostGeometryAdapters.hpp"
#include "ArborX_BoostRangeAdapters.hpp"
#include <ArborX_Box.hpp>
#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp> // is_accessible_from_host
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>         // lastElement
#include <ArborX_DetailsUtils.hpp>                        // exclusivePrefixSum
#include <ArborX_Point.hpp>
#include <ArborX_Predicates.hpp>
#include <ArborX_Sphere.hpp>
#ifdef ARBORX_ENABLE_MPI
#include <ArborX_PairIndexRank.hpp>
#endif

#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/algorithm/transform.hpp>
#include <boost/range/combine.hpp>

#ifdef ARBORX_ENABLE_MPI
#include <mpi.h>
#endif

namespace BoostRTreeHelpers
{
// NOTE: The balancing algorithm does not really matter since the tree is
// created using packing algorithm and there is no later insertion of new
// objects into the tree.  The values of the maximum and the minimum number
// of elements could be adjusted in principle but I didn't bother setting
// this type as a template argument.
using Parameter = boost::geometry::index::linear<16>;

template <typename Value, typename Index>
struct PairMaker
{
  using result_type = std::pair<Value, Index>;
  template <typename T>
  inline result_type operator()(T const &v) const
  {
    return result_type(v.value(), v.index());
  }
};

template <typename Indexable>
using RTree =
    boost::geometry::index::rtree<std::pair<Indexable, int>, Parameter>;

template <typename View>
static RTree<typename View::value_type> makeRTree(View const &objects)
{
  using Indexable = typename View::value_type;
  return RTree<Indexable>(
      objects | boost::adaptors::indexed() |
      boost::adaptors::transformed(PairMaker<Indexable, int>()));
}

// NOTE: Boost.Config defines BOOST_NO_CXX11_VARIADIC_TEMPLATES for nvcc
// with the current version of CUDA we are using.  In consequence we are
// not able to use std::tuple<Indexable, ...> which is unfortunate :(
class AppendRankToPairObjectIndex
{
public:
  AppendRankToPairObjectIndex(int rank)
      : _rank(rank)
  {}

  template <typename T1, typename T2>
  inline boost::tuple<T1, T2, int> operator()(std::pair<T1, T2> const &p) const
  {
    return boost::make_tuple(std::get<0>(p), std::get<1>(p), _rank);
  }

private:
  int _rank = -1;
};

#ifdef ARBORX_ENABLE_MPI
template <typename Indexable>
using ParallelRTree =
    boost::geometry::index::rtree<boost::tuple<Indexable, int, int>, Parameter>;

template <typename View>
static ParallelRTree<typename View::value_type> makeRTree(MPI_Comm comm,
                                                          View const &objects)
{
  using Indexable = typename View::value_type;

  // Fill buffer with pair (object, index)
  std::vector<std::pair<Indexable, int>> buffer;
  boost::copy(objects | boost::adaptors::indexed() |
                  boost::adaptors::transformed(PairMaker<Indexable, int>()),
              std::back_inserter(buffer));

  // Gather all buffers
  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  std::vector<int> counts(comm_size);
  counts[comm_rank] = buffer.size();
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, counts.data(), 1, MPI_INT,
                comm);
  std::vector<int> offsets(comm_size + 1);
  offsets[0] = 0;
  for (int i = 0; i < comm_size; ++i)
    offsets[i + 1] = counts[i] + offsets[i];
  decltype(buffer) all_buffers(offsets.back());
  auto const bytes_per_element = sizeof(typename decltype(buffer)::value_type);
  for (auto *pv : {&counts, &offsets})
    for (auto &x : *pv)
      x *= bytes_per_element;
  MPI_Allgatherv(buffer.data(), counts[comm_rank], MPI_BYTE, all_buffers.data(),
                 counts.data(), offsets.data(), MPI_BYTE, comm);
  for (auto &x : offsets)
    x /= bytes_per_element;

  std::vector<boost::tuple<Indexable, int, int>> all_objects;
  for (int i = 0; i < comm_size; ++i)
    boost::transform(std::make_pair(all_buffers.data() + offsets[i],
                                    all_buffers.data() + offsets[i + 1]),
                     std::back_inserter(all_objects),
                     AppendRankToPairObjectIndex(i));

  return ParallelRTree<Indexable>(all_objects);
}
#endif

template <typename Value>
struct UnaryPredicate
{
  using Function = std::function<bool(Value const &)>;
  UnaryPredicate(Function pred)
      : _pred(pred)
  {}
  inline bool operator()(Value const &val) const { return _pred(val); }
  Function _pred;
};

template <typename Value>
static auto translate(ArborX::Intersects<ArborX::Sphere> const &query)
{
  auto const sphere = getGeometry(query);
  auto const radius = sphere.radius();
  auto const centroid = sphere.centroid();
  ArborX::Box box;
  ArborX::Details::expand(box, sphere);
  return boost::geometry::index::intersects(box) &&
         boost::geometry::index::satisfies(
             UnaryPredicate<Value>([centroid, radius](Value const &val) {
               boost::geometry::index::indexable<Value> indexableGetter;
               auto const &geometry = indexableGetter(val);
               return boost::geometry::distance(centroid, geometry) <= radius;
             }));
}

template <typename Value>
static auto translate(ArborX::Intersects<ArborX::Box> const &query)
{
  ArborX::Box const &box = getGeometry(query);
  return boost::geometry::index::intersects(box);
}

template <typename Value, typename Geometry>
static auto translate(ArborX::Nearest<Geometry> const &query)
{
  auto const geometry = getGeometry(query);
  auto const k = getK(query);
  return boost::geometry::index::nearest(geometry, k);
}

template <typename Indexable, typename InputView,
          typename OutputView = Kokkos::View<int *, Kokkos::HostSpace>>
static std::tuple<OutputView, OutputView>
performQueries(RTree<Indexable> const &rtree, InputView const &queries)
{
  static_assert(KokkosExt::is_accessible_from_host<InputView>::value);

  using Value = typename RTree<Indexable>::value_type;
  auto const n_queries = queries.extent_int(0);
  OutputView offset("offset", n_queries + 1);
  std::vector<Value> returned_values;
  for (int i = 0; i < n_queries; ++i)
    offset(i) = rtree.query(translate<Value>(queries(i)),
                            std::back_inserter(returned_values));
  using ExecutionSpace = typename InputView::execution_space;
  ExecutionSpace space;
  ArborX::exclusivePrefixSum(space, offset);
  auto const n_results = KokkosExt::lastElement(space, offset);
  OutputView indices("indices", n_results);
  for (int i = 0; i < n_queries; ++i)
    for (int j = offset(i); j < offset(i + 1); ++j)
      indices(j) = returned_values[j].second;
  return std::make_tuple(offset, indices);
}

#ifdef ARBORX_ENABLE_MPI
template <typename Indexable, typename InputView,
          typename OutputView1 =
              Kokkos::View<ArborX::PairIndexRank *, Kokkos::HostSpace>,
          typename OutputView2 = Kokkos::View<int *, Kokkos::HostSpace>>
static std::tuple<OutputView2, OutputView1>
performQueries(ParallelRTree<Indexable> const &rtree, InputView const &queries)
{
  static_assert(KokkosExt::is_accessible_from_host<InputView>::value);
  using Value = typename ParallelRTree<Indexable>::value_type;
  auto const n_queries = queries.extent_int(0);
  OutputView2 offset("offset", n_queries + 1);
  std::vector<Value> returned_values;
  for (int i = 0; i < n_queries; ++i)
    offset(i) = rtree.query(translate<Value>(queries(i)),
                            std::back_inserter(returned_values));
  using ExecutionSpace = typename InputView::execution_space;
  ExecutionSpace space;
  ArborX::exclusivePrefixSum(space, offset);
  auto const n_results = KokkosExt::lastElement(space, offset);
  OutputView1 values("values", n_results);
  for (int i = 0; i < n_queries; ++i)
    for (int j = offset(i); j < offset(i + 1); ++j)
    {
      int index;
      int rank;
      boost::tie(boost::tuples::ignore, index, rank) = returned_values[j];
      values(j) = {index, rank};
    }
  return std::make_tuple(offset, values);
}
#endif
} // end namespace BoostRTreeHelpers

namespace BoostExt
{

// FIXME Goal is to match the BVH interface
template <typename Indexable>
class RTree
{
public:
  using DeviceType = Kokkos::DefaultHostExecutionSpace::device_type;
  using memory_space = typename DeviceType::memory_space;

  template <typename ExecutionSpace>
  RTree(ExecutionSpace, Kokkos::View<Indexable *, DeviceType> const &values)
  {
    static_assert(Kokkos::is_execution_space<ExecutionSpace>::value);

    _tree = BoostRTreeHelpers::makeRTree(values);
  }

  // WARNING trailing pack will match anything :/
  template <typename ExecutionSpace, typename Predicates, typename InputView,
            typename... TrailingArgs>
  void query(ExecutionSpace const &, Predicates const &predicates,
             InputView &indices, InputView &offset, TrailingArgs &&...) const
  {
    static_assert(Kokkos::is_execution_space<ExecutionSpace>::value);

    std::tie(offset, indices) =
        BoostRTreeHelpers::performQueries(_tree, predicates);
  }

  template <typename ExecutionSpace, typename Predicates, typename Callback,
            typename... TrailingArgs>
  void query(ExecutionSpace const &, Predicates const &, Callback const &,
             TrailingArgs &&...) const
  {
    static_assert(Kokkos::is_execution_space<ExecutionSpace>::value);

    throw std::runtime_error(
        "Boost RTree does not support callback only query overload.");
  }

private:
  BoostRTreeHelpers::RTree<Indexable> _tree;
};

#ifdef ARBORX_ENABLE_MPI
template <typename Indexable>
class ParallelRTree
{
public:
  using DeviceType = Kokkos::DefaultHostExecutionSpace::device_type;
  using memory_space = typename DeviceType::memory_space;

  template <typename ExecutionSpace>
  ParallelRTree(MPI_Comm comm, ExecutionSpace const &,
                Kokkos::View<Indexable *, DeviceType> const &values)
  {
    static_assert(Kokkos::is_execution_space<ExecutionSpace>::value);

    _tree = BoostRTreeHelpers::makeRTree(comm, values);
  }

  // WARNING trailing pack will match anything :/
  template <typename ExecutionSpace, typename Predicates, typename InputView1,
            typename InputView2, typename... TrailingArgs>
  void query(ExecutionSpace const &, Predicates const &predicates,
             InputView1 &indices, InputView2 &offset, TrailingArgs &&...) const
  {
    static_assert(Kokkos::is_execution_space<ExecutionSpace>::value);

    std::tie(offset, indices) =
        BoostRTreeHelpers::performQueries(_tree, predicates);
  }

private:
  BoostRTreeHelpers::ParallelRTree<Indexable> _tree;
};
#endif

} // namespace BoostExt

namespace ArborX
{
// Specialization of ArborX::query
template <typename Indexable, typename ExecutionSpace, typename Predicates,
          typename InputView, typename... TrailingArgs>
inline void query(BoostExt::RTree<Indexable> const &rtree,
                  ExecutionSpace const &space, Predicates const &predicates,
                  InputView &indices, InputView &offset, TrailingArgs &&...args)
{
  rtree.query(space, predicates, indices, offset,
              std::forward<TrailingArgs>(args)...);
}
} // namespace ArborX

#endif

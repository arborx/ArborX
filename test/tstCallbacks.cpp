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
namespace Traits
{
template <>
struct Access<NearestPredicates, PredicatesTag>
{
  using memory_space = Kokkos::HostSpace;
  static int size(NearestPredicates const &) { return 1; }
  static auto get(NearestPredicates const &, int) { return nearest(Point{}); }
};
template <>
struct Access<SpatialPredicates, PredicatesTag>
{
  using memory_space = Kokkos::HostSpace;
  static int size(SpatialPredicates const &) { return 1; }
  static auto get(SpatialPredicates const &, int)
  {
    return intersects(Point{});
  }
};
} // namespace Traits
} // namespace ArborX

// Custom callbacks
struct SpatialPredicateCallbackMissingTag
{
  template <typename Predicate, typename OutputFunctor>
  void operator()(Predicate const &, int, OutputFunctor const &) const
  {
  }
};

struct NearestPredicateCallbackMissingTag
{
  template <typename Predicate, typename OutputFunctor>
  void operator()(Predicate const &, int, float, OutputFunctor const &) const
  {
  }
};

int main()
{
  using ArborX::Details::check_valid_callback;

  // view type does not matter as long as we do not call the output functor
  Kokkos::View<float *> v;

  check_valid_callback(ArborX::Details::CallbackDefaultSpatialPredicate{},
                       SpatialPredicates{}, v);

  check_valid_callback(ArborX::Details::CallbackDefaultNearestPredicate{},
                       NearestPredicates{}, v);

  check_valid_callback(
      ArborX::Details::CallbackDefaultNearestPredicateWithDistance{},
      NearestPredicates{}, v);

  // Uncomment to see error messages

  check_valid_callback(SpatialPredicateCallbackMissingTag{},
                       SpatialPredicates{}, v);

  check_valid_callback(NearestPredicateCallbackMissingTag{},
                       NearestPredicates{}, v);

#ifndef __NVCC__
  check_valid_callback(
      [](auto const &predicate, int primitive, auto const &out) {},
      SpatialPredicates{}, v);

  check_valid_callback([](auto const &predicate, int primitive, float distance,
                          auto const &out) {},
                       NearestPredicates{}, v);
#endif
}

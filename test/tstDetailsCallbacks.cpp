#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsCallbacks.hpp>
#include <ArborX_Predicates.hpp>

struct DummyNearestPredicates
{
};

struct DummySpatialPredicates
{
};

namespace ArborX
{
namespace Traits
{
template <>
struct Access<DummyNearestPredicates, PredicatesTag>
{
  using memory_space = Kokkos::HostSpace;
  static inline int size(DummyNearestPredicates const &) { return 1; }
  static inline auto get(DummyNearestPredicates const &, int)
  {
    return nearest(Point{});
  }
};
template <>
struct Access<DummySpatialPredicates, PredicatesTag>
{
  using memory_space = Kokkos::HostSpace;
  static inline int size(DummySpatialPredicates const &) { return 1; }
  static inline auto get(DummySpatialPredicates const &, int)
  {
    return intersects(Point{});
  }
};
} // namespace Traits
} // namespace ArborX

using ArborX::Details::is_detected;

// Custom callbacks
struct SpatialPredicateCallbackMissingTag
{
  template <typename Predicate, typename OutputFunctor>
  void operator()(Predicate const &, int, OutputFunctor const &) const
  {
  }
};

static_assert(
    is_detected<
        ArborX::Details::SpatialPredicateInlineCallbackArchetypeExpression,
        SpatialPredicateCallbackMissingTag,
        ArborX::Details::PredicatesHelper<DummySpatialPredicates>,
        ArborX::Details::OutputFunctorHelper<
            Kokkos::View<int *, Kokkos::HostSpace>>>{},
    "");

struct NearestPredicateCallbackMissingTag
{
  template <typename Predicate, typename OutputFunctor>
  void operator()(Predicate const &, int, float, OutputFunctor const &) const
  {
  }
};

static_assert(
    is_detected<
        ArborX::Details::NearestPredicateInlineCallbackArchetypeExpression,
        NearestPredicateCallbackMissingTag,
        ArborX::Details::PredicatesHelper<DummySpatialPredicates>,
        ArborX::Details::OutputFunctorHelper<
            Kokkos::View<float *, Kokkos::HostSpace>>>{},
    "");

int main()
{
  // view type does not matter as long as we do not call the output functor
  Kokkos::View<float *> v;

  check_valid_callback(ArborX::Details::CallbackDefaultSpatialPredicate{},
                       DummySpatialPredicates{}, v);

  check_valid_callback(ArborX::Details::CallbackDefaultNearestPredicate{},
                       DummyNearestPredicates{}, v);

  check_valid_callback(
      ArborX::Details::CallbackDefaultNearestPredicateWithDistance{},
      DummyNearestPredicates{}, v);
}

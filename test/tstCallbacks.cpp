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

struct Wrong
{
};

struct SpatialPredicateCallbackDoesNotTakeCorrectArgument
{
  template <typename OutputFunctor>
  void operator()(Wrong, int, OutputFunctor const &) const
  {
  }
};

struct CustomCallbackNearestPredicate
{
  template <class Predicate>
  KOKKOS_FUNCTION void operator()(Predicate const &, int, float) const
  {
  }
};

struct CustomCallbackSpatialPredicate
{
  template <class Predicate>
  KOKKOS_FUNCTION void operator()(Predicate const &, int) const
  {
  }
};

struct CustomCallbackSpatialPredicateMissingConstQualifier
{
  template <class Predicate>
  KOKKOS_FUNCTION void operator()(Predicate const &, int)
  {
  }
};

struct CustomCallbackSpatialPredicateNonVoidReturnType
{
  template <class Predicate>
  KOKKOS_FUNCTION auto operator()(Predicate const &, int) const
  {
    return Wrong{};
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

  // not required to tag inline callbacks any more
  check_valid_callback(SpatialPredicateCallbackMissingTag{},
                       SpatialPredicates{}, v);

  check_valid_callback(NearestPredicateCallbackMissingTag{},
                       NearestPredicates{}, v);

  // generic lambdas are supported if not using NVCC
#ifndef __NVCC__
  check_valid_callback([](auto const & /*predicate*/, int /*primitive*/,
                          auto const & /*out*/) {},
                       SpatialPredicates{}, v);

  check_valid_callback([](auto const & /*predicate*/, int /*primitive*/,
                          float /*distance*/, auto const & /*out*/) {},
                       NearestPredicates{}, v);
#endif

  // Uncomment to see error messages

  // check_valid_callback(SpatialPredicateCallbackDoesNotTakeCorrectArgument{},
  //                     SpatialPredicates{}, v);

  // check_valid_callback(ArborX::Details::CallbackDefaultSpatialPredicate{},
  //                     NearestPredicates{}, v);

  // check_valid_callback(ArborX::Details::CallbackDefaultNearestPredicate{},
  //                     SpatialPredicates{}, v);

  check_valid_callback(CustomCallbackNearestPredicate{}, NearestPredicates{});

  check_valid_callback(CustomCallbackSpatialPredicate{}, SpatialPredicates{});

  // Uncomment to see error messages

  // check_valid_callback(CustomCallbackSpatialPredicateNonVoidReturnType{},
  //                     SpatialPredicates{});

  // check_valid_callback(CustomCallbackSpatialPredicateMissingConstQualifier{},
  //                     SpatialPredicates{});

  // check_valid_callback(CustomCallbackNearestPredicate{},
  // SpatialPredicates{});

  // check_valid_callback(CustomCallbackSpatialPredicate{},
  // NearestPredicates{});

#ifndef __NVCC__
  check_valid_callback([](auto const & /*predicate*/, int /*primitive*/) {},
                       SpatialPredicates{});

  check_valid_callback(
      [](auto const & /*predicate*/, int /*primitive*/, float /*distance*/) {},
      NearestPredicates{});

  // Uncomment to see error messages

  // check_valid_callback([](Wrong, int /*primitive*/) {}, SpatialPredicates{});
#endif
}

#include <ArborX_DetailsConcepts.hpp>

#include <Kokkos_Core.hpp>

#include <tuple>

using ArborX::Box;
using ArborX::Point;
using ArborX::Sphere;
using ArborX::Details::has_centroid;
using ArborX::Details::is_expandable;

static_assert(is_expandable<Box, Box>::value, "");
static_assert(is_expandable<Box, Box const>::value, "");
static_assert(!is_expandable<Box const, Box>::value, "");

static_assert(is_expandable<Box, Point>::value, "");
static_assert(is_expandable<Box, Sphere>::value, "");

static_assert(!is_expandable<Point, Point>::value, "");
static_assert(!is_expandable<Point, Box>::value, "");
static_assert(!is_expandable<Point, Sphere>::value, "");

// NOTE Possible but not implemented
static_assert(!is_expandable<Sphere, Point>::value, "");
static_assert(!is_expandable<Sphere, Box>::value, "");
static_assert(!is_expandable<Sphere, Sphere>::value, "");

static_assert(has_centroid<Box, Point>::value, "");
static_assert(has_centroid<Point, Point>::value, "");
static_assert(has_centroid<Sphere, Point>::value, "");

using ArborX::Details::is_complete;

template <typename T, typename Enable = void>
struct NotSpecializedForIntegralTypes;
template <typename T>
struct NotSpecializedForIntegralTypes<
    T, std::enable_if_t<!std::is_integral<T>::value>>
{
};

static_assert(is_complete<NotSpecializedForIntegralTypes<float>>::value, "");
static_assert(!is_complete<NotSpecializedForIntegralTypes<int>>::value, "");

using ArborX::Details::first_template_parameter_t;
static_assert(std::is_same<first_template_parameter_t<std::tuple<int, float>>,
                           int>::value,
              "");

using ArborX::Details::has_access_traits;
using ArborX::Traits::PredicatesTag;
using ArborX::Traits::PrimitivesTag;
struct HasNoAccessTraitsSpecialization
{
};
struct HasEmptySpecialization
{
};
namespace ArborX
{
namespace Traits
{
template <typename Tag>
struct Access<HasEmptySpecialization, Tag>
{
};
} // namespace Traits
} // namespace ArborX
static_assert(has_access_traits<Kokkos::View<double *>, PrimitivesTag>::value,
              "");
static_assert(has_access_traits<Kokkos::View<double *>, PredicatesTag>::value,
              "");
static_assert(
    !has_access_traits<HasNoAccessTraitsSpecialization, PrimitivesTag>::value,
    "");
static_assert(
    !has_access_traits<HasNoAccessTraitsSpecialization, PredicatesTag>::value,
    "");
static_assert(has_access_traits<HasEmptySpecialization, PrimitivesTag>::value,
              "");
static_assert(has_access_traits<HasEmptySpecialization, PredicatesTag>::value,
              "");

int main() {}

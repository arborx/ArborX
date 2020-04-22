#include <ArborX_Traits.hpp>

#include <Kokkos_Core.hpp>

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

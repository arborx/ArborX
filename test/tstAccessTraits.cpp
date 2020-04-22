#include <ArborX_AccessTraits.hpp>

#include <Kokkos_Core.hpp>

using ArborX::Details::check_valid_access_traits;
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

int main()
{
  Kokkos::View<ArborX::Point *> p;
  Kokkos::View<float **> v;
  check_valid_access_traits(PrimitivesTag{}, p);
  check_valid_access_traits(PrimitivesTag{}, v);

  using NearestPredicate = decltype(ArborX::nearest(ArborX::Point{}));
  Kokkos::View<NearestPredicate *> q;
  check_valid_access_traits(PredicatesTag{}, q);

  // Uncomment to see error messages
  // check_valid_access_traits(PrimitivesTag{},
  // HasNoAccessTraitsSpecialization{});

  // check_valid_access_traits(PrimitivesTag{}, HasEmptySpecialization{});
}


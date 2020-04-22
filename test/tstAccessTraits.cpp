#include <ArborX_AccessTraits.hpp>

#include <Kokkos_Core.hpp>

using ArborX::Details::check_valid_access_traits;
using ArborX::Traits::PredicatesTag;
using ArborX::Traits::PrimitivesTag;

struct NoAccessTraitsSpecialization
{
};
struct EmptySpecialization
{
};
struct InvalidMemorySpace
{
};
struct SizeMemberFunctionNotStatic
{
};
namespace ArborX
{
namespace Traits
{
template <typename Tag>
struct Access<EmptySpecialization, Tag>
{
};
template <typename Tag>
struct Access<InvalidMemorySpace, Tag>
{
  using memory_space = void;
};
template <typename Tag>
struct Access<SizeMemberFunctionNotStatic, Tag>
{
  using memory_space = Kokkos::HostSpace;
  int size(SizeMemberFunctionNotStatic) { return 255; }
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

  // check_valid_access_traits(PrimitivesTag{}, NoAccessTraitsSpecialization{});

  // check_valid_access_traits(PrimitivesTag{}, EmptySpecialization{});

  // check_valid_access_traits(PrimitivesTag{}, InvalidMemorySpace{});

  // check_valid_access_traits(PrimitivesTag{}, SizeMemberFunctionNotStatic{});
}


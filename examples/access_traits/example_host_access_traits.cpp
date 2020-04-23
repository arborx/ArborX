#include <ArborX_LinearBVH.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>
#include <random>
#include <vector>

namespace ArborX
{
namespace Traits
{
template <typename T, typename Tag>
struct Access<std::vector<T>, Tag>
{
  static std::size_t size(std::vector<T> const &v) { return v.size(); }
  KOKKOS_FUNCTION static T const &get(std::vector<T> const &v, std::size_t i)
  {
    return v[i];
  }
  using memory_space = Kokkos::HostSpace;
};
} // namespace Traits
} // namespace ArborX

int main(int argc, char *argv[])
{
  Kokkos::initialize(argc, argv);
  {

    std::vector<ArborX::Point> points;
    // Fill vector with random points in [-1, 1]^3
    std::uniform_real_distribution<float> dis{-1., 1.};
    std::default_random_engine gen;
    auto rd = [&]() { return dis(gen); };
    std::generate_n(std::back_inserter(points), 100, [&]() {
      return ArborX::Point{rd(), rd(), rd()};
    });

    // Pass directly the vector of points to use the access traits defined above
    ArborX::BVH<Kokkos::HostSpace> bvh{Kokkos::DefaultHostExecutionSpace{},
                                       points};

    // As a supported alternative, wrap the vector in an unmanaged View
    bvh = ArborX::BVH<Kokkos::HostSpace>{
        Kokkos::DefaultHostExecutionSpace{},
        Kokkos::View<ArborX::Point *, Kokkos::HostSpace,
                     Kokkos::MemoryUnmanaged>{points.data(), points.size()}};
  }
  Kokkos::finalize();

  return 0;
}

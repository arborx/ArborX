# MovingLeastSquares

**Header:** `<ArborX_InterpMovingLeastSquares.hpp>`  
**Namespace:** `ArborX::Interpolation`

---

## Overview

`MovingLeastSquares` (MLS) is a meshless interpolation technique that approximates a function value at target points using a weighted combination of values at nearby source points. The weight of each source point decreases with distance from the target, localising the interpolation.

ArborX provides a GPU-accelerated implementation that:
1. Searches for `k` nearest source points for each target point using the BVH.
2. Computes MLS coefficients using a compact radial basis function (CRBF) and a polynomial basis.
3. Applies the coefficients to user-supplied source values to produce approximations at the targets.

---

## Synopsis

```cpp
namespace ArborX::Interpolation {

template <typename MemorySpace,
          typename FloatingCalculationType = double>
class MovingLeastSquares
{
public:
    // Constructor: build the MLS object (search + coefficient computation)
    template <typename ExecutionSpace,
              typename SourcePoints,
              typename TargetPoints,
              typename CRBFunc         = CRBF::Wendland<0>,
              typename PolynomialDegree = PolynomialDegree<2>>
    MovingLeastSquares(ExecutionSpace  const& space,
                       SourcePoints    const& source_points,
                       TargetPoints    const& target_points,
                       CRBFunc                = {},
                       PolynomialDegree       = {},
                       std::optional<int>     num_neighbors = std::nullopt);

    // Interpolate source values to target points
    template <typename ExecutionSpace,
              typename SourceValues,
              typename ApproxValues>
    void interpolate(ExecutionSpace  const& space,
                     SourceValues    const& source_values,
                     ApproxValues         & approx_values) const;
};

} // namespace ArborX::Interpolation
```

---

## Template parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MemorySpace` | — | Kokkos memory space where internal data is stored. |
| `FloatingCalculationType` | `double` | Floating-point type used for coefficient computation. Using `double` improves numerical stability for ill-conditioned problems. |

---

## Constructor

```cpp
MovingLeastSquares(ExecutionSpace  const& space,
                   SourcePoints    const& source_points,
                   TargetPoints    const& target_points,
                   CRBFunc                = {},
                   PolynomialDegree       = {},
                   std::optional<int>     num_neighbors = std::nullopt);
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `space` | Kokkos execution space for all parallel operations. |
| `source_points` | Access-traits-compatible range of source points (must satisfy `GeometryTraits::is_point_v`). |
| `target_points` | Access-traits-compatible range of target points (same dimension as source points). |
| `CRBFunc` | Compact radial basis function type. Default: `CRBF::Wendland<0>`. |
| `PolynomialDegree` | Degree of the polynomial basis. Default: `PolynomialDegree<2>` (quadratic). |
| `num_neighbors` | Number of source neighbours per target. Defaults to the polynomial basis size for the given dimension and degree if `std::nullopt`. |

The constructor:
1. Builds a BVH over the source points.
2. Queries `num_neighbors` nearest source points for each target.
3. Computes and stores MLS coefficients.

### Preconditions

- Source and target points must have the same spatial dimension.
- `num_neighbors` must be positive and `<= source_points.size()`.
- `MemorySpace` must be accessible from `ExecutionSpace`.

---

## `interpolate`

```cpp
template <typename ExecutionSpace, typename SourceValues, typename ApproxValues>
void interpolate(ExecutionSpace const& space,
                 SourceValues   const& source_values,
                 ApproxValues        & approx_values) const;
```

Applies the precomputed MLS coefficients to `source_values` and writes the approximated values to `approx_values`.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `space` | Kokkos execution space. |
| `source_values` | 1-D `Kokkos::View` of values at the source points. Must have extent `source_points.size()`. |
| `approx_values` | 1-D writable `Kokkos::View` for output. Must have extent `target_points.size()`. |

### Notes

- `interpolate` can be called multiple times with different `source_values` (e.g., different time steps) without rebuilding the `MovingLeastSquares` object.
- Both views must be accessible from `ExecutionSpace`.

---

## Radial basis functions (`CRBFunc`)

The following compact radial basis functions are provided:

| Type | Description |
|------|-------------|
| `CRBF::Wendland<0>` | Wendland C² function (default) |
| `CRBF::Wendland<2>` | Wendland C⁴ function |
| `CRBF::Wendland<4>` | Wendland C⁶ function |
| `CRBF::Wu<2>` | Wu C² function |
| `CRBF::Wu<4>` | Wu C⁴ function |

Higher-order functions yield smoother approximations at the cost of more computation.

---

## Polynomial degrees

| Type | Polynomial basis |
|------|-----------------|
| `PolynomialDegree<0>` | Constant (1 term) |
| `PolynomialDegree<1>` | Linear |
| `PolynomialDegree<2>` | Quadratic (default) |

The polynomial degree determines the minimum number of neighbours required for the system to be well-conditioned.

---

## Example

```cpp
#include <ArborX_InterpMovingLeastSquares.hpp>

// source_points: Kokkos::View<ArborX::Point<3>*, MemorySpace>
// target_points: Kokkos::View<ArborX::Point<3>*, MemorySpace>
// source_values: Kokkos::View<double*, MemorySpace>

ArborX::Interpolation::MovingLeastSquares<MemorySpace> mls(
    exec_space, source_points, target_points);

Kokkos::View<double*, MemorySpace> approx_values("approx", n_targets);
mls.interpolate(exec_space, source_values, approx_values);
```

---

## See also

- [`BoundingVolumeHierarchy`](BVH.md)
- [`attach_indices`](attach_indices.md)
- [`AccessTraits`](GeometriesRequirements.md)

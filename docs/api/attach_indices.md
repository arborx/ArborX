# attach_indices

**Header:** `<detail/ArborX_AttachIndices.hpp>`  
**Namespace:** `ArborX::Experimental`

---

## Overview

`attach_indices` is a convenience wrapper that enriches a set of **values** or **predicates** with their zero-based position in the original sequence. The result is an access-trait-compatible range whose elements are `PairValueIndex<Value, Index>` objects carrying both the original value and its ordinal index.

This is the standard way to make tree results traceable back to the originating query or primitive when using a custom callback.

---

## Synopsis

```cpp
namespace ArborX::Experimental {

template <Details::Concepts::AccessTraits Values, typename Index>
struct AttachIndices {
    Values _values;
};

template <typename Index = /* default */,
          Details::Concepts::AccessTraits Values = void>
auto attach_indices(Values const& values)
    -> AttachIndices<Values, Index>;

} // namespace ArborX::Experimental
```

### Template parameters

| Parameter | Description |
|-----------|-------------|
| `Index` | Integer type used to store the position. Defaults to the `index_type` of `PairValueIndex<int>` (typically `int`). |
| `Values` | Any type that satisfies `AccessTraits`. Deduced from the argument. |

### Parameters

| Parameter | Description |
|-----------|-------------|
| `values` | The sequence of values or predicates to annotate with indices. |

### Return value

An `AttachIndices<Values, Index>` object.  
`AccessTraits` for this object provide:
- `size()` – same as the underlying values.
- `get(i)` – returns `PairValueIndex<value_type, Index>{values[i], Index(i)}` (or, for predicates, attaches the index as data).

---

## `PairValueIndex`

```cpp
template <typename Value, typename Index = int>
struct PairValueIndex {
    Value value;
    Index index;
};
```

Both fields are public. `index` contains the zero-based position of `value` in the original sequence.

---

## `AccessTraits` specialisation

`AttachIndices` has a full `AccessTraits` specialisation:

```cpp
template <typename Values, typename Index>
struct ArborX::AccessTraits<ArborX::Experimental::AttachIndices<Values, Index>>
{
    using memory_space = typename AccessTraits<Values>::memory_space;

    KOKKOS_FUNCTION static auto size(AttachIndices<Values,Index> const&);
    KOKKOS_FUNCTION static auto get (AttachIndices<Values,Index> const&, int i);
};
```

When the underlying `Values` satisfies the `Predicates` concept, `get(i)` attaches the index as predicate data via `attach(pred, Index(i))`. Otherwise it returns a `PairValueIndex`.

---

## Notes

- No copy of the underlying data is made; `AttachIndices` stores a reference-like wrapper. The original `values` object must outlive the `AttachIndices` wrapper.
- The default `Index` type matches the index type used internally by `BoundingVolumeHierarchy`, so no explicit template argument is usually required.
- When the wrapped type contains predicates (e.g., passed to `BVH::query`), `getData(predicate)` can be used inside a callback to retrieve the original position.

---

## Example

### Tracking which primitive matched which query

```cpp
#include <detail/ArborX_AttachIndices.hpp>
#include <ArborX_LinearBVH.hpp>

// Build a tree over points, storing (point, index) pairs
ArborX::BoundingVolumeHierarchy bvh(
    space,
    ArborX::Experimental::attach_indices(points));  // values carry their index

// Query with indexed predicates so callbacks know the query id
bvh.query(
    space,
    ArborX::Experimental::attach_indices(predicates),
    KOKKOS_LAMBDA(auto const& predicate, auto const& value) {
        int query_idx = ArborX::getData(predicate);   // position of predicate
        int prim_idx  = value.index;                  // position of primitive
        // ...
    });
```

---

## See also

- [`PairValueIndex`](PairValueIndex.md)
- [`Callbacks`](Callbacks.md)
- [`DefaultIndexableGetter`](DefaultIndexableGetter.md)

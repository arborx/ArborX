# Callbacks

**Header:** `<detail/ArborX_Callbacks.hpp>`  
**Namespace:** `ArborX`

---

## Overview

Callbacks are user-supplied function objects that are invoked during tree traversal whenever a predicate matches a stored value. They give the user full control over what happens with each match and, for spatial and ordered-spatial predicates, whether the traversal should continue or exit early.

---

## Callback requirements

A callback must be a callable object with a specific `operator()` signature that depends on the query flavour being used.

### Standard (output) callback

Used with the CRS-graph query overload (two-view or explicit output view form):

```cpp
struct MyCallback {
    template <typename Predicate, typename Value, typename OutputFunctor>
    KOKKOS_FUNCTION void operator()(Predicate const& predicate,
                                    Value    const& value,
                                    OutputFunctor const& out) const;
};
```

| Parameter | Description |
|-----------|-------------|
| `predicate` | The predicate that matched `value`. |
| `value` | The stored value in the tree. |
| `out` | A unary functor; call `out(x)` to append `x` to the output. |

The return type **must be `void`**.

### Inline (no-output) callback

Used with the single-callback form `bvh.query(space, predicates, callback)`:

```cpp
struct MyCallback {
    template <typename Predicate, typename Value>
    KOKKOS_FUNCTION void operator()(Predicate const& predicate,
                                    Value    const& value) const;
    // -or- return CallbackTreeTraversalControl for spatial predicates
};
```

The return type must be:
- `void` for all predicate types, **or**
- `ArborX::CallbackTreeTraversalControl` for **spatial** and **ordered-spatial** predicates only.

Returning `ArborX::CallbackTreeTraversalControl` for a **nearest** predicate is **ill-formed**.

---

## `CallbackTreeTraversalControl`

```cpp
enum class CallbackTreeTraversalControl {
    early_exit,
    normal_continuation
};
```

Return from an inline callback to steer tree traversal:

| Enumerator | Effect |
|------------|--------|
| `early_exit` | Stop traversal for the current predicate immediately. |
| `normal_continuation` | Continue traversal normally (default behaviour). |

---

## Default callback

```cpp
namespace ArborX::Details {

struct DefaultCallback {
    template <typename Predicate, typename Value, typename OutputFunctor>
    KOKKOS_FUNCTION void operator()(Predicate const&,
                                    Value    const& value,
                                    OutputFunctor const& out) const {
        out(value);
    }
};

} // namespace ArborX::Details
```

`DefaultCallback` simply forwards every matched value to the output functor unchanged. It is used internally when the user does not provide an explicit callback.

---

## Tagged post-callbacks

A callback can declare `using tag = ArborX::Details::PostCallbackTag;` to signal that it is a *post-callback*, i.e. it should be applied after the primary traversal output has been assembled. This is an advanced facility used by built-in helpers such as `attach_indices`.

---

## Notes

- Callbacks are invoked inside a `Kokkos` parallel region; they must be safe to call from device code. Use `KOKKOS_FUNCTION` (or `KOKKOS_LAMBDA`) to mark them as device-callable.
- Generic lambdas (`auto` parameters) are **not** supported with NVCC because CUDA does not allow `__host__ __device__` extended lambdas to be generic.
- The library performs compile-time validation of the callback signature via `check_valid_callback`. A clear static-assertion error is emitted when the signature does not match.

---

## Example

```cpp
// Inline callback that collects (query index, value) pairs
struct CollectPairs {
    Kokkos::View<int*, MemorySpace> query_ids;
    Kokkos::View<int*, MemorySpace> values;
    Kokkos::View<int , MemorySpace> count;

    template <typename Predicate, typename Value>
    KOKKOS_FUNCTION void operator()(Predicate const& pred,
                                    Value     const& val) const {
        int idx = Kokkos::atomic_fetch_inc(&count());
        query_ids(idx) = getData(pred);  // attached index
        values(idx)    = val;
    }
};
```

---

## See also

- [`attach_indices`](attach_indices.md)
- [`TraversalPolicy`](TraversalPolicy.md)
- [`BoundingVolumeHierarchy::query`](BVH.md)

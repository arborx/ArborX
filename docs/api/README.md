# ArborX API Reference

This directory contains the API reference documentation for the ArborX library, following a style similar to [cplusplus.com](https://cplusplus.com/).

---

## Spatial indexing

| Page | Description |
|------|-------------|
| [BoundingVolumeHierarchy](BVH.md) | Primary spatial index (BVH / LBVH) |
| [Callbacks](Callbacks.md) | User-supplied match handlers for `query` |
| [attach_indices](attach_indices.md) | Annotate values/predicates with their ordinal index |
| [QueryPerThread](QueryPerThread.md) | Per-thread (device-inline) single-predicate query |
| [DefaultIndexableGetter](DefaultIndexableGetter.md) | Default functor mapping values to geometries |
| [TraversalPolicy](TraversalPolicy.md) | Buffer size and predicate-sorting options |
| [SpaceFillingCurves](SpaceFillingCurves.md) | Morton32 / Morton64 ordering curves |

---

## Geometry types

| Page | Namespace | Description |
|------|-----------|-------------|
| [Point](Point.md) | `ArborX` | N-D point |
| [Box](Box.md) | `ArborX` | Axis-aligned bounding box |
| [Sphere](Sphere.md) | `ArborX` | N-D sphere |
| [Triangle](Triangle.md) | `ArborX` | N-D triangle |
| [Ray](Ray.md) | `ArborX::Experimental` | 3-D ray |
| [Segment](Segment.md) | `ArborX::Experimental` | N-D line segment |
| [Tetrahedron](Tetrahedron.md) | `ArborX::ExperimentalHyperGeometry` | 3-D tetrahedron |
| [KDOP](KDOP.md) | `ArborX::Experimental` | k-Discrete Oriented Polytope |

---

## Geometry traits and requirements

| Page | Description |
|------|-------------|
| [GeometryTraits](GeometryTraits.md) | Customisation-point traits for geometry types |
| [GeometriesRequirements](GeometriesRequirements.md) | What is needed to register a custom geometry |

---

## Algorithms

| Page | Description |
|------|-------------|
| [NeighborList](NeighborList.md) | `findHalfNeighborList` / `findFullNeighborList` |
| [OrderedSpatial](OrderedSpatial.md) | Distance-ordered spatial queries (`ordered_intersects`) |

---

## Interpolation

| Page | Description |
|------|-------------|
| [MovingLeastSquares](MovingLeastSquares.md) | Meshless MLS interpolation |

---

## Clustering

| Page | Description |
|------|-------------|
| [MinimumSpanningTree](MinimumSpanningTree.md) | Euclidean MST (Borůvka algorithm) |
| [Dendrogram](Dendrogram.md) | Hierarchical clustering tree from MST edges |

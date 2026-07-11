# ADR-001: Canonical Spatial Representation vs. Native Modality Representations

**Status:** Accepted (Class Project) / Future Reconsideration (Showcase)

## Background

The SIADS 696 project harmonizes all transportation modalities into a common analytical grain:

> **Taxi Zone × Date × Temporal Bucket**

This design dramatically simplified feature engineering, multimodal integration, representation learning, and downstream modeling by ensuring every modality shared a common spatial and temporal index.

While this proved to be an effective engineering decision for the course project, subsequent exploration raised a broader architectural question:

> **Should every transportation modality be forced into the same spatial representation?**

---

# The Current Architecture

```
Taxi
        │
Subway
        │
Bus Speed
        │
Traffic
        │
Bridges
        │
        ▼
Taxi Zone
        │
Date
        │
Temporal Bucket
        │
        ▼
Unified Feature Store
        │
        ▼
Representation Learning
        │
        ▼
Anomaly Detection
```

Advantages:

- Simple joins
- Consistent feature engineering
- One modeling table
- Straightforward visualization
- Easy multimodal comparisons

This architecture was the correct choice for the class project.

---

# The Limitation

Not every modality possesses the same native spatial resolution.

Some datasets naturally align with Taxi Zones.

Others do not.

Forcing every modality into Taxi Zones necessarily introduces different levels of spatial approximation.

---

# Spatial Fidelity by Modality

| Modality | Native Representation | Taxi Zone Localization | Spatial Confidence |
|-----------|----------------------|------------------------|--------------------|
| Taxi / Green Taxi | Origin & destination Taxi Zones | Native | High |
| FHVHV | Origin & destination Taxi Zones | Native | High |
| Subway | Station complexes | Direct spatial join | High |
| Bus Speed | GPS route segments | Spatial overlay | Medium |
| Traffic | Sensor locations | Spatial interpolation | Medium |
| Bridges & Tunnels | Facility locations | Gateway assignment | Medium |
| Bus Ridership | Route-level ridership | No direct localization | Low |

The important observation is that **all modalities should not be interpreted as equally localized**, even if they ultimately share a Taxi Zone feature table.

---

# Case Study: Bus Ridership

The hourly bus ridership dataset illustrates this limitation particularly well.

Available information includes:

- Timestamp
- Route
- Fare class
- Estimated ridership

The dataset does **not** contain:

- Stop identifiers
- Boarding locations
- Geographic coordinates
- Taxi Zones

Unlike subway ridership, there is no direct spatial key that enables a straightforward Taxi Zone assignment.

---

# Possible Future Enhancements

Several approaches could improve spatial localization of route-level ridership.

## Option 1 — Stop-Level Join (Preferred)

If future bus ridership datasets include GTFS stop identifiers:

```
Bus Stop
      ↓
Coordinates
      ↓
Spatial Join
      ↓
Taxi Zone
```

This would provide spatial fidelity comparable to the subway pipeline.

---

## Option 2 — Route Geometry Allocation

Use GTFS route geometries (or the existing bus-speed route network) to estimate each route's exposure to Taxi Zones.

For each route:

1. Identify all intersected Taxi Zones.
2. Compute the proportion of route length within each zone.
3. Allocate route-level ridership proportionally across those zones.

This would create a **route exposure estimate**, not true boarding counts, but would substantially improve localization over assigning all riders to a single point.

---

## Option 3 — Terminal Assignment

Assign ridership to the first or last stop of each route.

Simple but unlikely to reflect actual rider distribution.

---

## Option 4 — Network Exposure

Treat the entire route as influencing every Taxi Zone it traverses without attempting to estimate individual boarding locations.

This may be useful for exposure-based analyses but should not be interpreted as localized demand.

---

# Future Architectural Direction

The discussion also surfaced a more fundamental redesign for a future showcase version.

Rather than immediately forcing every modality into Taxi Zones, each transportation system could first be analyzed within its **native spatial representation**.

```
Taxi
        │
        ▼
Origin–Destination Network

Subway
        │
        ▼
Station Network

Bus
        │
        ▼
Route Network

Traffic
        │
        ▼
Road Segment Network

Bridge Crossings
        │
        ▼
Gateway Network
                │
                ▼
     Modality-Specific Features
                │
                ▼
      Common Latent Representation
                │
                ▼
Multimodal Analysis
```

Instead of sharing a common spatial unit, the modalities would share a **common analytical representation** learned after feature engineering.

This resembles modern multimodal machine learning pipelines, where each modality is first encoded within its native structure before being fused into a shared latent space.

---

# Spatial Uncertainty as a First-Class Concept

One promising extension is to explicitly model spatial uncertainty throughout the pipeline.

Rather than assuming every feature has equivalent spatial precision, each could carry a confidence classification reflecting how directly it is tied to geography.

For example:

| Confidence | Example |
|------------|---------|
| High | Native Taxi Zones, station complexes |
| Medium | GPS route segments, traffic sensors, bridge facilities |
| Low | Route-level ridership allocation, inferred exposure |

Potential future uses include:

- Displaying spatial confidence within the showcase application.
- Comparing analytical results under different spatial allocation assumptions.
- Incorporating uncertainty into clustering and anomaly detection.
- Providing users with transparency regarding observed versus inferred spatial relationships.

---

# Decision

The canonical Taxi Zone representation remains the appropriate architecture for the SIADS 696 class project because it enables robust multimodal integration within a consistent analytical framework.

However, future showcase versions should explicitly acknowledge differing levels of spatial fidelity across modalities and explore architectures that preserve native network representations before projecting them into a common analytical space.

Rather than replacing the existing pipeline, this represents a possible second-generation architecture that prioritizes methodological transparency over uniform spatial representation.
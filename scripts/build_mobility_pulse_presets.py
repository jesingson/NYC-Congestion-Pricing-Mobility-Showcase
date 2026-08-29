from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------
# Repository setup
# ---------------------------------------------------------------------
# Running:
#
#     python scripts/build_mobility_pulse_presets.py
#
# starts Python from /scripts. Add the repository root explicitly so
# imports from app.* work reliably.

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------

import imageio.v2 as imageio
import matplotlib

# Render without opening matplotlib windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from shapely.geometry import shape


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

from app.data_access.anomalies import (
    get_metric_driver_anomaly_events,
)
from app.data_access.loaders import (
    CONGESTION_PRICING_START_DATE,
    METRIC_LABELS,
)
from app.data_access.spatial_aggregations import (
    ALL_TEMPORAL_BUCKETS_LABEL,
)
from app.data_access.zone_profile_app_tables import (
    ZONE_PROFILE_DAILY_DIR,
)
from app.utils.project_branding import BRAND_COLORS


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

GEOJSON_PATH = (
    ROOT
    / "data"
    / "processed"
    / "app_tables"
    / "taxi_zones_simplified.geojson"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "app_tables"
    / "mobility_pulse_presets"
)


# ---------------------------------------------------------------------
# Render settings
# ---------------------------------------------------------------------

WIDTH_PX = 1280
HEIGHT_PX = 720
DPI = 100

# Focused saved animations should be slow enough to perceive changes.
DEFAULT_FPS = 8

# The full 2023-2026 history contains roughly 1,186 daily frames.
# 15 FPS yields a video of roughly 75-80 seconds instead of 2+ minutes.
FULL_HISTORY_FPS = 15

# H.264 quality:
# lower CRF = larger / higher quality
# higher CRF = smaller / lower quality
CRF = 24


# ---------------------------------------------------------------------
# Curated Mobility Pulse presets
# ---------------------------------------------------------------------
#
# IMPORTANT:
#
# Saved MP4s render every available observation date. They do NOT use
# Raw 10's custom-animation frame cap.
#
# Orange marker:
#   At least one selected-finalist anomaly occurred in that Taxi Zone
#   on that date where the displayed metric was identified as one of
#   the event drivers.
#
# Canonical anomaly events are Taxi Zone × date × daypart. For this
# daily animation, multiple anomalous dayparts in the same zone/date
# collapse to one visible marker.

PRESETS = [
    {
        "preset_id": "full_history_subway_citywide",
        "menu_label": "NYC mobility across the full study period",
        "title": "NYC mobility across the full study period",
        "description": (
            "Follow citywide Subway ridership from 2023 through early 2026 "
            "and watch how neighborhood patterns evolved before and after "
            "congestion pricing."
        ),
        "metric": "subway_ridership",
        "temporal_bucket": ALL_TEMPORAL_BUCKETS_LABEL,
        "start_date": "2023-01-01",
        "end_date": "2026-03-31",
        "borough": None,
        "fps": FULL_HISTORY_FPS,
    },
    {
        "preset_id": "cp_launch_taxi_citywide",
        "menu_label": "NYC at the congestion-pricing launch",
        "title": "NYC at the congestion-pricing launch",
        "description": (
            "Watch weekend-evening Taxi demand reorganize across the city "
            "around the January 2025 congestion-pricing launch."
        ),
        "metric": "taxi_trip_count",
        "temporal_bucket": "weekend_evening",
        "start_date": "2024-11-01",
        "end_date": "2025-04-30",
        "borough": None,
        "fps": DEFAULT_FPS,
    },
    {
        "preset_id": "post_cp_bronx_taxi_overnight",
        "menu_label": "Bronx overnight activity after launch",
        "title": "Bronx overnight activity after launch",
        "description": (
            "Follow weekend-overnight Taxi demand across Bronx neighborhoods "
            "during the first six months after congestion pricing began."
        ),
        "metric": "taxi_trip_count",
        "temporal_bucket": "weekend_overnight",
        "start_date": "2025-01-05",
        "end_date": "2025-06-30",
        "borough": "Bronx",
        "fps": DEFAULT_FPS,
    },
    {
        "preset_id": "early_2026_brooklyn_fhvhv",
        "menu_label": "Brooklyn for-hire activity",
        "title": "Brooklyn for-hire activity",
        "description": (
            "See how weekend-evening high-volume for-hire activity varied "
            "across Brooklyn neighborhoods in early 2026."
        ),
        "metric": "fhvhv_trip_count",
        "temporal_bucket": "weekend_evening",
        "start_date": "2026-01-01",
        "end_date": "2026-03-31",
        "borough": "Brooklyn",
        "fps": DEFAULT_FPS,
    },
    {
        "preset_id": "early_2026_queens_bus",
        "menu_label": "Queens weekend bus activity",
        "title": "Queens weekend bus activity",
        "description": (
            "Track weekend-midday Bus trip activity across Queens and see "
            "where neighborhood patterns diverged from their pre-CP norms."
        ),
        "metric": "bus_trip_count",
        "temporal_bucket": "weekend_midday",
        "start_date": "2026-01-01",
        "end_date": "2026-03-31",
        "borough": "Queens",
        "fps": DEFAULT_FPS,
    },
    {
        "preset_id": "early_2026_manhattan_subway_transfers",
        "menu_label": "Manhattan late-night transit",
        "title": "Manhattan late-night transit",
        "description": (
            "Follow weekend-overnight Subway transfer activity across "
            "Manhattan during early 2026."
        ),
        "metric": "subway_transfers",
        "temporal_bucket": "weekend_overnight",
        "start_date": "2026-01-01",
        "end_date": "2026-03-31",
        "borough": "Manhattan",
        "fps": DEFAULT_FPS,
    },
    {
        "preset_id": "early_2026_citywide_subway_transfers",
        "menu_label": "Citywide late-night transit",
        "title": "Citywide late-night transit",
        "description": (
            "Compare weekend-overnight Subway transfer patterns across the "
            "city during early 2026."
        ),
        "metric": "subway_transfers",
        "temporal_bucket": "weekend_overnight",
        "start_date": "2026-01-01",
        "end_date": "2026-03-31",
        "borough": None,
        "fps": DEFAULT_FPS,
    },
]


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_inputs() -> None:
    """Fail early when required local files are unavailable."""

    if not GEOJSON_PATH.exists():
        raise FileNotFoundError(
            f"Taxi Zone GeoJSON not found:\n{GEOJSON_PATH}"
        )

    if not Path(ZONE_PROFILE_DAILY_DIR).exists():
        raise FileNotFoundError(
            "Zone Profile daily app-table directory not found:\n"
            f"{ZONE_PROFILE_DAILY_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------

def calculate_robust_symmetric_bound(
    values: pd.Series,
    *,
    percentile: float = 0.95,
    minimum: float = 1.0,
) -> float:
    """Calculate a symmetric color bound from absolute-value quantiles."""

    clean = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .dropna()
        .abs()
    )

    if clean.empty:
        return minimum

    bound = float(
        clean.quantile(percentile)
    )

    if not np.isfinite(bound):
        return minimum

    return max(
        bound,
        minimum,
    )


def load_pulse_panel(
    *,
    metric: str,
    temporal_bucket: str,
    start_date: str,
    end_date: str,
    borough: str | None,
) -> pd.DataFrame:
    """Load one metric and calculate change from each zone's pre-CP mean.

    The reference value is calculated from the full available pre-CP
    history before the requested animation window is applied.
    """

    print(
        f"Loading metric: {metric}"
    )

    frame = pd.read_parquet(
        ZONE_PROFILE_DAILY_DIR,
        engine="pyarrow",
        columns=[
            "taxi_zone_id",
            "date",
            "zone",
            "borough",
            metric,
        ],
        filters=[
            (
                "temporal_bucket",
                "==",
                temporal_bucket,
            ),
        ],
    )

    frame = frame.copy()

    frame["date"] = pd.to_datetime(
        frame["date"]
    )

    frame[metric] = pd.to_numeric(
        frame[metric],
        errors="coerce",
    )

    cp_start = pd.Timestamp(
        CONGESTION_PRICING_START_DATE
    )

    # -------------------------------------------------------------
    # Calculate each Taxi Zone's full pre-CP reference.
    # -------------------------------------------------------------

    pre_reference = (
        frame.loc[
            frame["date"].lt(
                cp_start
            ),
            [
                "taxi_zone_id",
                metric,
            ],
        ]
        .groupby(
            "taxi_zone_id",
            observed=True,
            dropna=False,
        )[metric]
        .mean()
        .rename("pre_reference")
        .reset_index()
    )

    frame = frame.merge(
        pre_reference,
        on="taxi_zone_id",
        how="left",
        validate="many_to_one",
    )

    valid_reference = (
        frame[
            "pre_reference"
        ].notna()
        & frame[
            "pre_reference"
        ].gt(0)
    )

    frame[
        "pulse_change"
    ] = np.where(
        valid_reference,
        (
            frame[metric]
            - frame[
                "pre_reference"
            ]
        )
        / frame[
            "pre_reference"
        ]
        * 100,
        np.nan,
    )

    frame[
        "pulse_index"
    ] = np.where(
        valid_reference,
        frame[metric]
        / frame[
            "pre_reference"
        ]
        * 100,
        np.nan,
    )

    # -------------------------------------------------------------
    # Apply animation window only after reference calculation.
    # -------------------------------------------------------------

    frame = frame[
        frame[
            "date"
        ].between(
            pd.Timestamp(
                start_date
            ),
            pd.Timestamp(
                end_date
            ),
        )
    ].copy()

    if borough is not None:
        frame = frame[
            frame[
                "borough"
            ]
            .astype(str)
            .eq(
                borough
            )
        ].copy()

    return (
        frame.sort_values(
            [
                "date",
                "taxi_zone_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def get_available_frame_dates(
    panel: pd.DataFrame,
) -> list[pd.Timestamp]:
    """Return every available observation date in chronological order."""

    return (
        pd.Series(
            pd.to_datetime(
                panel[
                    "date"
                ].dropna()
            )
        )
        .drop_duplicates()
        .sort_values()
        .tolist()
    )


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def load_zone_polygons() -> list[tuple[int, np.ndarray]]:
    """Load Taxi Zone outer polygon rings from simplified GeoJSON."""

    geojson = json.loads(
        GEOJSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    records: list[
        tuple[int, np.ndarray]
    ] = []

    for feature in geojson.get(
        "features",
        [],
    ):
        properties = feature.get(
            "properties",
            {},
        )

        geometry = feature.get(
            "geometry",
            {},
        )

        taxi_zone_id = properties.get(
            "taxi_zone_id"
        )

        if taxi_zone_id is None:
            continue

        taxi_zone_id = int(
            taxi_zone_id
        )

        geometry_type = geometry.get(
            "type"
        )

        coordinates = geometry.get(
            "coordinates",
            [],
        )

        if geometry_type == "Polygon":
            polygons = [
                coordinates
            ]

        elif geometry_type == "MultiPolygon":
            polygons = coordinates

        else:
            continue

        for polygon in polygons:
            if not polygon:
                continue

            # Presentation animation only needs the outer ring.
            outer_ring = np.asarray(
                polygon[0],
                dtype=float,
            )

            if len(
                outer_ring
            ) < 3:
                continue

            records.append(
                (
                    taxi_zone_id,
                    outer_ring,
                )
            )

    return records


def load_zone_centroids() -> pd.DataFrame:
    """Return one centroid per Taxi Zone from simplified GeoJSON."""

    geojson = json.loads(
        GEOJSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    records: list[
        dict[str, object]
    ] = []

    for feature in geojson.get(
        "features",
        [],
    ):
        properties = feature.get(
            "properties",
            {},
        )

        geometry = feature.get(
            "geometry"
        )

        if not properties or not geometry:
            continue

        taxi_zone_id = properties.get(
            "taxi_zone_id"
        )

        if taxi_zone_id is None:
            continue

        centroid = shape(
            geometry
        ).centroid

        records.append(
            {
                "taxi_zone_id": int(
                    taxi_zone_id
                ),
                "lon": float(
                    centroid.x
                ),
                "lat": float(
                    centroid.y
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# ---------------------------------------------------------------------
# Anomaly helpers
# ---------------------------------------------------------------------

def load_preset_anomalies(
    *,
    metric: str,
    temporal_bucket: str,
    start_date: str,
    end_date: str,
    borough: str | None,
) -> pd.DataFrame:
    """Load metric-linked selected-finalist anomalies for one preset.

    The anomaly source is Taxi Zone × date × daypart. Multiple anomaly
    events in one zone/date collapse to one daily animation marker.
    """

    anomaly_bucket = (
        None
        if temporal_bucket
        == ALL_TEMPORAL_BUCKETS_LABEL
        else temporal_bucket
    )

    anomalies = (
        get_metric_driver_anomaly_events(
            metric=metric,
            temporal_bucket=(
                anomaly_bucket
            ),
        )
        .copy()
    )

    if anomalies.empty:
        return pd.DataFrame(
            columns=[
                "taxi_zone_id",
                "date",
                "anomaly_event_count",
                "lon",
                "lat",
            ]
        )

    anomalies[
        "date"
    ] = pd.to_datetime(
        anomalies[
            "date"
        ]
    )

    anomalies = anomalies[
        anomalies[
            "date"
        ].between(
            pd.Timestamp(
                start_date
            ),
            pd.Timestamp(
                end_date
            ),
        )
    ].copy()

    if borough is not None:
        anomalies = anomalies[
            anomalies[
                "borough"
            ]
            .astype(str)
            .eq(
                borough
            )
        ].copy()

    if anomalies.empty:
        return pd.DataFrame(
            columns=[
                "taxi_zone_id",
                "date",
                "anomaly_event_count",
                "lon",
                "lat",
            ]
        )

    # Collapse multiple anomalous dayparts into one daily zone marker.
    anomalies = (
        anomalies.groupby(
            [
                "taxi_zone_id",
                "date",
            ],
            observed=True,
            dropna=False,
            as_index=False,
        )
        .agg(
            anomaly_event_count=(
                "comparison_event_id",
                "nunique",
            ),
        )
    )

    anomalies = anomalies.merge(
        load_zone_centroids(),
        on="taxi_zone_id",
        how="left",
        validate="many_to_one",
    )

    return (
        anomalies.sort_values(
            [
                "date",
                "taxi_zone_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ---------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------

def build_pulse_colormap() -> LinearSegmentedColormap:
    """Create the project's diverging Mobility Pulse palette."""

    return (
        LinearSegmentedColormap.from_list(
            "mobility_pulse",
            [
                (
                    0.00,
                    BRAND_COLORS[
                        "terracotta"
                    ],
                ),
                (
                    0.35,
                    BRAND_COLORS[
                        "pale_peach"
                    ],
                ),
                (
                    0.50,
                    BRAND_COLORS[
                        "ice"
                    ],
                ),
                (
                    0.65,
                    BRAND_COLORS[
                        "seafoam"
                    ],
                ),
                (
                    1.00,
                    BRAND_COLORS[
                        "dark_teal"
                    ],
                ),
            ],
        )
    )


# ---------------------------------------------------------------------
# Preset renderer
# ---------------------------------------------------------------------

def render_preset(
    preset: dict[str, object],
) -> dict[str, object]:
    """Render one curated Mobility Pulse animation to MP4."""

    preset_id = str(
        preset[
            "preset_id"
        ]
    )

    title = str(
        preset[
            "title"
        ]
    )

    metric = str(
        preset[
            "metric"
        ]
    )

    temporal_bucket = str(
        preset[
            "temporal_bucket"
        ]
    )

    start_date = str(
        preset[
            "start_date"
        ]
    )

    end_date = str(
        preset[
            "end_date"
        ]
    )

    preset_fps = int(
        preset.get(
            "fps",
            DEFAULT_FPS,
        )
    )

    borough_raw = preset.get(
        "borough"
    )

    borough = (
        str(
            borough_raw
        )
        if borough_raw is not None
        else None
    )

    print()
    print(
        "=" * 72
    )

    print(
        title
    )

    print(
        "=" * 72
    )

    # -------------------------------------------------------------
    # Data
    # -------------------------------------------------------------

    panel = load_pulse_panel(
        metric=metric,
        temporal_bucket=(
            temporal_bucket
        ),
        start_date=start_date,
        end_date=end_date,
        borough=borough,
    )

    if panel.empty:
        raise ValueError(
            f"No rows available for preset: "
            f"{preset_id}"
        )

    anomalies = (
        load_preset_anomalies(
            metric=metric,
            temporal_bucket=(
                temporal_bucket
            ),
            start_date=start_date,
            end_date=end_date,
            borough=borough,
        )
    )

    # Saved MP4s deliberately use every available date.
    frame_dates = (
        get_available_frame_dates(
            panel
        )
    )

    if not frame_dates:
        raise ValueError(
            f"No animation dates available for preset: "
            f"{preset_id}"
        )

    color_bound = (
        calculate_robust_symmetric_bound(
            panel[
                "pulse_change"
            ]
        )
    )

    duration_seconds = (
        len(
            frame_dates
        )
        / preset_fps
    )

    print(
        f"Rows:                {len(panel):,}"
    )

    print(
        f"Dates:               {panel['date'].nunique():,}"
    )

    print(
        f"Frames:              {len(frame_dates):,}"
    )

    print(
        f"FPS:                 {preset_fps}"
    )

    print(
        f"Estimated duration:  {duration_seconds:,.1f} sec"
    )

    print(
        f"Color bound:         ±{color_bound:,.1f}%"
    )

    print(
        f"Anomaly zone-dates:  {len(anomalies):,}"
    )

    # -------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------

    zone_ids = set(
        panel[
            "taxi_zone_id"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    all_polygons = (
        load_zone_polygons()
    )

    polygon_records = [
        (
            taxi_zone_id,
            vertices,
        )
        for (
            taxi_zone_id,
            vertices,
        ) in all_polygons
        if taxi_zone_id
        in zone_ids
    ]

    if not polygon_records:
        raise ValueError(
            "No Taxi Zone geometry matched the selected panel."
        )

    polygon_zone_ids = [
        taxi_zone_id
        for taxi_zone_id, _
        in polygon_records
    ]

    vertices = [
        polygon_vertices
        for _, polygon_vertices
        in polygon_records
    ]

    all_vertices = np.vstack(
        vertices
    )

    xmin = float(
        all_vertices[
            :,
            0,
        ].min()
    )

    xmax = float(
        all_vertices[
            :,
            0,
        ].max()
    )

    ymin = float(
        all_vertices[
            :,
            1,
        ].min()
    )

    ymax = float(
        all_vertices[
            :,
            1,
        ].max()
    )

    x_padding = (
        xmax - xmin
    ) * 0.035

    y_padding = (
        ymax - ymin
    ) * 0.035

    # -------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------

    cmap = (
        build_pulse_colormap()
    )

    norm = Normalize(
        vmin=-color_bound,
        vmax=color_bound,
        clip=True,
    )

    fig, ax = plt.subplots(
        figsize=(
            WIDTH_PX / DPI,
            HEIGHT_PX / DPI,
        ),
        dpi=DPI,
    )

    fig.patch.set_facecolor(
        "white"
    )

    ax.set_facecolor(
        BRAND_COLORS[
            "ice"
        ]
    )

    collection = PolyCollection(
        vertices,
        edgecolors="white",
        linewidths=0.35,
    )

    ax.add_collection(
        collection
    )

    # Reusable anomaly layer. Only offsets change between frames.
    anomaly_scatter = ax.scatter(
        [],
        [],
        s=52,
        c=BRAND_COLORS[
            "terracotta"
        ],
        edgecolors="white",
        linewidths=1.2,
        alpha=0.95,
        zorder=10,
    )

    ax.set_xlim(
        xmin - x_padding,
        xmax + x_padding,
    )

    ax.set_ylim(
        ymin - y_padding,
        ymax + y_padding,
    )

    # Approximate geographic aspect correction around NYC.
    ax.set_aspect(
        1
        / np.cos(
            np.deg2rad(
                40.7
            )
        )
    )

    ax.axis(
        "off"
    )

    metric_label = (
        METRIC_LABELS.get(
            metric,
            metric.replace(
                "_",
                " ",
            ).title(),
        )
    )

    scope_label = (
        borough
        if borough is not None
        else "Citywide"
    )

    bucket_label = (
        temporal_bucket
        .replace(
            "_",
            " ",
        )
        .title()
    )

    fig.text(
        0.055,
        0.955,
        title,
        ha="left",
        va="top",
        fontsize=19,
        fontweight="bold",
        color=BRAND_COLORS[
            "dark_teal"
        ],
    )

    fig.text(
        0.055,
        0.915,
        (
            f"{metric_label} · "
            f"{scope_label} · "
            f"{bucket_label}"
        ),
        ha="left",
        va="top",
        fontsize=11,
    )

    fig.text(
        0.055,
        0.885,
        (
            "Change from each zone's own "
            "pre-congestion-pricing average"
        ),
        ha="left",
        va="top",
        fontsize=10,
    )

    date_text = fig.text(
        0.945,
        0.952,
        "",
        ha="right",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=BRAND_COLORS[
            "dark_teal"
        ],
    )

    policy_text = ax.text(
        0.015,
        0.025,
        "",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color=BRAND_COLORS[
            "dark_teal"
        ],
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
            "pad": 5,
        },
    )

    anomaly_legend = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor=(
            BRAND_COLORS[
                "terracotta"
            ]
        ),
        markeredgecolor="white",
        markeredgewidth=1.2,
        markersize=8,
        label="Metric-linked stress anomaly",
    )

    ax.legend(
        handles=[
            anomaly_legend
        ],
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor=(
            BRAND_COLORS[
                "seafoam"
            ]
        ),
        fontsize=9,
    )

    scalar_mappable = (
        ScalarMappable(
            norm=norm,
            cmap=cmap,
        )
    )

    scalar_mappable.set_array(
        []
    )

    colorbar = fig.colorbar(
        scalar_mappable,
        ax=ax,
        orientation="vertical",
        fraction=0.028,
        pad=0.02,
    )

    colorbar.set_label(
        "% change vs pre-CP average",
        fontsize=10,
    )

    fig.subplots_adjust(
        left=0.03,
        right=0.91,
        bottom=0.03,
        top=0.85,
    )

    # -------------------------------------------------------------
    # Video writer
    # -------------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / f"{preset_id}.mp4"
    )

    writer = (
        imageio.get_writer(
            str(
                output_path
            ),
            fps=preset_fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=16,
            ffmpeg_params=[
                "-crf",
                str(
                    CRF
                ),
                "-preset",
                "medium",
            ],
        )
    )

    missing_color = (
        0.88,
        0.88,
        0.88,
        1.0,
    )

    cp_start = pd.Timestamp(
        CONGESTION_PRICING_START_DATE
    )

    # -------------------------------------------------------------
    # Render frames
    # -------------------------------------------------------------

    try:
        for (
            frame_number,
            date_value,
        ) in enumerate(
            frame_dates,
            start=1,
        ):
            date_value = (
                pd.Timestamp(
                    date_value
                )
            )

            date_panel = panel[
                panel[
                    "date"
                ].eq(
                    date_value
                )
            ]

            # -----------------------------------------------------
            # Update map fills.
            # -----------------------------------------------------

            values = (
                date_panel
                .set_index(
                    "taxi_zone_id"
                )[
                    "pulse_change"
                ]
                .to_dict()
            )

            facecolors = []

            for taxi_zone_id in (
                polygon_zone_ids
            ):
                value = values.get(
                    taxi_zone_id,
                    np.nan,
                )

                if pd.isna(
                    value
                ):
                    facecolors.append(
                        missing_color
                    )

                else:
                    facecolors.append(
                        cmap(
                            norm(
                                float(
                                    value
                                )
                            )
                        )
                    )

            collection.set_facecolors(
                facecolors
            )

            # -----------------------------------------------------
            # Update anomaly markers.
            # -----------------------------------------------------

            date_anomalies = (
                anomalies[
                    anomalies[
                        "date"
                    ].eq(
                        date_value
                    )
                ]
                .copy()
            )

            if date_anomalies.empty:
                anomaly_scatter.set_offsets(
                    np.empty(
                        (
                            0,
                            2,
                        )
                    )
                )

            else:
                anomaly_offsets = (
                    date_anomalies[
                        [
                            "lon",
                            "lat",
                        ]
                    ]
                    .dropna()
                    .to_numpy()
                )

                anomaly_scatter.set_offsets(
                    anomaly_offsets
                )

            # -----------------------------------------------------
            # Update labels.
            # -----------------------------------------------------

            date_text.set_text(
                date_value.strftime(
                    "%b %d, %Y"
                )
            )

            if (
                date_value
                >= cp_start
            ):
                policy_text.set_text(
                    "Post-congestion pricing"
                )

            else:
                policy_text.set_text(
                    "Pre-congestion pricing"
                )

            # -----------------------------------------------------
            # Write frame.
            # -----------------------------------------------------

            fig.canvas.draw()

            rgba = np.asarray(
                fig.canvas.buffer_rgba()
            )

            rgb = (
                rgba[
                    ...,
                    :3,
                ]
                .copy()
            )

            writer.append_data(
                rgb
            )

            if (
                frame_number == 1
                or frame_number
                % 25
                == 0
                or frame_number
                == len(
                    frame_dates
                )
            ):
                print(
                    f"Rendered "
                    f"{frame_number:>4}/"
                    f"{len(frame_dates)} "
                    f"frames"
                )

    finally:
        writer.close()

        plt.close(
            fig
        )

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------

    file_size_bytes = (
        output_path
        .stat()
        .st_size
    )

    file_size_mb = (
        file_size_bytes
        / 1024
        / 1024
    )

    duration_seconds = (
        len(
            frame_dates
        )
        / preset_fps
    )

    print()
    print(
        f"Created:             {output_path}"
    )

    print(
        f"Size:                {file_size_mb:,.2f} MB"
    )

    print(
        f"Duration:            {duration_seconds:,.1f} sec"
    )

    print(
        f"Anomaly zone-dates:  {len(anomalies):,}"
    )

    return {
        **preset,
        "filename": (
            output_path.name
        ),
        "frames": len(
            frame_dates
        ),
        "fps": preset_fps,
        "duration_seconds": round(
            duration_seconds,
            2,
        ),
        "width_px": WIDTH_PX,
        "height_px": HEIGHT_PX,
        "crf": CRF,
        "file_size_bytes": (
            file_size_bytes
        ),
        "file_size_mb": round(
            file_size_mb,
            2,
        ),
        "color_bound_percent": round(
            color_bound,
            2,
        ),
        "anomaly_overlay": True,
        "anomaly_zone_date_count": int(
            len(
                anomalies
            )
        ),
        "date_frequency": (
            "all_available_dates"
        ),
    }


# ---------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------

def load_existing_manifest() -> list[dict[str, object]]:
    """Load previously rendered preset metadata when available."""

    manifest_path = (
        OUTPUT_DIR
        / "manifest.json"
    )

    if not manifest_path.exists():
        return []

    try:
        return json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

    except Exception:
        return []


def update_manifest(
    new_results: list[
        dict[str, object]
    ],
) -> Path:
    """Merge newly rendered presets into the existing manifest."""

    existing = (
        load_existing_manifest()
    )

    by_id: dict[
        str,
        dict[str, object],
    ] = {
        str(
            row[
                "preset_id"
            ]
        ): row
        for row in existing
        if "preset_id"
        in row
    }

    for result in new_results:
        by_id[
            str(
                result[
                    "preset_id"
                ]
            )
        ] = result

    # Preserve the editorial order defined in PRESETS.
    ordered_results = []

    for preset in PRESETS:
        preset_id = str(
            preset[
                "preset_id"
            ]
        )

        if preset_id in by_id:
            ordered_results.append(
                by_id[
                    preset_id
                ]
            )

    manifest_path = (
        OUTPUT_DIR
        / "manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            ordered_results,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return manifest_path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    """Build one or all curated Mobility Pulse presets."""

    validate_inputs()

    parser = argparse.ArgumentParser(
        description=(
            "Build pre-rendered Mobility Pulse MP4 presets."
        )
    )

    parser.add_argument(
        "--preset",
        default="all",
        help=(
            "Preset ID to build, or 'all'."
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help=(
            "List available presets and exit."
        ),
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # List available presets.
    # -------------------------------------------------------------

    if args.list:
        for preset in PRESETS:
            print(
                f"{preset['preset_id']}: "
                f"{preset['menu_label']}"
            )

        return

    # -------------------------------------------------------------
    # Resolve requested build set.
    # -------------------------------------------------------------

    if (
        args.preset
        == "all"
    ):
        selected_presets = (
            PRESETS
        )

    else:
        selected_presets = [
            preset
            for preset in PRESETS
            if preset[
                "preset_id"
            ]
            == args.preset
        ]

        if not selected_presets:
            valid_ids = ", ".join(
                str(
                    preset[
                        "preset_id"
                    ]
                )
                for preset
                in PRESETS
            )

            raise ValueError(
                f"Unknown preset: "
                f"{args.preset}. "
                f"Valid IDs: "
                f"{valid_ids}"
            )

    # -------------------------------------------------------------
    # Render.
    # -------------------------------------------------------------

    results = []

    for preset in (
        selected_presets
    ):
        result = (
            render_preset(
                preset
            )
        )

        results.append(
            result
        )

    # -------------------------------------------------------------
    # Update manifest.
    # -------------------------------------------------------------

    manifest_path = (
        update_manifest(
            results
        )
    )

    # -------------------------------------------------------------
    # Summary.
    # -------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "BUILD SUMMARY"
    )

    print(
        "=" * 80
    )

    for result in results:
        print(
            f"{result['filename']:<48} "
            f"{result['file_size_mb']:>7.2f} MB   "
            f"{result['frames']:>4} frames   "
            f"{result['fps']:>2} FPS   "
            f"{result['duration_seconds']:>6.1f} sec   "
            f"{result['anomaly_zone_date_count']:>6} anomaly zone-dates"
        )

    print()
    print(
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()

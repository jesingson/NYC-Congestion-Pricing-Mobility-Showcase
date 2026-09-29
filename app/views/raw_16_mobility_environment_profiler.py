from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.graph_objects as go

from app.data_access.loaders import APP_ROOT
from app.data_access.spatial_visuals import get_zone_geojson
from app.utils.project_branding import (
    BRAND_COLORS,
    apply_branding,
    exploration_section,
    inject_app_css,
    render_chart_insight,
)


PAGE_CAPTION = "MOBILITY ENVIRONMENT PROFILER"
PAGE_TITLE = "What defines this mobility environment?"
REPORT_TITLE = "RAW 16 — MOBILITY-ENVIRONMENT SCOUTING REPORT"

CLUSTER_DIR = APP_ROOT / "data" / "processed" / "3.2.2.final_tables"
ASSIGNMENTS_PATH = (
    CLUSTER_DIR / "canonical_cluster_assignments-20260803-193800.parquet"
)
PROFILES_PATH = CLUSTER_DIR / "canonical_cluster_profiles.parquet"
NAME_LOOKUP_PATH = CLUSTER_DIR / "canonical_cluster_name_lookup.csv"

CLUSTER_INPUT_DIR = APP_ROOT / "data" / "processed" / "3.1.1.final_tables"
SCALED_MATRIX_PATH = CLUSTER_INPUT_DIR / "mobility_profile_scaled_matrix.parquet"
RAW_FEATURES_PATH = CLUSTER_INPUT_DIR / "mobility_profile_raw_features.parquet"
FEATURE_COLUMNS_PATH = CLUSTER_INPUT_DIR / "mobility_profile_feature_columns.csv"

GRAIN_COLUMNS = ["taxi_zone_id", "pre_post_cp"]
PERIOD_ORDER = ["pre_cp", "post_cp"]
HERO_CLUSTER = "Long-Trip Fast-Mobility Zones"
HERO_PERIOD = "post_cp"
HERO_ZONE_ID = 30

METRIC_LABELS = {
    "avg_bus_speed": "Average bus speed",
    "bus_trip_count": "Bus trips",
    "fhvhv_avg_trip_duration": "FHVHV average duration",
    "fhvhv_avg_trip_speed": "FHVHV average speed",
    "fhvhv_trip_count": "FHVHV trips",
    "subway_ridership": "Subway ridership",
    "subway_transfers": "Subway transfers",
    "taxi_avg_trip_duration": "Taxi average duration",
    "taxi_avg_trip_speed": "Taxi average speed",
    "taxi_trip_count": "Taxi trips",
}


POLICY_GEOGRAPHY_LABELS = {
    "cbd": "CBD",
    "adjacent": "Adjacent",
    "gateway": "Gateway",
    "non_cbd": "Non-CBD",
    "non-cbd": "Non-CBD",
    "non cbd": "Non-CBD",
}

MAP_CENTER = {"lat": 40.7128, "lon": -74.0060}
MAP_ZOOM = 9.15


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required cluster asset: {path}")


def _csv(frame: pd.DataFrame, *, decimals: int = 3) -> str:
    if frame.empty:
        return "No rows"
    result = frame.copy()
    float_columns = result.select_dtypes(
        include=["float", "float32", "float64"]
    ).columns
    result[float_columns] = result[float_columns].round(decimals)
    return result.to_csv(index=False).strip()


def _section(title: str, body: str) -> str:
    return f"\n## {title}\n{body.strip()}\n"


def _period_label(value: object) -> str:
    return {"pre_cp": "Pre-CP", "post_cp": "Post-CP"}.get(str(value), str(value))


def _feature_contract(scaled: pd.DataFrame) -> list[str]:
    if FEATURE_COLUMNS_PATH.exists():
        contract = pd.read_csv(FEATURE_COLUMNS_PATH)
        candidate_columns = [
            column
            for column in ["feature_column", "feature", "column"]
            if column in contract.columns
        ]
        if candidate_columns:
            features = contract[candidate_columns[0]].dropna().astype(str).tolist()
            return [feature for feature in features if feature in scaled.columns]

    return [
        column
        for column in scaled.columns
        if column not in GRAIN_COLUMNS
        and pd.api.types.is_numeric_dtype(scaled[column])
    ]


def _cluster_profiles(assignments: pd.DataFrame) -> pd.DataFrame:
    return (
        assignments.groupby(
            ["cluster_label", "canonical_cluster_name", "pre_post_cp"],
            observed=True,
            dropna=False,
        )
        .agg(
            taxi_zone_periods=("taxi_zone_id", "size"),
            distinct_taxi_zones=("taxi_zone_id", "nunique"),
            boroughs=("borough", "nunique"),
        )
        .reset_index()
        .assign(pre_post_cp=lambda frame: frame["pre_post_cp"].map(_period_label))
        .sort_values(["cluster_label", "pre_post_cp"])
    )


def _transition_summary(assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    transition = assignments.pivot(
        index=["taxi_zone_id", "zone", "borough"],
        columns="pre_post_cp",
        values=["cluster_label", "canonical_cluster_name"],
    ).reset_index()
    transition.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in transition.columns
    ]
    transition = transition.rename(
        columns={
            "taxi_zone_id": "taxi_zone_id",
            "zone": "zone",
            "borough": "borough",
            "cluster_label_pre_cp": "pre_cluster_label",
            "cluster_label_post_cp": "post_cluster_label",
            "canonical_cluster_name_pre_cp": "pre_cluster_name",
            "canonical_cluster_name_post_cp": "post_cluster_name",
        }
    )
    transition["changed_cluster"] = transition["pre_cluster_label"].ne(
        transition["post_cluster_label"]
    )

    matrix = (
        transition.groupby(
            ["pre_cluster_name", "post_cluster_name"],
            observed=True,
            dropna=False,
        )
        .size()
        .rename("taxi_zones")
        .reset_index()
        .sort_values("taxi_zones", ascending=False)
    )
    changed = transition.loc[transition["changed_cluster"]].sort_values(
        ["pre_cluster_name", "post_cluster_name", "zone"]
    )
    return matrix, changed


def _geography_mix(assignments: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = (
        assignments.groupby(
            ["canonical_cluster_name", "pre_post_cp", column],
            observed=True,
            dropna=False,
        )
        .size()
        .rename("taxi_zone_periods")
        .reset_index()
    )
    grouped["share_within_cluster_period"] = grouped.groupby(
        ["canonical_cluster_name", "pre_post_cp"], observed=True
    )["taxi_zone_periods"].transform(lambda values: values / values.sum())
    grouped["pre_post_cp"] = grouped["pre_post_cp"].map(_period_label)
    return grouped.sort_values(
        ["canonical_cluster_name", "pre_post_cp", "taxi_zone_periods"],
        ascending=[True, True, False],
    )


def _profile_signals(profiles: pd.DataFrame) -> pd.DataFrame:
    result = profiles.copy()
    result["absolute_signal_zscore"] = result["cluster_signal_zscore"].abs()
    result["signal_direction"] = np.where(
        result["cluster_signal_zscore"].ge(0), "Above city norm", "Below city norm"
    )
    return result.sort_values(
        ["canonical_cluster_name", "absolute_signal_zscore"],
        ascending=[True, False],
    )[
        [
            "canonical_cluster_name",
            "metric_label",
            "cluster_mean_value",
            "cluster_signal_zscore",
            "signal_direction",
        ]
    ]


def _member_diagnostics(
    assignments: pd.DataFrame,
    scaled: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = _feature_contract(scaled)
    if not features:
        raise ValueError("No clustering feature columns could be identified.")

    merged = assignments.merge(
        scaled[GRAIN_COLUMNS + features],
        on=GRAIN_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if merged[features].isna().any().any():
        missing_rows = int(merged[features].isna().any(axis=1).sum())
        raise ValueError(
            f"Scaled feature alignment produced {missing_rows:,} incomplete rows."
        )

    centroids = merged.groupby("cluster_label", observed=True)[features].mean()
    centroid_labels = centroids.index.to_numpy()
    matrix = merged[features].to_numpy(dtype=float)
    centroid_matrix = centroids.to_numpy(dtype=float)
    squared_distances = ((matrix[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(
        axis=2
    )
    distances = np.sqrt(squared_distances)

    label_to_position = {
        int(label): position for position, label in enumerate(centroid_labels)
    }
    assigned_positions = merged["cluster_label"].astype(int).map(label_to_position).to_numpy()
    row_positions = np.arange(len(merged))
    assigned_distance = distances[row_positions, assigned_positions]
    alternate_distances = distances.copy()
    alternate_distances[row_positions, assigned_positions] = np.inf
    nearest_alt_positions = alternate_distances.argmin(axis=1)
    nearest_alt_distance = alternate_distances[row_positions, nearest_alt_positions]
    nearest_alt_labels = centroid_labels[nearest_alt_positions]

    name_map = (
        assignments[["cluster_label", "canonical_cluster_name"]]
        .drop_duplicates()
        .set_index("cluster_label")["canonical_cluster_name"]
        .to_dict()
    )
    merged["assigned_distance"] = assigned_distance
    merged["nearest_alternative_cluster_label"] = nearest_alt_labels
    merged["nearest_alternative_cluster_name"] = merged[
        "nearest_alternative_cluster_label"
    ].map(name_map)
    merged["nearest_alternative_distance"] = nearest_alt_distance
    merged["assignment_margin"] = nearest_alt_distance - assigned_distance
    merged["distance_ratio"] = np.divide(
        assigned_distance,
        nearest_alt_distance,
        out=np.full(len(merged), np.nan),
        where=nearest_alt_distance > 0,
    )
    merged["member_percentile_within_cluster"] = merged.groupby(
        "cluster_label", observed=True
    )["assigned_distance"].rank(method="average", pct=True)
    merged["pre_post_cp"] = merged["pre_post_cp"].map(_period_label)

    member_columns = [
        "taxi_zone_id",
        "zone",
        "borough",
        "pre_post_cp",
        "canonical_cluster_name",
        "assigned_distance",
        "nearest_alternative_cluster_name",
        "nearest_alternative_distance",
        "assignment_margin",
        "distance_ratio",
        "member_percentile_within_cluster",
    ]
    members = merged[member_columns].copy()
    representative = (
        members.sort_values(["canonical_cluster_name", "assigned_distance"])
        .groupby("canonical_cluster_name", observed=True)
        .head(5)
    )
    boundary = (
        members.sort_values(["canonical_cluster_name", "assignment_margin"])
        .groupby("canonical_cluster_name", observed=True)
        .head(8)
    )
    coherence = (
        members.groupby("canonical_cluster_name", observed=True)
        .agg(
            taxi_zone_periods=("taxi_zone_id", "size"),
            median_assigned_distance=("assigned_distance", "median"),
            p90_assigned_distance=("assigned_distance", lambda values: values.quantile(0.90)),
            median_assignment_margin=("assignment_margin", "median"),
            minimum_assignment_margin=("assignment_margin", "min"),
            weak_margin_rows=("distance_ratio", lambda values: int(values.ge(0.90).sum())),
        )
        .reset_index()
        .sort_values("median_assignment_margin")
    )

    # Explain boundary assignments using the exact squared-distance difference.
    boundary_index = boundary.index.to_numpy()
    contribution_records: list[dict[str, object]] = []
    for index in boundary_index:
        assigned_position = assigned_positions[index]
        alternate_position = nearest_alt_positions[index]
        row = matrix[index]
        support = (
            (row - centroid_matrix[alternate_position]) ** 2
            - (row - centroid_matrix[assigned_position]) ** 2
        )
        top_positions = np.argsort(support)[::-1][:8]
        for rank, feature_position in enumerate(top_positions, start=1):
            contribution_records.append(
                {
                    "taxi_zone_id": int(merged.iloc[index]["taxi_zone_id"]),
                    "zone": merged.iloc[index]["zone"],
                    "pre_post_cp": merged.iloc[index]["pre_post_cp"],
                    "assigned_cluster": merged.iloc[index]["canonical_cluster_name"],
                    "nearest_alternative": merged.iloc[index][
                        "nearest_alternative_cluster_name"
                    ],
                    "rank": rank,
                    "feature_column": features[feature_position],
                    "assignment_support": float(support[feature_position]),
                }
            )
    contributions = pd.DataFrame(contribution_records)
    return members, coherence, representative, boundary, contributions


def _base_metric(feature_column: str) -> str:
    if "_shape_" in feature_column:
        return feature_column.split("_shape_", 1)[0]
    if feature_column.endswith("_period_mean"):
        return feature_column.removesuffix("_period_mean")
    return feature_column


def _feature_label(feature_column: str) -> str:
    metric = _base_metric(feature_column)
    metric_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    if feature_column.endswith("_period_mean"):
        return f"{metric_label} — Period mean"
    if "_shape_" in feature_column:
        bucket = feature_column.split("_shape_", 1)[1]
        bucket_label = bucket.replace("_", " ").title().replace("Am", "AM").replace("Pm", "PM")
        return f"{metric_label} — {bucket_label} pattern"
    return metric_label


@st.cache_data(show_spinner="Building the mobility-environment hero...")
def _policy_geography_label(value: object) -> str:
    """Return the reader-facing label for the manually defined geography."""
    normalized = str(value).strip().lower()
    return POLICY_GEOGRAPHY_LABELS.get(normalized, str(value))


def _categorical_zone_map(
    frame: pd.DataFrame,
    *,
    category_column: str,
    category_order: list[str],
    color_lookup: dict[str, str],
    hover_fields: list[str],
    hover_template: str,
    uirevision: str,
) -> go.Figure:
    """Map one categorical Taxi Zone classification with a stable legend."""
    geojson = get_zone_geojson()
    figure = go.Figure()

    for category in category_order:
        subset = frame.loc[frame[category_column].eq(category)].copy()
        if subset.empty:
            continue

        customdata = subset[hover_fields].astype(object).to_numpy()

        # WHY: one trace per category gives a true categorical legend rather than
        # implying that the cluster labels live on a continuous numeric scale.
        figure.add_trace(
            go.Choroplethmap(
                geojson=geojson,
                locations=subset["taxi_zone_id"],
                featureidkey="properties.taxi_zone_id",
                z=np.ones(len(subset)),
                zmin=0,
                zmax=1,
                colorscale=[
                    [0.0, color_lookup[category]],
                    [1.0, color_lookup[category]],
                ],
                marker={"line": {"width": 0.55, "color": "white"}},
                customdata=customdata,
                hovertemplate=hover_template,
                showscale=False,
                name=category,
                showlegend=True,
            )
        )

    apply_branding(figure)
    figure.update_layout(
        title={"text": ""},
        map={"style": "carto-positron", "center": MAP_CENTER, "zoom": MAP_ZOOM},
        height=620,
        margin={"l": 0, "r": 0, "t": 10, "b": 5},
        hovermode="closest",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
            "title": {"text": ""},
        },
        uirevision=uirevision,
    )
    return figure


def _mobility_environment_map(
    assignments: pd.DataFrame,
    *,
    period: str,
) -> go.Figure:
    """Map the unsupervised mobility-environment assignment for one policy period."""
    period_label = _period_label(period)
    frame = assignments.loc[
        assignments["pre_post_cp"].eq(period)
    ].copy()

    cluster_order = sorted(frame["canonical_cluster_name"].dropna().astype(str).unique())
    palette = [
        BRAND_COLORS["dark_teal"],
        BRAND_COLORS["seafoam"],
        BRAND_COLORS["terracotta"],
        BRAND_COLORS["pale_peach"],
        "#667085",
    ]
    color_lookup = {
        cluster: palette[index % len(palette)]
        for index, cluster in enumerate(cluster_order)
    }

    frame["_period_label"] = period_label
    return _categorical_zone_map(
        frame,
        category_column="canonical_cluster_name",
        category_order=cluster_order,
        color_lookup=color_lookup,
        hover_fields=[
            "zone",
            "borough",
            "canonical_cluster_name",
            "_period_label",
        ],
        hover_template=(
            "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
            "Mobility environment: %{customdata[2]}<br>"
            "Period: %{customdata[3]}<extra></extra>"
        ),
        uirevision=f"raw16-mobility-environments-{period}",
    )


def _policy_geography_map(assignments: pd.DataFrame) -> go.Figure:
    """Map the manually defined policy-geography classification."""
    frame = (
        assignments.sort_values(["taxi_zone_id", "pre_post_cp"])
        .drop_duplicates("taxi_zone_id", keep="first")
        .copy()
    )
    frame["policy_geography"] = frame["cbd_spatial_category"].map(
        _policy_geography_label
    )

    category_order = [
        category
        for category in ["CBD", "Adjacent", "Gateway", "Non-CBD"]
        if category in set(frame["policy_geography"])
    ]
    # Preserve any unexpected but valid contract label instead of silently dropping it.
    category_order += sorted(
        set(frame["policy_geography"].dropna().astype(str)) - set(category_order)
    )

    policy_palette = [
        BRAND_COLORS["terracotta"],
        BRAND_COLORS["pale_peach"],
        BRAND_COLORS["seafoam"],
        BRAND_COLORS["dark_teal"],
    ]
    color_lookup = {
        category: policy_palette[index % len(policy_palette)]
        for index, category in enumerate(category_order)
    }

    return _categorical_zone_map(
        frame,
        category_column="policy_geography",
        category_order=category_order,
        color_lookup=color_lookup,
        hover_fields=["zone", "borough", "policy_geography"],
        hover_template=(
            "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
            "Policy geography: %{customdata[2]}<extra></extra>"
        ),
        uirevision="raw16-policy-geography",
    )


def _load_hero_data() -> dict[str, object]:
    for path in [ASSIGNMENTS_PATH, PROFILES_PATH, SCALED_MATRIX_PATH]:
        _require_file(path)

    assignments = pd.read_parquet(ASSIGNMENTS_PATH).copy()
    profiles = pd.read_parquet(PROFILES_PATH).copy()
    scaled = pd.read_parquet(SCALED_MATRIX_PATH).copy()
    features = _feature_contract(scaled)
    members, _, _, _, _ = _member_diagnostics(assignments, scaled)

    hero_members = members.loc[
        members["canonical_cluster_name"].eq(HERO_CLUSTER)
        & members["pre_post_cp"].eq(_period_label(HERO_PERIOD))
    ].copy()
    hero_member = hero_members.loc[hero_members["taxi_zone_id"].eq(HERO_ZONE_ID)]
    if hero_member.empty:
        raise ValueError("The frozen Broad Channel Post-CP hero member is unavailable.")
    hero_member = hero_member.iloc[0]

    hero_pre_rows = assignments.loc[
        assignments["taxi_zone_id"].eq(HERO_ZONE_ID)
        & assignments["pre_post_cp"].eq("pre_cp"),
        "canonical_cluster_name",
    ]
    if hero_pre_rows.empty:
        raise ValueError("Broad Channel's Pre-CP mobility environment is unavailable.")
    hero_pre_cluster = str(hero_pre_rows.iloc[0])

    merged = assignments.merge(
        scaled[GRAIN_COLUMNS + features],
        on=GRAIN_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    centroids = merged.groupby("cluster_label", observed=True)[features].mean()
    selected = merged.loc[
        merged["taxi_zone_id"].eq(HERO_ZONE_ID)
        & merged["pre_post_cp"].eq(HERO_PERIOD)
    ].iloc[0]
    assigned_label = int(selected["cluster_label"])
    alternative_name = str(hero_member["nearest_alternative_cluster_name"])
    alternative_label = int(
        assignments.loc[
            assignments["canonical_cluster_name"].eq(alternative_name),
            "cluster_label",
        ].iloc[0]
    )
    values = selected[features].to_numpy(dtype=float)
    assigned_centroid = centroids.loc[assigned_label].to_numpy(dtype=float)
    alternative_centroid = centroids.loc[alternative_label].to_numpy(dtype=float)
    support = (values - alternative_centroid) ** 2 - (values - assigned_centroid) ** 2

    feature_support = pd.DataFrame(
        {"feature_column": features, "assignment_support": support}
    )
    feature_support["metric"] = feature_support["feature_column"].map(_base_metric)
    metric_support = (
        feature_support.groupby("metric", observed=True)["assignment_support"]
        .sum()
        .rename("assignment_support")
        .reset_index()
    )
    metric_support["metric_label"] = metric_support["metric"].map(METRIC_LABELS).fillna(
        metric_support["metric"]
    )
    metric_support["favors"] = np.where(
        metric_support["assignment_support"].ge(0), HERO_CLUSTER, alternative_name
    )
    metric_support = metric_support.sort_values("assignment_support")

    archetype = profiles.loc[
        profiles["canonical_cluster_name"].eq(HERO_CLUSTER)
    ].copy()
    archetype["direction"] = np.where(
        archetype["cluster_signal_zscore"].ge(0), "Above city norm", "Below city norm"
    )
    archetype = archetype.sort_values("cluster_signal_zscore")

    transitions = assignments.pivot(
        index="taxi_zone_id",
        columns="pre_post_cp",
        values="canonical_cluster_name",
    )
    entrants = int(
        (
            transitions.get("post_cp", pd.Series(dtype=str)).eq(HERO_CLUSTER)
            & ~transitions.get("pre_cp", pd.Series(dtype=str)).eq(HERO_CLUSTER)
        ).sum()
    )
    strongest = archetype.loc[archetype["cluster_signal_zscore"].abs().idxmax()]
    return {
        "members": hero_members,
        "member": hero_member,
        "pre_cluster": hero_pre_cluster,
        "archetype": archetype,
        "metric_support": metric_support,
        "entrants": entrants,
        "strongest": strongest,
        "feature_support": feature_support.sort_values("assignment_support", ascending=False),
    }


def _archetype_chart(archetype: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(archetype)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "cluster_signal_zscore:Q",
                title="Environment mean relative to the citywide feature norm",
                axis=alt.Axis(format=".1f"),
            ),
            y=alt.Y("metric_label:N", title=None, sort="x"),
            color=alt.Color(
                "direction:N",
                title=None,
                scale=alt.Scale(
                    domain=["Below city norm", "Above city norm"],
                    range=[BRAND_COLORS["terracotta"], BRAND_COLORS["dark_teal"]],
                ),
                legend=alt.Legend(orient="top", direction="horizontal"),
            ),
            tooltip=[
                alt.Tooltip("metric_label:N", title="Metric"),
                alt.Tooltip("cluster_mean_value:Q", title="Cluster mean", format=",.3f"),
                alt.Tooltip(
                    "cluster_signal_zscore:Q", title="Standardized signal", format="+.3f"
                ),
            ],
        )
        .properties(height=330)
    )


def _member_distance_chart(
    members: pd.DataFrame,
    *,
    selected_zone_id: int,
    selected_zone_name: str,
) -> alt.Chart:
    chart = members.copy().sort_values("assigned_distance", ascending=False)
    chart["member_label"] = chart["zone"] + " · " + chart["borough"]
    chart["selected_member"] = np.where(
        chart["taxi_zone_id"].eq(selected_zone_id), selected_zone_name, "Other members"
    )
    return (
        alt.Chart(chart)
        .mark_circle(size=105, opacity=0.88)
        .encode(
            x=alt.X(
                "assigned_distance:Q",
                title="Distance from assigned environment centroid",
                scale=alt.Scale(zero=True),
            ),
            y=alt.Y(
                "member_label:N",
                title=None,
                sort=alt.SortField(field="assigned_distance", order="descending"),
                axis=alt.Axis(labelLimit=240),
            ),
            color=alt.Color(
                "selected_member:N",
                title=None,
                scale=alt.Scale(
                    domain=[selected_zone_name, "Other members"],
                    range=[BRAND_COLORS["terracotta"], BRAND_COLORS["dark_teal"]],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("zone:N", title="Taxi Zone"),
                alt.Tooltip("borough:N", title="Borough"),
                alt.Tooltip("assigned_distance:Q", title="Assigned distance", format=".3f"),
                alt.Tooltip(
                    "nearest_alternative_cluster_name:N", title="Nearest alternative"
                ),
                alt.Tooltip(
                    "nearest_alternative_distance:Q",
                    title="Alternative distance",
                    format=".3f",
                ),
                alt.Tooltip("assignment_margin:Q", title="Distance advantage", format=".3f"),
            ],
        )
        .properties(height=max(440, 22 * len(chart)))
    )


def _support_chart(
    metric_support: pd.DataFrame,
    *,
    assigned_cluster: str,
    alternative_cluster: str,
    zone_name: str,
) -> alt.Chart:
    return (
        alt.Chart(metric_support)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "assignment_support:Q",
                title="Relative-fit contribution — left favors alternative; right favors selected environment",
                axis=alt.Axis(format=".1f"),
            ),
            y=alt.Y(
                "metric_label:N",
                title=None,
                sort=alt.SortField(field="assignment_support", order="ascending"),
                axis=alt.Axis(labelLimit=240),
            ),
            color=alt.Color(
                "favors:N",
                title=None,
                scale=alt.Scale(
                    domain=[assigned_cluster, alternative_cluster],
                    range=[BRAND_COLORS["dark_teal"], BRAND_COLORS["terracotta"]],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("metric_label:N", title="Mobility metric"),
                alt.Tooltip("favors:N", title="Favors"),
                alt.Tooltip("assignment_support:Q", title="Distance contribution", format="+.3f"),
            ],
        )
        .properties(height=390)
    )


def _render_support_key(
    *, assigned_cluster: str, alternative_cluster: str, zone_name: str
) -> None:
    left, right = st.columns(2)
    left.markdown(
        f"**← Peach: closer to {alternative_cluster}**  \n"
        "Negative bars are evidence for the closest competing cluster."
    )
    right.markdown(
        f"**Teal: closer to {assigned_cluster} →**  \n"
        "Positive bars are evidence for the selected cluster."
    )
    st.caption(
        f"For {zone_name}, each bar combines 11 standardized inputs for one mobility "
        "metric: its period mean plus ten time-of-week shape features. Bar length is "
        "the metric family's contribution to the difference in squared centroid "
        "distance—not a raw trip, speed, or duration value."
    )


def _all_cluster_distance_chart(distances: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(distances)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "distance:Q",
                title="Euclidean distance across all 110 standardized features",
                scale=alt.Scale(zero=True),
            ),
            y=alt.Y(
                "cluster_name:N",
                title=None,
                sort=alt.SortField(field="distance", order="ascending"),
                axis=alt.Axis(labelLimit=260),
            ),
            color=alt.Color(
                "relationship:N",
                title=None,
                scale=alt.Scale(
                    domain=["Selected environment", "Closest alternative", "Other environment"],
                    range=[
                        BRAND_COLORS["dark_teal"],
                        BRAND_COLORS["terracotta"],
                        BRAND_COLORS["seafoam"],
                    ],
                ),
                legend=alt.Legend(orient="top", direction="horizontal"),
            ),
            tooltip=[
                alt.Tooltip("rank:Q", title="Distance rank"),
                alt.Tooltip("cluster_name:N", title="Mobility environment"),
                alt.Tooltip("relationship:N", title="Relationship"),
                alt.Tooltip("distance:Q", title="Distance", format=".3f"),
            ],
        )
        .properties(height=260)
    )


def _centroid_contrast_chart(
    exact: pd.DataFrame,
    *,
    zone_name: str,
    assigned_cluster: str,
    alternative_cluster: str,
    feature_count: int = 10,
) -> alt.LayerChart:
    contrast = exact.nlargest(feature_count, "assignment_support", keep="all").copy()
    opposing = exact.nsmallest(feature_count, "assignment_support", keep="all").copy()
    contrast = (
        pd.concat([contrast, opposing], ignore_index=True)
        .drop_duplicates("feature_column")
        .assign(absolute_support=lambda frame: frame["assignment_support"].abs())
        .nlargest(feature_count, "absolute_support")
    )
    contrast["feature_label"] = contrast["feature_column"].map(_feature_label)
    feature_order = contrast.sort_values("assignment_support")["feature_label"].tolist()

    rules = contrast.copy()
    rules["centroid_low"] = rules[
        ["assigned_centroid_value", "alternative_centroid_value"]
    ].min(axis=1)
    rules["centroid_high"] = rules[
        ["assigned_centroid_value", "alternative_centroid_value"]
    ].max(axis=1)
    rule_chart = (
        alt.Chart(rules)
        .mark_rule(color=BRAND_COLORS["seafoam"], strokeWidth=4, opacity=0.75)
        .encode(
            x=alt.X(
                "centroid_low:Q",
                title="Standardized feature value",
                axis=alt.Axis(format=".1f"),
            ),
            x2="centroid_high:Q",
            y=alt.Y(
                "feature_label:N",
                title=None,
                sort=feature_order,
                axis=alt.Axis(labelLimit=310),
            ),
        )
    )

    points = contrast.melt(
        id_vars=["feature_column", "feature_label", "assignment_support", "favors"],
        value_vars=[
            "zone_value",
            "assigned_centroid_value",
            "alternative_centroid_value",
        ],
        var_name="series",
        value_name="standardized_value",
    )
    series_labels = {
        "zone_value": zone_name,
        "assigned_centroid_value": assigned_cluster,
        "alternative_centroid_value": alternative_cluster,
    }
    points["series_label"] = points["series"].map(series_labels)
    point_chart = (
        alt.Chart(points)
        .mark_point(filled=True, size=115, stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("standardized_value:Q", title="Standardized feature value"),
            y=alt.Y("feature_label:N", title=None, sort=feature_order),
            color=alt.Color(
                "series_label:N",
                title=None,
                scale=alt.Scale(
                    domain=[zone_name, assigned_cluster, alternative_cluster],
                    range=[
                        BRAND_COLORS["pale_peach"],
                        BRAND_COLORS["dark_teal"],
                        BRAND_COLORS["terracotta"],
                    ],
                ),
                legend=None,
            ),
            shape=alt.Shape(
                "series_label:N",
                title=None,
                scale=alt.Scale(
                    domain=[zone_name, assigned_cluster, alternative_cluster],
                    range=["diamond", "circle", "square"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("feature_label:N", title="Feature"),
                alt.Tooltip("series_label:N", title="Profile"),
                alt.Tooltip(
                    "standardized_value:Q", title="Standardized value", format="+.3f"
                ),
                alt.Tooltip("favors:N", title="Feature favors"),
                alt.Tooltip(
                    "assignment_support:Q", title="Relative-fit contribution", format="+.3f"
                ),
            ],
        )
    )
    return (rule_chart + point_chart).properties(height=max(360, 37 * len(feature_order)))


def _render_centroid_key(
    *, zone_name: str, assigned_cluster: str, alternative_cluster: str
) -> None:
    zone_column, assigned_column, alternative_column = st.columns(3)
    zone_column.markdown(f"**◆ Selected zone:** {zone_name}")
    assigned_column.markdown(f"**● Selected centroid:** {assigned_cluster}")
    alternative_column.markdown(f"**■ Competing centroid:** {alternative_cluster}")


def _feature_support_for_member(
    assignments: pd.DataFrame,
    scaled: pd.DataFrame,
    features: list[str],
    member: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Explain one assignment against its nearest competing centroid."""
    merged = assignments.merge(
        scaled[GRAIN_COLUMNS + features],
        on=GRAIN_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    centroids = merged.groupby("cluster_label", observed=True)[features].mean()
    selected = merged.loc[
        merged["taxi_zone_id"].eq(int(member["taxi_zone_id"]))
        & merged["pre_post_cp"].eq(
            {"Pre-CP": "pre_cp", "Post-CP": "post_cp"}.get(
                str(member["pre_post_cp"]), str(member["pre_post_cp"])
            )
        )
    ].iloc[0]
    assigned_label = int(selected["cluster_label"])
    alternative_name = str(member["nearest_alternative_cluster_name"])
    alternative_label = int(
        assignments.loc[
            assignments["canonical_cluster_name"].eq(alternative_name),
            "cluster_label",
        ].iloc[0]
    )
    values = selected[features].to_numpy(dtype=float)
    assigned_centroid = centroids.loc[assigned_label].to_numpy(dtype=float)
    alternative_centroid = centroids.loc[alternative_label].to_numpy(dtype=float)
    support = (values - alternative_centroid) ** 2 - (
        values - assigned_centroid
    ) ** 2

    exact = pd.DataFrame(
        {
            "feature_column": features,
            "assignment_support": support,
            "zone_value": values,
            "assigned_centroid_value": assigned_centroid,
            "alternative_centroid_value": alternative_centroid,
        }
    )
    exact["assigned_absolute_gap"] = (
        exact["zone_value"] - exact["assigned_centroid_value"]
    ).abs()
    exact["alternative_absolute_gap"] = (
        exact["zone_value"] - exact["alternative_centroid_value"]
    ).abs()
    exact["metric"] = exact["feature_column"].map(_base_metric)
    exact["favors"] = np.where(
        exact["assignment_support"].ge(0),
        str(member["canonical_cluster_name"]),
        alternative_name,
    )
    grouped = (
        exact.groupby("metric", observed=True)["assignment_support"]
        .sum()
        .rename("assignment_support")
        .reset_index()
    )
    grouped["metric_label"] = grouped["metric"].map(METRIC_LABELS).fillna(
        grouped["metric"]
    )
    grouped["favors"] = np.where(
        grouped["assignment_support"].ge(0),
        str(member["canonical_cluster_name"]),
        alternative_name,
    )

    cluster_names = (
        assignments[["cluster_label", "canonical_cluster_name"]]
        .drop_duplicates("cluster_label")
        .set_index("cluster_label")["canonical_cluster_name"]
        .to_dict()
    )
    distance_records = []
    for label, centroid in centroids.iterrows():
        distance_records.append(
            {
                "cluster_label": int(label),
                "cluster_name": cluster_names.get(int(label), f"Cluster {label}"),
                "distance": float(np.sqrt(((values - centroid.to_numpy(dtype=float)) ** 2).sum())),
                "relationship": (
                    "Selected environment"
                    if int(label) == assigned_label
                    else "Closest alternative"
                    if int(label) == alternative_label
                    else "Other environment"
                ),
            }
        )
    distances = pd.DataFrame(distance_records).sort_values("distance")
    distances["rank"] = np.arange(1, len(distances) + 1)
    return (
        grouped.sort_values("assignment_support"),
        exact.sort_values("assignment_support", ascending=False),
        distances,
    )


def _transition_diagnostics(
    assignments: pd.DataFrame,
    scaled: pd.DataFrame,
    features: list[str],
    members: pd.DataFrame,
    taxi_zone_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    zone_members = members.loc[members["taxi_zone_id"].eq(taxi_zone_id)].copy()
    zone_members["period_order"] = zone_members["pre_post_cp"].map(
        {"Pre-CP": 0, "Post-CP": 1}
    )
    zone_members = zone_members.sort_values("period_order")
    if len(zone_members) < 2:
        return zone_members, pd.DataFrame()

    merged = assignments.merge(
        scaled[GRAIN_COLUMNS + features],
        on=GRAIN_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    centroids = merged.groupby("cluster_label", observed=True)[features].mean()
    selected = merged.loc[merged["taxi_zone_id"].eq(taxi_zone_id)].set_index(
        "pre_post_cp"
    )
    if not {"pre_cp", "post_cp"}.issubset(selected.index):
        return zone_members, pd.DataFrame()

    pre_label = int(selected.loc["pre_cp", "cluster_label"])
    post_label = int(selected.loc["post_cp", "cluster_label"])
    if pre_label == post_label:
        return zone_members, pd.DataFrame()

    pre_values = selected.loc["pre_cp", features].to_numpy(dtype=float)
    post_values = selected.loc["post_cp", features].to_numpy(dtype=float)
    old_centroid = centroids.loc[pre_label].to_numpy(dtype=float)
    new_centroid = centroids.loc[post_label].to_numpy(dtype=float)
    pre_new_advantage = (pre_values - old_centroid) ** 2 - (
        pre_values - new_centroid
    ) ** 2
    post_new_advantage = (post_values - old_centroid) ** 2 - (
        post_values - new_centroid
    ) ** 2
    movement = post_new_advantage - pre_new_advantage
    exact = pd.DataFrame({"feature_column": features, "relative_fit_shift": movement})
    exact["metric"] = exact["feature_column"].map(_base_metric)
    grouped = (
        exact.groupby("metric", observed=True)["relative_fit_shift"]
        .sum()
        .reset_index()
    )
    grouped["metric_label"] = grouped["metric"].map(METRIC_LABELS).fillna(
        grouped["metric"]
    )
    grouped["direction"] = np.where(
        grouped["relative_fit_shift"].ge(0),
        "Toward Post-CP cluster",
        "Toward Pre-CP cluster",
    )
    return zone_members, grouped.sort_values("relative_fit_shift")


def _transition_chart(transition_support: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(transition_support)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "relative_fit_shift:Q",
                title="Change in relative squared-distance fit — left old environment; right new environment",
            ),
            y=alt.Y(
                "metric_label:N",
                title=None,
                sort=alt.SortField(field="relative_fit_shift", order="ascending"),
                axis=alt.Axis(labelLimit=230),
            ),
            color=alt.Color(
                "direction:N",
                title=None,
                scale=alt.Scale(
                    domain=["Toward Post-CP cluster", "Toward Pre-CP cluster"],
                    range=[BRAND_COLORS["dark_teal"], BRAND_COLORS["terracotta"]],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("metric_label:N", title="Metric family"),
                alt.Tooltip("direction:N", title="Movement"),
                alt.Tooltip(
                    "relative_fit_shift:Q", title="Relative-fit shift", format="+.3f"
                ),
            ],
        )
        .properties(height=380)
    )


@st.cache_data(show_spinner="Preparing the mobility-environment explorer...")
def _load_explorer_data() -> dict[str, object]:
    assignments = pd.read_parquet(ASSIGNMENTS_PATH).copy()
    profiles = pd.read_parquet(PROFILES_PATH).copy()
    scaled = pd.read_parquet(SCALED_MATRIX_PATH).copy()
    features = _feature_contract(scaled)
    members, _, _, _, _ = _member_diagnostics(assignments, scaled)
    return {
        "assignments": assignments,
        "profiles": profiles,
        "scaled": scaled,
        "features": features,
        "members": members,
    }


@st.cache_data(show_spinner=False)
def _validate_production_contract(
    assignments: pd.DataFrame,
    profiles: pd.DataFrame,
    scaled: pd.DataFrame,
) -> dict[str, int]:
    """Fail closed when the app cannot reproduce the canonical K-Means solution."""
    features = _feature_contract(scaled)
    failures: list[str] = []

    if len(features) != 110:
        failures.append(f"expected 110 clustering features; found {len(features)}")
    if len(assignments) != 526:
        failures.append(f"expected 526 Taxi Zone-period assignments; found {len(assignments)}")
    duplicate_assignments = int(assignments.duplicated(GRAIN_COLUMNS).sum())
    if duplicate_assignments:
        failures.append(f"found {duplicate_assignments} duplicate assignment keys")

    scaled_keys = scaled[GRAIN_COLUMNS]
    duplicate_scaled = int(scaled_keys.duplicated().sum())
    if duplicate_scaled:
        failures.append(f"found {duplicate_scaled} duplicate scaled-matrix keys")

    merged = assignments.merge(
        scaled[GRAIN_COLUMNS + features],
        on=GRAIN_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = int(merged["_merge"].ne("both").sum())
    if unmatched:
        failures.append(f"found {unmatched} assignments without a scaled feature row")
    missing_feature_rows = int(merged[features].isna().any(axis=1).sum())
    if missing_feature_rows:
        failures.append(f"found {missing_feature_rows} rows with missing clustering features")

    expected_periods = {"pre_cp", "post_cp"}
    incomplete_zone_pairs = int(
        assignments.groupby("taxi_zone_id", observed=True)["pre_post_cp"]
        .agg(lambda values: set(values.astype(str)) != expected_periods)
        .sum()
    )
    if incomplete_zone_pairs:
        failures.append(f"found {incomplete_zone_pairs} Taxi Zones without a complete Pre/Post pair")

    profile_duplicates = int(
        profiles.duplicated(["cluster_label", "feature_column"]).sum()
    )
    if profile_duplicates:
        failures.append(f"found {profile_duplicates} duplicate cluster-profile rows")

    assignment_mismatches = 0
    unresolved_ties = 0
    invalid_alternatives = 0
    if not failures:
        centroids = merged.groupby("cluster_label", observed=True)[features].mean()
        centroid_labels = centroids.index.to_numpy(dtype=int)
        values = merged[features].to_numpy(dtype=float)
        centroid_values = centroids.to_numpy(dtype=float)
        distances = np.sqrt(
            ((values[:, None, :] - centroid_values[None, :, :]) ** 2).sum(axis=2)
        )
        nearest_positions = distances.argmin(axis=1)
        reproduced_labels = centroid_labels[nearest_positions]
        canonical_labels = merged["cluster_label"].to_numpy(dtype=int)
        assignment_mismatches = int((reproduced_labels != canonical_labels).sum())
        sorted_distances = np.sort(distances, axis=1)
        unresolved_ties = int(
            np.isclose(sorted_distances[:, 0], sorted_distances[:, 1], atol=1e-9).sum()
        )
        invalid_alternatives = int(
            (~np.isfinite(sorted_distances[:, 1]) | (sorted_distances[:, 1] < 0)).sum()
        )
        if assignment_mismatches:
            failures.append(
                f"{assignment_mismatches} canonical assignments are not the nearest reconstructed centroid"
            )
        if unresolved_ties:
            failures.append(f"found {unresolved_ties} unresolved nearest-centroid ties")
        if invalid_alternatives:
            failures.append(f"found {invalid_alternatives} rows without a valid alternative centroid")

    if failures:
        raise ValueError("Mobility-environment validation failed: " + "; ".join(failures) + ".")

    return {
        "assignment_rows": len(assignments),
        "features": len(features),
        "clusters": int(assignments["cluster_label"].nunique()),
        "taxi_zones": int(assignments["taxi_zone_id"].nunique()),
        "assignment_mismatches": assignment_mismatches,
        "centroid_ties": unresolved_ties,
    }


@st.cache_data(show_spinner="Scouting mobility environments...")
def _build_scouting_report() -> str:
    _require_file(ASSIGNMENTS_PATH)
    _require_file(PROFILES_PATH)
    _require_file(NAME_LOOKUP_PATH)

    assignments = pd.read_parquet(ASSIGNMENTS_PATH).copy()
    profiles = pd.read_parquet(PROFILES_PATH).copy()
    lookup = pd.read_csv(NAME_LOOKUP_PATH).copy()

    required_assignment_columns = {
        "solution_id",
        "cluster_method",
        "cluster_count",
        "taxi_zone_id",
        "zone",
        "borough",
        "cbd_spatial_category",
        "pre_post_cp",
        "cluster_label",
        "canonical_cluster_name",
    }
    required_profile_columns = {
        "cluster_label",
        "feature_column",
        "cluster_mean_value",
        "metric_label",
        "cluster_signal_zscore",
        "canonical_cluster_name",
    }
    missing_assignments = sorted(required_assignment_columns - set(assignments.columns))
    missing_profiles = sorted(required_profile_columns - set(profiles.columns))
    if missing_assignments or missing_profiles:
        raise ValueError(
            "Cluster contract is incomplete. "
            f"Assignment fields missing: {missing_assignments or 'none'}; "
            f"profile fields missing: {missing_profiles or 'none'}."
        )

    transition_matrix, changed_zones = _transition_summary(assignments)
    key_duplicates = int(assignments.duplicated(GRAIN_COLUMNS).sum())
    profile_duplicates = int(
        profiles.duplicated(["cluster_label", "feature_column"]).sum()
    )
    lookup_mismatches = int(
        assignments[["cluster_label", "canonical_cluster_name"]]
        .drop_duplicates()
        .merge(
            lookup,
            on="cluster_label",
            how="outer",
            suffixes=("_assignment", "_lookup"),
        )
        .query("canonical_cluster_name_assignment != canonical_cluster_name_lookup")
        .shape[0]
    )

    contract = pd.DataFrame(
        [
            ("Assignment rows", len(assignments)),
            ("Distinct Taxi Zones", assignments["taxi_zone_id"].nunique()),
            ("Policy periods", assignments["pre_post_cp"].nunique()),
            ("Canonical clusters", assignments["cluster_label"].nunique()),
            ("Assignment key duplicates", key_duplicates),
            ("Profile rows", len(profiles)),
            ("Profile key duplicates", profile_duplicates),
            ("Lookup mismatches", lookup_mismatches),
            ("Zones changing cluster", len(changed_zones)),
            ("Zones remaining stable", assignments["taxi_zone_id"].nunique() - len(changed_zones)),
        ],
        columns=["check", "result"],
    )

    report = REPORT_TITLE + "\n"
    report += _section("CONTRACT QA", _csv(contract, decimals=1))
    report += _section("CLUSTER SIZE BY PERIOD", _csv(_cluster_profiles(assignments), decimals=1))
    report += _section("CLUSTER ARCHETYPE SIGNALS", _csv(_profile_signals(profiles), decimals=3))
    report += _section("PRE/POST TRANSITION MATRIX", _csv(transition_matrix, decimals=1))
    report += _section("TAXI ZONES THAT CHANGED ENVIRONMENT", _csv(changed_zones, decimals=1))
    report += _section("BOROUGH MIX", _csv(_geography_mix(assignments, "borough"), decimals=3))
    report += _section(
        "POLICY-GEOGRAPHY MIX",
        _csv(_geography_mix(assignments, "cbd_spatial_category"), decimals=3),
    )

    if SCALED_MATRIX_PATH.exists():
        scaled = pd.read_parquet(SCALED_MATRIX_PATH).copy()
        members, coherence, representative, boundary, contributions = (
            _member_diagnostics(assignments, scaled)
        )
        scaled_status = pd.DataFrame(
            [
                ("Scaled matrix available", True),
                ("Scaled matrix rows", len(scaled)),
                ("Clustering features", len(_feature_contract(scaled))),
                ("Assignment/scaled key alignment", set(map(tuple, assignments[GRAIN_COLUMNS].to_numpy())) == set(map(tuple, scaled[GRAIN_COLUMNS].to_numpy()))),
            ],
            columns=["check", "result"],
        )
        report += _section("MEMBER-DIAGNOSTIC CONTRACT", _csv(scaled_status, decimals=1))
        report += _section("CLUSTER COHERENCE", _csv(coherence, decimals=3))
        report += _section("FIVE MOST REPRESENTATIVE MEMBERS PER CLUSTER", _csv(representative, decimals=3))
        report += _section("EIGHT WEAKEST-MARGIN MEMBERS PER CLUSTER", _csv(boundary, decimals=3))
        report += _section("FEATURES SUPPORTING BOUNDARY ASSIGNMENTS", _csv(contributions, decimals=3))
        decision = (
            "The exact 110-feature scaled matrix is available. Phase 2 can build a "
            "faithful member-distance view, nearest-alternative comparison, and "
            "feature-level assignment explanation without approximating the model."
        )
    else:
        missing = pd.DataFrame(
            [
                ("Required asset", str(SCALED_MATRIX_PATH)),
                ("Expected grain", "taxi_zone_id × pre_post_cp"),
                ("Expected feature count", 110),
                ("Why required", "Exact K-Means centroid distances and nearest alternatives"),
            ],
            columns=["item", "value"],
        )
        report += _section("MISSING MEMBER-ATTRIBUTION ASSET", _csv(missing, decimals=1))
        decision = (
            "The existing 3.2.2 app tables explain cluster archetypes and membership "
            "but cannot faithfully rank representative, boundary, or nearest-alternative "
            "members. Add the original 3.1.1 scaled matrix to the Showcase or build a "
            "compact 526-row assignment-diagnostic export upstream before Phase 2."
        )

    report += _section(
        "PHASE-1 DECISION",
        decision
        + "\n\nThe recommended frozen hero is the cluster with a recognizable archetype, "
        "meaningful internal spread, and several interpretable boundary members—not "
        "automatically the largest or most separated cluster.",
    )
    return report.strip()


inject_app_css()

# -----------------------------------------------------------------------------
# PHASE 5 PRODUCTION EXPERIENCE
# -----------------------------------------------------------------------------
st.caption(PAGE_CAPTION)
st.title(PAGE_TITLE)
st.write(
    "Neighborhoods can differ in more than one mobility measure at a time. This page "
    "groups Taxi Zones by their complete mobility pattern—demand, speed, duration, and "
    "time-of-week shape—then shows what defines each environment, how typical an "
    "individual zone is, and whether that zone's environment changed after congestion "
    "pricing began."
)

try:
    hero = _load_hero_data()
    explorer = _load_explorer_data()
    production_contract = _validate_production_contract(
        explorer["assignments"], explorer["profiles"], explorer["scaled"]
    )
except Exception as error:
    st.error(f"The mobility-environment profiler could not be built: {error}")
    st.stop()

hero_members = hero["members"]
hero_member = hero["member"]
archetype = hero["archetype"]
strongest = hero["strongest"]
explorer_members = explorer["members"]

st.header("How do we group NYC Taxi Zones?")
st.write(
    "Two recurring geography systems appear throughout the Showcase. **Mobility "
    "environments** are learned from the data: Taxi Zones with similar 110-feature "
    "mobility profiles are grouped together even when they are far apart geographically. "
    "**Policy geography** is manually defined from each zone's relationship to the "
    "congestion-pricing geography. The maps below make both systems visible before we "
    "use them as filters elsewhere."
)

environment_tab, policy_tab = st.tabs(
    ["Mobility environments", "Policy geography"]
)

with environment_tab:
    map_period_label = st.radio(
        "Mobility-environment period",
        options=["Pre-CP", "Post-CP"],
        index=1,
        horizontal=True,
        key="raw16_orientation_environment_period",
        help=(
            "Mobility environments are period-specific because a Taxi Zone's complete "
            "mobility profile can move closer to a different learned cluster over time."
        ),
    )
    map_period = "pre_cp" if map_period_label == "Pre-CP" else "post_cp"

    st.plotly_chart(
        _mobility_environment_map(
            explorer["assignments"],
            period=map_period,
        ),
        width="stretch",
        config={"displayModeBar": False},
        key=f"raw16_environment_map_{map_period}",
    )
    st.caption(
        "Colors identify the learned mobility environment assigned to each physical "
        "Taxi Zone. The categories are behavioral clusters, not contiguous geographic "
        "regions; use the period control to see whether assignments changed."
    )

with policy_tab:
    st.plotly_chart(
        _policy_geography_map(explorer["assignments"]),
        width="stretch",
        config={"displayModeBar": False},
        key="raw16_policy_geography_map",
    )
    st.caption(
        "Policy geography is a fixed, manually defined spatial classification used "
        "throughout the Showcase: CBD, adjacent, gateway, and non-CBD."
    )

render_chart_insight(
    "These two maps group the same Taxi Zones for different reasons: **mobility "
    "environments ask which places behave alike**, while **policy geography asks where "
    "a place sits relative to the congestion-pricing geography**. Keeping those ideas "
    "separate makes the recurring filters elsewhere in the Showcase easier to interpret."
)

st.header("What defines a mobility environment?")
st.write(
    "Now look inside one of the learned groups. The fixed example is the Post-CP "
    "**Long-Trip Fast-Mobility Zones** environment. Each bar shows one headline "
    "mobility measure relative to the citywide feature norm: right of zero is above "
    "the norm and left is below. These ten averages make the environment readable, "
    "while the actual cluster assignment uses the full 110-feature mobility pattern."
)

st.altair_chart(_archetype_chart(archetype), width="stretch")
hero_low = archetype.iloc[0]
hero_high = archetype.iloc[-1]
render_chart_insight(
    f"This environment is distinguished most by **low {hero_low['metric_label']}** "
    f"({float(hero_low['cluster_signal_zscore']):+.1f} relative to the city norm) and "
    f"**high {hero_high['metric_label']}** "
    f"({float(hero_high['cluster_signal_zscore']):+.1f}). Membership is determined by "
    "the complete 110-feature mobility pattern—not by geography or either metric alone."
)

hero_1, hero_2, hero_3, hero_4 = st.columns(4)
hero_1.metric("Post-CP members", f"{len(hero_members):,}")
hero_2.metric("New since Pre-CP", f"{int(hero['entrants']):,}")
hero_3.metric(
    "Boundary members",
    f"{int(hero_members['distance_ratio'].ge(0.90).sum()):,}",
    help="Members whose closest competing centroid is almost as near as the selected centroid.",
)
hero_4.metric(
    "Strongest defining signal",
    f"{float(strongest['cluster_signal_zscore']):+.1f}",
    help=f"{strongest['metric_label']}; standardized units from the city norm.",
)

st.markdown("#### A useful boundary example: Broad Channel")
boundary_1, boundary_2, boundary_3 = st.columns(3)
boundary_1.metric(
    "Boundary example period",
    "Post-CP",
    help=f"Broad Channel belonged to {hero['pre_cluster']} Pre-CP.",
)
boundary_2.metric(
    "Selected-centroid distance", f"{float(hero_member['assigned_distance']):.3f}"
)
boundary_3.metric(
    "Closest-competitor distance",
    f"{float(hero_member['nearest_alternative_distance']):.3f}",
    help=str(hero_member["nearest_alternative_cluster_name"]),
)
render_chart_insight(
    f"Broad Channel changed from **{hero['pre_cluster']} Pre-CP** to "
    f"**{HERO_CLUSTER} Post-CP**, but the Post-CP winner leads its closest competitor "
    f"by only **{float(hero_member['assignment_margin']):.3f} distance units**. It is "
    "therefore a boundary assignment worth investigating—not a textbook example."
)

with exploration_section(
    key="raw16_exploration_area",
    title="Investigate an environment and Taxi Zone",
    description=(
        "Choose a mobility environment and policy period, then inspect one Taxi "
        "Zone's typicality, nearest alternatives, feature-level fit, and Pre/Post "
        "environment transition."
    ),
):
    cluster_options = sorted(explorer_members["canonical_cluster_name"].dropna().unique())
    control_1, control_2 = st.columns([2, 1])
    selected_cluster = control_1.selectbox(
        "Mobility environment",
        options=cluster_options,
        index=cluster_options.index(HERO_CLUSTER) if HERO_CLUSTER in cluster_options else 0,
        key="raw16_phase6_cluster",
    )
    selected_period = control_2.segmented_control(
        "Policy period",
        options=["Pre-CP", "Post-CP"],
        default="Post-CP",
        key="raw16_phase6_period",
    ) or "Post-CP"

    scope_members = explorer_members.loc[
        explorer_members["canonical_cluster_name"].eq(selected_cluster)
        & explorer_members["pre_post_cp"].eq(selected_period)
    ].copy()
    scope_members["period_distance_percentile"] = scope_members[
        "assigned_distance"
    ].rank(method="average", pct=True)

    zone_choices = scope_members.sort_values(["zone", "borough"]).assign(
        zone_choice=lambda frame: frame["zone"] + " · " + frame["borough"]
    )
    default_zone_index = 0
    if selected_cluster == HERO_CLUSTER and selected_period == "Post-CP":
        broad_channel_matches = zone_choices.index[
            zone_choices["taxi_zone_id"].eq(HERO_ZONE_ID)
        ].tolist()
        if broad_channel_matches:
            default_zone_index = zone_choices.index.get_loc(broad_channel_matches[0])
    selected_zone_choice = st.selectbox(
        "Taxi Zone to explain",
        options=zone_choices["zone_choice"].tolist(),
        index=default_zone_index,
        key=f"raw16_phase6_zone_{selected_cluster}_{selected_period}",
    )
    selected_member = zone_choices.loc[
        zone_choices["zone_choice"].eq(selected_zone_choice)
    ].iloc[0]
    selected_zone_name = str(selected_member["zone"])
    member_support, member_exact, member_distances = _feature_support_for_member(
        explorer["assignments"],
        explorer["scaled"],
        explorer["features"],
        selected_member,
    )
    scope_profile = explorer["profiles"].loc[
        explorer["profiles"]["canonical_cluster_name"].eq(selected_cluster)
    ].copy()
    scope_profile["direction"] = np.where(
        scope_profile["cluster_signal_zscore"].ge(0),
        "Above city norm",
        "Below city norm",
    )
    scope_profile = scope_profile.sort_values("cluster_signal_zscore")

    overview_tab, diagnosis_tab, transition_tab = st.tabs(
        ["Environment overview", "Why this Taxi Zone?", "Pre/Post transition"]
    )

    with overview_tab:
        overview_1, overview_2, overview_3, overview_4 = st.columns(4)
        overview_1.metric("Members in period", f"{len(scope_members):,}")
        overview_2.metric(
            "Typical centroid distance", f"{scope_members['assigned_distance'].median():.3f}"
        )
        overview_3.metric(
            "Boundary members", f"{int(scope_members['distance_ratio'].ge(0.90).sum()):,}"
        )
        overview_4.metric(
            "Selected member percentile",
            f"{float(selected_member['period_distance_percentile']) * 100:.0f}th",
            help="Percentile of centroid distance among members of this environment in the selected period; higher is less typical.",
        )

        st.subheader("Environment signature")
        st.altair_chart(_archetype_chart(scope_profile), width="stretch")
        overview_low = scope_profile.iloc[0]
        overview_high = scope_profile.iloc[-1]
        render_chart_insight(
            f"**{selected_cluster}** is most below the city norm on "
            f"**{overview_low['metric_label']}** and most above it on "
            f"**{overview_high['metric_label']}**. These ten headline averages describe "
            "the environment; the actual assignment also uses their time-of-week shapes."
        )

        st.subheader("Which members are most—and least—typical?")
        st.caption(
            f"Terracotta highlights **{selected_zone_name}**; teal marks the other "
            f"{selected_period} members of **{selected_cluster}**."
        )
        st.altair_chart(
            _member_distance_chart(
                scope_members,
                selected_zone_id=int(selected_member["taxi_zone_id"]),
                selected_zone_name=selected_zone_name,
            ),
            width="stretch",
        )
        closest_member = scope_members.loc[scope_members["assigned_distance"].idxmin()]
        farthest_member = scope_members.loc[scope_members["assigned_distance"].idxmax()]
        render_chart_insight(
            f"**{closest_member['zone']}** is the most representative member in this "
            f"period; **{farthest_member['zone']}** is the most distant. "
            f"**{selected_zone_name}** sits at the "
            f"**{float(selected_member['period_distance_percentile']) * 100:.0f}th "
            "distance percentile within this period**. Distance from the centroid measures typicality; the "
            "winning margin measures how strongly this environment beat its alternatives."
        )

    with diagnosis_tab:
        ranked_distances = member_distances.sort_values("distance").reset_index(drop=True)
        winner = ranked_distances.iloc[0]
        runner_up = ranked_distances.iloc[1]
        third_place = ranked_distances.iloc[2]
        diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4 = st.columns(4)
        diagnosis_1.metric("Selected distance", f"{float(winner['distance']):.3f}")
        diagnosis_2.metric("Competitor distance", f"{float(runner_up['distance']):.3f}")
        diagnosis_3.metric(
            "Distance advantage",
            f"{float(runner_up['distance'] - winner['distance']):.3f}",
        )
        diagnosis_4.metric(
            "Distance ratio",
            f"{float(winner['distance'] / runner_up['distance']):.3f}",
            help="Values near 1 indicate a close boundary assignment.",
        )
        st.caption(
            f"Selected: **{winner['cluster_name']}** · Closest competitor: "
            f"**{runner_up['cluster_name']}**"
        )

        st.subheader("Why this environment—and why not the others?")
        st.altair_chart(_all_cluster_distance_chart(member_distances), width="stretch")
        render_chart_insight(
            f"**{winner['cluster_name']}** wins because its centroid is closest at "
            f"**{float(winner['distance']):.3f}**. **{runner_up['cluster_name']}** is the "
            f"nearest alternative at **{float(runner_up['distance']):.3f}**; the third "
            f"choice, **{third_place['cluster_name']}**, is farther at "
            f"**{float(third_place['distance']):.3f}**. K-Means simply chooses the "
            "smallest complete-profile distance to an environment centroid."
        )

        st.subheader("Where the zone sits relative to both centroids")
        _render_centroid_key(
            zone_name=selected_zone_name,
            assigned_cluster=selected_cluster,
            alternative_cluster=str(selected_member["nearest_alternative_cluster_name"]),
        )
        st.altair_chart(
            _centroid_contrast_chart(
                member_exact,
                zone_name=selected_zone_name,
                assigned_cluster=selected_cluster,
                alternative_cluster=str(selected_member["nearest_alternative_cluster_name"]),
            ),
            width="stretch",
        )
        strongest_for = member_exact.iloc[0]
        strongest_against = member_exact.iloc[-1]
        counter_copy = (
            f"The clearest counterargument is **{_feature_label(str(strongest_against['feature_column']))}**, "
            f"which is closer to **{selected_member['nearest_alternative_cluster_name']}**."
            if float(strongest_against["assignment_support"]) < 0
            else "Even the weakest displayed feature is closer to the selected centroid."
        )
        render_chart_insight(
            f"The clearest reason to choose **{selected_cluster}** is "
            f"**{_feature_label(str(strongest_for['feature_column']))}**: "
            f"{selected_zone_name} is **{float(strongest_for['assigned_absolute_gap']):.3f}** "
            f"standardized units from the selected centroid versus "
            f"**{float(strongest_for['alternative_absolute_gap']):.3f}** from the "
            f"alternative. {counter_copy}"
        )

        with st.expander("How all 110 feature differences add up"):
            _render_support_key(
                assigned_cluster=selected_cluster,
                alternative_cluster=str(selected_member["nearest_alternative_cluster_name"]),
                zone_name=selected_zone_name,
            )
            st.altair_chart(
                _support_chart(
                    member_support,
                    assigned_cluster=selected_cluster,
                    alternative_cluster=str(selected_member["nearest_alternative_cluster_name"]),
                    zone_name=selected_zone_name,
                ),
                width="stretch",
            )
            best_group = member_support.loc[member_support["assignment_support"].idxmax()]
            opposing_group = member_support.loc[member_support["assignment_support"].idxmin()]
            render_chart_insight(
                f"**{best_group['metric_label']}** provides the strongest grouped support "
                f"for **{selected_cluster}**. **{opposing_group['metric_label']}** is the "
                "strongest grouped counterweight or, when still positive, the weakest "
                "supporting family."
            )

        with st.expander("Inspect all 110 individual feature contributions"):
            exact_display = member_exact.copy()
            exact_display["feature"] = exact_display["feature_column"].map(_feature_label)
            exact_display["absolute_support"] = exact_display["assignment_support"].abs()
            st.dataframe(
                exact_display.sort_values("absolute_support", ascending=False)[
                    ["feature", "assignment_support", "favors"]
                ].style.format({"assignment_support": "{:+,.3f}"}),
                hide_index=True,
                width="stretch",
                height=460,
            )

    with transition_tab:
        zone_periods, transition_support = _transition_diagnostics(
            explorer["assignments"],
            explorer["scaled"],
            explorer["features"],
            explorer_members,
            int(selected_member["taxi_zone_id"]),
        )
        if len(zone_periods) < 2:
            st.info("A complete Pre/Post pair is not available for this Taxi Zone.")
        else:
            pre_row = zone_periods.loc[zone_periods["pre_post_cp"].eq("Pre-CP")].iloc[0]
            post_row = zone_periods.loc[zone_periods["pre_post_cp"].eq("Post-CP")].iloc[0]
            changed = str(pre_row["canonical_cluster_name"]) != str(
                post_row["canonical_cluster_name"]
            )
            transition_1, transition_2, transition_3, transition_4 = st.columns(4)
            transition_1.metric(
                f"Pre-CP · {pre_row['canonical_cluster_name']}",
                f"{float(pre_row['assigned_distance']):.3f}",
                help="Distance to the Pre-CP selected centroid.",
            )
            transition_2.metric(
                f"Post-CP · {post_row['canonical_cluster_name']}",
                f"{float(post_row['assigned_distance']):.3f}",
                help="Distance to the Post-CP selected centroid.",
            )
            transition_3.metric(
                "Pre-CP winning margin", f"{float(pre_row['assignment_margin']):.3f}"
            )
            transition_4.metric(
                "Post-CP winning margin", f"{float(post_row['assignment_margin']):.3f}"
            )

            if changed and not transition_support.empty:
                st.subheader("What moved the zone toward its Post-CP cluster?")
                st.caption(
                    f"Left favors the old cluster, {pre_row['canonical_cluster_name']}; "
                    f"right favors the new cluster, {post_row['canonical_cluster_name']}."
                )
                st.altair_chart(_transition_chart(transition_support), width="stretch")
                transition_leader = transition_support.loc[
                    transition_support["relative_fit_shift"].idxmax()
                ]
                transition_resistor = transition_support.loc[
                    transition_support["relative_fit_shift"].idxmin()
                ]
                render_chart_insight(
                    f"**{selected_zone_name} changed environments**. The largest movement "
                    f"toward **{post_row['canonical_cluster_name']}** came from "
                    f"**{transition_leader['metric_label']}**. "
                    f"**{transition_resistor['metric_label']}** moved most in the opposite "
                    "direction or supplied the least support for the transition."
                )
            else:
                render_chart_insight(
                    f"**{selected_zone_name} remained in {post_row['canonical_cluster_name']}** "
                    "across both periods. Its winning margin changed from "
                    f"**{float(pre_row['assignment_margin']):.3f}** Pre-CP to "
                    f"**{float(post_row['assignment_margin']):.3f}** Post-CP, indicating "
                    "whether the stable assignment became more or less decisive."
                )


st.markdown("### What this page establishes")
st.markdown(
    "The Showcase uses two complementary ways to group place. **Policy geography** "
    "describes where a Taxi Zone sits relative to the congestion-pricing geography; "
    "**mobility environments** describe which zones have similar multivariate mobility "
    "patterns, regardless of where they sit on the map. Within the learned environments, "
    "some zones are highly typical while others are genuine boundary cases, and the same "
    "zone can move toward a different environment as its overall mobility profile changes."
)

with st.expander("How this page works", expanded=False):
    st.markdown(
        """
        **1. Keep the two geography systems conceptually separate.** **Policy geography**
        is a manually defined spatial classification—CBD, adjacent, gateway, and non-CBD.
        **Mobility environments** are learned from mobility behavior and can group
        geographically distant Taxi Zones together.

        **3. Represent each Taxi Zone-period with its full mobility pattern.** Ten
        mobility metrics each contribute one period mean and ten time-of-week shape
        features: **10 × (1 + 10) = 110 standardized inputs**.

        **2. Group similar profiles with K-Means.** The selected solution contains
        **five clusters**, fitted once across the shared Pre/Post matrix. The closest
        cluster centroid determines membership; the reader-facing environment names
        were assigned after clustering to describe their dominant mobility signatures.

        **4. Separate an environment's signature from an individual member's fit.**
        The headline profile summarizes the environment average. **Centroid distance**
        measures how far an individual Taxi Zone-period sits from that average; a lower
        distance means a more typical member.

        **5. Measure how decisive an assignment is.** The **winning margin** compares
        the selected centroid with the nearest alternative. A small margin—or a distance
        ratio near 1—marks a boundary case where another environment is almost as close.

        **6. Explain assignments with the same features used by the model.** Feature
        contributions show which parts of the 110-feature profile pull a zone toward
        its selected environment rather than its alternatives. They explain statistical
        fit, not why the underlying mobility pattern occurred.

        **7. Compare the same Taxi Zone across policy periods.** Pre/Post transitions
        show whether the zone's nearest mobility environment changed and which features
        most supported that shift.
        """
    )
    st.caption(
        f"Validation check: {production_contract['assignment_rows']:,} Taxi Zone-period "
        f"rows, {production_contract['features']} standardized features, "
        f"{production_contract['clusters']} environments, zero reconstructed "
        "assignment mismatches, and zero nearest-centroid ties."
    )

st.caption(
    "Evidence scope: descriptive clustering of Taxi Zone mobility profiles before and "
    "after the January 2025 congestion-pricing launch. Environment membership and "
    "feature contributions summarize similarity in the observed mobility data; they do "
    "not establish that congestion pricing caused a zone to enter or leave an environment."
)

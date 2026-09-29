from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from app.utils.project_branding import inject_app_css


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "app"
IMAGE_DIR = APP_DIR / "images"

PULSE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "app_tables"
    / "mobility_pulse_presets"
)
PULSE_MANIFEST = PULSE_DIR / "manifest.json"


# ---------------------------------------------------------------------
# External links
# ---------------------------------------------------------------------

SHOWCASE_REPO_URL = (
    "https://github.com/jesingson/"
    "NYC-Congestion-Pricing-Mobility-Showcase"
)

SOURCE_REPO_URL = (
    "https://github.com/jesingson/"
    "NYC-Congestion-Pricing-Mobility-v2"
)


# ---------------------------------------------------------------------
# Homepage-only styling
# ---------------------------------------------------------------------

inject_app_css()

st.markdown(
    """
    <style>
    /* Canonical project palette:
       dark teal #006D77
       seafoam #83C5BE
       ice #EDF6F9
       pale peach #FFDDD2
       terracotta #E29578
    */

    .home-hero {
        background: linear-gradient(
            135deg,
            #EDF6F9 0%,
            #EDF6F9 72%,
            #FFDDD2 100%
        );
        border: 1px solid rgba(0, 109, 119, 0.16);
        border-radius: 24px;
        padding: 2.15rem 2.3rem 1.8rem 2.3rem;
        margin-bottom: 1.15rem;
    }

    .home-eyebrow,
    .section-kicker {
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .home-eyebrow {
        color: #006D77;
        margin-bottom: 0.45rem;
    }

    .section-kicker {
        color: #E29578;
        margin-bottom: 0.20rem;
    }

    .home-hero h1 {
        color: #006D77;
        font-size: clamp(2.35rem, 4.4vw, 4.5rem);
        line-height: 1.02;
        letter-spacing: -0.04em;
        margin: 0 0 0.95rem 0;
        max-width: 1120px;
    }

    .home-hero-copy {
        font-size: 1.08rem;
        line-height: 1.62;
        max-width: 1060px;
    }

    .path-card {
        min-height: 220px;
        border: 1px solid rgba(0, 109, 119, 0.18);
        border-radius: 19px;
        padding: 1.28rem 1.35rem 1.05rem 1.35rem;
        background: white;
        margin: 0.30rem 0 0.50rem 0;
    }

    .path-number {
        display: inline-block;
        color: #006D77;
        background: #EDF6F9;
        border-radius: 999px;
        padding: 0.18rem 0.60rem;
        font-size: 0.70rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        margin-bottom: 0.68rem;
    }

    .path-card h3 {
        color: #006D77;
        margin-top: 0;
        margin-bottom: 0.48rem;
        line-height: 1.14;
    }

    .path-card p {
        margin-bottom: 0;
        line-height: 1.56;
    }

    .finding-copy {
        border: 1px solid rgba(0, 109, 119, 0.16);
        border-radius: 20px;
        padding: 1.18rem 1.28rem;
        background: white;
        margin: 0.55rem 0 0.90rem 0;
    }

    .finding-number {
        color: #E29578;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        margin-bottom: 0.30rem;
    }

    .finding-copy h3 {
        color: #006D77;
        line-height: 1.16;
        margin-top: 0;
        margin-bottom: 0.58rem;
    }

    .finding-copy p {
        line-height: 1.60;
        margin-bottom: 0;
    }

    .method-intro {
        background: #EDF6F9;
        border-left: 5px solid #83C5BE;
        border-radius: 13px;
        padding: 1.05rem 1.18rem;
        margin-bottom: 0.85rem;
        line-height: 1.58;
    }

    .bridge-panel {
        background: #FFDDD2;
        border: 1px solid rgba(226, 149, 120, 0.35);
        border-radius: 20px;
        padding: 1.38rem 1.55rem;
        margin-bottom: 1.05rem;
    }

    .bridge-panel h2 {
        color: #006D77;
        line-height: 1.12;
        margin-top: 0;
        margin-bottom: 0.55rem;
    }

    .bridge-panel p {
        line-height: 1.58;
        margin-bottom: 0;
    }

    .gallery-intro {
        max-width: 980px;
        line-height: 1.58;
        margin-bottom: 0.7rem;
    }

    .zone-panel {
        background: linear-gradient(135deg, #EDF6F9 0%, #FFFFFF 100%);
        border: 1px solid rgba(0, 109, 119, 0.16);
        border-radius: 20px;
        padding: 1.35rem 1.45rem;
        margin-bottom: 0.75rem;
    }

    .zone-number {
        color: #E29578;
        font-size: 3.0rem;
        font-weight: 850;
        line-height: 1;
        letter-spacing: -0.05em;
        margin-bottom: 0.35rem;
    }

    .zone-label,
    .showcase-kicker {
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.10em;
        text-transform: uppercase;
    }

    .zone-label {
        color: #006D77;
        margin-bottom: 0.85rem;
    }

    .showcase-kicker {
        color: #E29578;
        margin-bottom: 0.30rem;
    }

    .zone-panel h3,
    .showcase-copy h3 {
        color: #006D77;
        line-height: 1.16;
        margin-top: 0;
        margin-bottom: 0.50rem;
    }

    .zone-panel p,
    .showcase-copy p {
        line-height: 1.56;
        margin-bottom: 0;
    }

    .showcase-copy {
        border: 1px solid rgba(0, 109, 119, 0.16);
        border-radius: 18px;
        padding: 1.05rem 1.12rem;
        background: white;
        min-height: 185px;
        margin: 0.40rem 0 0.65rem 0;
    }

    /* Fixed visual stage: differently shaped source images now occupy the
       same vertical footprint instead of determining the card height. */
    .showcase-stage {
        height: 430px;
        border: 1px solid rgba(0, 109, 119, 0.14);
        border-radius: 18px;
        background: white;
        padding: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        margin: 0.35rem 0 0.65rem 0;
    }

    .showcase-stage img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
        border-radius: 10px;
    }

    .showcase-tab-note {
        color: rgba(31, 41, 55, 0.76);
        font-size: 0.92rem;
        line-height: 1.50;
        margin: 0.15rem 0 0.55rem 0;
    }

    .go-deeper {
        background: #EDF6F9;
        border-radius: 18px;
        padding: 1.15rem 1.25rem;
        min-height: 165px;
    }

    .go-deeper h3 {
        color: #006D77;
        margin-top: 0;
        margin-bottom: 0.45rem;
    }

    .go-deeper p {
        line-height: 1.52;
        margin-bottom: 0;
    }

    .evidence-boundary {
        background: #EDF6F9;
        border-top: 3px solid #006D77;
        border-radius: 14px;
        padding: 1.0rem 1.15rem;
        margin-top: 1.20rem;
        line-height: 1.56;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid rgba(0, 109, 119, 0.14);
        border-radius: 14px;
        padding: 0.72rem 0.82rem;
    }

    div.stButton > button {
        border-radius: 999px;
        border: 1px solid #83C5BE;
        min-height: 2.42rem;
    }

    div.stButton > button:hover {
        border-color: #006D77;
    }

    /* Give images a consistent editorial edge without boxing them heavily. */
    div[data-testid="stImage"] img {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _image_path(filename: str) -> Path:
    """Return the local path for one homepage image asset."""
    return IMAGE_DIR / filename


def _render_image(
    filename: str,
    *,
    caption: str | None = None,
) -> None:
    """Render one homepage image and fail softly if the asset is absent."""
    path = _image_path(filename)

    if path.exists():
        st.image(
            str(path),
            use_container_width=True,
        )
    else:
        st.warning(
            f"Homepage image is missing: `app/images/{filename}`"
        )

    if caption:
        st.caption(caption)


def _render_showcase_stage(
    filename: str,
    *,
    alt: str,
) -> None:
    """
    Render a static homepage asset inside a fixed-height visual stage.

    WHY: the curated Showcase PNGs have intentionally different aspect ratios.
    A fixed container makes neighboring cards align without cropping the chart.
    """
    path = _image_path(filename)

    if not path.exists():
        st.warning(
            f"Homepage image is missing: `app/images/{filename}`"
        )
        return

    import base64

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("ascii")

    suffix = path.suffix.lower()
    mime = (
        "image/png"
        if suffix == ".png"
        else "image/jpeg"
    )

    st.markdown(
        f"""
        <div class="showcase-stage">
            <img
                src="data:{mime};base64,{encoded}"
                alt="{alt}"
            />
        </div>
        """,
        unsafe_allow_html=True,
    )


def _nav_button(
    label: str,
    target: str,
    *,
    key: str,
) -> None:
    """
    Navigate with st.switch_page instead of st.page_link.

    WHY: page_link can fail against st.navigation page metadata in some
    Streamlit versions. switch_page resolves the registered page only after
    the visitor clicks, avoiding that url_pathname failure path.
    """
    if st.button(
        label,
        key=key,
        use_container_width=True,
    ):
        st.switch_page(target)


def _render_path_card(
    *,
    number: str,
    eyebrow: str,
    title: str,
    body: str,
    links: list[tuple[str, str, str]],
) -> None:
    """Render one Start Here card with two curated destinations."""
    st.markdown(
        f"""
        <div class="path-card">
            <div class="path-number">{number} · {eyebrow}</div>
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for label, target, key in links:
        _nav_button(
            label,
            target,
            key=key,
        )


def _render_finding_copy(
    *,
    number: str,
    title: str,
    body: str,
) -> None:
    """Render the editorial copy for one Key Finding."""
    st.markdown(
        f"""
        <div class="finding-copy">
            <div class="finding-number">Key Finding {number}</div>
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_pulse_video() -> bool:
    """Render the first packaged Mobility Pulse MP4 when one is available."""
    if not PULSE_MANIFEST.exists():
        return False

    try:
        manifest = json.loads(
            PULSE_MANIFEST.read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError):
        return False

    if isinstance(manifest, dict):
        presets = (
            manifest.get("presets")
            or manifest.get("stories")
            or manifest.get("items")
            or []
        )
    elif isinstance(manifest, list):
        presets = manifest
    else:
        presets = []

    for preset in presets:
        if not isinstance(preset, dict):
            continue

        filename = preset.get("filename")
        if not filename:
            continue

        video_path = PULSE_DIR / str(filename)

        if video_path.exists():
            st.video(
                str(video_path),
                format="video/mp4",
            )
            return True

    return False


# ---------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------

st.markdown(
    """
    <div class="home-hero">
        <div class="home-eyebrow">NYC Congestion Pricing Mobility Showcase</div>
        <h1>How did New York City mobility change after congestion pricing?</h1>
        <div class="home-hero-copy">
            Follow NYC mobility from <strong>January 2023 through March 2026</strong>,
            spanning the January 5, 2025 congestion-pricing launch. Explore what changed
            across neighborhoods, times of day, and transportation modes — then compare
            what actually happened with a model-estimated world in which congestion
            pricing had not begun.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_metric_1, hero_metric_2, hero_metric_3, hero_metric_4 = st.columns(4)

hero_metric_1.metric(
    "Study window",
    "Jan 2023–Mar 2026",
)
hero_metric_2.metric(
    "Taxi Zones",
    "263",
)
hero_metric_3.metric(
    "Unified mobility panel",
    "1.56M rows",
)
hero_metric_4.metric(
    "Congestion pricing begins",
    "Jan 5, 2025",
)

st.caption(
    "Observed Pre/Post views describe mobility around the policy launch. "
    "Counterfactual views estimate an alternative no-CP path; they are not direct "
    "causal measurements."
)

st.divider()


# ---------------------------------------------------------------------
# Start here
# ---------------------------------------------------------------------

st.markdown(
    '<div class="section-kicker">Choose your route</div>',
    unsafe_allow_html=True,
)
st.header("Start here")

st.write(
    "You do not need to move through the Showcase in order. Pick the question "
    "closest to what you want to understand, then follow one of the suggested views."
)

start_1, start_2 = st.columns(2)

with start_1:
    _render_path_card(
        number="01",
        eyebrow="Observed mobility",
        title="What changed?",
        body=(
            "Start with the observed mobility record: when change appeared, where it "
            "was concentrated, and how differently neighborhoods responded."
        ),
        links=[
            (
                "Start with Temporal Overview",
                "views/raw_01_temporal_did_mobility_change.py",
                "start_temporal",
            ),
            (
                "Then explore Spatial Patterns",
                "views/raw_03_spatial_where_did_it_change.py",
                "start_spatial",
            ),
        ],
    )

with start_2:
    _render_path_card(
        number="02",
        eyebrow="Mobility structure & stress",
        title="When and where did mobility become unusual?",
        body=(
            "See how the city's mobility system moved together, when it departed "
            "from its usual patterns, and which transportation modes drove those "
            "unusual conditions."
        ),
        links=[
            (
                "Start with Mobility Drivers",
                "views/raw_27_mobility_drivers.py",
                "start_drivers",
            ),
            (
                "Then explore Stress Anomaly Patterns",
                "views/raw_14_mobility_stress_patterns.py",
                "start_stress_patterns",
            ),
        ],
    )

start_3, start_4 = st.columns(2)

with start_3:
    _render_path_card(
        number="03",
        eyebrow="Forecasting",
        title="Can we predict what happens next?",
        body=(
            "See how closely the forecasting system followed mobility it had never "
            "seen, where it struggled, and whether it improved on simply repeating "
            "the previous week's value."
        ),
        links=[
            (
                "Start with Forecast Scorecard",
                "views/raw_18_forecast_scorecard.py",
                "start_forecast_scorecard",
            ),
            (
                "Then explore Forecast Reliability",
                "views/raw_19_forecast_reliability.py",
                "start_forecast_reliability",
            ),
        ],
    )

with start_4:
    _render_path_card(
        number="04",
        eyebrow="Counterfactual",
        title="What might have happened without congestion pricing?",
        body=(
            "Compare observed post-launch mobility with a model-estimated no-CP "
            "trajectory, then test which conclusions hold up across alternative "
            "analytical choices."
        ),
        links=[
            (
                "Start with Counterfactual Overview",
                "views/raw_21_counterfactual_overview.py",
                "start_cf_overview",
            ),
            (
                "Then test Counterfactual Robustness",
                "views/raw_24_counterfactual_robustness.py",
                "start_cf_robustness",
            ),
        ],
    )

st.divider()


# ---------------------------------------------------------------------
# Key findings
# ---------------------------------------------------------------------

st.markdown(
    '<div class="section-kicker">What the analysis found</div>',
    unsafe_allow_html=True,
)
st.header("Four findings worth starting with")

st.write(
    "These are the strongest substantive results from across the Showcase — not a "
    "summary of every page, but four places where the citywide story becomes clear."
)

# Finding 01 — image left, copy right
kf1_image, kf1_copy = st.columns(
    [1.18, 0.82],
    vertical_alignment="center",
)

with kf1_image:
    _render_image(
        "kf_citywide_demand_shift.png"
    )

with kf1_copy:
    _render_finding_copy(
        number="01",
        title="Citywide demand shifted upward around January 2025",
        body=(
            "Before congestion pricing began, Taxi, FHVHV, and Subway demand spent "
            "long stretches below their usual same-weekday and same-daypart levels. "
            "Afterward, all three spent much more time above those typical levels, "
            "often together. The timing is striking, but this descriptive view alone "
            "does not establish causation."
        ),
    )
    _nav_button(
        "Explore Mobility Drivers",
        "views/raw_27_mobility_drivers.py",
        key="kf1_link",
    )

# Finding 02 — copy left, image right
kf2_copy, kf2_image = st.columns(
    [0.82, 1.18],
    vertical_alignment="center",
)

with kf2_copy:
    _render_finding_copy(
        number="02",
        title="Taxi growth drives most Taxi–FHVHV neighborhood divergence",
        body=(
            "Among the 54 Taxi Zones where Taxi and FHVHV demand moved in opposite "
            "directions, 51 paired rising Taxi activity with declining FHVHV activity, "
            "and 49 of those 54 zones were in Manhattan. The largest divergences were "
            "therefore overwhelmingly Taxi-growth stories."
        ),
    )
    _nav_button(
        "Explore Mode Divergences",
        "views/raw_06_mode_divergences_disagree.py",
        key="kf2_link",
    )

with kf2_image:
    _render_image(
        "kf_taxi_growth_fhvhv_divergence.png"
    )

# Finding 03 — image left, copy right
kf3_image, kf3_copy = st.columns(
    [1.18, 0.82],
    vertical_alignment="center",
)

with kf3_image:
    _render_image(
        "kf_stress_anomaly_geography_composite.png"
    )

with kf3_copy:
    _render_finding_copy(
        number="03",
        title="Mobility stress reorganized geographically",
        body=(
            "After a sharp demand-stress surge around the January 2025 transition, "
            "stress incidence settled lower in the CBD and gateway areas but higher "
            "outside them. Outside those core areas, demand-related stress remained "
            "substantial even as congestion-only stress receded."
        ),
    )
    _nav_button(
        "Explore Stress Geography",
        "views/raw_13_stress_anomaly_spatial_explorer.py",
        key="kf3_link",
    )

# Finding 04 — copy left, image right
kf4_copy, kf4_image = st.columns(
    [0.82, 1.18],
    vertical_alignment="center",
)

with kf4_copy:
    _render_finding_copy(
        number="04",
        title="The modeled no-CP difference varies sharply by borough and mode",
        body=(
            "The counterfactual does not tell one uniform citywide story. Manhattan "
            "shows the strongest Taxi-trip separation, while Brooklyn, Queens, and "
            "the Bronx are more strongly distinguished by FHVHV-trip differences. "
            "These are model-estimated comparisons between observed post-launch "
            "mobility and an estimated no-CP path."
        ),
    )
    _nav_button(
        "Explore Multimodal Counterfactuals",
        "views/raw_23_multimodal_counterfactuals.py",
        key="kf4_link",
    )

with kf4_image:
    _render_image(
        "kf_counterfactual_borough_contrast.png"
    )

st.divider()


# ---------------------------------------------------------------------
# Taxi-Zone depth
# ---------------------------------------------------------------------

st.markdown(
    '<div class="section-kicker">Go local</div>',
    unsafe_allow_html=True,
)
st.header("Explore NYC Taxi Zone by Taxi Zone")

st.write(
    "The citywide story is only the starting point. Drill into individual Taxi Zones "
    "to see how a place behaves across transportation modes, unusual mobility periods, "
    "forecast reliability, and the estimated no-congestion-pricing path."
)

zone_image, zone_copy = st.columns(
    [1.55, 0.75],
    vertical_alignment="center",
)

with zone_image:
    _render_image(
        "showcase_taxi_zone_accuracy_quilt.png",
        caption=(
            "Example: forecast accuracy inside one Taxi Zone across mobility measures "
            "and forecast horizons."
        ),
    )

with zone_copy:
    st.markdown(
        """
        <div class="zone-panel">
            <div class="zone-number">263</div>
            <div class="zone-label">Taxi Zones to explore</div>
            <h3>One place. Many analytical lenses.</h3>
            <p>
                The Accuracy Quilt shows how forecast reliability can change inside
                one Taxi Zone depending on the mobility measure and how far ahead the
                system is asked to predict. The same geography can also be explored
                through observed change, stress, relationships, and the no-CP
                counterfactual.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    zone_tab_1, zone_tab_2, zone_tab_3 = st.tabs(
        [
            "Profile & compare",
            "Stress & forecast",
            "No-CP",
        ]
    )

    with zone_tab_1:
        _nav_button(
            "Profile a Taxi Zone",
            "views/raw_07_zone_profile_what_happened_here.py",
            key="zone_profile",
        )
        _nav_button(
            "Compare Zone Rankings",
            "views/raw_04_zone_rankings_changed_most.py",
            key="zone_rankings",
        )

    with zone_tab_2:
        _nav_button(
            "Explore Stress Geography",
            "views/raw_13_stress_anomaly_spatial_explorer.py",
            key="zone_stress",
        )
        _nav_button(
            "Explore Forecast Reliability",
            "views/raw_19_forecast_reliability.py",
            key="zone_forecast",
        )

    with zone_tab_3:
        _nav_button(
            "Explore Counterfactual Geography",
            "views/raw_22_counterfactual_geography.py",
            key="zone_counterfactual",
        )


st.divider()


# ---------------------------------------------------------------------
# Analytical validation
# ---------------------------------------------------------------------

st.markdown(
    '<div class="section-kicker">How much should you trust the machinery?</div>',
    unsafe_allow_html=True,
)
st.header("How we tested the analytical system")

st.markdown(
    """
    <div class="method-intro">
        The Showcase moves from describing observed mobility to detecting unusual
        conditions, forecasting mobility the models have not seen, and finally
        estimating an unobserved no-CP path. Each step introduces a different kind
        of uncertainty, so each is tested in a different way.
    </div>
    """,
    unsafe_allow_html=True,
)

anomaly_tab, forecast_tab, counterfactual_tab = st.tabs(
    [
        "Anomaly detection",
        "Forecasting",
        "Counterfactual",
    ]
)

with anomaly_tab:
    st.subheader(
        "Can we tell what kind of unusual mobility event occurred?"
    )
    st.write(
        "The anomaly system does not treat every unusual observation as the same "
        "event. It separates demand-related and congestion-related stress, then "
        "identifies which transportation modes were involved together. Here, "
        "congestion-involved anomalies became less common after January 2025 while "
        "the remaining events became more concentrated in Bus-only stress."
    )
    _render_image(
        "method_anomaly_congestion_retreat.png"
    )
    _nav_button(
        "Open Stress Anomaly Patterns",
        "views/raw_14_mobility_stress_patterns.py",
        key="method_anomaly_link",
    )

with forecast_tab:
    st.subheader(
        "Can the forecasting system track mobility it has never seen?"
    )
    st.write(
        "Forecasts were evaluated against an untouched Jan–Mar 2026 holdout rather "
        "than the observations used to choose the forecasting system. In the "
        "Manhattan Subway example below, all three forecast horizons stay close to "
        "what actually happened while the Last-week baseline provides a simple "
        "reference to beat."
    )
    _render_image(
        "method_forecast_validation.png"
    )
    _nav_button(
        "Open Forecast Scorecard",
        "views/raw_18_forecast_scorecard.py",
        key="method_forecast_link",
    )

with counterfactual_tab:
    st.subheader(
        "Do the no-CP conclusions survive different analytical views?"
    )
    st.write(
        "The counterfactual is challenged across population, weighting, time, "
        "geography, forecast horizon, and calibration. Most demand and ridership "
        "results remain directionally consistent across those checks, while Taxi "
        "average speed is noticeably more sensitive to the analytical lens."
    )
    _render_image(
        "method_counterfactual_robustness.png"
    )
    _nav_button(
        "Open Counterfactual Robustness",
        "views/raw_24_counterfactual_robustness.py",
        key="method_cf_link",
    )

st.divider()


# ---------------------------------------------------------------------
# Counterfactual bridge
# ---------------------------------------------------------------------

st.markdown(
    '<div class="section-kicker">From prediction to alternative history</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="bridge-panel">
        <h2>From predicting the city to imagining the city without pricing</h2>
        <p>
            Once the forecasting system has been tested against mobility that actually
            occurred, it can be used for the harder question: <strong>what might mobility
            have looked like if congestion pricing had not begun?</strong> The chart below
            makes that transition concrete by showing observed mobility alongside the
            model-estimated no-CP trajectories rather than reducing the counterfactual to
            a single summary number.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

_render_image(
    "kf_counterfactual_fhvhv_speed_boroughs.png",
    caption=(
        "FHVHV average speed by borough. The post-launch paths compare observed "
        "mobility with the model-estimated no-congestion-pricing trajectory."
    ),
)

bridge_left, bridge_right = st.columns(2)

with bridge_left:
    _nav_button(
        "Open Counterfactual Overview",
        "views/raw_21_counterfactual_overview.py",
        key="bridge_overview",
    )

with bridge_right:
    _nav_button(
        "Explore Counterfactual Geography",
        "views/raw_22_counterfactual_geography.py",
        key="bridge_geography",
    )

st.divider()


# ---------------------------------------------------------------------
# Explore the Showcase
# ---------------------------------------------------------------------

st.markdown(
    '<div class="section-kicker">Explore the Showcase</div>',
    unsafe_allow_html=True,
)
st.header("See the mobility system from a different angle")

st.markdown(
    """
    <div class="gallery-intro">
        The same mobility record can reveal very different structure depending on
        how you look at it. Start with two featured views, then open the gallery
        tabs for relationships, forecast evaluation, counterfactual robustness,
        animated geography, seasonal cycles, and recurring mobility day types.
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Two featured views — equal visual stages and equal copy boxes
# ------------------------------------------------------------------

featured_left, featured_right = st.columns(2)

with featured_left:
    _render_showcase_stage(
        "raw_14_mobility_stress_patterns.png",
        alt="Multimodal stress timeline",
    )

    st.markdown(
        """
        <div class="showcase-copy">
            <div class="showcase-kicker">Stress & anomalies</div>
            <h3>When did mobility modes become unusual together?</h3>
            <p>
                The multimodal stress timeline turns unusual periods into shared
                events, revealing when Taxi, FHVHV, Subway, and Bus stress overlapped
                — and when one mode broke away.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _nav_button(
        "Explore Stress Anomaly Patterns",
        "views/raw_14_mobility_stress_patterns.py",
        key="showcase_stress",
    )


with featured_right:
    _render_showcase_stage(
        "showcase_forecast_reliability_bivariate.png",
        alt="Forecast reliability bivariate Taxi Zone map",
    )

    st.markdown(
        """
        <div class="showcase-copy">
            <div class="showcase-kicker">Forecasting</div>
            <h3>Where is mobility hard to predict — and where does the model help?</h3>
            <p>
                Forecast error and improvement over the Last-week baseline answer
                different questions. This map puts both on the same geography,
                including places where mobility is difficult to predict but the
                forecast still adds value.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _nav_button(
        "Explore Forecast Reliability",
        "views/raw_19_forecast_reliability.py",
        key="showcase_reliability",
    )


# ------------------------------------------------------------------
# Tabbed gallery — all other visual experiences remain available
# without extending the homepage into another long gallery.
# ------------------------------------------------------------------

st.subheader("More ways to explore")

st.markdown(
    """
    <div class="showcase-tab-note">
        Pick a visual lens. Each tab previews one analytical experience and links
        directly to the interactive page behind it.
    </div>
    """,
    unsafe_allow_html=True,
)

(
    relationships_tab,
    win_miss_tab,
    robustness_tab,
    pulse_tab,
    spiral_tab,
    day_types_tab,
) = st.tabs(
    [
        "Mode Relationships",
        "Win–Miss Plane",
        "Robustness",
        "Mobility Pulse",
        "Annual Spiral",
        "Day Types",
    ]
)


with relationships_tab:
    tab_image, tab_copy = st.columns(
        [1.35, 0.65],
        vertical_alignment="center",
    )

    with tab_image:
        _render_showcase_stage(
            "showcase_mode_relationships_taxi_fhvhv.png",
            alt="Taxi and FHVHV relationship chart",
        )

    with tab_copy:
        st.markdown(
            """
            <div class="showcase-copy">
                <div class="showcase-kicker">Observed mobility</div>
                <h3>Watch two modes move together — or apart</h3>
                <p>
                    Connect each Taxi Zone's Pre- and Post-CP position to see whether
                    Taxi and FHVHV demand moved together, diverged, or changed at very
                    different magnitudes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _nav_button(
            "Explore Mode Relationships",
            "views/raw_05_mode_relationships_move_together.py",
            key="showcase_relationships",
        )


with win_miss_tab:
    tab_image, tab_copy = st.columns(
        [1.35, 0.65],
        vertical_alignment="center",
    )

    with tab_image:
        _render_showcase_stage(
            "showcase_forecast_win_miss_plane.png",
            alt="Forecast Win-Miss Plane",
        )

    with tab_copy:
        st.markdown(
            """
            <div class="showcase-copy">
                <div class="showcase-kicker">Forecast evaluation</div>
                <h3>Separate forecast difficulty from forecast usefulness</h3>
                <p>
                    The Win–Miss Plane shows why a forecast can have relatively high
                    error and still improve substantially on the Last-week baseline.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _nav_button(
            "Explore Forecast Wins & Misses",
            "views/raw_20_forecast_wins_misses.py",
            key="showcase_win_miss",
        )


with robustness_tab:
    tab_image, tab_copy = st.columns(
        [1.35, 0.65],
        vertical_alignment="center",
    )

    with tab_image:
        _render_showcase_stage(
            "showcase_counterfactual_robustness_profiles.png",
            alt="Counterfactual robustness radar profiles",
        )

    with tab_copy:
        st.markdown(
            """
            <div class="showcase-copy">
                <div class="showcase-kicker">Counterfactual</div>
                <h3>Robustness has a shape, not a single score</h3>
                <p>
                    A result can hold up strongly under one analytical choice and be
                    more sensitive under another. The profile exposes that structure
                    rather than hiding it behind one confidence-like number.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _nav_button(
            "Explore Counterfactual Robustness",
            "views/raw_24_counterfactual_robustness.py",
            key="showcase_robustness",
        )


with pulse_tab:
    pulse_visual, pulse_copy = st.columns(
        [1.35, 0.65],
        vertical_alignment="center",
    )

    with pulse_visual:
        if not _render_pulse_video():
            st.info(
                "The packaged Mobility Pulse video was not found from the local "
                "preset manifest. The interactive page can still be opened directly."
            )

    with pulse_copy:
        st.markdown(
            """
            <div class="showcase-copy">
                <div class="showcase-kicker">Animated geography</div>
                <h3>Watch mobility move across the city</h3>
                <p>
                    Mobility Pulse animates Taxi Zones through time, showing how each
                    zone moves away from its own pre-pricing reference while
                    metric-linked stress events appear across the map.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _nav_button(
            "Explore Mobility Pulse",
            "views/raw_10_mobility_pulse.py",
            key="showcase_pulse",
        )


with spiral_tab:
    tab_image, tab_copy = st.columns(
        [1.35, 0.65],
        vertical_alignment="center",
    )

    with tab_image:
        _render_showcase_stage(
            "vp_annual_mobility_spiral.png",
            alt="Annual mobility spiral",
        )

    with tab_copy:
        st.markdown(
            """
            <div class="showcase-copy">
                <div class="showcase-kicker">Cyclical time</div>
                <h3>See the year as a cycle instead of a straight line</h3>
                <p>
                    The annual spiral keeps seasonal position visible, making
                    recurring periods and year-over-year departures easier to compare.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _nav_button(
            "Explore Cyclical Time",
            "views/raw_11_cyclical_time.py",
            key="showcase_spiral",
        )


with day_types_tab:
    tab_image, tab_copy = st.columns(
        [1.35, 0.65],
        vertical_alignment="center",
    )

    with tab_image:
        _render_showcase_stage(
            "vp_mobility_day_types.png",
            alt="Mobility Day Types calendar",
        )

    with tab_copy:
        st.markdown(
            """
            <div class="showcase-copy">
                <div class="showcase-kicker">Mobility environments</div>
                <h3>See what kind of mobility day NYC was experiencing</h3>
                <p>
                    The day-type calendar compresses multiple mobility measures and
                    dayparts into recurring daily states, then shows how those states
                    cluster and persist through time.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _nav_button(
            "Explore Mobility Day Types",
            "views/raw_26_mobility_day_types.py",
            key="showcase_day_types",
        )


st.divider()


# ---------------------------------------------------------------------
# Go deeper
# ---------------------------------------------------------------------

st.markdown(
    '<div class="section-kicker">Behind the Showcase</div>',
    unsafe_allow_html=True,
)
st.header("Go deeper")

deep_1, deep_2, deep_3 = st.columns(3)

with deep_1:
    st.markdown(
        """
        <div class="go-deeper">
            <h3>Source pipeline</h3>
            <p>
                See how the analytical stages connect and browse the infographics
                created throughout the project.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _nav_button(
        "About the Source Pipeline",
        "views/about_source_pipeline.py",
        key="deep_pipeline",
    )

with deep_2:
    st.markdown(
        """
        <div class="go-deeper">
            <h3>Analytical repository</h3>
            <p>
                Open the canonical notebook-based Mobility v2 project that produced
                the forecasting and counterfactual outputs used here.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.link_button(
        "Open Mobility v2 on GitHub ↗",
        SOURCE_REPO_URL,
        use_container_width=True,
    )

with deep_3:
    st.markdown(
        """
        <div class="go-deeper">
            <h3>Showcase repository</h3>
            <p>
                Open the Streamlit application, packaged data, and presentation assets
                that power this interactive Showcase.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.link_button(
        "Open Showcase on GitHub ↗",
        SHOWCASE_REPO_URL,
        use_container_width=True,
    )

st.markdown(
    """
    <div class="evidence-boundary">
        <strong>Evidence boundary.</strong> Pre/Post views describe observed mobility
        around the January 5, 2025 policy launch. Forecasting views evaluate prediction
        against held-out observations. No-CP counterfactual views compare observed
        post-launch mobility with model-estimated alternative trajectories and should
        not be read as direct causal measurements.
    </div>
    """,
    unsafe_allow_html=True,
)

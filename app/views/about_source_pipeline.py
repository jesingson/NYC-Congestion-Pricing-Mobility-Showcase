from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

import streamlit as st

from app.utils.project_branding import inject_app_css


REPO_URL = "https://github.com/jesingson/NYC-Congestion-Pricing-Mobility-v2"
INFOGRAPHICS_URL = f"{REPO_URL}/tree/main/infographics"
RAW_BASE_URL = (
    "https://raw.githubusercontent.com/jesingson/"
    "NYC-Congestion-Pricing-Mobility-v2/main/infographics"
)


@dataclass(frozen=True)
class Infographic:
    """Describe one infographic stored in the analytical repository."""

    filename: str
    title: str
    description: str
    chapter: str

    @property
    def image_url(self) -> str:
        """Return the raw GitHub URL used to render the image."""
        return f"{RAW_BASE_URL}/{quote(self.filename)}"

    @property
    def github_url(self) -> str:
        """Return the GitHub file page for the original asset."""
        return f"{REPO_URL}/blob/main/infographics/{quote(self.filename)}"


INFOGRAPHICS = [
    Infographic(
        "NYC_Congestion_Pricing_Mobility_v2_Pipeline.png",
        "NYC Congestion Pricing Mobility v2 Pipeline",
        "The full analytical journey from the multimodal mobility foundation through anomaly detection, forecasting, counterfactual estimation, and validation.",
        "Project overview",
    ),
    Infographic(
        "3.2.1 Time Series Clustering.png",
        "Time-Series Clustering",
        "How recurring mobility behavior can be represented and grouped to reveal transportation environments that are not defined by administrative boundaries alone.",
        "Mobility structure & anomalies",
    ),
    Infographic(
        "3.2.2 Clustering Visual Verification via PCA and UMAP.png",
        "Clustering Verification with PCA and UMAP",
        "A visual check of whether the selected mobility groups remain recognizable when high-dimensional mobility profiles are projected into lower-dimensional views.",
        "Mobility structure & anomalies",
    ),
    Infographic(
        "3.2.2 Evaluating Clustering.png",
        "Evaluating Clustering",
        "How candidate cluster solutions are compared before selecting the retained mobility-environment structure.",
        "Mobility structure & anomalies",
    ),
    Infographic(
        "3.2.2 T-SNE.png",
        "t-SNE",
        "An intuitive guide to another nonlinear projection used to inspect structure in high-dimensional mobility profiles.",
        "Mobility structure & anomalies",
    ),
    Infographic(
        "3.3.1. Residualized Time Series.png",
        "Residualized Time Series",
        "Why unusual mobility is measured relative to the behavior normally expected for a particular place and time rather than from raw values alone.",
        "Mobility structure & anomalies",
    ),
    Infographic(
        "3.3.2 DBScan for Anomaly Detection.png",
        "DBSCAN for Anomaly Detection",
        "How density-based anomaly detection identifies observations that sit outside locally dense patterns of mobility behavior.",
        "Mobility structure & anomalies",
    ),
    Infographic(
        "3.3.3 Isolation Forest Anomaly Detection.png",
        "Isolation Forest Anomaly Detection",
        "How tree-based isolation provides a structurally different view of unusual mobility observations.",
        "Mobility structure & anomalies",
    ),
    Infographic(
        "3.3.4 GMM Anomaly Detection.png",
        "Gaussian Mixture Model Anomaly Detection",
        "How probabilistic mixture modeling provides another candidate surface for identifying unusual mobility conditions.",
        "Mobility structure & anomalies",
    ),
    Infographic(
        "3.5.5 Outlier Ensemble Evaluation.png",
        "Outlier Ensemble Evaluation",
        "How multiple anomaly-detection surfaces are compared before translating the retained signal into interpretable mobility-stress events.",
        "Mobility structure & anomalies",
    ),
    Infographic(
        "4.2.1 Temporal Decomposition.png",
        "Temporal Decomposition",
        "How recurring temporal structure is separated from the changing mobility signal before forecasting models are compared.",
        "Forecasting",
    ),
    Infographic(
        "4.3.1 About ARIMA SARIMAX and VARIMA.png",
        "ARIMA, SARIMAX, and VARIMA",
        "A visual explanation of the classical time-series families evaluated in the forecasting system.",
        "Forecasting",
    ),
    Infographic(
        "4.3.1 Classic Prob Models -- Monthly Refits.png",
        "Classical Models with Monthly Refits",
        "How the classical forecasting models are refreshed as time advances rather than being trained once and treated as permanently fixed.",
        "Forecasting",
    ),
    Infographic(
        "4.3.1 Rolling validation and refreshes -- ARIMA SARIMAX and VARIMA.png",
        "Rolling Validation for Classical Models",
        "How time-aware validation and model refreshes preserve the direction of time while testing classical forecasts.",
        "Forecasting",
    ),
    Infographic(
        "4.4.1 Extra Trees.png",
        "Extra Trees",
        "How an Extra Trees ensemble turns many randomized decision trees into a global panel forecasting model.",
        "Forecasting",
    ),
    Infographic(
        "4.4.1 Gain and SHAP.png",
        "Gain and SHAP",
        "Two complementary ways of asking which predictors a tree-based forecasting system relies on.",
        "Forecasting",
    ),
    Infographic(
        "4.4.1 Gain vs SHAP.png",
        "Gain vs. SHAP",
        "Why model-internal split importance and prediction-level attribution answer related but different feature-importance questions.",
        "Forecasting",
    ),
    Infographic(
        "4.4.1 Global vs Local Tree Ensemble Forecasting.png",
        "Global vs. Local Tree-Ensemble Forecasting",
        "Why a single model that learns across Taxi Zones can differ from fitting a separate forecasting model for every place.",
        "Forecasting",
    ),
    Infographic(
        "4.4.1 LightGBT.png",
        "LightGBM",
        "A visual explanation of gradient-boosted trees as one of the panel-forecasting families evaluated in the project.",
        "Forecasting",
    ),
    Infographic(
        "4.4.1 Rolling validation and refreshes -- Ensembles.png",
        "Rolling Validation for Tree Ensembles",
        "How ensemble forecasts are evaluated through time while respecting the information that would actually have been available at each prediction point.",
        "Forecasting",
    ),
    Infographic(
        "4.5.1 Feature importance in a neural network.png",
        "Feature Importance in a Neural Network",
        "How the project investigates what a neural forecasting model learned rather than treating it only as a prediction engine.",
        "Forecasting",
    ),
    Infographic(
        "4.5.1 Ingesting Temporal Structures into NNs.png",
        "Feeding Temporal Structure into Neural Networks",
        "How recent mobility history is organized so neural sequence models can learn recurring temporal behavior.",
        "Forecasting",
    ),
    Infographic(
        "4.5.1 Lag-window MLP GRU and TCN Overview.png",
        "Lag-Window MLP, GRU, and TCN",
        "A side-by-side view of the neural sequence architectures compared during model selection.",
        "Forecasting",
    ),
    Infographic(
        "4.5.2 Forecasting with Transformers.png",
        "Forecasting with Transformers",
        "How Transformer-style sequence modeling is adapted from token sequences to multivariate mobility history.",
        "Forecasting",
    ),
    Infographic(
        "4.5.2 What Changes When We Change a Transformer.png",
        "What Changes When We Change a Transformer?",
        "A guide to how architectural choices alter the information flow and capacity of the time-series Transformer.",
        "Forecasting",
    ),
    Infographic(
        "4.6.1 Complete system scoring.png",
        "Complete Forecasting-System Scoring",
        "How candidate forecasting families are brought onto a common evaluation surface before the final model is selected for each job.",
        "Forecasting",
    ),
    Infographic(
        "4.6.1 Paired block bootstrap.png",
        "Paired Block Bootstrap",
        "How forecast comparisons preserve temporal dependence while asking whether observed performance differences are durable.",
        "Forecasting",
    ),
    Infographic(
        "4.6.2 Failure Analysis.png",
        "Forecast Failure Analysis",
        "How the project moves beyond average error to investigate the places and mobility conditions where forecasts struggle.",
        "Forecasting",
    ),
    Infographic(
        "4.6.2 Feature Ablation.png",
        "Feature Ablation",
        "How removing information from the forecasting system helps reveal which feature groups contribute useful predictive signal.",
        "Forecasting",
    ),
    Infographic(
        "5.1.1 Counterfactual Forecasting.png",
        "Counterfactual Forecasting",
        "The core idea behind using the forecasting system to estimate an alternative post-pricing mobility trajectory.",
        "Counterfactual analysis",
    ),
    Infographic(
        "5.1.1 State Space Models.png",
        "State-Space Models",
        "A visual explanation of latent-state forecasting concepts considered while designing the counterfactual system.",
        "Counterfactual analysis",
    ),
    Infographic(
        "5.1.1 Synthetic World Counterfactual Forecasting .png",
        "Synthetic-World Counterfactual Forecasting",
        "How the no-congestion-pricing trajectory is constructed without recursively contaminating the synthetic world with observed post-pricing outcomes.",
        "Counterfactual analysis",
    ),
    Infographic(
        "5.2.1. Feature Reliance Across Forecasting Environments.png",
        "Feature Reliance Across Forecasting Environments",
        "How reliance on recent history and contextual information can change across forecasting environments.",
        "Counterfactual analysis",
    ),
    Infographic(
        "5.3.1 Robustness Analysis.png",
        "Robustness Analysis",
        "How the final counterfactual conclusions are challenged across multiple analytical dimensions before being labeled Stable, Mixed, or Sensitive.",
        "Counterfactual analysis",
    ),
]

CHAPTER_ORDER = [
    "Mobility structure & anomalies",
    "Forecasting",
    "Counterfactual analysis",
]


def render_infographic(item: Infographic) -> None:
    """Render one selected infographic and links to its canonical source."""
    st.markdown(f"### {item.title}")
    st.write(item.description)
    st.image(item.image_url, use_container_width=True)
    st.caption(
        "The image is loaded from the canonical Mobility v2 analytical repository. "
        "If GitHub is temporarily unavailable, the rest of the Showcase will still work."
    )
    st.link_button("Open original infographic on GitHub ↗", item.github_url)


def main() -> None:
    """Render the Project Infographics gallery."""
    st.set_page_config(
        page_title="About the Source Pipeline",
        page_icon="🔗",
        layout="wide",
    )
    inject_app_css()

    st.title("About the Source Pipeline for This Showcase")
    st.write(
        """
        This Showcase is the interactive presentation layer of the larger **NYC
        Congestion Pricing Mobility v2** analytical project. The source pipeline brings
        together NYC transportation, weather, temporal, and spatial data; identifies
        recurring mobility patterns and unusual conditions; builds and evaluates
        forecasting models; and uses that forecasting system to estimate and validate
        a no-congestion-pricing counterfactual.

        This page shows how that analytical pipeline feeds the Showcase and collects
        the infographics created throughout the project to explain its major methods,
        modeling decisions, and analytical stages.
        """
    )

    st.link_button("Explore the Mobility v2 source repository ↗", REPO_URL)

    st.divider()

    pipeline = INFOGRAPHICS[0]
    st.markdown("## From data to counterfactual")
    st.write(
        "The source project moves from a common multimodal mobility foundation into "
        "mobility structure and anomaly detection, forecasting, and finally the "
        "model-estimated no-CP counterfactual that powers the final Showcase chapter."
    )
    st.image(pipeline.image_url, use_container_width=True)
    st.link_button("Open pipeline infographic on GitHub ↗", pipeline.github_url)

    st.markdown("### The analytical pipeline")
    st.markdown(
        """
        **Build the mobility foundation**  
        Align Taxi, FHVHV, Subway, Bus, weather, and spatial context around a shared
        Taxi Zone × Date × Temporal Bucket structure.

        **Understand mobility and detect unusual conditions**  
        Learn recurring mobility environments, then identify departures from what is
        normally expected for a particular place and time.

        **Forecast expected mobility**  
        Compare classical time-series, tree-based, neural, and Transformer approaches
        across multiple mobility measures and forecast horizons.

        **Estimate and validate the no-CP counterfactual**  
        Use the forecasting system to construct an alternative post-2025 mobility
        trajectory, compare it with observed mobility, and test how robust those
        differences are to reasonable analytical choices.
        """
    )

    st.divider()
    st.markdown("## Explore the infographics")
    st.write(
        "The infographics below were created throughout Mobility v2 to make important "
        "methods and modeling decisions easier to understand. Choose an analytical "
        "stage, then select any infographic to view it at full page width."
    )
    st.link_button("Browse all infographics on GitHub ↗", INFOGRAPHICS_URL)

    tabs = st.tabs(CHAPTER_ORDER)

    for tab, chapter in zip(tabs, CHAPTER_ORDER):
        with tab:
            chapter_items = [item for item in INFOGRAPHICS if item.chapter == chapter]
            selected_title = st.selectbox(
                "Choose an infographic",
                options=[item.title for item in chapter_items],
                key=f"infographic_{chapter.lower().replace(' ', '_').replace('&', 'and')}",
            )
            selected = next(item for item in chapter_items if item.title == selected_title)
            render_infographic(selected)

    st.divider()
    st.markdown("## Explore the source project")
    st.write(
        """
        These infographics summarize pieces of a much larger notebook-based analytical
        workflow. **Mobility v2** is the canonical analytical record; this Showcase uses
        packaged outputs from that pipeline to make selected findings and diagnostics
        interactive.
        """
    )
    source_col, infographic_col = st.columns(2)
    with source_col:
        st.link_button("Open the Mobility v2 repository ↗", REPO_URL, use_container_width=True)
    with infographic_col:
        st.link_button("Browse every infographic ↗", INFOGRAPHICS_URL, use_container_width=True)

    st.caption(
        "The infographics are loaded from the public Mobility v2 repository, while the "
        "interactive Showcase remains a separate application and repository."
    )


if __name__ == "__main__":
    main()

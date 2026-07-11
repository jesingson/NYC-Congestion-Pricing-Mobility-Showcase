"""Build compact spatial assets used by the Streamlit app.

Run from the repository root:

    python scripts/build_spatial_app_tables.py

Set REBUILD_SPATIAL_APP_TABLES=true when launching Streamlit only when you
explicitly want the app itself to rebuild a missing or stale summary.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.data_access.spatial_aggregations import (  # noqa: E402
    build_spatial_zone_pre_post_summary,
    write_spatial_app_table,
)
from app.data_access.spatial_visuals import (  # noqa: E402
    build_simplified_zone_geojson,
)


def main() -> None:
    print("Building spatial Taxi Zone summary...")
    summary = build_spatial_zone_pre_post_summary()
    summary_path = write_spatial_app_table(summary)

    print("Building simplified Taxi Zone GeoJSON...")
    geojson_path = build_simplified_zone_geojson()

    print(f"Summary rows: {len(summary):,}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {geojson_path}")


if __name__ == "__main__":
    main()
"""Build only the Mobility regime cluster comparison parquet for Raw 07.

Run from the repository root:

    python scripts/build_zone_profile_cluster_comparison_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


from app.data_access.zone_profile_app_tables import (  # noqa: E402
    build_zone_profile_cluster_comparison_table,
)


if __name__ == "__main__":
    build_zone_profile_cluster_comparison_table(
        verbose=True
    )

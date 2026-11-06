"""Step 52 - Cue-stage map (Appendix Table A5).

Input : outputs/tables/table_nb_funnel.csv (step 10)
Output: outputs/tables/table_cue_stage_map.csv

For each cue, the sign and incidence rate ratio of its association with video exposure
(model ``reach``), like conversion and save conversion, grouped by whether the cue moves
all three outcomes in the same direction, acts on conversion only, or moves exposure
against conversion.
"""
from __future__ import annotations

import pandas as pd

from common import DIMENSION, LABELS, TABLES

GROUPS = {
    "Same direction at all stages": ["hashtag_count", "video_duration"],
    "Conversion only (no association with exposure)": ["has_text_overlay", "is_vertical", "video_quality_score", "loudness"],
    "Opposite directions at exposure and conversion": ["semantic_richness", "text_length", "emoji_count", "has_poi", "is_ad"],
}
MODELS = {"reach": "Video exposure", "like_conversion": "Like conversion", "save_conversion": "Save conversion"}


def cell(t: pd.DataFrame, model: str, var: str) -> str:
    r = t[(t.model == model) & (t.variable == var)].iloc[0]
    sign = "+" if r.b > 0 else "−"
    return f"{sign} ({r.irr:.2f})" if r.p < .05 else f"n.s. ({r.irr:.2f})"


def main() -> None:
    t = pd.read_csv(TABLES / "table_nb_funnel.csv")
    rows = []
    for group, cues in GROUPS.items():
        for v in cues:
            rows.append({"group": group, "cue": LABELS[v], "cue_family": DIMENSION[v],
                         **{name: cell(t, m, v) for m, name in MODELS.items()}})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "table_cue_stage_map.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

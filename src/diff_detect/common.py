import os
from typing import Literal

CHALLENGE_NAMES = {
    "explain_dummy": "Dummy",
    "explain_butterfly_easy": "Butterfly Wings (Easy)",
    "explain_butterfly_difficult": "Butterfly Wings (Difficult)",
    "explain_flybutter_easy": "FlyButter Wings (Easy)",
    "rate_dummy": "Dummy",
    "rate_butterfly_easy": "Butterfly Wings (Easy)",
    "rate_butterfly_difficult": "Butterfly Wings (Difficult)",
    "rate_flybutter_easy": "FlyButter Wings (Easy)",
}
DifferenceLabel = Literal["wing outline", "pattern shape", "color"]
DIFFERENCE_LABEL_STYLES: dict[DifferenceLabel, dict[str, str]] = {
    "wing outline": {"color": "#ffb000"},
    "pattern shape": {"color": "#007bff"},
    "color": {"color": "#e83e8c"},
}
DIFFERENCE_LABELS = tuple(DIFFERENCE_LABEL_STYLES)
EXPLAIN_CANVAS_SCALE = float(os.getenv("EXPLAIN_CANVAS_SCALE", 0.23))
EXPLAIN_STROKE_WIDTH = 8

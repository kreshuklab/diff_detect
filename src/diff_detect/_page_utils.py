from typing import Literal

from diff_detect.common import DIFFERENCE_LABEL_STYLES, DifferenceLabel

PageKey = Literal["login", "challenge", "task", "leaderboard"]


def format_label(label: DifferenceLabel) -> str:
    return (
        f":color[{label.title()}]"
        f'{{foreground="{DIFFERENCE_LABEL_STYLES[label]["color"]}"}}'
    )

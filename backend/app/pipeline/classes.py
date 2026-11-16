"""The classes each crop head can predict.

The order of each list is part of the model contract (contracts/model_io.md):
the real model must output its scores in exactly this order.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassInfo:
    label: str
    display_name: str
    category: str  # "disease", "pest" or "healthy"


CLASSES: dict[str, list[ClassInfo]] = {
    "tomato": [
        ClassInfo("tomato_healthy", "Healthy tomato leaf", "healthy"),
        ClassInfo("tomato_early_blight", "Early blight", "disease"),
        ClassInfo("tomato_late_blight", "Late blight", "disease"),
        ClassInfo("tomato_septoria_leaf_spot", "Septoria leaf spot", "disease"),
        ClassInfo("tomato_spider_mites", "Two-spotted spider mite damage", "pest"),
    ],
    "apple": [
        ClassInfo("apple_healthy", "Healthy apple leaf", "healthy"),
        ClassInfo("apple_scab", "Apple scab", "disease"),
        ClassInfo("apple_cedar_rust", "Cedar apple rust", "disease"),
        ClassInfo("apple_black_rot", "Black rot (frogeye leaf spot)", "disease"),
    ],
    "corn": [
        ClassInfo("corn_healthy", "Healthy corn leaf", "healthy"),
        ClassInfo("corn_northern_leaf_blight", "Northern corn leaf blight", "disease"),
        ClassInfo("corn_common_rust", "Common rust", "disease"),
        ClassInfo("corn_gray_leaf_spot", "Gray leaf spot", "disease"),
    ],
}

BY_LABEL: dict[str, ClassInfo] = {c.label: c for items in CLASSES.values() for c in items}

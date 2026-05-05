"""
UDMS Category Mapping — Single source of truth for all label translations.

This module defines how each source dataset's class labels map to UDMS categories.
NEVER hardcode category mappings elsewhere — always import from here.
"""

UDMS_CATEGORIES = [
    "bad_drainage",
    "damaged_signage",
    "illegal_dumping",
    "potholes",
    "vegetation_overgrowth",
]

CATEGORY_LABELS = {
    "bad_drainage": "Bad Drainage / Water Issues",
    "damaged_signage": "Damaged Signage / Infrastructure",
    "illegal_dumping": "Illegal Dumping / Garbage",
    "potholes": "Potholes / Road Damage",
    "vegetation_overgrowth": "Vegetation Overgrowth",
}

NUM_CLASSES = 5

# Source → UDMS mapping. None means discard.
ROAD_ISSUES_HF_MAP = {
    "Littering/Garbage": "illegal_dumping",
    "Damaged Road Issues": "potholes",
    "Pothole Issues": "potholes",
    "Broken Road Sign Issues": "damaged_signage",
    "Mixed Issues": None,       # discard — no 'other' category
    "Vandalism/Graffiti": None, # discard
    "Illegal Parking": None     # discard
}

URBAN_VISUAL_POLLUTION_MAP = {
    "GARBAGE": "illegal_dumping",
    "POTHOLES": "potholes",
    "POTHOLE": "potholes",
    "CONSTRUCTION_ROAD": "potholes",
    "ROAD_CONSTRUCTION": "potholes",
    "BAD_STREETLIGHT": None,    # discard — broken_lighting removed
    "BROKEN_SIGNAGE": "damaged_signage",
    "FADED_SIGNAGE": "damaged_signage",
    "CLUTTER_SIDEWALK": None,   # discard
    "CLUTTERED_SIDEWALK": None, # discard
    "GRAFFITI": None,           # discard
    "BAD_BILLBOARD": None,
    "SAND_ON_ROAD": None,
    "SAND_ON_ROADS": None,
    "UNKEPT_FACADE": None
}

ROAD_HAZARDS_MAP = {
    "pothole": "potholes",
    "crack": "potholes",
    "cracks": "potholes",
    "open_manhole": "bad_drainage",
    "good_road": None  # negative sample, discard
}

CIVIC_ISSUES_MAP = {
    "pothole": "potholes",
    "garbage": "illegal_dumping",
    "plain_road": None,  # negative sample, discard
    "non_garbage": None  # negative sample, discard
}

FLOOD_CLASSIFICATION_MAP = {
    "flooded": "bad_drainage",
    "Flooded": "bad_drainage",
    "non_flooded": None,  # discard
    "Non Flooded": None,
    "non-flooded": None,
}

DAMAGED_SIGNS_MAP = {
    # All damaged sign images map to damaged_signage
    "damaged": "damaged_signage",
    "Damaged": "damaged_signage",
}


def get_udms_category(source_dataset: str, source_label: str) -> str | None:
    """Return the UDMS category for a given source dataset and label, or None to discard."""
    maps = {
        "road_issues_hf": ROAD_ISSUES_HF_MAP,
        "urban_visual_pollution": URBAN_VISUAL_POLLUTION_MAP,
        "road_hazards": ROAD_HAZARDS_MAP,
        "civic_issues_qr4change": CIVIC_ISSUES_MAP,
        "flood_classification": FLOOD_CLASSIFICATION_MAP,
        "damaged_signs": DAMAGED_SIGNS_MAP,
    }
    mapping = maps.get(source_dataset, {})
    return mapping.get(source_label, None)


def get_label_index(category: str) -> int:
    """Return integer index for a UDMS category string."""
    return UDMS_CATEGORIES.index(category)


def get_category_from_index(index: int) -> str:
    """Return UDMS category string from integer index."""
    return UDMS_CATEGORIES[index]

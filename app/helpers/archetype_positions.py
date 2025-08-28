from __future__ import annotations
from collections import defaultdict
from typing import Dict, List

# Curated mapping POS -> archetypes

POSITION_ARCHETYPES: Dict[str, List[str]] = {
    "PG": [
        "Pass-First Point Guard / Floor General",
        "Floor General",
        "Pick-and-Roll Ball Handler",
        "Drive-and-Kick Playmaker",
        "Volume Distributor",
        "Low-Turnover Connector",
        "Primary Creator / High-Usage Scorer",
        "Shooting-Creative Guard",
        "Isolation Scorer",
        "Transition Creator",
        "Defensive Ballhawk",
        "Point-of-Attack Stopper",
        "Two-Way Guard",
        "Microwave Bench Scorer",
    ],
    "SG": [
        "Efficient Spot-Up Shooter",
        "Movement Shooter",
        "Deep Range Specialist",
        "High-Volume 3-Point Gunner",
        "Corner Specialist",
        "Shooting-Creative Guard",
        "Isolation Scorer",
        "Transition Creator",
        "Primary Creator / High-Usage Scorer",
        "Off-Ball Movement Creator",
        "Secondary Creator / Off-Ball Facilitator",
        "Two-Way Guard",
        "Perimeter Shutdown Defender",
        "Point-of-Attack Stopper",
        "Low-Turnover Connector",
        "Volume Distributor",
        "Microwave Bench Scorer",
        "Midrange Maestro",
        "Free-Throw Magnet Scorer",
        "Effort Defensive Role Player",
        "Low-Usage Secure Role Player",
        "Non-Impact Role Player",
    ],
    "SF": [
        "Playmaking Wing",
        "Two-Way Wing",
        "Gravity/Space Creator",
        "Closeout Attacker Wing",
        "Efficient Spot-Up Shooter",
        "Deep Range Specialist",
        "Corner Specialist",
        "Off-Ball Movement Creator",
        "Secondary Creator / Off-Ball Facilitator",
        "Perimeter Shutdown Defender",
        "Switchable Forward Defender",
        "Glue Role Player / Hustle Forward",
        "Midrange Maestro",
        "Transition Finisher",
        "Microwave Bench Scorer",
        "Effort Defensive Role Player",
        "Low-Usage Secure Role Player",
        "Non-Impact Role Player",
        "Free-Throw Magnet Scorer",
    ],
    "PF": [
        "Floor-Spacer Stretch Four",
        "Switchable Forward Defender",
        "Glue Role Player / Hustle Forward",
        "Second-Chance Specialist",
        "Putback Specialist",
        "Pick-and-Pop Big",
        "Versatile Facilitator / Combo Big",
        "High-Usage Volume Big / Post Scorer",
        "Post-Up Bruiser",
        "Roll Man / Screen Finisher",
        "Small-Ball Center",
        "3-and-D Big",
        "Post Hub Playmaker",
        "Efficient Finisher / Low-Usage Big",
        "Rim Protector / Rebounding Big",
        "Effort Defensive Role Player",
        "Low-Usage Secure Role Player",
        "Offensive Liability / Role Defensive Specialist",
        "Non-Impact Role Player",
    ],
    "C": [
        "Rim Protector / Rebounding Big",
        "Efficient Finisher / Low-Usage Big",
        "High-Usage Volume Big / Post Scorer",
        "Post-Up Bruiser",
        "Roll Man / Screen Finisher",
        "Small-Ball Center",
        "Versatile Facilitator / Combo Big",
        "Post Hub Playmaker",
        "Pick-and-Pop Big",
        "Second-Chance Specialist",
        "Putback Specialist",
        "3-and-D Big",
        "Offensive Liability / Role Defensive Specialist",
    ],
}

# ---- Reverse map: archetype -> typical positions ----
ARCHETYPE_TO_POS: Dict[str, List[str]] = {}
_tmp = defaultdict(list)
for pos, arcs in POSITION_ARCHETYPES.items():
    for a in arcs:
        if pos not in _tmp[a]:
            _tmp[a].append(pos)
ARCHETYPE_TO_POS = dict(_tmp)


# ---- Convenience helpers -----------------------------------------------------

def archetypes_for_position(pos: str) -> List[str]:
    """Get archetypes commonly seen at a position (PG/SG/SF/PF/C)."""
    key = normalize_position(pos)
    return POSITION_ARCHETYPES.get(key, [])

def positions_for_archetype(archetype: str) -> List[str]:
    """Get the typical positions for an archetype."""
    return ARCHETYPE_TO_POS.get(archetype, [])

def normalize_position(pos: str) -> str:
    """
    Normalize messy position strings to one of: PG, SG, SF, PF, C.
    Handles things like 'G', 'F', 'PG/SG', 'SF-PF', 'C-F', etc.
    """
    p = (pos or "").upper().replace(" ", "")
    if not p:
        return ""

    # direct one-letter buckets first
    if p in ("PG", "SG", "SF", "PF", "C"):
        return p
    if p in ("G", "GUARD"):
        return "PG"  # lean PG for generic guards
    if p in ("F", "FORWARD"):
        return "SF"  # lean SF for generic forwards
    if p in ("C", "CENTER"):
        return "C"

    # split on common separators and choose a primary
    for sep in ("/", "-", ","):
        if sep in p:
            parts = [x for x in p.split(sep) if x]
            # priority order to pick a single bucket
            priority = ["PG", "SG", "SF", "PF", "C"]
            for pref in priority:
                if pref in parts:
                    return pref
            # fallbacks
            if "G" in parts:
                return "PG"
            if "F" in parts:
                return "SF"

    # last-chance heuristics
    if "PG" in p:
        return "PG"
    if "SG" in p:
        return "SG"
    if "SF" in p:
        return "SF"
    if "PF" in p:
        return "PF"
    if "C" in p:
        return "C"
    if "GUARD" in p:
        return "PG"
    if "FORWARD" in p:
        return "SF"
    return ""

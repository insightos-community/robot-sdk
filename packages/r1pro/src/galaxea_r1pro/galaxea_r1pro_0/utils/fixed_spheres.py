"""Fixed link-local R1Pro collision geometry, shared by every planning request."""

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=1)
def _asset():
    return json.loads((Path(__file__).parent / "data/r1pro_collision_spheres.json").read_text())


def collision_spheres():
    """Return an independent configuration; cuRobo may normalize it in place."""
    return deepcopy(_asset())

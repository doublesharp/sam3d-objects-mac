"""Reuse a strict report for exactly the same in-memory validated geometry.

Certificates are created by the exporter after its strict validation. They are
never read from GLB metadata or a saved report. UV seam duplication and triangle
ordering may change without changing the oriented embedded surface.
"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from .surface_repair import _arrays


def _canonical_geometry(vertices, faces):
    vertices, faces = _arrays(vertices, faces)
    vertices = vertices.copy()
    vertices[vertices == 0] = 0.0
    vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    faces = inverse[faces]
    # Cyclic rotation preserves winding. Do not reverse or deduplicate faces.
    normalized = faces.copy()
    for shift in (1, 2):
        candidate = np.roll(faces, shift, axis=1)
        less = ((candidate[:, 0] < normalized[:, 0])
                | ((candidate[:, 0] == normalized[:, 0]) & (candidate[:, 1] < normalized[:, 1]))
                | ((candidate[:, 0] == normalized[:, 0]) & (candidate[:, 1] == normalized[:, 1])
                   & (candidate[:, 2] < normalized[:, 2])))
        normalized[less] = candidate[less]
    order = np.lexsort((normalized[:, 2], normalized[:, 1], normalized[:, 0]))
    return vertices, normalized[order]


@dataclass(frozen=True)
class GeometryCertificate:
    _vertices: np.ndarray
    _faces: np.ndarray
    _report: Mapping

    def report_for(self, vertices, faces):
        current_vertices, current_faces = _canonical_geometry(vertices, faces)
        if not np.array_equal(current_vertices, self._vertices) or not np.array_equal(current_faces, self._faces):
            return None
        return dict(self._report)


def from_validated_geometry(vertices, faces, report):
    """Snapshot exporter-owned geometry and its already successful strict report."""
    if not report or report.get('strict_watertight') is not True or report.get('orientable') is not True:
        return None
    canonical_vertices, canonical_faces = _canonical_geometry(vertices, faces)
    canonical_vertices.flags.writeable = False
    canonical_faces.flags.writeable = False
    return GeometryCertificate(canonical_vertices, canonical_faces, MappingProxyType(dict(report)))

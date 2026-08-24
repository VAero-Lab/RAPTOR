#!/usr/bin/env python3
"""
Import Ecuadorian UAS zones into a RAPTOR airspace file.

Converts the GeoJSON layers published by the *mapa-zonas-no-vuelo-ecuador*
viewer (https://github.com/alegriagabriel05-collab/mapa-zonas-no-vuelo-ecuador,
G. Alegría, EPN) into the JSON schema that :func:`raptor.airspace.
load_airspace_from_file` reads.

This script is a **one-off conversion tool, not a dependency.** Run it
when the published layers change; RAPTOR imports nothing from it and
nothing Ecuador-specific at runtime. The output is a plain JSON file in
``data/``, interchangeable with a zone file for any other country.

Why this matters more than convenience
--------------------------------------
The hand-written zone file previously shipped with the package carried
eleven zones drawn as circles from approximate coordinates. The published
layers carry the real footprints: 44 zones inside the Quito DEM alone,
with the runway-aligned aerodrome polygon rather than a 9 km disc, and
with each zone already classified as *Prohibida*, *Restringida* or
*Peligrosa*. That classification is exactly the permission axis the
optimizer reasons about, so importing it replaces guesswork with the
source the regulator and the environment ministry actually publish.

Source layers
-------------
``zonas_poligonales_dgac_1``
    AIP ENR 5.1 prohibited and restricted areas (national).
``zonas_no_vuelo_aeropuertos_prohibida_2``
    Runway-aligned prohibited surfaces at 18 airports.
``zonas_no_vuelo_buffer_4``
    Safety buffers around aerodromes, heliports, hospitals, government
    and military installations, industry, crowds and security zones.
``maate_snap_poligonos_3``
    MAATE/SNAP protected areas — permit from the environment ministry.

Usage
-----
    python -m scripts.import_zones_ecuador \\
        --source /path/to/mapa-zonas-no-vuelo-ecuador/data \\
        --out data/ecuador_uas_zones.json \\
        --bbox -0.42 0.08 -78.72 -78.28        # optional: clip to the DEM

Omit ``--bbox`` to import the whole country.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple

# ── Source layer → how to read it ──────────────────────────────────────────

LAYERS = {
    "zonas_poligonales_dgac_1.js": "dgac_enr",
    "zonas_no_vuelo_aeropuertos_prohibida_2.js": "aerodrome_surface",
    "zonas_no_vuelo_buffer_4.js": "buffer",
    "maate_snap_poligonos_3.js": "maate",
}

# ── Feature ``tipo`` → RAPTOR zone_type ────────────────────────────────────
# Permission is *not* set here: it comes from the feature's own
# Prohibida/Restringida classification, which is the authoritative field.
TIPO_TO_ZONE_TYPE = {
    "aeropuerto": "aerodrome_ctr",
    "aerodromo": "aerodrome_ctr",
    "helipuerto": "heliport",
    "hospital": "hospital",
    "instalacion gobierno": "government",
    "instalacion militar": "military",
    "instalacion industrial": "industrial",
    "infraestructura estrategica": "strategic",
    "concentracion personas": "crowd",
    "zona de seguridad": "security",
    "zona restringida enr 5.1": "restricted",
    "zona peligrosa enr 5.1": "restricted",
    "area militar de instruccion": "military",
}

ZONA_TO_PERMISSION = {
    "prohibida": "prohibited",
    "restringida": "permit",
    "peligrosa": "permit",
}

#: Ecuadorian AIP flight levels appear as "FL120"/"GND"/"UNL" strings.
_FL_RE = re.compile(r"^FL\s*(\d+)$", re.I)


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _norm(text: Optional[str]) -> str:
    return _strip_accents(str(text or "")).strip().lower()


def _parse_level(value, default: float) -> float:
    """
    Read an AIP vertical limit into metres AMSL.

    ``GND`` is the surface, ``UNL`` unlimited, ``FLnnn`` a flight level in
    hundreds of feet.
    """
    if value is None:
        return default
    s = str(value).strip().upper()
    if s in ("GND", "SFC", "0"):
        return 0.0
    if s in ("UNL", "UNLIMITED"):
        return 99999.0
    m = _FL_RE.match(s)
    if m:
        return float(m.group(1)) * 100.0 * 0.3048
    try:
        return float(s)
    except ValueError:
        return default


def load_layer(path: str) -> dict:
    """Read a qgis2web ``var name = {...};`` file as GeoJSON."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if "=" not in text:
        raise ValueError(f"{path}: not a qgis2web layer file")
    payload = text.split("=", 1)[1].strip().rstrip(";")
    return json.loads(payload)


def _rings(geometry: dict) -> List[List[Tuple[float, float]]]:
    """Outer rings of a Polygon or MultiPolygon, as (lat, lon) lists."""
    gtype = geometry.get("type")
    if gtype == "Polygon":
        raw = [geometry["coordinates"][0]]
    elif gtype == "MultiPolygon":
        raw = [poly[0] for poly in geometry["coordinates"]]
    else:
        return []
    return [[(pt[1], pt[0]) for pt in ring] for ring in raw]


def _intersects_bbox(ring: Iterable[Tuple[float, float]], bbox) -> bool:
    if bbox is None:
        return True
    lat0, lat1, lon0, lon1 = bbox
    return any(lat0 <= la <= lat1 and lon0 <= lo <= lon1 for la, lo in ring)


def _simplify_ring(ring: List[Tuple[float, float]],
                   max_vertices: int) -> List[Tuple[float, float]]:
    """
    Thin a ring to at most ``max_vertices`` points.

    Buffer polygons arrive as 60–80-point circles; the containment test
    walks every edge, so keeping all of them costs real time for no
    accuracy that survives a 186 m DEM cell.
    """
    if len(ring) <= max_vertices:
        return ring
    if ring[0] == ring[-1]:
        ring = ring[:-1]
    step = max(len(ring) // max_vertices, 1)
    thinned = ring[::step]
    if len(thinned) < 3:
        thinned = ring[:3]
    return thinned


def convert(
    source_dir: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    max_vertices: int = 24,
    default_ceiling_m: float = 6000.0,
) -> dict:
    """Build a RAPTOR airspace dict from the published layers."""
    zones: List[dict] = []
    seen: Dict[str, int] = {}

    def zone_id(base: str) -> str:
        key = re.sub(r"[^A-Za-z0-9]+", "_", _strip_accents(base)).strip("_")[:40]
        key = key or "ZONE"
        seen[key] = seen.get(key, 0) + 1
        return key if seen[key] == 1 else f"{key}_{seen[key]}"

    for filename, kind in LAYERS.items():
        path = os.path.join(source_dir, filename)
        if not os.path.exists(path):
            print(f"  ! missing layer, skipped: {filename}", file=sys.stderr)
            continue

        gj = load_layer(path)
        kept = 0
        for feat in gj.get("features", []):
            props = feat.get("properties", {})
            geom = feat.get("geometry") or {}

            for ring in _rings(geom):
                if not _intersects_bbox(ring, bbox):
                    continue

                name = (props.get("nombre") or props.get("Nombre_Zona")
                        or props.get("ID_Zona") or "zona")
                tipo = _norm(props.get("tipo") or props.get("Tipo"))
                zona = _norm(props.get("Zona") or props.get("Tipo_Restriccion"))

                if kind == "maate":
                    ztype = "ecological"
                    permission = "permit"
                    ceiling = float(props.get("Altura_Maxima_Metros") or 120.0)
                    is_agl = True
                    floor = 0.0
                    authority = "MAATE / SNAP"
                    citation = "RDAC 101.190(a)(6)"
                else:
                    ztype = TIPO_TO_ZONE_TYPE.get(tipo, "restricted")
                    permission = ZONA_TO_PERMISSION.get(zona, "permit")
                    floor = _parse_level(props.get("Limite_Inferior"), 0.0)
                    ceiling = _parse_level(
                        props.get("Limite_Superior"), default_ceiling_m
                    )
                    is_agl = False
                    authority = str(props.get("provincia") or "DGAC")
                    citation = ("RDAC 101.190(b)" if permission == "prohibited"
                                else "RDAC 101.190(a)")

                    if ztype == "aerodrome_ctr" and permission == "permit":
                        # 101.195(a)(3): outside the prohibited surfaces but
                        # within 5 km of a *controlled aerodrome*, flight is
                        # allowed up to 40 m AGL with the operator's
                        # coordination — a ceiling, not a wall.
                        #
                        # This does not extend to heliports: 101.195(c) gives
                        # them a 0.9 km radius requiring coordination, with no
                        # height allowance, so they stay lateral permit zones.
                        # Applying the 40 m cap to them would put a hard
                        # ceiling under every hospital helipad in the city and
                        # make medical delivery infeasible by construction.
                        ztype = "aerodrome_ring"
                        ceiling = 40.0
                        is_agl = True
                        floor = 0.0
                        citation = "RDAC 101.195(a)(3)"

                zones.append({
                    "zone_id": zone_id(f"{ztype.upper()}_{name}"),
                    "zone_type": ztype,
                    "permission": permission,
                    "geometry_type": "polygon",
                    "vertices": [[round(la, 6), round(lo, 6)]
                                 for la, lo in _simplify_ring(ring, max_vertices)],
                    "altitude_floor_m": floor,
                    "altitude_ceiling_m": ceiling,
                    "is_agl": is_agl,
                    "buffer_m": 0.0,
                    "authority": authority,
                    "citation": citation,
                    "description": str(
                        props.get("observacion") or props.get("Observacion") or name
                    )[:300],
                })
                kept += 1

        print(f"  {filename:<44s} {kept:4d} zone(s)")

    return {
        "metadata": {
            "name": "Ecuador UAS restriction zones",
            "regulation": "RDAC 101 (DGAC Ecuador, Res. DGAC-DGAC-2024-0100-R)",
            "source": ("mapa-zonas-no-vuelo-ecuador (G. Alegria, EPN) — "
                       "DGAC, MAATE and IGM layers"),
            "source_url": ("https://github.com/alegriagabriel05-collab/"
                           "mapa-zonas-no-vuelo-ecuador"),
            "bbox": list(bbox) if bbox else None,
            "n_zones": len(zones),
        },
        "zones": zones,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="data/ directory of the mapa-zonas-no-vuelo-ecuador checkout")
    ap.add_argument("--out", required=True, help="output airspace JSON path")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("LAT0", "LAT1", "LON0", "LON1"),
                    help="clip to this bounding box (defaults to the whole country)")
    ap.add_argument("--max-vertices", type=int, default=24,
                    help="thin polygon rings to at most this many vertices")
    args = ap.parse_args(argv)

    bbox = tuple(args.bbox) if args.bbox else None
    print(f"Reading layers from {args.source}")
    data = convert(args.source, bbox=bbox, max_vertices=args.max_vertices)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    n = data["metadata"]["n_zones"]
    print(f"\nWrote {n} zone(s) to {args.out}")

    by_perm: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    for z in data["zones"]:
        by_perm[z["permission"]] = by_perm.get(z["permission"], 0) + 1
        by_type[z["zone_type"]] = by_type.get(z["zone_type"], 0) + 1
    print("  by permission: " + ", ".join(f"{k}={v}" for k, v in sorted(by_perm.items())))
    print("  by type:       " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# RAPTOR v0.8.0

**RAPTOR:** Energy-Optimal **R**egulation-**A**ware **P**ath and **T**rajectory **O**ptimization for **R**otorcraft (eVTOLs, multirotors)

A Python package for 3D flight path optimization over real terrain (DEM) for
eVTOL/multirotor UAV operations. Supply your own DEM, facility coordinates (pick-up centers, delivery destinations, etc.), and airspace zone
definitions (restricted zones) to apply it to any region. Includes a complete case study for the
Quito Metropolitan District, Ecuador (2,850 m ASL).

<img src="raptor_package_logo.svg" alt="RAPTOR Logo" width="500">

## Installation

```bash
pip install -e .    # editable (development)
pip install .       # standard
```

**Requirements:** Python >= 3.9, numpy, scipy, matplotlib

Optional, each enabling one capability and degrading cleanly when absent:

| Package | Enables | Without it |
|---|---|---|
| `aerosandbox` | wing drag polars computed from geometry | analytic parabolic polar, labelled as such |
| `tifffile` + `imagecodecs` | building DEMs from local GeoTIFF tiles (NASADEM tiles are LZW-compressed and need both) | use OpenTopoData or a prebuilt `.npz` |

```bash
pip install -e '.[all]'    # everything
```

## Documentation

| Document | What it covers |
|---|---|
| [`docs/CASES.md`](docs/CASES.md) | Every mission case, its settings, its results and the command that reproduces it |
| [`docs/ROUND4.md`](docs/ROUND4.md) | What changed in the latest round of work, why, and what is still open |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history |

## What is computed and what is assumed

Every number the package produces rests on a model, and the models differ
in how much they know. This is the map:

| Quantity | Source | Says so via |
|---|---|---|
| Wing lift and drag | AeroSandbox nonlinear lifting line, cached | `ac.aero_source` |
| Fuselage, booms, gear, stopped rotors | component build-up from dimensions | `ac.drag_report()` |
| Rotor power | momentum theory + blade profile drag | valid outside the vortex-ring band, `vortex_ring_margin()` |
| Battery OCV and resistance | measured discharge log, or textbook default | `ac.battery_provenance()` |
| Terrain | SRTM 1-arc-second, voids interpolated | `void_report()`, `cross_check_corridor()` |
| Wind | steady log-profile model | not a forecast; `WindField` |
| Airspace zones | published DGAC/MAATE footprints | `zone.citation`, `zone.authority` |

Nothing above is tuned to make a result come out. Where a value is a
judgement — rotor parking, permit effort, penalty weights — it is named,
defaulted conservatively, and meant to be swept.

```python
from raptor import AircraftEnergyParams, UAVConfig, check_flight_envelope

ac = AircraftEnergyParams().build_aero()   # wing polar + airframe build-up
print(ac.aero_source)                       # wing: aerosandbox, airframe: build-up
print(ac.drag_report())                     # where the drag comes from
print(ac.battery_provenance())              # measured vs assumed
for p in ac.validate():                     # implausible parameters
    print("!", p)
uav = UAVConfig.for_vehicle(ac, altitude=2800)   # envelope from the aerodynamics
```

## The vertical corridor — read this first

UAS rules cap height above *ground*, not above sea level. RDAC 101.185
allows 122 m AGL. A cruise leg flown at one fixed altitude cannot obey
that over anything but a plain: hold 2 950 m AMSL across a Quito corridor
whose ground runs 2 429–2 904 m and the height above ground swings from
46 m to 521 m — simultaneously too low over the ridge and 400 m over the
ceiling in the valley.

So RAPTOR plans in an **AGL corridor**. `MissionConstraints` defines the
band (clearance floor to regulatory ceiling), `RoutedPath(corridor_mode=
"agl")` tracks the terrain inside it, and `constraints.validate()` refuses
to let you plan inside a band with no room:

```python
con = MissionConstraints()          # 60–122 m AGL, 62 m of room
print(con.describe_corridor())
for p in con.validate():
    print("!", p)                   # empty when the corridor is flyable
```

Before optimizing anything, triage the corridors:

```bash
python -m scripts.check_feasibility
```

It separates the three reasons a route fails — an impossible constraint
set, relief steeper than the aircraft's descent rate, or prohibited
airspace across the line — because each needs a different fix.

## Package Structure

```
raptor/
├── setup.py                          # pip install
├── LICENSE                           # MIT
├── README.md
├── data/                             # Region-specific data (replaceable)
│   ├──TIF_files/                     # Source rasters for DEM generation
│   │ ├──rasters_NASADEM.tar.gz       # NASADEM bare-earth, void-free (shipped DEM)
│   │ ├──n00_w079_1arc_v3.tif         # SRTM v3, kept for cross-checking
│   │ ├──s01_w079_1arc_v3.tif
│   ├── dmq_dem_30m.npz               # Quito DEM (NASADEM, 30 m, zero voids)
│   ├── vehicles/                     # va17.json, va23.json, va25.json
│   ├── dmq_facilities.json           # Medical facility coordinates
│   └── ecuador_uas_zones.json        # DGAC/MAATE zones (49) — plain data
├── raptor/                # Core package
│   ├── atmosphere.py                 # ISA standard atmosphere
│   ├── aero.py                       # Wing polar from geometry (AeroSandbox)
│   ├── drag.py                       # Component build-up for the airframe
│   ├── battery.py                    # Pack fitting from discharge logs
│   ├── wind.py                       # Shear, crab angle, round-trip penalty
│   ├── config.py                     # UAV envelope + vertical corridor
│   ├── dem_build.py                  # DEMs from GeoTIFF tiles or OpenTopoData
│   ├── dem.py                        # DEM interface, haversine, bearing
│   ├── segments.py                   # 6 flight segment types
│   ├── path.py                       # FlightPath construction
│   ├── terrain.py                    # Clearance floor + AGL ceiling checks
│   ├── builder.py                    # DEM-aware path builder
│   ├── energy.py                     # Power models + stall detection
│   ├── regulations.py                # RDAC 101 limits + permission classes
│   ├── compliance.py                 # Graded standing + waiver quantification
│   ├── missions.py                   # Round trips, tours, hub-and-spoke
│   ├── airspace.py                   # Zone geometry and permission lookup
│   ├── corridor.py                   # Terrain-following reference profiles
│   ├── penalties.py                  # Regulations → optimizer search pressure
│   ├── routed_path.py                # Lateral routing + waypoints
│   ├── optimizer.py                  # DE engine, feasibility-first
│   ├── mission_planner.py            # Multi-leg SOC-coupled orchestration
│   ├── scenarios.py                  # 8-scenario medical catalog (Quito)
│   ├── vehicles.py                   # 4 parametric UAV configurations
│   ├── astar_baseline.py             # A* grid benchmark
│   ├── viz_corridor.py               # AGL corridor, permission and waiver maps
│   ├── visualization.py              # Matplotlib publication figures
├── scripts/                          # Runner scripts
│   ├── build_dem.py                  # Build terrain from tiles or the network
│   ├── check_feasibility.py          # Corridor triage — run this first
│   ├── make_figures.py               # All figures
│   ├── run_studies.py                # The five scenario studies
│   ├── build_vehicles.py             # Aircraft from published manufacturer data
│   └── import_zones_ecuador.py       # Import published DGAC/MAATE zones
└── examples/                         # Demo scripts
    ├── demo_path_planning.py         # Basic pipeline
    ├── demo_energy_analysis.py       # Energy model
    ├── demo_missions.py              # Mission profiles and payload order
    ├── demo_vehicle.py               # Aerodynamics, drag, battery provenance
    ├── demo_regulatory.py            # Corridor, permissions, credentials
    └── demo_optimization.py          # Optimization with airspace
```

## Quick Start

```python
from raptor import *
from raptor.airspace import build_airspace
from raptor.routed_path import RoutedPath
from raptor.optimizer import PathOptimizer, OptMode

dem = DEMInterface(find_dem())        # data/dmq_dem_30m.npz
airspace = build_airspace(dem=dem)
con = MissionConstraints()
optimizer = PathOptimizer(dem, UAVConfig(), con, AircraftEnergyParams())

orig = FacilityNode('Hospital', -0.2000, -78.4930, 2788.0)
dest = FacilityNode('Clinic',   -0.3080, -78.4710, 2510.0)

# Terrain-following corridor: the cruise tracks the ground inside the
# legal AGL band instead of holding one altitude across the relief.
rp = RoutedPath(orig, dest, dem, UAVConfig(), con,
                n_intermediate=2, corridor_mode='agl')

result = optimizer.optimize_routed(
    rp, mode=OptMode.ENERGY, payload_kg=1.5,
    airspace=airspace, maxiter=200, popsize=20, verbose=True)

print(result.summary())
print(result.regulatory_summary)      # LEGAL with 2 authorisation(s)
print(result.permits_required)        # {'HOSPITAL_...': 'hospital', ...}
```

## Terrain

The DEM decides how good every clearance number is, so building a good
one is a first-class step rather than a prerequisite you are assumed to
have met.

```bash
# From local SRTM/NASADEM tiles — free, offline, source resolution.
# .tif files and .tar.gz archives of them are both accepted, and each
# tile's own GDAL_NODATA tag and pixel convention are honoured, so the
# two products can be mixed in one call.
python -m scripts.build_dem tiles --tiff 'data/TIF_files/*.tar.gz' \
    --bbox -0.42 0.08 -78.72 -78.28 --spacing 30 \
    --product 'NASADEM v001 bare-earth, 1 arc-second' \
    --out data/dmq_dem_30m.npz

# From OpenTopoData — no files, works anywhere on Earth
python -m scripts.build_dem corridor --from -0.2000 -78.4930 \
    --to -0.3242 -78.5493 --width 2000 --spacing 30 --out data/corridor.npz
```

Resolution is not a detail. Rebuilding the bundled Quito grid from
~186 m to 30 m moved every corridor's ceiling exceedance *upward* — a
coarse grid smooths away exactly the ridges a 60 m corridor has to clear
— and flipped one route from feasible to not:

| Corridor | 186 m grid | 30 m grid |
|---|---|---|
| Espejo → Calderón | fits | 59 m over ceiling, 2 % of route |
| Espejo → Tumbaco | 120 m over, 33 % | 153 m over, 38 % |
| Garcés → Lloa | 80 m over, 24 % | 103 m over, 30 % |

The public OpenTopoData instance allows 100 000 points a day, so a 30 m
raster over a whole city is out of reach — fetch the corridor you plan
over, which is masked to a swath around the route rather than a
rectangle. `--estimate` prices any request before it spends quota, and
`--api` points at a self-hosted instance with no limits.

`raptor.find_dem()` resolves to the finest grid present, so rebuilding
terrain takes effect everywhere without editing paths.

**Is the terrain any good?** SRTM returns nothing from steep slopes and
water; 8.2 % of the DMQ mosaic is interpolated. That matters only if the
holes are where you fly:

```bash
python -m scripts.build_dem check --from -0.2000 -78.4930 \
    --to -0.2158 -78.4085 --against srtm30m
```

```
  8.2% of the terrain here was interpolated, not measured.
  along this corridor (2000 m swath): 0.9% interpolated
  local minus srtm30m: bias +1.1 m, scatter 7.7 m — consistent
```

The voids sit away from the study corridors, and the mosaic agrees with
an independent query to within a metre — so the local 30 m DEM is good
enough and OpenTopoData is not needed for Quito. Use `--against
aster30m` for a genuinely independent comparison (expect ~10 m: that is
ASTER's own accuracy, not your error). NASADEM would be the ideal
reference but the public instance does not serve it.

## Aerodynamics

Two halves, summed, never overlapping. The **wing** comes from
AeroSandbox's nonlinear lifting line, computed once over a grid of
incidence and Reynolds number and cached — a solve takes ~3.5 s, so it
cannot run inside an optimizer objective. The **rest of the airframe**
comes from a component build-up: skin friction × form factor ×
interference × wetted area, part by part.

```python
from raptor.vehicles import get_vehicle
ac = get_vehicle("va23").build_aero()
print(ac.drag_report(V=27, altitude=2800))
```

The build-up matters more than it sounds, because a lift+cruise quadplane
carries its stopped lift rotors through cruise and a stopped rotor is
close to a bluff plate. On the baseline airframe they out-drag the wing:

| Component | Drag at 30 m/s | Share |
|---|---|---|
| stopped rotors ×4 | 5.5 N | 35 % |
| **wing** (profile + induced) | 4.2 N | 27 % |
| landing gear | 1.7 N | 11 % |
| nacelles ×4 | 1.4 N | 9 % |
| fuselage | 1.1 N | 7 % |
| tail, booms, miscellaneous | 1.7 N | 11 % |

How the rotors come to rest moves the total by a factor of two, so it is
named rather than tuned — `rotor_parking` is `tilted`, `folded`,
`aligned` or `unaligned`, and the choice is recorded in the drag report.
A tilt-rotor parks nothing (`tilted`) and pays instead in cruise
propulsive efficiency, because its cruise propellers are its hover
rotors, running badly off-design.

`UAVConfig.for_vehicle()` derives the **whole** flight envelope from the
aerodynamics — not just the speeds but the climb and descent angles, the
climb and descent rates, the turn radius and the VTOL descent rate — so
none of them can drift apart from the aircraft or from each other:

```python
uav = UAVConfig.for_vehicle(ac.with_payload(1.5), altitude=2800.0)
print(uav.manoeuvre_problems())      # [] when the limits are self-consistent
```

The hand-typed constants that `UAVConfig()` still carries do *not* pass
that check, which is what it is for:

| Quantity | Typed in | Implied by the others |
|---|---:|---:|
| minimum turn radius | 100 m | **159 m** at 30° bank, 30 m/s |
| max climb rate | 5.0 m/s | **7.8 m/s** at 15°, 30 m/s |
| max climb angle | 15° | **10.7°** from the VA23's power and L/D |

Deriving the climb angle from power rather than typing it in matters most:
it is what the corridor envelope uses to decide whether terrain can be
followed at all, and 15° made corridors look flyable that are not.

## Battery

The pack is an open-circuit voltage curve behind an internal resistance,
with usable-energy derating and C-rate reporting.

The curve is **chemistry-specific**, because a LiPo falls steadily from
4.2 V while a high-nickel Li-ion holds a long plateau and then drops in
the last fifth — and the plateau is what decides how much of the pack
sits above the cut-off:

```python
from raptor.battery import cell_ocv_curve, check_pack_against_datasheet
check_pack_against_datasheet("li-ion", 4.2, 15.12,
                             current_a=0.84, resistance_ohm=0.025)
```

The Li-ion curve is **measured**, not assumed: a C/30 characterisation of
a Samsung INR21700-30T from a CC BY 4.0 dataset
([doi:10.17632/fywnpsjfpc.1](https://doi.org/10.17632/fywnpsjfpc.1)),
re-extractable with `python -m scripts.extract_ocv --all --download`.

The check above is the only validation available without a bench: a
datasheet states capacity in amp-hours *and* energy in watt-hours, and
their ratio is a mean voltage the curve has to reproduce — at the
terminals, under the datasheet's own discharge current, which is why it
takes the current and the resistance. Comparing an open-circuit mean
against a loaded one asks the curve to be wrong by exactly one `I·R`, and
an earlier version of this check passed a curve that was.

Pack energy comes from that curve, not from `mass × specific_energy`.
That is the way round it has to be: with energy coming from mass, the
curve set only the current and the resistive loss, and swapping a
measured curve for a generic one changed the endurance by nothing at
all.

Both curve and resistance can be **fitted from a measured discharge
log**, which replaces all of the above with measurement:

```python
from raptor import DischargeLog, fit_battery

log = DischargeLog.from_csv("bench/pack_discharge.csv")
ac.fit_battery_from_log(log)
print(ac.battery_provenance())      # MEASURED vs ASSUMED, line by line
```

The fit solves `V_terminal = V_oc(SOC) − I·R` as one linear least-squares
problem. Current variation is what separates the two terms, so a log
flown at constant current cannot identify the resistance — the fit says
so and holds it at a prior rather than returning a confident wrong
number.

## Wind

Not modelled before, and over Andean terrain in a 62 m corridor that is
not a small omission:

```python
from raptor.wind import QUITO_ANNUAL_MEAN, QUITO_REANALYSIS
print(QUITO_ANNUAL_MEAN.source)          # where the number came from
print(QUITO_ANNUAL_MEAN.speed_at(90.0))  # 3.96 m/s in the corridor
```

Three things this gets right that a single-number wind does not.

**A mast measurement is not a corridor wind.** An airport anemometer
stands on mown grass; the corridor crosses a city. Reading 3.08 m/s
straight up to 90 m with the corridor's roughness gives 6.02 m/s;
converting through a blending height using both roughnesses gives
3.96 m/s. `WindField` carries `reference_roughness_m` for this reason.

**Quito's two sources disagree and both are shipped.** A thirty-year
station record gives 3.08 m/s from the NNW; MERRA-2 reanalysis gives
1.65 m/s from the east. `QUITO_ANNUAL_MEAN` and `QUITO_REANALYSIS` are
both there, a factor of 1.9 apart, so a conclusion can be tested against
both rather than resting on whichever is more convenient.

**A round trip loses in both directions** — the slow leg lasts longer
than the fast leg saves — while a one-way leg can win outright. That
asymmetry is why the mission cases in `docs/CASES.md` treat them
separately.

## When no legal path exists

Some corridors have no compliant profile at any altitude — the ground
falls away faster than the aircraft can descend. Reporting that as
"infeasible" is true and useless, so compliance is graded and a waiver is
quantified:

```python
result = optimizer.plan_route(rp, airspace=airspace, payload_kg=1.5)
print(result.compliance.verdict())
print(result.compliance.filing_summary())
```

```
WAIVER — +17 m AGL over 9% of route
  Height waiver required (RDAC 101.185 → 101.035)
    Limit:      122 m AGL
    Requested:  139 m AGL (+17 m)
    Affected:   0.49 km of 5.31 km (9.3 %), in 3 stretch(es)
      1. km  1.67– 1.88  peak 139 m AGL at -0.2535, -78.5620
```

`plan_route` tries strict compliance first and only relaxes the height
ceiling if that fails, minimising the relief needed. Prohibited airspace
is never relaxed — RDAC 101.190(b) admits no exception, so a route
blocked by it is reported as not flyable rather than given a waiver that
does not exist.

| Level | Meaning |
|---|---|
| `COMPLIANT` | fly it, nothing to file |
| `AUTHORIZATION_REQUIRED` | restricted airspace, 101.190(a) |
| `WAIVER_REQUIRED` | height relief under 101.035, quantified |
| `NON_COMPLIANT` | prohibited airspace — the route must change |

A waiver is asked for **per stretch**, not as one ceiling over the whole
route. Requesting relief across the 90 % of track that never needed any
is the easy thing to ask for and the hard thing to be granted; asking
per excursion typically cuts the requested airspace by 90 %.

Some permits cannot be routed around at all. A medical network is built
around hospitals, and hospitals are permit zones under 101.190(a)(3), so
the pads themselves usually sit inside one — Hospital Eugenio Espejo is
333 m inside the Baca Ortiz buffer. `compliance.unavoidable_permits`
separates those from the ones routing can actually avoid, which is what
stops a permit-avoidance study chasing a trade that does not exist.

## Mission profiles

```python
from raptor import out_and_back, sample_collection, MissionPlanner

mission = sample_collection(hospital, [centre_a, centre_b], per_stop_kg=0.7)
print(mission.payload_schedule())     # [0.0, 0.7, 1.4] — heavier as it goes
result = MissionPlanner(dem, uav, con, ac, airspace).plan(mission)
```

`out_and_back`, `sample_collection`, `supply_tour`, `hub_and_spoke` and
`shuttle` differ in what happens *between* legs, which is what decides
whether a multi-stop mission is one flight or several: a stop that can
swap the pack resets the state of charge, and one that cannot does not.
A collection round flies its heaviest leg on its lowest charge — the
coupling a per-leg analysis cannot see.

## Regulation-aware planning

Zones carry a **permission class**, not just a type:

| Class | Meaning | Optimizer treatment |
|---|---|---|
| `FREE` | fly without asking | no cost |
| `PERMIT` | legal with prior authorisation (RDAC 101.190(a)) | priced, reported |
| `PROHIBITED` | never overflown (RDAC 101.190(b)) | hard barrier |

A permit zone does **not** make a route infeasible. The optimizer weighs
crossing it and filing for permission against flying around, and the
result lists every authorisation the plan depends on. That distinction is
the regulation's own, and collapsing it into "violation" throws away the
trade the planner wants to make.

Credentials change which limits bind:

```python
from raptor import OperationalContext, RDAC_101, PathOptimizer

ctx = OperationalContext(
    has_uoc=True,                     # required for medical cargo (101.170)
    bvlos=True,                       # specific category (101.210)
    authorized_zone_ids=['HOSPITAL_Hospital_Metropolitano'],  # already held
)
optimizer = PathOptimizer(dem, uav, con, ac, regulation=RDAC_101, context=ctx)
print(ctx.compliance_findings(RDAC_101))   # route-independent obligations
```

To use a different jurisdiction, build a different `RegulatoryProfile` —
nothing else changes.

## Importing published zone data

The Ecuadorian DGAC/MAATE zone layers published by
[mapa-zonas-no-vuelo-ecuador](https://github.com/alegriagabriel05-collab/mapa-zonas-no-vuelo-ecuador)
(G. Alegría, EPN) import directly, with their own
*Prohibida* / *Restringida* classification preserved:

```bash
python -m scripts.import_zones_ecuador \
    --source /path/to/mapa-zonas-no-vuelo-ecuador/data \
    --out data/ecuador_uas_zones.json \
    --bbox -0.42 0.08 -78.72 -78.28
```

## Figures

Two scripts. `make_figures` draws the **network** — every corridor, the
airspace, the atlas — and there is one of each. `make_case_figures`
draws a **flight**, and there is one of each per case and per pair:

```bash
python -m scripts.make_figures --out figures/current
python -m scripts.make_case_figures --list
python -m scripts.make_case_figures --case one_way --pair garces-pintag
python -m scripts.make_case_figures --all --maxiter 8 --popsize 6
```

Per flight it produces the route from above with its altitude, height
above ground and lateral offset on one distance axis; the same route in
three dimensions; the routes the optimiser tried beside the objective
and the constraint penalties behind it; the pack's state of charge,
voltage, current and cumulative energy with the flight phases marked;
and which phase spent the energy against which spent the time.

The objective and the urgency are selectable:

```bash
python -m scripts.make_case_figures --case one_way --pair espejo-pintag \
    --objective time --urgency emergency
```


```bash
python -m scripts.make_figures --out figures/current
```

| Figure | Shows |
|---|---|
| `corridor_amsl_vs_agl` | the central figure: the legal band drawn as a ribbon following the ground, constant-altitude against terrain-following |
| `corridor_agl_profile` | the same in the frame the rule is written in — height above ground |
| `corridors_all` | every study corridor |
| `airspace_permission_map` | zones coloured by permission, not by type |
| `waiver_map` | where on the network height relief is needed, and how much |
| `drag_buildup` | which parts of the airframe make the drag |
| `aero_polar` | computed polar and lift curve against the constants they replaced |
| `dem_resolution_effect` | what a coarse DEM hides |
| `wind_sensitivity` | shear from anemometer to corridor, round-trip penalty |

## Results on the DMQ network

Six study corridors, 30 m terrain, computed aerodynamics, all 49 published
zones. Optimizer settings are modest — these are existence results, and
the energies are upper bounds:

| Corridor | Standing | E [Wh] | Filing |
|---|---|---|---|
| Garcés → Píntag | COMPLIANT | 168 | nothing |
| Espejo → Guamaní | AUTHORISED | 166 | 1 hospital permit |
| Espejo → Calderón | AUTHORISED | 114 | 1 hospital permit |
| Garcés → Lloa | AUTHORISED | 120 | 1 industrial permit |
| Suárez → IESS Sur | AUTHORISED | 123 | 1 hospital permit |
| Espejo → Tumbaco | WAIVER | 105 | +29 m AGL over 4 % |

None of the six is legal flown as a straight line; all six are flyable
once routed. Across the wider network, only **3 of 15** hospital pairs
admit a legal *straight* corridor — which is what the lateral routing is
for.

One mission-level result is decisive enough to quote without hedging:
hub-and-spoke over the same two destinations, differing only in whether
base has a spare battery.

| Ground equipment | Energy | SOC at worst |
|---|---|---|
| spare pack at base | 609 Wh | 55.7 % |
| no spare pack | 609 Wh | **27.8 %** |

Identical routing and identical energy to the tenth of a watt-hour, and
28 percentage points of margin between them. Ground equipment buys margin
here; extend the tour and it decides whether the mission exists at all.

```bash
python -m scripts.run_studies              # all five studies
```

## Using Your Own Data

The framework is region-agnostic. To apply it to a different area:

1. **DEM:** Provide a `.npz` file with keys `lat`, `lon`, `elevation` (2D grid)
2. **Facilities:** JSON with `name`, `lat`, `lon`, `ground_elev` per facility
3. **Airspace:** JSON file defining restricted zones:

```json
{
  "zones": [
    {
      "zone_id": "MY_AIRPORT",
      "zone_type": "aerodrome_ctr",
      "geometry_type": "circle",
      "center_lat": 40.6413,
      "center_lon": -73.7781,
      "radius_m": 9000,
      "altitude_ceiling_m": 3000,
      "is_agl": false,
      "description": "JFK Airport exclusion zone"
    }
  ]
}
```

Zone types: `prohibited`, `restricted`, `aerodrome_ctr`, `aerodrome_ring`,
`heliport`, `controlled_airspace`, `altitude_limit`, `populated_area`,
`ecological`, `military`, `government`, `security`, `hospital`,
`strategic`, `industrial`, `crowd`, `temporal`

Geometry types: `circle` (center + radius), `polygon` (vertex list), `global` (everywhere)

Optional per-zone keys: `permission` (`free` / `permit` / `prohibited`,
overriding the type default), `permit_family`, `authority`, `citation`.

## Running Scripts

```bash
# Corridor triage — which routes are legal at all. Run this first.
python -m scripts.check_feasibility

# Rebuild the three aircraft from published manufacturer data
python -m scripts.build_vehicles
python -m scripts.build_vehicles --check     # report only, write nothing

# Every figure except the method comparison, which runs the optimizer
python -m scripts.make_figures --out figures/current
python -m scripts.make_figures --only methods --maxiter 10 --popsize 8

# The five scenario studies
python -m scripts.run_studies --only atlas
python -m scripts.run_studies --maxiter 60 --repeats 5

# Terrain quality
python -m scripts.build_dem check --from LAT LON --to LAT LON --against aster30m

# Demos
python -m examples.demo_vehicle                        # the three aircraft
python -m examples.demo_vehicle --provenance va23      # where each number came from
python -m examples.demo_energy_analysis                # where the energy goes
python -m examples.demo_path_planning --plot
python -m examples.demo_optimization --plot --maxiter 100
```

## License

MIT — See LICENSE file.

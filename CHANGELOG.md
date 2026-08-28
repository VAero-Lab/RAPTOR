# Changelog

## v0.8.0 — Terrain, aircraft, battery, wind

Full account with before/after numbers in [`docs/ROUND4.md`](docs/ROUND4.md);
every mission case and its settings in [`docs/CASES.md`](docs/CASES.md).

### Fixed — results-changing

- **The mission planner tracked state of charge against a hard-coded
  600 Wh pack**, whatever aircraft was flying. On the VA23's 867 Wh of
  usable energy that overstated depletion by 45 %. It now uses the
  aircraft's own.
- **The planner's energy re-analysis dropped the payload.** The
  optimizer was given it and the re-analysis was not, so the two
  disagreed by 15 % on a 2 kg delivery — and the number the mission
  reported was the one computed for an empty aircraft.
- **State of charge advanced on energy delivered to the drivetrain, not
  drawn from the cells.** The I²R heat comes out of the same pack and
  never arrives, so the mission was charged only for what got through.
- **Pack energy came from `mass × specific_energy`, so the
  open-circuit-voltage curve reached nothing.** Swapping a measured
  Li-ion curve for a generic LiPo one changed the endurance by 0.00 %.
  Energy now comes from the curve; mass and specific energy are the
  cross-check.
- **Battery mass was cell mass**, with no allowance for wiring, BMS and
  case. Counting them (15 %) means none of the three aircraft can carry
  its largest listed pack inside its own published empty mass —
  the VA23 ships with 22 Ah rather than 30 Ah as a result.
- **The datasheet check compared an open-circuit mean against a loaded
  one**, and so was satisfied by a curve wrong by exactly the `I·R` it
  omitted. It now compares at the terminals and states its basis.

- **The corridor profile reported clearance and ceiling figures from the
  dense reference path, not from the simplified polyline the aircraft
  actually flies.** On 30 m terrain the flown path oscillated to 165 m
  AGL over flat ground and dipped to 40 m where the floor is 60 m, while
  the profile reported zero violations. Simplification now repairs
  clearance by construction, every number is measured on the flown path,
  and the breakpoint budget went from 12 to 24. **The headline result
  changed: 2 of 15 hospital pairs admit a legal straight corridor, not
  3.**
- **The GeoTIFF reader assumed one pixel convention and one void
  sentinel.** SRTM v3 is `RasterPixelIsPoint` with −32767; NASADEM is
  `RasterPixelIsArea` with −32768. Reading both alike misplaced NASADEM
  by half a cell — cross-check scatter dropped from sd 10.2 m to 3.3 m
  once fixed. `GTRasterTypeGeoKey` and `GDAL_NODATA` are now read per
  tile, with a hard floor at −1000 m as a second guard.
- **`m_tow` meant "take-off mass" in its docstring and "mass before
  payload" to `with_payload()`.** A file holding the datasheet MTOW put
  every loaded mission over MTOW. Split into `m_tow` (empty),
  `m_tow_max` and `payload_capacity_kg`, with a warning on overload.
- **`cell_capacity_ah` held the pack's capacity**, collapsing the
  parallel-string count to 1 and inflating pack resistance by roughly
  the real parallel count (≈7× for the VA23).
- **`electrical_power_max_w` sized the pack from a single cell**, giving
  a seventh of the real power — which then bound the derived climb angle
  and halved it.
- **`SOC` meant "usable energy exhausted" to the energy budget and "cell
  empty" to the voltage curve**, so the pack read 2.5 V/cell at the
  reserve. Added `charge_fraction`.
- **`build_polar()` attached a drag table without updating the scalar
  `C_L_max`** that `stall_speed_at()` reads, so every aircraft computed
  its speed envelope from the default wing.
- **`to_dict()` dropped four fields `from_dict()` reads** — rotor
  diameter, both efficiencies and the airframe geometry — so saving a
  vehicle and loading it back returned a different aircraft.
- **`DEMInterface` flipped the elevation grid for a descending latitude
  axis but not the void mask**, slope grid or coordinate meshes.
- **`AStarResult` provided `get_waypoints_array` but not
  `get_waypoints`**, so profile figures silently rejected A* output —
  the exact comparison those figures exist to make.
- **Figures re-analysed a whole mission at the departure payload.** An
  out-and-back carries the cargo out and comes home empty, so the
  battery panel and the results table disagreed by 8 % on a two-leg
  round trip. Mission-level energy is now stitched from the planner's
  own per-leg results (`stitch_energy`).
- **Every leg after the first was analysed from a full pack.**
  `analyze_path_energy` takes `SOC_initial` and the planner never
  passed it, so leg 3 of a supply tour was modelled at 4.2 V per cell
  and the state-of-charge trace jumped back to 100 % at each stop.
  Terminal voltage sets the current, so later legs were reported
  cooler and cheaper than they are.
- **The path simplifier could emit a tidy route through the ceiling.**
  Where the AGL band is tighter than the airframe's climb limit no
  straight segment fits, and the fallback invented an altitude — 145 m
  AGL against a 122 m ceiling, reported as a success. It now falls back
  to the altitude actually flown, so every emitted waypoint is inside
  the band or a point the aircraft already reached.

### Added

- **`raptor/simplify.py` — optimised profiles turned into flight
  plans.** A post-processing pass that redraws a leg's altitude profile
  as the fewest straight segments that fit inside the 60–122 m AGL
  tube, respecting the aircraft's own climb and descent limits, and
  reports what the tidying cost in energy rather than assuming it is
  free. On the VA17's Espejo→Suárez leg: 28 vertical manoeuvres to 8,
  thirteen waypoints, +7 % cruise energy. It sits *beside* the flown
  path — every compliance and energy number is still measured on what
  the aircraft flies.
- **Numbered waypoints across every view.** The same number means the
  same point in the map, altitude, height-above-ground, lateral-offset
  and three-dimensional panels, and in the exported `waypoints.csv`.
- **`examples/PAPER/` — the production run.** Eight case scripts, each
  sweeping three aircraft × two payload modes × three mission drivers,
  plus `compare_all.py` for the cross-case figures. Output is
  git-ignored and regenerable.

- **A measured open-circuit-voltage curve** for a high-nickel NMC 21700,
  extracted from a CC BY 4.0 dataset (Samsung INR21700-30T at C/30,
  doi:10.17632/fywnpsjfpc.1) by `scripts/extract_ocv.py`. Three of the
  four cells agree to 1.3 mV; the fourth returns twice the charge it
  delivers and is excluded by a coulombic-efficiency check.
- **A mass budget** in `scripts/build_vehicles.py`: pack plus airframe
  plus propulsion plus avionics must fit inside the published empty
  mass, and the pack that closes it is chosen from the manufacturer's
  listed options.
- **An endurance verdict** that compares a published claim against the
  band between cruise-speed and minimum-power endurance, rather than
  against one speed the manufacturer never stated.
- **`RegulatoryProfile.unenforceable_rules()`** — the rules RDAC 101
  states that this package cannot test, now named in every compliance
  report instead of being silently absent.
- **`MissionConstraints.max_range` is enforced**, having been declared
  and read by nothing.

- **Three aircraft derived from published manufacturer data** — T-Drones
  VA17, T-Drones VA23 (default), Alafija VA25 — generated by
  `scripts/build_vehicles.py`. Every parameter tagged `referenced`,
  `derived` or `assumed` and readable as `ac.provenance`. The four
  invented vehicles and their hard-coded fallbacks are gone.
- **`UAVConfig.for_vehicle()` now derives the whole envelope**: climb and
  descent angles from power and glide, climb and descent rates from angle
  and speed, turn radius from bank and speed, VTOL descent rate from the
  vortex-ring band. `manoeuvre_problems()` reports any set that
  contradicts itself.
- **Chemistry-specific battery curves** with `check_pack_against_datasheet()`,
  the one validation available without a bench. SOC-dependent internal
  resistance. A `tilted` rotor-parking mode for tilt-rotors.
- **Cited Quito wind defaults** (`QUITO_ANNUAL_MEAN`,
  `QUITO_WINDIEST_MONTH`, `QUITO_CALMEST_MONTH`, `QUITO_REANALYSIS`) and
  a two-roughness boundary layer — reading a mast measurement up with the
  corridor's roughness alone overstated corridor wind by 47 %.
- **Plan-view figures**: routes from above on shaded terrain,
  lateral-deviation profiles, a combined plan-profile-deviation figure,
  and a five-method comparison (straight line, terrain-following, A*, DE
  at N=1 and N=3) on one page. Hillshading is computed from the DEM, so
  it works for any region.
- **`raptor.terrain.check_path_envelope()`** — the call site the six
  `UAVConfig.validate_*` methods never had.
- `DEMInterface.decimated()` and `from_arrays()`; DEM provenance via
  `metadata.provenance()`; tile archives readable directly from
  `.tar.gz`.

### Changed

- DEM is NASADEM bare-earth, zero voids, replacing an SRTM mosaic that
  was 8.2 % interpolated.
- Visualization is matplotlib only; `visualization_plotly.py` removed
  along with `run_experiments.py`, `run_scenario_catalog.py` and
  `paper_cases.py`.
- `max_path_segments` raised from 20 (unenforced, and wrong for
  terrain-following) to 80, and reported rather than imposed.
- `imagecodecs` added to the `terrain` extra — NASADEM tiles are
  LZW-compressed.

### Testing

131 → 166 tests, passing with `-W error::UserWarning`.

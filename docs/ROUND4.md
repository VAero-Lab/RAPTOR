# Round 4 — Terrain, Aircraft, Battery, Wind

**What this document is.** A record of what changed in this round of work
on RAPTOR, why each change was made, what it did to the numbers, and what
is still open. It is written to be read by someone who was not present
for the work.

## How to read this

Every finding below has the same four parts, in the same order:

1. **What the code was doing** — the behaviour, described plainly, with
   no assumption that you remember the module.
2. **Why that is wrong** — the mechanism, not just the label.
3. **What changed in the numbers** — before and after, measured.
4. **What to do about it** — whether anything is still required of you.

Findings are ordered by how much they change the answers, not by which
module they live in.

A note on tone: several of these are bugs I introduced in earlier rounds.
They are described the same way as the ones that were already there,
because for a reader checking a number the provenance of a mistake does
not matter.

---

## Summary

Ten stages of work, in four areas.

| Area | What happened |
|---|---|
| **Terrain** | DEM switched from a void-filled SRTM mosaic to void-free NASADEM; two georeferencing bugs found and fixed by cross-checking the two |
| **Aircraft** | Four invented vehicles replaced by three derived from published data, with every parameter tagged as referenced, derived or assumed |
| **Battery** | Chemistry-specific voltage curves, a validation against a cell datasheet, and three field-meaning confusions resolved |
| **Wind** | Cited Quito defaults, a two-roughness boundary layer, and Beaufort limits read conservatively |
| **Closing out** | Three questions left open at the end of the first pass, all three now answered |
| **Figures** | Four defects a reader found, a missing altitude panel, per-flight figures for every case and pair, and a wind figure that says something about a route |

Test count went from 131 to 166, and the suite passes with warnings
promoted to errors.

**The single most consequential finding** is not in that table. It is
that the corridor profile was reporting clearance and ceiling numbers
computed from a reference path the aircraft does not fly. See Finding 1.

**The most surprising** is Finding 15: replacing an assumed battery
curve with a measured one changed the endurance by *nothing*, because
the energy budget never consulted the curve. Two independent statements
of the same quantity, and only the one with no chemistry in it reached
the answer.

---

## Finding 1 — The corridor was checking a path the aircraft does not fly

**What the code was doing.** `build_corridor_profile` computes a dense
terrain-following reference — the lowest altitude the aircraft can hold
while staying the target height above ground, given its climb and descent
angles. It then simplifies that reference to about a dozen breakpoints,
because each breakpoint becomes a flight segment and segments cost
evaluation time. It then reported the profile's clearance and ceiling
figures **from the dense reference, not from the simplified polyline.**

**Why that is wrong.** Those are two different paths. The aircraft flies
the polyline; the reference is what the polyline was fitted to. Between
breakpoints the polyline is a straight line in altitude while the ground
rolls underneath, so it can cut below the corridor floor and bulge above
the ceiling — and the report, reading the reference, said neither had
happened.

The gap was small on the 186 m grid the package used to ship. On 30 m
terrain it was not, and there was a second mechanism making it worse:
when the breakpoint budget could not be met, the simplification tolerance
was relaxed by a factor of 1.6 up to twelve times — a permitted vertical
error of two kilometres against a corridor 62 m tall.

**What changed in the numbers.** On Espejo → Guamaní, the flown path
oscillated to **165 m AGL over flat ground and dipped to 40 m where the
floor is 60 m**, while the profile reported zero violations. After the
fix that leg flies 0–110 m AGL with no violations at all.

Three changes: simplification now runs a repair pass that lifts
breakpoints until the flown chord never cuts below the reference, so
clearance holds by construction; every reported number is measured on the
flown polyline; and the breakpoint budget went from 12 to 24, which is
where ceiling exceedance on that corridor reaches zero. Path build cost
is 11.4 ms.

One headline number changed as a result. **2 of 15 hospital pairs admit a
legal straight corridor, not 3.** The third was passing only because the
check read the reference.

**What to do about it.** Nothing — but any corridor result quoted from
before this round should be re-run.

---

## Finding 2 — Two terrain products, two georeferencing conventions

**What the code was doing.** The GeoTIFF reader took a tile's tiepoint as
the centre of its first pixel, and assumed every tile declared −32767 as
its void marker.

**Why that is wrong.** Neither is a property of GeoTIFF; both are
properties of a particular product. `GTRasterTypeGeoKey` says whether the
tiepoint marks a pixel's *corner* or its *centre*, and SRTM v3 ships as
centre while NASADEM ships as corner. Reading both as centre misplaces
NASADEM by half a cell — 15 m at 1 arc-second — and on Andean slopes that
becomes elevation error pointing consistently downhill. Separately, SRTM
declares −32767 as its void value and NASADEM declares −32768, so one
hard-coded number cannot serve both: the wrong one leaves a −32 768 m
sentinel sitting in the grid as though it were terrain.

**What changed in the numbers.** The misregistration was found by
cross-checking the two products against each other, which is what a
cross-check is for. Before the fix they disagreed by **sd 10.2 m on cells
where both had measured data, with 24 % differing by more than 10 m**.
After, sd 3.3 m and 1.6 %. A residual +4.0 m median offset remains, which
is a vertical-reference difference between the products and well inside
both their stated accuracies.

Both the declared sentinel and a hard floor at −1000 m are now applied,
so a mislabelled tile cannot inject a canyon the planner would dive into.

**What to do about it.** Nothing.

---

## Finding 3 — The DEM has no invented cells any more, and that changed less than expected

**What the code was doing.** Shipping a DEM built from an SRTM v3 mosaic
in which **8.2 % of cells returned no radar signal** and were filled by
averaging their neighbours.

**Why that matters.** Interpolated ground is smooth and real ground is
not, and over the Andes the voids cluster on exactly the steep terrain
that threatens a low corridor.

**What changed in the numbers.** `data/dmq_dem_30m.npz` is now built from
NASADEM's bare-earth product: **zero voids** over the whole DMQ box.
Where SRTM had voids, the two products disagree by sd 107 m with a worst
case of 922 m, and **98 % of all cells differing by more than 50 m were
SRTM voids**.

And yet: straight-corridor feasibility is 2 of 15 pairs before and after,
with no verdict flipping. The reason is that the voids cluster on the
Pichincha flanks and the eastern cordillera, off to the side of most
inter-hospital corridors — along the actual routes the void fraction is
0–5 %, not 8.2 %. NASADEM is the right DEM to ship, but **not because it
changed these answers.**

The DEM also now records where it came from. `save_dem` had been writing
provenance that nothing read; `DEMInterface.metadata.provenance()` now
returns a citable line.

**What to do about it.** Nothing. `data/dmq_dem.npz` (the old 186 m grid)
was deleted; the resolution-effect figure now decimates the shipped DEM
instead, which is a better comparison anyway.

---

## Finding 4 — The vehicle files were an invented aircraft

**What the code was doing.** Shipping four vehicles — `baseline`,
`heavy_cargo`, `long_range`, `high_altitude` — whose forty-odd parameters
each were plausible numbers with nothing behind them. The factory
functions fell back to hard-coded values when a file was missing, so a
missing file produced a complete aircraft with no provenance and no sign
anything had gone wrong.

**Why that is wrong.** A performance figure computed from invented
parameters is not a result. And with everything in one undifferentiated
list, a reader has no way to tell which numbers can be checked against a
datasheet and which are somebody's judgement.

**What changed.** Three vehicles, generated by
`scripts/build_vehicles.py` from the manufacturer pages: **T-Drones VA17**
(4.3 kg tilt-rotor), **T-Drones VA23** (12.5 kg quadplane, the default)
and **Alafija VA25** (13 kg quadplane). Every parameter carries a tag:

- **referenced** — quoted from the page
- **derived** — computed from referenced values by a stated relation
- **assumed** — judgement, with the basis given

readable at runtime as `ac.provenance`, and printable with
`python -m examples.demo_vehicle --provenance va23`. The factories now
raise rather than inventing an aircraft.

**The check that makes this more than bookkeeping.** The endurance and
cruise-speed claims were deliberately held *out* of the derivation, so
they could serve as an independent test of it. Run through the full pack
model on cruise power, against the published endurance: **VA23 1.24×,
VA25 0.76×, VA17 1.58×**. The first two straddle the claim as expected
once VTOL phases are counted, which cost far more than cruise. The VA17
does not, and the build script says so rather than my tuning it away —
120 minutes from a 418 Wh pack on a 4.3 kg airframe is either a
conservative claim or a pack smaller than the listing states.

**What to do about it.** If you have measured data for any of these
aircraft, the `assumed` entries are the list of things to replace, in
provenance order.

---

## Finding 5 — `m_tow` meant two different things

**What the code was doing.** The field is documented as "total take-off
mass". `with_payload()` *adds* payload to it.

**Why that is wrong.** Both readings are defensible and they are
incompatible. A file holding the datasheet MTOW put every loaded mission
over MTOW, with hover power, stall speed and the entire derived speed
envelope computed for an aircraft heavier than any that can legally fly.
Nothing detects this: the numbers that come back are perfectly
reasonable-looking numbers for the wrong aeroplane.

**What changed.** `m_tow` now explicitly holds the mass *without*
payload. `m_tow_max` records the certificated ceiling separately,
`payload_capacity_kg` the quoted payload, and `validate()` checks the
three close. Exceeding MTOW raises a warning naming the overload.

**What to do about it.** Any vehicle JSON you wrote by hand needs
checking — if its `m_tow` was the datasheet MTOW, it is now the empty
mass and the aircraft is heavier than you meant.

---

## Finding 6 — The manoeuvre limits contradicted each other

**What the code was doing.** `UAVConfig` carried `min_turn_radius`,
`max_bank_angle`, `fw_cruise_airspeed`, `fw_max_climb_angle` and
`fw_max_climb_rate` as five independent constants.

**Why that is wrong.** They are not independent. Turn radius, bank angle
and speed are three names for two facts: `R = V²/(g·tan φ)`. Climb angle,
climb rate and speed likewise: `V_y = V·sin γ`. Typed in separately they
drift, and the drift is invisible because **different parts of the
planner read different members of each set** — the corridor envelope uses
the climb *angle*, the segment builder uses the climb *rate*. The two
halves of the planner were flying different aircraft.

**What changed in the numbers.** The shipped defaults were internally
inconsistent by:

| quantity | declared | implied by the others |
|---|---:|---:|
| minimum turn radius | 100 m | **159 m** at 30° bank, 30 m/s |
| max climb rate | 5.0 m/s | **7.8 m/s** at 15°, 30 m/s |
| max descent rate | 4.0 m/s | **8.7 m/s** at 12°, 42 m/s |

`UAVConfig.for_vehicle()` now derives all of them from the aircraft, and
`manoeuvre_problems()` reports any set that disagrees. The historical
constants remain as the bare `UAVConfig()` defaults and are used by
`examples/demo_vehicle.py` to demonstrate the failure.

**What to do about it.** Use `UAVConfig.for_vehicle(ac, altitude=...)`
rather than `UAVConfig()`. The latter is now effectively a
demonstration of what not to do.

---

## Finding 7 — The climb angle was about 50 % steeper than the propulsion supports

**What the code was doing.** `fw_max_climb_angle = 15.0` degrees, typed
in, independent of the aircraft, the payload and the air density — at a
point in the code where the corridor envelope uses it to decide whether
terrain can be followed at all.

**Why that is wrong.** Steady climb is a power balance. With the motor
sized at a multiple of cruise power, it collapses to
`sin γ = (ratio − 1)/(L/D)` — the climb angle is set by lift-to-drag and
installed power margin, and by nothing else. A climb angle assumed
steeper than the propulsion supports makes corridors look flyable that
are not.

**What changed in the numbers.** Derived at 2 800 m and MTOW, with the
motor at 2.5× cruise power: **VA17 8.7°, VA23 10.7°, VA25 9.7°**, against
the hardcoded 15°. Descent, derived as the steepest idle glide the speed
envelope allows, comes out at 9.1–11.8° against the hardcoded 12° — so
descent was roughly right and climb was not.

Straight-corridor feasibility still counts 2 of 15 pairs, but severity
rises sharply on the legs that climb into the Pichincha flanks:

| corridor | with 15° | derived |
|---|---|---|
| Espejo → Lloa | 112 m over 30 % of the leg | **208 m over 59 %** |
| P.A. Suárez → Lloa | 295 m over 51 % | **434 m over 65 %** |

**What to do about it.** The 2.5× power ratio is the single assumption
this rests on and is the first thing to replace against a real motor
curve. It is `AircraftEnergyParams.CLIMB_POWER_RATIO`.

---

## Finding 8 — The VTOL descent rate had drifted into the vortex-ring band

**What the code was doing.** `vtol_max_descent_rate = 2.0` m/s, a
constant.

**Why that is wrong.** A descending rotor re-ingests its own wake between
roughly 0.25 and 1.5 times the hover induced velocity, and inside that
band momentum theory does not merely lose accuracy — it inverts, and
thrust can collapse. Where the band starts depends on disk loading, so it
moves with the aircraft.

**What changed in the numbers.** 2.0 m/s was outside the band for the old
12 kg aircraft on 0.82 m² of rotor disk and **inside** it for the VA23's
10 kg on 0.98 m² — the safer aircraft, judged by the wrong number. The
rate is now derived at 0.8 × the band's lower edge, which for the VA23 at
2 800 m is 1.50 m/s.

**What to do about it.** Nothing.

---

## Finding 9 — One battery curve was serving every chemistry

**What the code was doing.** A single tabulated LiPo open-circuit-voltage
curve, used for every pack.

**Why that is wrong.** A LiPo falls steadily from 4.2 V; a high-nickel
Li-ion holds a long plateau and then drops in the last fifth. The plateau
is what decides how much of the pack sits above the cut-off, which is to
say how much of it is real.

**What changed, and what I could not get.** I could not find a citable
OCV curve for a high-nickel 21700. I found two published curves and
rejected both, which is worth recording:

- **Chen & Rincón-Mora (2006)** is a LiCoO₂ polymer cell charged to
  4.1 V. Rescaling its curve onto a 2.5–4.2 V window puts the pack's mean
  voltage about 0.2 V too high — **a 5 % energy overestimate hiding
  inside a citation.**
- The unified-OCV model of Weng et al. is LiFePO₄, a different shape
  again.

So the Li-ion curve is an NMC-typical *shape* anchored on three published
Molicel INR21700-P42A numbers (4.2 V charged, 2.5 V cut-off, 4.2 Ah /
15.12 Wh typical), and it is labelled as exactly that. The
Chen–Rincón-Mora equation remains in the module, cited, for polymer
packs, with a comment explaining why it is not the default.

**The check that can fail.** A datasheet gives capacity in amp-hours and
energy in watt-hours, and their ratio is a mean voltage the curve has to
reproduce. The Li-ion curve lands within **0.9 %** of the P42A's 3.600 V.
A test confirms the LiPo curve *fails* the same check, so the check is
testing something.

Internal resistance now follows the SOC dependence from the same paper —
flat above 20 % charge, climbing steeply below. In practice a flight
never reaches the steep part, so the effect is about 11 % more resistance
at mission end, acting exactly where margin is thinnest.

**What to do about it.** If a pack comes in hand, `fit_battery()` will
replace all of this from a discharge log, and `battery_provenance()`
reports which numbers are measured and which assumed.

---

## Finding 10 — Three more fields that meant two things

Three instances of the same shape as Finding 5, all in the pack model.

**`cell_capacity_ah` held the pack's capacity.** That field divided into
pack capacity is the parallel-string count, and the parallel count
divides the pack's internal resistance. Holding the pack figure collapsed
the count to 1 and inflated modelled resistance by roughly the real
parallel count — about 7× for the VA23. The packs are now specified as
12S of a named cell — 5.2 parallel strings once the pack that actually
fits the airframe is chosen (Finding 13), giving 57 mΩ.

**`electrical_power_max_w` sized the pack from a single cell.** Same
field, second site, found later: C-rate multiplies the *pack's* capacity,
and reading the cell field gave a seventh of the real power. That then
became the binding constraint on the derived climb angle and **roughly
halved it** — 4.3° instead of 10.7° for the VA23.

**`SOC` meant "usable energy exhausted" to the energy budget and "cell
empty" to the voltage curve.** Reading the curve at the mission SOC put
the pack at a flat 2.5 V/cell the moment the reserve was reached, which
is a voltage no flight ever sees. There is now a separate
`charge_fraction`; at the reserve the VA23 reads 3.38 V/cell, above its
3.00 V cut-off.

`validate()` now checks all three, plus that the pack's two independent
statements of its own energy — mass × specific energy, and the OCV curve
integrated over its charge — agree to within 8 %.

---

## Finding 11 — Wind was read up the wrong boundary layer

**What the code was doing.** `WindField` had one roughness length,
applied both at the anemometer and along the corridor.

**Why that is wrong.** An airport anemometer stands on mown grass; the
corridor crosses a city. One logarithmic profile cannot describe both.

**What changed in the numbers.** Reading a 3.08 m/s mast measurement
straight up to 90 m with the corridor's urban roughness gives **6.02
m/s**. Converting properly — up from the mast on the site's own
roughness to a blending height above both surface layers, then back down
on the corridor's — gives **3.96 m/s**. A 47 % error, propagating into
every crab angle and round-trip penalty computed from it.

**Quito's defaults now carry their source, and the two sources
disagree.** A thirty-year station record gives 3.08 m/s from the NNW;
MERRA-2 reanalysis gives 1.65 m/s from the east. Both are kept — a factor
of 1.9 apart in speed and 250° in direction — because a conclusion that
survives both does not depend on which is right. The station magnitudes
are the default, being a real measurement; direction is a parameter to
sweep.

**Beaufort ratings are bands and are read at the bottom.** "Wind
resistance level 5" is 8.0–10.7 m/s. An aircraft rated to force 5 has
been shown to handle 8.0, not 10.7. VA17 and VA23 come out at 8.0 m/s,
VA25 at 10.8.

**What to do about it.** Nothing, but note that the wind picture is the
weakest-evidenced part of this model. Two sources a factor of two apart
is not a climatology.

---

## Finding 12 — Checks that nothing called

The audit this round looked specifically for one shape, because four
earlier bugs shared it: **a correct mechanism that nothing exercised, or
a partial check standing in for a complete one.**

**Six envelope validators were dead code.** `UAVConfig` has carried
`validate_fw_airspeed`, `validate_fw_climb_angle`,
`validate_fw_descent_angle`, `validate_vtol_climb`,
`validate_vtol_descent` and `validate_altitude` since the package was
written, and **nothing called any of them.** The path builder constructs
segments from those same limits, so in practice the results came out
inside the envelope — but that was a property of the builder, not a
checked fact, and it would have gone on being true silently right up
until it stopped. `raptor.terrain.check_path_envelope()` is the call site
they were missing. Run against the current planner it comes back clean,
which is now a verified fact rather than an accidental one.

**`max_path_segments` was declared, documented, and read by nothing.** It
was 20. A terrain-following leg over 30 m terrain is forty-one segments,
so the constraint was both unenforced and wrong. It is now a cost bound
with a realistic default, reported rather than imposed — too many
segments makes the search slow, not the route illegal.

**`AStarResult` provided `get_waypoints_array` but not `get_waypoints`.**
It was written to satisfy the array interface, and the profile plotting
reached for the object form — so a figure comparing A* against
differential evolution silently accepted one planner's output and
rejected the other's, which is exactly the comparison the figure exists
to make.

**`build_polar` attached a drag table without updating the scalar
`C_L_max`** that `stall_speed_at` actually reads. All three aircraft were
computing their speed envelopes from the default wing. There was even a
warning about the mismatch in `check_flight_envelope` — a warning where an
update belonged.

**`to_dict` dropped four fields `from_dict` reads.** Saving a vehicle and
loading it back returned a different aircraft: no rotor diameter, no
separate VTOL and cruise efficiencies, and no airframe geometry to
compute drag from.

**`DEMInterface` flipped the elevation grid for a descending latitude
axis but not the void mask**, so a void report on such a DEM pointed at
the wrong ground.

---

## What did not change, and why

**Temperature and ageing are still out of the battery model.**
Deliberately — you asked for them to stay out, and it is worth naming
what that costs. The pack is modelled at one temperature, and Quito's
diurnal range at 2 800 m is enough to move both capacity and internal
resistance by several per cent. A cold dawn departure has less pack than
the model gives it.

**Wind is steady.** No gusts, no turbulence, no diurnal cycle. The
`gust_factor` field feeds the margin check, but the field itself does not
vary in time. Given that Quito's two wind sources disagree by a factor of
1.9 on the *mean*, modelling the variance would be false precision.

**Two of the three rules RDAC 101 states that RAPTOR cannot check are
still unchecked** — separation from persons and structures, and
visibility. They now appear in every compliance report as explicitly
untested (Finding 17), which is as far as this can be taken without a
building layer and a weather feed.

## One result worth flagging on its own

The permit-sensitivity study sweeps the weight the optimiser puts on
crossing airspace that needs an authorisation. At a weight of 0.50 it
found a route with **zero permits** — and put seventeen waypoints below
the 60 m corridor floor to get one.

It traded a safety constraint for a regulatory one, and a report showing
only the permit count would have made that row look like the best result
in the table.

This is exactly what the `safe` / `flyable` distinction added in an
earlier round exists to catch, and it is the first time the machinery has
caught something rather than merely being correct. It also argues against
the obvious way to use the package: **tuning a penalty weight until the
compliance summary looks good is a way to produce a route nobody should
fly.** The full table is in `docs/CASES.md`, Case 3.2.

---

## What a reader should check first

If you are picking this up to verify something, these are the four
places where a mistake would matter most, in order:

1. **`scripts/build_vehicles.py`** — the `assumed` provenance entries.
   Aspect ratio, rotor diameter, the specific energy and the climb power
   ratio are the four that propagate furthest.
2. **`raptor/corridor.py`, `_simplify`** — the clearance-repair pass.
   Everything about corridor feasibility rests on the flown polyline
   never dipping below the reference, and that property is enforced
   there and nowhere else.
3. **`raptor/battery.py`, `_NMC_CELL_V`** — the OCV table. Measured now
   (Finding 14), and since Finding 15 it drives the pack's energy
   rather than only its voltage, so it is the input to every endurance
   number in a way it was not before.
4. **`raptor/wind.py`, the Quito block** — two sources a factor of two
   apart, with the choice between them stated rather than resolved.

Each has a test that will fail if it is changed carelessly, but a test
only checks the property it was written for.

---

# Part 2 — Closing the three open questions

The first pass ended with three things unresolved. This part is what
happened when each was pushed on. All three turned out to be answerable,
and two of them exposed further bugs on the way.

---

## Finding 13 — The VA17's endurance: not an anomaly, a mass budget

**The open question.** The pack model outlasted the VA17's published
120-minute endurance by 1.58×. The VA23 and VA25 straddled their claims;
the VA17 did not, and I could not say why from published data.

**What was actually wrong.** Two things, neither of them about the VA17.

The first is that **battery mass was cell mass.** The pack's watt-hours
were divided by a specific energy of 220 Wh/kg to get its mass, and
220 Wh/kg is roughly what a high-power 21700 achieves *on its own*.
Interconnects, BMS, balance leads, connectors and case are 10–20 % on a
real pack, and they were simply absent. Adding them is not a
refinement — leaving them out is not neutral, it is an optimistic choice
dressed as an omission.

The second is that **nothing checked the aircraft added up.** A vehicle
definition can be internally consistent and describe an aircraft that
cannot be built: the manufacturer publishes an empty mass *and* a pack,
and if pack plus airframe plus propulsion exceeds the empty mass, one of
the published figures is wrong. Every component is plausible on its own,
so nothing noticed.

**What changed in the numbers.** With pack overhead counted, none of the
three aircraft could carry its largest listed pack inside its own
published empty mass:

| aircraft | largest listed pack | largest that fits | shortfall |
|---|---:|---:|---:|
| VA17 | 14.5 Ah | 11.6 Ah | −0.44 kg |
| VA23 | 30 Ah | 25.8 Ah | −0.96 kg |
| VA25 | 30 Ah | 26.9 Ah | −0.62 kg |

The VA23 result is the checkable one. **T-Drones lists 22, 27 and 30 Ah
options, and the mass budget says the 30 Ah does not fit.** The shipped
VA23 now carries the 22 Ah pack — the largest *listed* option that
closes — and the other two carry the capacity the budget allows, marked
as derived rather than referenced.

**And then the VA17 stopped being an anomaly.** Its gap fell from 1.58×
to 1.32× on the smaller pack. But the rest of the explanation was a
second error of mine: I had been comparing the model at the
manufacturer's *cruise* speed, and an endurance figure is not quoted at
cruise. A fixed wing loiters well below the speed it cruises at, and
neither manufacturer states which speed their endurance figure uses.

Comparing against the whole band instead — from cruise-speed endurance
to minimum-power endurance — every claim is bracketed:

| aircraft | claim | at cruise speed | at minimum-power speed | verdict |
|---|---:|---:|---:|---|
| VA17 | 120 min | 158 min | 181 min | below the band — conservative |
| VA23 | 180 min | **170 min** | **199 min** | inside the band |
| VA25 | 210 min | **146 min** | **260 min** | inside the band |

**Two of three claims fall inside the band, and the third is
conservative.** The VA17 was never the outlier; the comparison was.

**The cross-check that survives.** The published endurance was
deliberately held out of the derivation so it could test it. Inverting
the VA23's claim gives an implied cruise propulsive efficiency of
**0.76**, against the 0.72 assumed from first principles. That agreement
is the strongest single validation in the package, because the two
numbers come from completely different places — one from a datasheet's
endurance claim, the other from an assumption about propeller design.

**What to do about it.** `PACK_OVERHEAD_FRACTION` is 0.15 and is an
assumption. A photograph of any of these aircraft's battery bays would
settle both it and the pack question.

---

## Finding 14 — A measured battery curve, and a validation that could not fail

**The open question.** I could not find a citable open-circuit-voltage
curve for a high-nickel 21700, so the shipped curve was an NMC-typical
*shape* anchored on datasheet endpoints.

**What I found.** A CC BY 4.0 dataset of C/30 charge–discharge
characterisations of the Samsung INR21700-30T — a high-power NMC 21700
of exactly the right class:

> P. Pillai, S. Sundaresan, P. Kumar, K. Pattipati and B. Balasingam,
> "Open-Circuit Voltage Models for Battery Management Systems: A
> Review", *Energies* 15(18), 2022.
> Dataset doi:[10.17632/fywnpsjfpc.1](https://doi.org/10.17632/fywnpsjfpc.1)

At C/30 the terminal voltage is the open-circuit voltage to within a
couple of millivolts, so the curve comes straight out. It is extracted
by `scripts/extract_ocv.py`, which downloads the data and rebuilds the
table, so the number in the source can be checked against its origin.

**A bug found in the extraction, worth recording.** The dataset has four
cells. Averaging all four gives a curve 100 mV higher than averaging
three, because one file returns **twice as much charge as it delivers** —
its two branches are not halves of one cycle. A coulombic-efficiency
check catches it and the script excludes it, saying why. The remaining
three agree to **1.3 mV in mean open-circuit voltage** and recover
3.020–3.030 Ah against the cell's 3.000 Ah nominal, which is about as
tight a corroboration as three independent cells can give.

**What changed in the numbers.** The assumed curve was systematically
**0.10–0.16 V low between 50 % and 80 % state of charge** — the part of
the curve a mission actually spends its time on.

**And the validation that had been passing was not testing anything.**
The datasheet check compared the curve's mean *open-circuit* voltage
against the datasheet's watt-hours divided by amp-hours — which is a
mean *terminal* voltage, measured under load. Those differ by one `I·R`.
The old curve passed at +0.9 % precisely because it was too low by
roughly the drop the check was ignoring.

**A validation that omits a term can be satisfied by an error of the
same size, and will then certify the very thing it was meant to catch.**
The check now takes the discharge current and the resistance, compares
at the terminals, and says which basis it used. The LiPo curve still
fails it, so it still discriminates.

---

## Finding 15 — The battery curve reached nothing

**What the code was doing.** Pack energy came from
`battery_mass × specific_energy`. State of charge advanced by dividing
energy drawn by that figure. The open-circuit-voltage curve set the
terminal voltage, and from it the current and the `I²R` loss — and
nothing else.

**Why that is wrong.** It means the pack held two independent statements
of its own energy, and the one that reached every answer had no
chemistry in it. Swapping the measured Li-ion curve for the generic LiPo
one — curves whose mean voltages differ by 2 % and whose shapes differ
completely — changed the mission energy by **0.00 %** and the endurance
by **0.00 %**. Only the ohmic loss moved, by 0.9 % of a term that is
itself 2 % of the total.

That is the whole of Finding 14 landing on nothing. It would have gone
on being true silently, and every conclusion drawn about batteries would
have been drawn from a specific-energy figure that is itself an
assumption.

**What changed.** `battery_energy_wh` now comes from the curve — mean
open-circuit voltage × cells in series × capacity — whenever the
chemistry, cell count and capacity are known, and from mass and specific
energy only as a fallback. `validate()` checks the two routes agree, and
flags a pack claiming a specific energy above what its own cells reach.

Endurance at 900 W is now 57 min on the measured Li-ion curve against
58 min on the LiPo one. Small, but it is no longer zero, and it moves for
the right reason.

**The general lesson, which is the same one as Finding 12 in a different
costume.** When one quantity has two sources, work out which one the
answer actually flows through. It is not always the one the work went
into.

---

## Finding 16 — Payload sequencing, settled by not re-planning

**The open question.** Does it matter whether a multi-stop mission
carries its mass early or late? Two earlier attempts gave opposite
answers, and a five-seed repeat gave 511.8 ± 13.3 Wh against
517.6 ± 15.3 Wh — a 6 Wh effect inside a 15 Wh spread.

**What was wrong with the experiment, not the code.** Collection and
supply visit the same facilities in the same order and differ *only* in
what each leg carries. Planning them separately put the optimiser in the
loop twice, and its run-to-run spread swamped the effect being measured.
More compute would not have fixed it; more seeds would have narrowed the
error bar on a quantity that was still partly optimiser noise.

**The fix is to stop re-planning.** Fix the route once, then fly both
payload schedules down it. The optimiser appears once instead of twice,
its noise is common to both arms and cancels exactly, and the result is
deterministic.

**What came out.**

| route flown | payload order | energy | SOC min |
|---|---|---:|---:|
| collection's route | collection | **527.37 Wh** | 37.4 % |
| collection's route | supply | 533.32 Wh | 36.7 % |
| supply's route | collection | **520.14 Wh** | 38.3 % |
| supply's route | supply | 521.76 Wh | 38.1 % |

**Within a fixed route the comparison is exact** — collection cheaper by
5.95 Wh on its own route, 1.62 Wh on the supply tour's. No seeds, no
spread.

The 4.3 Wh between those two figures is not noise in the comparison; it
is the two routes being genuinely different. It is also **the same size
as the effect**, which is its own finding: which route the optimiser
settles on matters about as much as which way round the payload is
carried.

**But the mechanism is not the one I had written down, and it is the
more interesting half.** I had explained an earlier version of this in
terms of state of charge — a collection round carries its heaviest load
on its last leg, which is also its lowest charge. That explanation
predicts collection should be *worse* for margin. It is better. The
explanation was wrong.

What actually drives it:

| leg | km | net climb | cost of payload |
|---|---:|---:|---:|
| Espejo → Lloa | 11.5 | **+234 m** | **0.899 Wh/kg/km** |
| Lloa → Píntag | 30.3 | −230 m | 0.248 Wh/kg/km |
| Píntag → Espejo | 23.8 | −4 m | 0.258 Wh/kg/km |

**Carrying a kilogram costs 3.6× more on the climbing leg than on the
descending ones**, and collection wins because it flies the climb empty
and picks up on the way back down. Potential energy accounts for about a
third of the gap — lifting mass up a hill buys energy the descent does
not give back — and the rest is that a climbing leg spends longer at
climb power.

**So the rule is not "carry heavy early" or "carry heavy late". It is:
carry the mass on the legs that descend.**

**And the mechanism makes a prediction that holds.** If it is terrain
rather than mission shape, the answer should *reverse* over facilities
with no relief between them. Run the same two shapes over Pablo Arturo
Suárez and Docente Calderón, both on the basin floor, and it does:
267.5 Wh for collection against 264.5 Wh for supply — supply wins.

Same aircraft, same code, same mission shapes, opposite answers. That is
why two earlier versions of this study disagreed with each other and why
neither was wrong: they were asking a question whose answer is a property
of the terrain, and reporting it as a property of the mission.

That generalises past this package. In flat country payload ordering is
close to irrelevant, because every leg costs the same per kilogram. In
mountains it is a first-order routing decision, and it is invisible to
any planner that treats the legs as independent.

---

## Finding 16b — Three bugs in six lines of the mission planner

Found while re-running the cases against the corrected aircraft: the
mission's reported energy and its reported state of charge did not
reconcile. A 63.6 Wh delivery was draining 92 Wh from a pack. Three
separate faults, in the same six lines.

**The planner tracked state of charge against a hard-coded 600 Wh
pack.** `MissionPlanner.__init__` took `battery_wh=600.0` and used it
for every SOC calculation, whatever aircraft was flying. On the VA23's
867 Wh of usable energy that overstated depletion by 45 %. It is the
same shape as Finding 15 — two sources of truth for the pack's capacity,
and the one that reached the answer had nothing to do with the aircraft.

**The planner's energy re-analysis dropped the payload.** The optimizer
was given `payload_kg` and the re-analysis called
`analyze_path_energy(fp, self.ac)` with the *empty* aircraft. The two
disagreed by 15 % on a 2 kg delivery, and the figure the mission
reported was the one computed for an aeroplane carrying nothing. This is
the payload-propagation bug from an earlier round, resurfacing one level
up: the optimizer had been fixed and the planner had not.

**State of charge advanced on energy delivered, not energy drawn.** The
`I²R` heat comes out of the same pack and never reaches the drivetrain,
so charging the mission only for what arrived left the pack fuller than
it was.

**What changed in the numbers.** Two published results moved, and one of
them was a headline:

| result | before | after |
|---|---|---|
| hub-and-spoke without a spare pack | 6.5 % SOC — **battery exhausted** | **27.8 % SOC — feasible** |
| holding hospital permits | *more* expensive than not holding them | 18 % cheaper, as it should be |

The first is the one to be uncomfortable about. "Identical routing,
identical energy, and one of them runs the pack flat" was a good line and
it was an artefact. What survives is weaker and still worth saying:
identical energy, and 28 percentage points of margin between them.
Ground infrastructure buys margin on this mission rather than deciding
whether it exists — extend the tour or fly it on a cold morning and it
decides.

The second was visible as a contradiction and I had flagged it as
optimiser noise rather than chasing it. It was not noise. **A result that
points the wrong way physically is worth more attention than I gave it**,
even when the magnitude is small enough to be explicable.

---

## Finding 17 — Rules named but never tested

**What the code was doing.** `RegulatoryProfile` carried
`min_separation_persons_m` (30 m), `min_separation_buildings_m` (30 m)
and `min_visibility_m` (5 km), from RDAC 101.160 and 101.215. Nothing
read any of them.

**Why that is worse than it looks.** These *cannot* be checked here.
Testing separation from persons and structures needs a building footprint
and a crowd layer; visibility needs a weather feed. RAPTOR has terrain
and airspace zones.

But an unenforceable rule silently absent from a report is
indistinguishable from a rule that passed. A reader seeing
`Compliance: COMPLIANT` has no way to know that two rules in force were
never examined.

**What changed.** `RegulatoryProfile.unenforceable_rules()` names them,
`ComplianceAssessment` carries them, and the filing summary prints them
under a heading that says what they are:

```
Rules in force that this assessment did NOT test:
  · RDAC 101.160: 30 m from uninvolved persons and 30 m from structures
    in populated areas — needs a building and crowd layer; not modelled.
  · RDAC 101.215: 5 km minimum flight visibility — a weather minimum;
    not modelled.
A clean verdict above covers the rules RAPTOR can see, and not these.
```

The limitation has not gone away. It is now declared, which is the
difference between a limitation and an oversight.

---

## Finding 18 — Smaller things from the final sweep

**`MissionConstraints.max_range` was declared and read by nothing.** A
route could be twice the aircraft's stated range with no part of the
model saying so. Now checked by `check_path_envelope`.

**`assess_compliance` had an argument-order trap.** Its fourth positional
is `regulation`; `constraints` is fifth. Both are objects a caller has to
hand, the wrong order is the natural one to write, and passing
constraints as the regulation failed thirty lines later with a
missing-attribute error that named neither argument. It now raises
immediately and says which keyword to use. I know the trap is real
because I fell into it writing a test for Finding 17.

**Checks that were run and found nothing.** Worth recording, because a
clean result from a real check is information:

- Every power relation against an independent expression: hover as the
  limit of ascent and descent, climb and descent at zero flight-path
  angle equalling cruise, monotonicity in mass and altitude, the
  vortex-ring band bracketing the hover induced velocity. All hold.
- Nine internal-consistency identities on a planned route — segment
  lengths against waypoint hops, durations against the energy model's
  total, altitude bookkeeping against endpoint difference, state of
  charge against energy divided by usable energy. All agree to floating
  point.
- Sign conventions: bearings, meteorological wind direction, and that a
  tailwind raises ground speed while a headwind lowers it.
- No mutable default arguments, no bare `except`, no `TODO` or `FIXME`
  left in the package.

---

## Finding 19 — Time-constrained missions were unreachable

**What the code was doing.** `MedicalUrgency` defines three levels with
a time cap and a state-of-charge floor each — routine 60 min and 15 %,
urgent 30 min and 15 %, emergency 15 min and 20 %. `OptPriority` picks
between energy, time and balanced objectives. `plan_mission` read the
priority, mapped it to an optimizer mode, and **passed neither the time
cap nor the SOC floor to the optimizer.**

**Why that is wrong.** An EMERGENCY mission planned exactly like a
routine resupply. The whole time-constrained half of the model — the
part the original brief specifically asked for — existed and was
unreachable through the entry point the missions actually use. Nothing
failed; the results were simply the routine ones wearing an urgency
label.

`optimize_scenario` did use them, which is why this survived: the
machinery was exercised somewhere, just not on the path anything called.

**What changed in the numbers.** The same delivery, Espejo → Píntag at
2.0 kg, now behaves differently under different objectives:

| objective | urgency | cap | energy | flight time | SOC at end |
|---|---|---:|---:|---:|---:|
| energy | routine | 60 min | **169.6 Wh** | 16.2 min | 80.0 % |
| time | emergency | 15 min | 224.0 Wh | **15.0 min** | 73.4 % |

**Hurrying costs 32 % more energy to save 7 % of the time**, and arrives
with 6.6 points less charge. Before this fix the two rows were identical.

**What to do about it.** Nothing, but note that this is the third time a
figure or a study has turned out to be reporting something other than
what its label said. The pattern in Finding 12 is not exhausted.

---

## Finding 20 — Figure defects, found by a reader

Four, all raised by someone reading the figures rather than the code.
Recorded because the *kind* of defect is instructive: none of them was
a wrong number, and all four made a correct number unreadable.

**The red dots had no legend entry.** The corridor sheet marks points
above the 122 m ceiling in red. The series is labelled in the plotting
code — but the shared figure legend was built from panel 0's handles,
and panel 0 is the one corridor with *zero* exceedance, so the series
never existed there. Five of six panels showed unexplained dots. The
legend is now collected across all panels.

**The deviation panel was on a different x-axis from the profiles.**
Distance along the direct line, against distance flown for everything
else. A detour is longer than the line it leaves, so a reader lining the
panels up vertically was comparing points that are not the same point.
All three profile panels now share one axis.

**There was no altitude panel at all.** Height above ground answers the
regulatory question and is the wrong frame for picturing a flight: a
constant-altitude cruise appears in it as an inverted terrain profile,
which reads as violent manoeuvring when the aircraft is holding one
number. That is exactly how it was misread. The figure now carries
altitude AMSL as well, and map plus altitude is enough to reconstruct
the flight.

**The corridor sheet said "origin" and "destination" on every panel.**
Which says nothing on a sheet where every panel has one of each. They
are named now.

**And one thing that was not a figure defect but reads like one.** A\*
appears in the comparison sitting neatly inside the corridor band while
the differential-evolution routes sometimes leave it. That is not A\*
being better: its grid nodes only exist between terrain + 50 m and
terrain + 120 m, so **it cannot represent a ceiling violation.** It also
carries no turn-radius limit, no climb-angle limit, a fixed cruise speed
and a single propulsive efficiency, and it returns waypoints rather than
flight segments — so its energy is not comparable with the others'
either. The comparison is still worth drawing and it needed the caveat
stated, which it now is, in `docs/CASES.md`.

**A gap the same question exposed.** There is no turn segment in the
model. `SegmentType` has six members and none is a turn; heading changes
at waypoints are instantaneous; and `min_turn_radius` — derived this
round from bank angle and cruise speed — is computed, reported by
`manoeuvre_problems()`, and **used by nothing**. So the lateral detours
in every plan view are polylines with corners the aircraft could not
fly, and the energy of turning is uncounted. It is the Finding 12
pattern again, and I missed it because I checked whether the value was
*consistent* rather than whether anything *consumed* it.

---

# What this round means for using the package

Twenty findings is a lot to hold at once. Four of them change how
results should be read.

### Corridor verdicts are now measured on the flown path

Before Finding 1, a corridor could be reported as fitting the legal band
while the path the aircraft would actually fly left it. Any corridor
result from before this round should be re-run. The headline number
moved: **2 of 15 hospital pairs**, not 3.

### Mission-level numbers moved, and one headline reversed

Finding 16b corrected three faults in the mission planner's accounting.
A published result flipped: the hub-and-spoke round without a spare pack
was reported as running the battery flat, and does not — it finishes at
27.8 % with the pack the aircraft actually carries. The finding survives
in weaker form and the earlier statement should not be quoted.

### The aircraft are smaller than their datasheets suggest

Findings 13 and 15 together moved the VA23 from a 30 Ah pack at
1296 Wh to a 22 Ah pack at 985 Wh — a **24 % reduction in energy** — on
the strength of a mass budget and a measured voltage curve. Every mission
energy, every state-of-charge margin and every endurance figure in this
package is smaller than it was, and closer to what the airframe can
actually carry.

If you have flown any of these aircraft, the pack question is the single
most valuable thing you could check.

### "Compliant" means compliant with the rules RAPTOR can see

Finding 17 makes this explicit in every report rather than leaving it to
be inferred. Two rules of RDAC 101 in force over Quito are named but not
tested, and they are the two that would matter most in a dense urban
approach.

### The interesting results are about terrain, not about the aircraft

The three findings a reader is most likely to carry away all turn on
relief rather than on hardware:

- **Distance predicts almost nothing about corridor feasibility.** A
  5.3 km leg fails over 56 % of its length; a 23 km one misses by 11 m.
  What matters is relief per kilometre.
- **A coarse DEM biases corridor verdicts optimistic**, one-sidedly and
  on every corridor, because averaging removes ridges and never adds
  them.
- **Payload ordering is a terrain question.** Carrying a kilogram costs
  3.7× more on a climbing leg than a descending one, so the rule is
  carry the mass downhill — and in flat country the question does not
  arise at all.

That pattern is worth stating on its own. The package was built to
optimise paths for an aircraft, and most of what it has to say is about
the ground underneath it.

---

# A note on how these were found

Sixteen of the twenty findings share one shape, and it is worth
naming because it predicts where the next one will be:

**a correct mechanism that nothing exercised, or a partial check
standing in for a complete one.**

The six unused envelope validators. The declared range limit nothing
read. The provenance `save_dem` wrote and no reader consulted. The
scalar `C_L_max` that `build_polar` left stale. The `to_dict` that
dropped four fields. The void mask that did not flip with its grid. The
regulatory limits named but never tested. The datasheet check that
omitted the `I·R` term and was therefore satisfied by an error of
exactly that size. And the largest of them: a battery curve, carefully
measured, reaching an answer that was computed from something else
entirely.

The hard-coded 600 Wh pack in the mission planner, and the payload it
dropped from its own energy analysis.

None of these produced an error. Every one of them produced a plausible
number. That is what makes the shape worth hunting: the failure mode is
not a crash, it is a result that looks fine and is quietly built on
something other than what you think.

The sweep that found them was mechanical — list every function and field,
count the references, look at anything mentioned once. It is cheap, it
can be re-run in a minute, and on this package it has now produced
findings three times running.

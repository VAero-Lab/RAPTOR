"""
UAV Path Planning Framework for eVTOL Medical Delivery
=======================================================

A modular framework for parameterizing, building, optimizing, and
visualizing 3D flight paths over real terrain (DEM) for eVTOL UAV
operations. Designed for any region — supply your own DEM, facility
coordinates, and airspace zone definitions.

Includes a case study for the Quito Metropolitan District, Ecuador.

Modules:
    config         — UAV performance envelope and mission constraints
    dem            — Digital Elevation Model interface (terrain queries)
    segments       — Flight segment type definitions and parameterization
    path           — FlightPath construction from ordered segment sequences
    terrain        — Terrain clearance analysis and constraint evaluation
    builder        — Intelligent path builder (DEM-aware feasible paths)
    atmosphere     — ISA standard atmosphere model (density vs altitude)
    energy         — Power/energy models and battery SOC tracking (with stall detection)
    aero           — Drag polars from geometry (AeroSandbox surrogate)
    dem_build      — Building DEMs from GeoTIFF tiles or OpenTopoData
    regulations    — Codified UAS rules (RDAC 101) and permission classes
    compliance     — Graded regulatory standing and waiver quantification
    missions       — Mission profiles: round trip, tours, hub-and-spoke
    airspace       — 3D regulatory zone management and geofencing
    corridor       — Terrain-following reference profiles (AGL corridors)
    penalties      — Constraint handling: regulations as search pressure
    scenarios      — Medical delivery flight scenario definitions
    optimizer      — Path optimization engine (DE-based)
    mission_planner— Multi-leg mission orchestration with SOC coupling
    visualization         — Matplotlib publication-quality figures
    visualization_plotly  — Interactive Plotly/HTML figures (same API)

Author: Victor (LUAS-EPN / KU Leuven)
"""

from .config import UAVConfig, MissionConstraints
from .dem import DEMInterface, find_dem
from .segments import (
    SegmentType, FlightSegment,
    VTOLAscend, VTOLDescend,
    FWClimb, FWDescend, FWCruise,
    Transition
)
from .path import FlightPath, Waypoint, PathMetrics
from .terrain import TerrainAnalyzer, TerrainReport
from .builder import PathBuilder, FacilityNode, PathStrategy
from .atmosphere import isa_density, isa_temperature, isa_pressure
from .energy import (
    AircraftEnergyParams,
    BatteryModel, BatteryState,
    SegmentEnergyResult, MissionEnergyResult,
    analyze_path_energy,
    power_vertical_ascent, power_hover, power_transition,
    power_fw_climb, power_fw_cruise, power_fw_descent,
    power_vertical_descent,
)
from .regulations import (
    PermissionClass, RegulatoryProfile, OperationalContext,
    RDAC_101, MEDICAL_DELIVERY_CONTEXT,
)
from .corridor import (
    CorridorProfile, CorridorSurvey,
    build_corridor_profile, survey_corridor, climb_limited_envelope,
)
from .penalties import PenaltyWeights, PenaltyBreakdown, evaluate_penalties
from .aero import (
    AeroPolar, WingGeometry, build_polar, get_polar, parabolic_polar,
    reynolds, check_flight_envelope, aerosandbox_available,
)
from .dem_build import (
    OpenTopoDataClient, build_dem_from_geotiff, build_dem_from_opentopodata,
    corridor_bbox, route_mask, save_dem,
)
from .compliance import (
    ComplianceLevel, ComplianceAssessment, HeightWaiver, Excursion,
    assess_compliance, mission_compliance, find_excursions,
)
from .missions import (
    MissionProfile, BatteryAction, Stop, MissionPlan,
    one_way, out_and_back, sample_collection, supply_tour,
    hub_and_spoke, shuttle,
)
from .airspace import (
    ZoneType, AirspaceZone, AirspaceManager,
    PermissionClass as _PermissionClass,
    CircularZone, PolygonalZone,
    ZoneViolation, AirspaceReport,
    DEFAULT_PERMISSION, DEFAULT_PERMIT_FAMILY,
    build_airspace,
    load_airspace_from_file,
)
from .routed_path import RoutedPath
from .scenarios import (
    Facility, FlightScenario, MissionLeg,
    OptPriority, MedicalUrgency,
    build_scenario_catalog, get_scenario, list_scenarios,
    update_facility_elevations,
    ALL_FACILITIES, HANGARS, SUB_CENTERS,
)
from .optimizer import (
    PathOptimizer, OptMode,
    OptimizationResult,
)
from .vehicles import (
    load_vehicle_from_json, load_default_vehicle,
    list_vehicle_configs, get_vehicle,
    baseline_config, heavy_cargo_config, long_range_config, high_altitude_config,
    compare_vehicles_at_altitude, VEHICLE_CONFIGS,
)
from .visualization import (
    plot_convergence, plot_path_2d, plot_path_3d,
    plot_path_evolution, plot_energy_profile,
    plot_pareto_front, plot_scenario_dashboard,
    plot_airspace_map, plot_path_vs_ceiling,
    plot_topology_comparison, plot_constraint_budget,
    plot_mission_soc, plot_stall_envelope,
    plot_three_path_comparison, plot_vehicle_comparison,
    plot_astar_vs_de, plot_all,
)
from .visualization_plotly import (
    plot_convergence as iplot_convergence,
    plot_path_2d as iplot_path_2d,
    plot_path_3d as iplot_path_3d,
    plot_path_evolution as iplot_path_evolution,
    plot_energy_profile as iplot_energy_profile,
    plot_pareto_front as iplot_pareto_front,
    plot_scenario_dashboard as iplot_scenario_dashboard,
    plot_airspace_map as iplot_airspace_map,
    plot_path_vs_ceiling as iplot_path_vs_ceiling,
    plot_topology_comparison as iplot_topology_comparison,
    plot_constraint_budget as iplot_constraint_budget,
    plot_mission_soc as iplot_mission_soc,
    plot_stall_envelope as iplot_stall_envelope,
    plot_three_path_comparison as iplot_three_path_comparison,
    plot_vehicle_comparison as iplot_vehicle_comparison,
    plot_astar_vs_de as iplot_astar_vs_de,
    plot_all as iplot_all,
)

__version__ = "0.6.0"
__all__ = [
    "UAVConfig", "MissionConstraints",
    "DEMInterface", "find_dem",
    "SegmentType", "FlightSegment",
    "VTOLAscend", "VTOLDescend",
    "FWClimb", "FWDescend", "FWCruise", "Transition",
    "FlightPath", "Waypoint", "PathMetrics",
    "TerrainAnalyzer", "TerrainReport",
    "PathBuilder", "FacilityNode", "PathStrategy",
    "isa_density", "isa_temperature", "isa_pressure",
    "AircraftEnergyParams",
    "BatteryModel", "BatteryState",
    "SegmentEnergyResult", "MissionEnergyResult",
    "analyze_path_energy",
    "PermissionClass", "RegulatoryProfile", "OperationalContext",
    "RDAC_101", "MEDICAL_DELIVERY_CONTEXT",
    "CorridorProfile", "CorridorSurvey", "build_corridor_profile",
    "survey_corridor", "climb_limited_envelope",
    "PenaltyWeights", "PenaltyBreakdown", "evaluate_penalties",
    "AeroPolar", "WingGeometry", "build_polar", "get_polar",
    "parabolic_polar", "reynolds", "check_flight_envelope",
    "aerosandbox_available",
    "OpenTopoDataClient", "build_dem_from_geotiff",
    "build_dem_from_opentopodata", "corridor_bbox", "route_mask", "save_dem",
    "ComplianceLevel", "ComplianceAssessment", "HeightWaiver", "Excursion",
    "assess_compliance", "mission_compliance", "find_excursions",
    "MissionProfile", "BatteryAction", "Stop", "MissionPlan",
    "one_way", "out_and_back", "sample_collection", "supply_tour",
    "hub_and_spoke", "shuttle",
    "ZoneType", "AirspaceZone", "AirspaceManager",
    "DEFAULT_PERMISSION", "DEFAULT_PERMIT_FAMILY",
    "CircularZone", "PolygonalZone",
    "ZoneViolation", "AirspaceReport",
    "build_airspace",
    "load_airspace_from_file",
    "Facility", "FlightScenario", "MissionLeg",
    "OptPriority", "MedicalUrgency",
    "build_scenario_catalog", "get_scenario", "list_scenarios",
    "ALL_FACILITIES", "HANGARS", "SUB_CENTERS",
    "PathOptimizer", "OptMode", "OptimizationResult",
    "load_vehicle_from_json", "load_default_vehicle",
    "list_vehicle_configs", "get_vehicle", "VEHICLE_CONFIGS",
    "baseline_config", "heavy_cargo_config", "long_range_config", "high_altitude_config",
    "compare_vehicles_at_altitude",
    # matplotlib (publication)
    "plot_convergence", "plot_path_2d", "plot_path_3d",
    "plot_path_evolution", "plot_energy_profile",
    "plot_pareto_front", "plot_scenario_dashboard",
    "plot_airspace_map", "plot_path_vs_ceiling",
    "plot_topology_comparison", "plot_constraint_budget",
    "plot_mission_soc", "plot_stall_envelope",
    "plot_three_path_comparison", "plot_vehicle_comparison",
    "plot_astar_vs_de", "plot_all",
    # plotly (interactive HTML)
    "iplot_convergence", "iplot_path_2d", "iplot_path_3d",
    "iplot_path_evolution", "iplot_energy_profile",
    "iplot_pareto_front", "iplot_scenario_dashboard",
    "iplot_airspace_map", "iplot_path_vs_ceiling",
    "iplot_topology_comparison", "iplot_constraint_budget",
    "iplot_mission_soc", "iplot_stall_envelope",
    "iplot_three_path_comparison", "iplot_vehicle_comparison",
    "iplot_astar_vs_de", "iplot_all",
]


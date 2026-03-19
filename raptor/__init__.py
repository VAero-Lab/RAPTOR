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
    airspace       — 3D regulatory zone management and geofencing (RDAC 101)
    scenarios      — Medical delivery flight scenario definitions
    optimizer      — Path optimization engine (DE-based)
    mission_planner— Multi-leg mission orchestration with SOC coupling
    visualization  — Convergence, path, DEM, and Pareto visualization

Author: Victor (EPN / LUAS-EPN)
"""

from .config import UAVConfig, MissionConstraints
from .dem import DEMInterface
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
from .airspace import (
    ZoneType, AirspaceZone, AirspaceManager,
    CircularZone, PolygonalZone,
    ZoneViolation, AirspaceReport,
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
from .visualization import (
    plot_convergence, plot_path_2d, plot_path_3d,
    plot_path_evolution, plot_energy_profile,
    plot_pareto_front, plot_scenario_dashboard,
)

__version__ = "0.4.0"
__all__ = [
    "UAVConfig", "MissionConstraints",
    "DEMInterface",
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
    "ZoneType", "AirspaceZone", "AirspaceManager",
    "CircularZone", "PolygonalZone",
    "ZoneViolation", "AirspaceReport",
    "build_airspace",
    "load_airspace_from_file",
    "Facility", "FlightScenario", "MissionLeg",
    "OptPriority", "MedicalUrgency",
    "build_scenario_catalog", "get_scenario", "list_scenarios",
    "ALL_FACILITIES", "HANGARS", "SUB_CENTERS",
    "PathOptimizer", "OptMode", "OptimizationResult",
    "plot_convergence", "plot_path_2d", "plot_path_3d",
    "plot_path_evolution", "plot_energy_profile",
    "plot_pareto_front", "plot_scenario_dashboard",
]

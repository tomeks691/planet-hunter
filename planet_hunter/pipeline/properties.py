import logging
import math
from typing import Optional
from planet_hunter.models import StarInfo, AnalysisResult

log = logging.getLogger(__name__)

# Constants
R_SUN_M = 6.957e8       # solar radius in meters
R_EARTH_M = 6.371e6     # earth radius in meters
SIGMA_SB = 5.670374e-8  # Stefan-Boltzmann constant
AU_M = 1.496e11         # AU in meters
G_SI = 6.674e-11        # gravitational constant
M_SUN_KG = 1.989e30     # solar mass in kg


def estimate_planet_radius(depth: float, star_radius_solar: Optional[float]) -> Optional[float]:
    """Estimate planet radius in Earth radii from transit depth and stellar radius.

    depth is fractional (e.g., 0.01 for 1%).
    star_radius_solar in solar radii.
    Returns planet radius in Earth radii.
    """
    if depth is None or depth <= 0:
        return None
    if star_radius_solar is None or star_radius_solar <= 0:
        star_radius_solar = 1.0  # assume Sun-like

    # R_p / R_star = sqrt(depth)
    r_planet_solar = math.sqrt(depth) * star_radius_solar
    r_planet_earth = (r_planet_solar * R_SUN_M) / R_EARTH_M

    return round(r_planet_earth, 2)


def estimate_equilibrium_temp(
    star_teff: Optional[float],
    star_radius_solar: Optional[float],
    period_days: Optional[float],
    albedo: float = 0.3,
) -> Optional[float]:
    """Estimate planet equilibrium temperature in Kelvin.

    Uses Kepler's third law to get semi-major axis from period,
    then computes T_eq assuming uniform heat redistribution.
    """
    if star_teff is None or period_days is None:
        return None
    if star_radius_solar is None or star_radius_solar <= 0:
        star_radius_solar = 1.0

    # Semi-major axis from Kepler's third law (assume M_star ~ R_star^1 in solar units, rough)
    # a^3 / P^2 = G M / (4 pi^2)
    m_star_kg = star_radius_solar * M_SUN_KG  # very rough mass estimate
    period_s = period_days * 86400
    a = (G_SI * m_star_kg * period_s**2 / (4 * math.pi**2)) ** (1/3)

    r_star_m = star_radius_solar * R_SUN_M

    # T_eq = T_star * sqrt(R_star / (2*a)) * (1 - albedo)^(1/4)
    if a <= 0:
        return None

    t_eq = star_teff * math.sqrt(r_star_m / (2 * a)) * (1 - albedo) ** 0.25
    return round(t_eq, 0)


def compute_properties(result: AnalysisResult, star: StarInfo):
    """Fill in planet_radius and equilibrium_temp on the result."""
    result.planet_radius = estimate_planet_radius(result.depth, star.radius)
    result.equilibrium_temp = estimate_equilibrium_temp(
        star.teff, star.radius, result.period,
    )
    log.info(
        "TIC %d properties: R_p=%.2f R_earth, T_eq=%.0f K",
        result.tic_id,
        result.planet_radius or 0,
        result.equilibrium_temp or 0,
    )

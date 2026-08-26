import pytest


@pytest.fixture(scope="session")
def built_vehicle():
    """The default aircraft with both halves of its aero model attached.

    Deriving an envelope needs the airframe build-up as well as the wing
    polar; without it the L/D is roughly double the real value and every
    derived limit is wrong. Tests that exercise ``for_vehicle`` should
    use this rather than a bare vehicle.
    """
    from raptor.vehicles import get_vehicle
    return get_vehicle("va23").build_aero(verbose=False)

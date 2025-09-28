import numpy as np
import pytest

from hybridsim import RigidBody6DOF, Gravity, Drag

def make_body():
    I = 0.1 * np.eye(3)
    b = RigidBody6DOF("b", mass=5.0, inertia_tensor_body=I,
                      position=np.zeros(3), orientation=np.array([1,0,0,0], float))
    return b

def test_gravity_magnitude_and_direction():
    b = make_body()
    g = np.array([0.0, 0.0, -9.81])
    Gravity(g).apply(b, t=0.0)
    # resultant force should be m * g
    np.testing.assert_allclose(b.F, b.mass * g, rtol=0, atol=1e-12)

def test_drag_opposes_velocity_linear_and_quadratic():
    b = make_body()
    # give body some velocity
    b.v[:] = np.array([3.0, -4.0, 1.0])
    v = b.v.copy()
    # linear drag
    dlin = Drag(rho=1.0, Cd=1.0, area=1.0, mode="linear")
    dlin.apply(b, t=0.0)
    assert np.dot(b.F, v) <= 1e-12  # non-positive work
    # quadratic drag (fresh body)
    b2 = make_body()
    b2.v[:] = v
    dquad = Drag(rho=1.225, Cd=1.2, area=0.5, mode="quadratic")
    dquad.apply(b2, t=0.0)
    assert np.dot(b2.F, v) <= 1e-12

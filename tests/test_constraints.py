import numpy as np
from hybridsim import RigidBody6DOF, RigidTetherJoint

def test_distance_constraint_residual_and_jacobian_shapes():
    # two bodies 5 m apart on z
    I = 0.1 * np.eye(3)
    a = RigidBody6DOF("a", 5.0, I, position=np.array([0,0,0], float),
                      orientation=np.array([1,0,0,0], float))
    b = RigidBody6DOF("b", 5.0, I, position=np.array([0,0,5], float),
                      orientation=np.array([1,0,0,0], float))
    bodies = [a, b]
    tether = RigidTetherJoint(0, 1, [0,0,0], [0,0,0], length=5.0)
    c = tether.attach(bodies)  # DistanceConstraint
    # residual zero at perfect length
    C = c.evaluate()
    assert C.shape == (1,)
    assert abs(C[0]) < 1e-12
    # local Jacobian is (1,12) for two-body constraint
    Jloc = c.jacobian()
    assert Jloc.shape == (1, 12)

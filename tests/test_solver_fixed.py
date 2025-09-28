import numpy as np
from hybridsim import World, RigidBody6DOF, Gravity, Drag, RigidTetherJoint, HybridSolver

def test_kkt_velocity_level_satisfaction_single_step():
    # build world
    w = World(ground_z=-1e6, payload_index=0)  # avoid early stop
    I = 0.1 * np.eye(3)
    a = RigidBody6DOF("a", 5.0, I, position=np.array([0,0,1], float),
                      orientation=np.array([1,0,0,0], float))
    b = RigidBody6DOF("b", 2.0, I, position=np.array([0,0,6], float),
                      orientation=np.array([1,0,0,0], float))
    i = w.add_body(a); j = w.add_body(b); w.payload_index = i
    w.add_global_force(Gravity(np.array([0,0,-9.81])))
    a.per_body_forces.append(Drag(1.225, 1.0, 0.2, mode="quadratic"))
    b.per_body_forces.append(Drag(1.225, 1.5, 0.8, mode="quadratic"))
    # rigid tether length 5 m
    joint = RigidTetherJoint(i, j, [0,0,0], [0,0,0], length=5.0)
    c = joint.attach(w.bodies)
    w.add_constraint(c)

    # take one fixed step
    solver = HybridSolver(alpha=5.0, beta=0.2)
    dt = 0.005
    # pre-step Jv
    v_loc = np.concatenate([a.v, a.w, b.v, b.w])
    Jv_before = np.linalg.norm(c.jacobian() @ v_loc)
    w.step(solver, dt)
    # post-step Jv (use updated velocities)
    v_loc2 = np.concatenate([a.v, a.w, b.v, b.w])
    Jv_after = np.linalg.norm(c.jacobian() @ v_loc2)
    # expect velocity-level violation decreased and small
    assert Jv_after < Jv_before + 1e-9
    assert Jv_after < 1e-6

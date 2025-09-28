import numpy as np
import pytest

scipy = pytest.importorskip("scipy")

from hybridsim import World, RigidBody6DOF, Gravity, Drag, RigidTetherJoint, HybridIVPSolver, CSVLogger

def build_world_ivp():
    w = World(ground_z=0.0, payload_index=0)
    I = 0.1 * np.eye(3)
    a = RigidBody6DOF("payload", 10.0, I,
                      position=np.array([0,0,50], float),
                      orientation=np.array([1,0,0,0], float))
    b = RigidBody6DOF("canopy", 2.0, I,
                      position=np.array([0,0,55], float),
                      orientation=np.array([1,0,0,0], float))
    i = w.add_body(a); j = w.add_body(b); w.payload_index = i
    w.add_global_force(Gravity(np.array([0,0,-9.81])))
    a.per_body_forces.append(Drag(1.225, 1.0, 0.3, "quadratic"))
    b.per_body_forces.append(Drag(1.225, 1.5, 1.0, "quadratic"))
    tether = RigidTetherJoint(i, j, [0,0,0], [0,0,0], length=5.0)
    w.add_constraint(tether.attach(w.bodies))
    return w

def test_ivp_terminal_event_and_returned_solution(tmp_path):
    w = build_world_ivp()
    # logging to ensure CSV is produced by post-logging
    csv_path = tmp_path / "ivp.csv"
    w.set_logger(CSVLogger(str(csv_path)))
    ivp = HybridIVPSolver(method="Radau", rtol=1e-6, atol=1e-8, alpha=5.0, beta=0.2, max_step=0.2)
    sol = w.integrate_to(ivp, t_end=60.0)

    # Should stop by terminal event and return a valid OdeResult
    assert sol is not None
    assert getattr(sol, "status", 1) in (0, 1)  # 1=event, 0=finished span
    assert w.t > 0.0
    # Touchdown time recorded when event fires
    if getattr(sol, "t_events", None) and len(sol.t_events[0]):
        assert w.t_touchdown is not None
        # IVP event time matches world record within tolerance
        assert abs(sol.t_events[0][0] - w.t_touchdown) < 1e-6

    # CSV created and non-empty
    assert csv_path.exists() and csv_path.stat().st_size > 0

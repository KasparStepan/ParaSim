import numpy as np
from hybridsim import World, RigidBody6DOF, Gravity, HybridSolver

def test_fixed_step_stops_on_ground_and_estimates_touchdown_time():
    g = 9.81
    h0 = 2.0
    w = World(ground_z=0.0, payload_index=0)
    I = 0.1 * np.eye(3)
    body = RigidBody6DOF("payload", 1.0, I,
                         position=np.array([0,0,h0], float),
                         orientation=np.array([1,0,0,0], float))
    w.add_body(body)
    w.add_global_force(Gravity(np.array([0,0,-g])))

    solver = HybridSolver(alpha=0.0, beta=0.0)
    dt = 0.005
    w.run(solver, duration=5.0, dt=dt)

    # Analytic free-fall touchdown (no drag, no constraints)
    t_star = np.sqrt(2*h0/g)
    assert w.t_touchdown is not None
    # Fixed-step interpolation should be close (O(dt))
    assert abs(w.t_touchdown - t_star) < 5*dt

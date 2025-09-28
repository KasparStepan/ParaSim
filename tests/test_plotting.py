import os
import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")

from hybridsim import World, RigidBody6DOF, Gravity, Drag, RigidTetherJoint, HybridSolver, CSVLogger

@pytest.mark.filterwarnings("ignore:Matplotlib")
def test_world_save_plots_creates_images(tmp_path):
    import matplotlib
    matplotlib.use("Agg", force=True)

    w = World(ground_z=-1000.0, payload_index=0)
    I = 0.1 * np.eye(3)
    payload = RigidBody6DOF("payload", 5.0, I, position=np.array([0,0,20], float),
                            orientation=np.array([1,0,0,0], float))
    canopy  = RigidBody6DOF("canopy", 2.0, I, position=np.array([0,0,25], float),
                            orientation=np.array([1,0,0,0], float))
    ip = w.add_body(payload); ic = w.add_body(canopy); w.payload_index = ip
    w.add_global_force(Gravity(np.array([0,0,-9.81])))
    payload.per_body_forces.append(Drag(1.225, 1.0, 0.2, "quadratic"))
    canopy.per_body_forces.append(Drag(1.225, 1.5, 0.8, "quadratic"))
    tether = RigidTetherJoint(ip, ic, [0,0,0], [0,0,0], 5.0)
    w.add_constraint(tether.attach(w.bodies))

    # Log a brief run
    csv_path = tmp_path / "fixed.csv"
    w.set_logger(CSVLogger(str(csv_path)))
    solver = HybridSolver(alpha=5.0, beta=0.2)
    w.run(solver, duration=0.2, dt=0.01)
    assert csv_path.exists() and csv_path.stat().st_size > 0

    # Save plots via World API
    outdir = tmp_path / "plots"
    w.save_plots(str(csv_path), bodies=["payload", "canopy"], plots_dir=str(outdir), show=False)

    # Expect 3 images per body
    for name in ("payload", "canopy"):
        for suffix in ("_traj.png", "_vel_acc.png", "_forces.png"):
            fp = outdir / f"{name}{suffix}"
            assert fp.exists(), f"Expected plot not created: {fp}"
            assert os.path.getsize(fp) > 0

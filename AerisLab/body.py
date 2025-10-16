from __future__ import annotations
import numpy as np
from .mathutil import quat_to_rotmat, quat_normalize, quat_derivative

Array = np.ndarray

class RigidBody6DOF:
    """
    6-DoF rigid body in world frame with quaternion orientation.

    Frames & state:
    - p (3,): world position of body origin [m]
    - q (4,): unit quaternion body->world (scalar-first) [-]
    - v (3,): world linear velocity [m/s]
    - w (3,): world angular velocity [rad/s]

    Properties:
    - mass m [kg]
    - I_body (3,3): inertia in body principal frame [kg m^2]

    World inertia: I_world = R(q) @ I_body @ R(q)^T

    Forces & torques accumulate per step:
    - f (3,), tau (3,)

    Use __slots__ to reduce per-step allocations.
    """
    __slots__ = (
        "name", "p", "q", "v", "w",
        "mass", "I_body", "I_body_inv",
        "inv_mass", "radius",
        "f", "tau",
        "per_body_forces",
    )

    def __init__(
        self,
        name: str,
        mass: float,
        inertia_tensor_body: Array,
        position: Array,
        orientation: Array,
        linear_velocity: Array | None = None,
        angular_velocity: Array | None = None,
        radius: float = 0.0,
    ) -> None:
        self.name = name
        self.mass = float(mass)
        self.inv_mass = 0.0 if self.mass == 0.0 else 1.0 / self.mass
        self.I_body = np.asarray(inertia_tensor_body, dtype=np.float64)
        self.I_body_inv = np.linalg.inv(self.I_body)
        self.p = np.asarray(position, dtype=np.float64).copy()
        self.q = quat_normalize(np.asarray(orientation, dtype=np.float64).copy())
        self.v = np.zeros(3, dtype=np.float64) if linear_velocity is None else np.asarray(linear_velocity, dtype=np.float64).copy()
        self.w = np.zeros(3, dtype=np.float64) if angular_velocity is None else np.asarray(angular_velocity, dtype=np.float64).copy()
        self.radius = float(radius)
        self.f = np.zeros(3, dtype=np.float64)
        self.tau = np.zeros(3, dtype=np.float64)
        self.per_body_forces: list = []

    # --- basic ops ---
    def clear_forces(self) -> None:
        self.f.fill(0.0)
        self.tau.fill(0.0)

    def rotation_world(self) -> Array:
        return quat_to_rotmat(self.q)

    def inertia_world(self) -> Array:
        R = self.rotation_world()
        return R @ self.I_body @ R.T

    def mass_matrix_world(self) -> Array:
        """
        Return block-diagonal generalized mass for this body:
        M_i = diag(m I3, I_world) with shape (6,6).
        """
        M = np.zeros((6, 6), dtype=np.float64)
        M[0:3, 0:3] = self.mass * np.eye(3)
        M[3:6, 3:6] = self.inertia_world()
        return M

    # --- force application ---
    def apply_force(self, f: Array, point_world: Array | None = None) -> None:
        """
        Add world force f (3,). If point_world is provided, applies torque τ += r × f
        where r = point_world - body origin in world.
        """
        f = np.asarray(f, dtype=np.float64)
        self.f += f
        if point_world is not None:
            r = np.asarray(point_world, dtype=np.float64) - self.p
            self.tau += np.cross(r, f)

    def apply_torque(self, tau: Array) -> None:
        self.tau += np.asarray(tau, dtype=np.float64)

    def generalized_force(self) -> Array:
        """Return concatenated generalized force [f; tau] (6,)."""
        out = np.zeros(6, dtype=np.float64)
        out[:3] = self.f
        out[3:] = self.tau
        return out

    # --- integration ---
    def integrate_semi_implicit(self, dt: float, a_lin: Array, a_ang: Array) -> None:
        """
        Semi-implicit (symplectic) Euler:
        v_{n+1} = v_n + a_lin dt
        w_{n+1} = w_n + a_ang dt
        p_{n+1} = p_n + v_{n+1} dt
        q_{n+1} = normalize( q_n + qdot(v=w_{n+1}) dt )
        """
        self.v += a_lin * dt
        self.w += a_ang * dt
        self.p += self.v * dt
        qdot = quat_derivative(self.q, self.w)
        self.q = quat_normalize(self.q + qdot * dt)


    
class Parachute_RigidBody6DOF(RigidBody6DOF):
    """
    6-DoF rigid body with parachute capabilities.
    Inherits from RigidBody6DOF and adds parachute-specific properties and methods.
    """
    
    def __init__(
        self,
        name: str,
        mass: float,
        inertia_tensor_body: Array,
        position: Array,
        orientation: Array,
        linear_velocity: Array | None = None,
        angular_velocity: Array | None = None,
        radius: float = 0.0,
        activation_velocity: float = 30.0,  # m/s
        gate_sharpness: float = 10.0,       # controls smoothness of activation
        area_collapsed: float = 0.1         # m², small area when parachute is collapsed
    ) -> None:
        super().__init__(
            name, mass, inertia_tensor_body, position, orientation,
            linear_velocity, angular_velocity, radius
        )


    # Parachute specific added mass due to inflation
    def add_added_mass(self, updated_mass: float, density: float, volume: float, area_projected: float, diameter_equivalent: float) -> None:
        
        #TODO: implement model of added mass
        # Version 1
        updated_mass = 2.586 * density * diameter_equivalent**3 + 0.908 * density * volume + self.mass
        # Version 2
        updated_mass = 0.464*density*area_projected**(3/2) + 0.908*density*volume + self.mass

        #TODO: implement model for predicting parachute

        self.mass=updated_mass
        
        
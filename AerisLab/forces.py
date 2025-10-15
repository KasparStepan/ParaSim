from __future__ import annotations
import numpy as np
from typing import Callable, Optional, Protocol
from .mathutil import Array
from .body import RigidBody6DOF

class Force(Protocol):
    def apply(self, body: RigidBody6DOF, t: Optional[float] = None) -> None: ...


class Gravity:
    """Uniform gravity (world frame). Adds m*g force."""
    def __init__(self, g: Array) -> None:
        self.g = np.asarray(g, dtype=np.float64)

    def apply(self, body: RigidBody6DOF, t: Optional[float] = None) -> None:
        body.apply_force(body.mass * self.g)


class Drag:
    """
    Aerodynamic drag in world frame.

    Modes:
      - 'quadratic': F = -0.5 ρ Cd A |v| v
      - 'linear'   : F = -k v     (set k via k_linear)

    Parameters can be changed at runtime (e.g., canopy area growth).
    A can be float or callable A(t, body)->float. Same for Cd.
    """
    def __init__(
        self,
        rho: float = 1.225,
        Cd: float | Callable[[float, RigidBody6DOF], float] = 1.0,
        area: float | Callable[[float, RigidBody6DOF], float] = 1.0,
        mode: str = "quadratic",
        k_linear: float = 0.0,
    ) -> None:
        self.rho = float(rho)
        self.Cd = Cd
        self.area = area
        self.mode = mode
        self.k_linear = float(k_linear)

    def _value(self, val, t: float, body: RigidBody6DOF) -> float:
        return float(val(t, body)) if callable(val) else float(val)

    def apply(self, body: RigidBody6DOF, t: Optional[float] = None) -> None:
        tval = 0.0 if t is None else float(t)
        v = body.v
        if self.mode == "quadratic":
            Cd = self._value(self.Cd, tval, body)
            A = self._value(self.area, tval, body)
            speed = np.linalg.norm(v)
            if speed > 0.0:
                f = -0.5 * self.rho * Cd * A * speed * v
                body.apply_force(f)
        elif self.mode == "linear":
            body.apply_force(-self.k_linear * v)
        else:
            raise ValueError("Drag.mode must be 'quadratic' or 'linear'.")
        
class ParachuteDrag(Drag):
    """
    Specialized Drag for parachutes with time-dependent area.
    """
    def __init__(
        self,
        rho: float = 1.225,
        Cd: float | Callable[[float, RigidBody6DOF], float] = 1.5,
        area: float | Callable[[float, RigidBody6DOF], float] = 1.0,
        mode: str = "quadratic",
        activation_time: float = 0.0,
        activation_altitude: float | None = None,
        activation_velocity: float = 50.0,
        # Variables used for smooth transition for IVP solver
        gate_sharpness: float = 40.0,
        area_collapsed:float = 1e-3,
    ) -> None:
        super().__init__(rho=rho, Cd=Cd, area=area, mode=mode)
        self.activation_time = activation_time
        self.activation_altitude = (None if activation_altitude is None
                                else float(activation_altitude))
        self.activation_velocity = activation_velocity
        self.activation_status = False
        self.gate_sharpness = float(gate_sharpness)
        self.area_collapsed = float(area_collapsed)


    # TODO: implement physical model of parachute with better accuracy
    # TODO: Add added mass of parachute to body mass    
    def apply(self, body: RigidBody6DOF, t: Optional[float] = None) -> None:    
        tval = 0.0 if t is None else float(t)
        v = body.v
        v_mag = np.linalg.norm(v)
        
        # Check activation conditions
        if v_mag >= abs(self.activation_velocity) and self.activation_status == False:
            #print(f"Parachute activated at t={tval:.2f}s, v={v_mag:.2f}m/s")
            self.activation_status = True
            self.activation_time = tval

        # Apply drag only if activated
        if self.activation_status == True:
            #print(self.activation_status)
            Cd = self._value(self.Cd, tval, body)
            A = self.eval_area(tval, body)
            speed = v_mag
            if speed > 0.0: 
                f = -0.5 * self.rho * Cd * A * speed * v
                body.apply_force(f)
                #print(f"Parachute drag force at t={tval:.2f}s: F={f},A={self.eval_area(tval, body):.2f}m^2")
        else:
            f = np.zeros(3)
            body.apply_force(f)
            #print(f"Parachute drag force at t={tval:.2f}s: F={f},A={self.area:.2f}m^2")
           


    def eval_area(self, tval, body) -> float:
        t = 0.0 if tval is None else float(tval)
        k = getattr(self, "gate_sharpness", 40.0)      # higher k => sharper (but still smooth) transition
        A0 = getattr(self, "area_collapsed", 0.0)      # tiny baseline if you use it; else 0
        g = 0.5 * (1.0 + np.tanh(k * (t - self.activation_time)))  # smooth gate in (0,1)
        return A0 + g * (float(self.area) - A0)


class Spring:
    """
    Soft connection between two bodies (Hooke + line damping).
    Not a rigid constraint; acts like a tether spring.

    F = -k (|d|-L0) d_hat - c * ((v_rel ⋅ d_hat) d_hat)
    applied at attachment points in world frame (equal/opposite).

    Usage through joints.SoftTetherJoint for convenience.
    """
    def __init__(
        self,
        body_a: RigidBody6DOF,
        body_b: RigidBody6DOF,
        attach_a_local: Array,
        attach_b_local: Array,
        k: float,
        c: float,
        rest_length: float,
    ) -> None:
        self.a = body_a
        self.b = body_b
        self.ra_local = np.asarray(attach_a_local, dtype=np.float64)
        self.rb_local = np.asarray(attach_b_local, dtype=np.float64)
        self.k = float(k)
        self.c = float(c)
        self.L0 = float(rest_length)

    def apply_pair(self, t: Optional[float] = None) -> None:
        Ra = self.a.rotation_world(); Rb = self.b.rotation_world()
        ra_w = Ra @ self.ra_local
        rb_w = Rb @ self.rb_local
        pa = self.a.p + ra_w
        pb = self.b.p + rb_w
        d = pa - pb
        dist = np.linalg.norm(d)
        if dist == 0.0:
            return
        d_hat = d / dist

        va = self.a.v + np.cross(self.a.w, ra_w)
        vb = self.b.v + np.cross(self.b.w, rb_w)
        vrel = va - vb
        vrel_line = np.dot(vrel, d_hat)

        f = -self.k * (dist - self.L0) * d_hat - self.c * vrel_line * d_hat
        self.a.apply_force(+f, point_world=pa)
        self.b.apply_force(-f, point_world=pb)

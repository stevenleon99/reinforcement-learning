"""
Manual finite-horizon LQR cart-pole example with animation.

Python version of the MATLAB script:
    manual_lqr_cartpole_with_animation()

This example uses a LINEARIZED discrete-time cart-pole model near the upright equilibrium.

State:
    x = [cart position error,
         pole angle error from upright,
         cart velocity,
         pole angular velocity]

Control:
    u = cart force

The LQR is solved manually using backward Riccati recursion, without scipy/control dlqr().
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


def manual_lqr_cartpole_with_animation():
    # ============================================================
    # 1. Time setup
    # ============================================================
    dt = 0.05
    T = 5.0
    N = int(np.floor(T / dt)) + 1

    # ============================================================
    # 2. Linearized discrete-time cart-pole system
    # ============================================================

    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0.5 * dt, 1, 0],
        [0, 10 * dt, 0, 1]
    ], dtype=float)

    B = np.array([
        [0],
        [0],
        [dt],
        [2 * dt]
    ], dtype=float)

    # ============================================================
    # 3. Cost matrices
    # ============================================================

    Q = np.diag([1, 100, 1, 10])
    R = np.array([[0.1]])
    Qf = Q.copy()

    # ============================================================
    # 4. Manually solve finite-horizon LQR
    #
    # P_t = Q + A^T P_{t+1} A
    #       - A^T P_{t+1} B (R + B^T P_{t+1} B)^(-1) B^T P_{t+1} A
    #
    # K_t = (R + B^T P_{t+1} B)^(-1) B^T P_{t+1} A
    #
    # u_t = -K_t x_t
    # ============================================================

    P = np.zeros((N, 4, 4))
    K = np.zeros((N - 1, 1, 4))

    # Terminal condition
    P[-1, :, :] = Qf

    # Backward Riccati recursion
    for k in range(N - 2, -1, -1):
        Pnext = P[k + 1, :, :]

        S = R + B.T @ Pnext @ B

        # Equivalent to MATLAB: S \ (B' * Pnext * A)
        K[k, :, :] = np.linalg.solve(S, B.T @ Pnext @ A)

        P[k, :, :] = (
            Q
            + A.T @ Pnext @ A
            - A.T @ Pnext @ B @ K[k, :, :]
        )

    print("Initial LQR gain K[0]:")
    print(K[0, :, :])

    print("\nInitial cost-to-go matrix P[0]:")
    print(P[0, :, :])

    # ============================================================
    # 5. Simulate closed-loop system
    # ============================================================

    x = np.zeros((4, N))
    u = np.zeros((1, N - 1))
    t = np.zeros(N)

    # Initial condition
    x[:, 0] = np.array([
        0.3,    # cart position error
        1.21,   # pole angle error from upright, radians
        0.0,    # cart velocity
        0.0     # pole angular velocity
    ])

    for k in range(N - 1):
        # Manual LQR control law
        u[:, k] = -K[k, :, :] @ x[:, k]

        # Linear system update
        x[:, k + 1] = A @ x[:, k] + (B * u[0, k]).flatten()

        # Time update
        t[k + 1] = t[k] + dt

    # ============================================================
    # 6. Plot state trajectories
    # ============================================================

    plt.figure()
    plt.plot(t, x[0, :], linewidth=1.5, label="cart position")
    plt.plot(t, x[1, :], linewidth=1.5, label="pole angle error")
    plt.plot(t, x[2, :], linewidth=1.5, label="cart velocity")
    plt.plot(t, x[3, :], linewidth=1.5, label="pole angular velocity")
    plt.grid(True)

    plt.xlabel("Time [s]")
    plt.ylabel("State value")
    plt.title("Manual LQR Closed-Loop State Trajectory")
    plt.legend()

    # ============================================================
    # 7. Plot control input
    # ============================================================

    plt.figure()
    plt.step(t[:-1], u[0, :], where="post", linewidth=1.5)
    plt.grid(True)

    plt.xlabel("Time [s]")
    plt.ylabel("Control input u")
    plt.title("Manual LQR Control Force")

    # Show static plots without blocking the animation
    plt.show(block=False)

    # ============================================================
    # 8. Animate cart-pole
    # ============================================================

    animate_cartpole(t, x)


def animate_cartpole(t, x):
    fig, ax = plt.subplots()

    for k in range(x.shape[1]):
        draw_cartpole(ax, t[k], x[:, k])
        plt.pause(0.03)

    plt.show()


def draw_cartpole(ax, t, x):
    ax.clear()
    hold = True  # kept only to mirror MATLAB logic conceptually

    # Drawing parameters
    cart_width = 0.4
    cart_height = 0.2
    wheel_radius = 0.05
    pole_length = 0.8

    # State variables
    cart_pos = x[0]

    # x[1] is angle error from upright.
    # Upright corresponds to theta = pi.
    theta = np.pi + x[1]

    # Ground
    ax.plot([-3, 3], [-0.15, -0.15], "k", linewidth=1.5)

    # Cart body
    cart_x = cart_pos - cart_width / 2
    cart_y = 0.0

    cart = Rectangle(
        (cart_x, cart_y),
        cart_width,
        cart_height,
        facecolor=(0.3, 0.7, 0.9),
        edgecolor="k",
        linewidth=1.5,
    )
    ax.add_patch(cart)

    # Wheels
    wheel_y = cart_y - wheel_radius

    left_wheel = Circle(
        (cart_pos - cart_width / 3, wheel_y),
        wheel_radius,
        facecolor="k",
        edgecolor="k",
    )
    right_wheel = Circle(
        (cart_pos + cart_width / 3, wheel_y),
        wheel_radius,
        facecolor="k",
        edgecolor="k",
    )

    ax.add_patch(left_wheel)
    ax.add_patch(right_wheel)

    # Pivot point
    pivot_x = cart_pos
    pivot_y = cart_y + cart_height

    # Pole end point
    pole_x = pivot_x + pole_length * np.sin(theta)
    pole_y = pivot_y - pole_length * np.cos(theta)

    # Pole
    ax.plot([pivot_x, pole_x], [pivot_y, pole_y], "r", linewidth=4)

    # Pole mass
    ax.plot(
        pole_x,
        pole_y,
        "ko",
        markersize=10,
        markerfacecolor="y",
    )

    # Pivot point
    ax.plot(
        pivot_x,
        pivot_y,
        "ko",
        markersize=6,
        markerfacecolor="k",
    )

    # Plot settings
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-1.0, 1.5])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    ax.set_xlabel("Cart position")
    ax.set_ylabel("Height")
    ax.set_title(f"Manual LQR Cart-Pole Animation, t = {t:.2f} s")


if __name__ == "__main__":
    manual_lqr_cartpole_with_animation()

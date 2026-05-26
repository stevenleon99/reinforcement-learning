"""
Iterative LQR / iLQR for a nonlinear cart-pole system.

This is a Python version of the MATLAB cart-pole iLQR script.

State:
    x = [cart_position, pole_angle, cart_velocity, pole_angular_velocity]

Control:
    u = [cart_force]

Goal:
    Drive the pole angle toward pi radians with desired state:
    xd = [0, pi, 0, 0]

Dependencies:
    numpy
    matplotlib

Run:
    python iterative_lqr_cart_pole.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


# ==========================================================================
# Main script
# ==========================================================================

def main():
    T = 5.0
    dt = 0.05
    N = int(np.floor(T / dt)) + 1

    nX = 4
    nU = 1

    param = {
        "mc": 10.0,
        "mp": 2.0,
        "l": 0.5,
        "g": 9.8,
        "b": 0.1,
        "d": 0.1,
    }

    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    xd = np.array([0.0, np.pi, 0.0, 0.0])

    Q = np.diag([0.1, 1.0, 0.1, 1.0])
    Qf = np.diag([1.2, 30000.0, 1.5, 30000.0])
    R = np.array([[1e-6]])

    xtraj = np.zeros((nX, N))
    utraj = np.zeros((nU, N - 1))
    ktraj = np.zeros((nU, N - 1))
    Ktraj = np.zeros((nU, nX, N - 1))

    xtraj, utraj, ktraj, Ktraj, costs = ilqr(
        x0=x0,
        xtraj=xtraj,
        utraj=utraj,
        ktraj=ktraj,
        Ktraj=Ktraj,
        N=N,
        dt=dt,
        param=param,
        Q=Q,
        R=R,
        Qf=Qf,
        xd=xd,
        max_iter=300,
        tol=1e-4,
    )

    t = np.arange(N) * dt

    print("Final state:", xtraj[:, -1])
    print("Final cost:", costs[-1] if costs else None)

    plot_results(t, xtraj, utraj, costs)
    animate_cartpole(t, xtraj, param)


# ==========================================================================
# iLQR
# ==========================================================================

def ilqr(
    x0,
    xtraj,
    utraj,
    ktraj,
    Ktraj,
    N,
    dt,
    param,
    Q,
    R,
    Qf,
    xd,
    max_iter=300,
    tol=1e-4,
):
    """
    Iterative LQR algorithm.

    1. Roll out current control sequence.
    2. Backward pass computes feedforward k and feedback K.
    3. Forward pass performs line search using:
           u_new = u_old + alpha*k + K*(x_new - x_old)
    """

    xtraj, cost = rollout(x0, utraj, N, dt, param, Q, R, Qf, xd)
    costs = [cost]

    for iteration in range(max_iter):
        Ktraj, ktraj = backward_pass(
            xtraj=xtraj,
            utraj=utraj,
            Q=Q,
            R=R,
            Qf=Qf,
            xd=xd,
            param=param,
            N=N,
            dt=dt,
        )

        xnew, unew, new_cost, accepted = forward_pass(
            x0=x0,
            xtraj0=xtraj,
            utraj0=utraj,
            ktraj=ktraj,
            Ktraj=Ktraj,
            N=N,
            dt=dt,
            param=param,
            Q=Q,
            R=R,
            Qf=Qf,
            xd=xd,
            old_cost=cost,
        )

        if not accepted:
            print(f"Iteration {iteration:03d}: no line-search improvement; stopping.")
            break

        improvement = cost - new_cost
        xtraj = xnew
        utraj = unew
        cost = new_cost
        costs.append(cost)

        print(
            f"Iteration {iteration:03d}: "
            f"cost={cost:.6f}, improvement={improvement:.6f}"
        )

        if abs(improvement) < tol:
            print("Converged.")
            break

    return xtraj, utraj, ktraj, Ktraj, costs


# ==========================================================================
# Forward pass
# ==========================================================================

def forward_pass(
    x0,
    xtraj0,
    utraj0,
    ktraj,
    Ktraj,
    N,
    dt,
    param,
    Q,
    R,
    Qf,
    xd,
    old_cost,
):
    alphas = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001]

    for alpha in alphas:
        xtraj = np.zeros_like(xtraj0)
        utraj = np.zeros_like(utraj0)

        x = x0.copy()
        xtraj[:, 0] = x
        J = 0.0

        for i in range(N - 1):
            dx = x - xtraj0[:, i]
            u = utraj0[:, i] + alpha * ktraj[:, i] + Ktraj[:, :, i] @ dx

            # Optional control clamp for numerical stability.
            # Increase or remove if needed.
            u = np.clip(u, -200.0, 200.0)

            utraj[:, i] = u
            J += stage_cost(x, u, xd, Q, R)

            xdot = cartpole_dynamics(x, u, param)
            x = x + dt * xdot
            xtraj[:, i + 1] = x

        J += final_cost(xtraj[:, -1], xd, Qf)

        if np.isfinite(J) and J < old_cost:
            return xtraj, utraj, J, True

    return xtraj0, utraj0, old_cost, False


def rollout(x0, utraj, N, dt, param, Q, R, Qf, xd):
    xtraj = np.zeros((4, N))
    x = x0.copy()
    xtraj[:, 0] = x
    J = 0.0

    for i in range(N - 1):
        u = utraj[:, i]
        J += stage_cost(x, u, xd, Q, R)

        xdot = cartpole_dynamics(x, u, param)
        x = x + dt * xdot
        xtraj[:, i + 1] = x

    J += final_cost(xtraj[:, -1], xd, Qf)
    return xtraj, J


# ==========================================================================
# Backward pass
# ==========================================================================

def backward_pass(xtraj, utraj, Q, R, Qf, xd, param, N, dt):
    nX = xtraj.shape[0]
    nU = utraj.shape[0]

    Ktraj = np.zeros((nU, nX, N - 1))
    ktraj = np.zeros((nU, N - 1))

    # Terminal value derivatives
    Vx = Qf @ (xtraj[:, -1] - xd)
    Vxx = Qf.copy()

    reg = 1e-6

    for i in reversed(range(N - 1)):
        x = xtraj[:, i]
        u = utraj[:, i]

        gx, gu, gxx, gux, guu = cost_gradients(x, u, xd, Q, R)

        # Continuous dynamics derivatives
        Ac, Bc = cartpole_grads(x, u, param)

        # Discrete-time linearization:
        # x_{t+1} = x_t + dt*f(x_t,u_t)
        fx = np.eye(nX) + dt * Ac
        fu = dt * Bc

        Qx, Qu, Qxx, Qux, Quu, Qxu = q_terms(
            gx, gu, gxx, gux, guu, fx, fu, Vx, Vxx
        )

        # Regularize Quu to keep it positive definite / invertible.
        Quu_reg = Quu + reg * np.eye(nU)

        k, K = gains(Qu, Qux, Quu_reg)

        ktraj[:, i] = k
        Ktraj[:, :, i] = K

        Vx, Vxx = v_terms(Qx, Qu, Qxx, Qux, Quu, Qxu, K, k)

        # Symmetrize Vxx to reduce numerical drift.
        Vxx = 0.5 * (Vxx + Vxx.T)

    return Ktraj, ktraj


def q_terms(gx, gu, gxx, gux, guu, fx, fu, Vx, Vxx):
    Qx = gx + fx.T @ Vx
    Qu = gu + fu.T @ Vx
    Qxx = gxx + fx.T @ Vxx @ fx
    Qux = gux + fu.T @ Vxx @ fx
    Quu = guu + fu.T @ Vxx @ fu
    Qxu = Qux.T
    return Qx, Qu, Qxx, Qux, Quu, Qxu


def gains(Qu, Qux, Quu):
    """
    iLQR local control law:
        du = alpha*k + K*dx

    The minimizing gains are:
        k = -Quu^{-1} Qu
        K = -Quu^{-1} Qux
    """
    k = -np.linalg.solve(Quu, Qu)
    K = -np.linalg.solve(Quu, Qux)
    return k, K


def v_terms(Qx, Qu, Qxx, Qux, Quu, Qxu, K, k):
    """
    Value function derivative update.

    This form is equivalent to substituting:
        du* = k + K dx

    into the local quadratic Q-function.
    """
    Vx = Qx + K.T @ Qu + Qxu @ k + K.T @ Quu @ k
    Vxx = Qxx + K.T @ Qux + Qxu @ K + K.T @ Quu @ K
    return Vx, Vxx


# ==========================================================================
# Cost functions
# ==========================================================================

def stage_cost(x, u, xd, Q, R):
    dx = x - xd
    return 0.5 * dx.T @ Q @ dx + 0.5 * u.T @ R @ u


def final_cost(x, xd, Qf):
    dx = x - xd
    return 0.5 * dx.T @ Qf @ dx


def cost_gradients(x, u, xd, Q, R):
    gx = Q @ (x - xd)
    gu = R @ u
    gxx = Q
    gux = np.zeros((len(u), len(x)))
    guu = R
    return gx, gu, gxx, gux, guu


# ==========================================================================
# Cart-pole dynamics
# ==========================================================================

def cartpole_dynamics(x, u, param):
    mc = param["mc"]
    mp = param["mp"]
    l = param["l"]
    g = param["g"]
    b = param["b"]
    d = param["d"]

    u = float(np.asarray(u).reshape(-1)[0])

    x1, x2, x3, x4 = x
    s = np.sin(x2)
    c = np.cos(x2)

    denom = mc + mp * s**2

    xdot = np.zeros(4)
    xdot[0] = x3
    xdot[1] = x4
    xdot[2] = (
        u
        - b * x3
        + d * x4 * c / l
        + mp * s * (l * x4**2 + g * c)
    ) / denom

    xdot[3] = (
        -u * c
        + b * x3 * c
        - d * (mc + mp) * x4 / (mp * l)
        - mp * l * x4**2 * c * s
        - (mc + mp) * g * s / (mp * l)
    ) / (l * denom)

    return xdot


def cartpole_grads(x, u, param):
    """
    Numerical Jacobian of continuous cart-pole dynamics.

    Returns:
        A = df/dx
        B = df/du

    This is more robust than carrying a long symbolic derivative.
    """
    nX = len(x)
    nU = len(u)

    A = np.zeros((nX, nX))
    B = np.zeros((nX, nU))

    eps_x = 1e-5
    eps_u = 1e-5

    f0 = cartpole_dynamics(x, u, param)

    for j in range(nX):
        dx = np.zeros(nX)
        dx[j] = eps_x
        fp = cartpole_dynamics(x + dx, u, param)
        fm = cartpole_dynamics(x - dx, u, param)
        A[:, j] = (fp - fm) / (2.0 * eps_x)

    for j in range(nU):
        du = np.zeros(nU)
        du[j] = eps_u
        fp = cartpole_dynamics(x, u + du, param)
        fm = cartpole_dynamics(x, u - du, param)
        B[:, j] = (fp - fm) / (2.0 * eps_u)

    return A, B


# ==========================================================================
# Plotting and animation
# ==========================================================================

def plot_results(t, xtraj, utraj, costs):
    plt.figure()
    plt.plot(costs)
    plt.xlabel("iLQR iteration")
    plt.ylabel("Cost")
    plt.title("iLQR Cost Convergence")
    plt.grid(True)

    plt.figure()
    plt.plot(t, xtraj[0, :], label="cart position")
    plt.plot(t, xtraj[1, :], label="pole angle")
    plt.plot(t, xtraj[2, :], label="cart velocity")
    plt.plot(t, xtraj[3, :], label="pole angular velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("State")
    plt.title("State Trajectory")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.step(t[:-1], utraj[0, :], where="post")
    plt.xlabel("Time [s]")
    plt.ylabel("Control force")
    plt.title("Control Trajectory")
    plt.grid(True)

    plt.show(block=False)


def animate_cartpole(t, xtraj, param):
    l = param["l"]

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    cart_w = 0.3
    cart_h = 0.15
    wheel_r = 0.05

    for k in range(xtraj.shape[1]):
        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-1.5, 1.0)
        ax.set_title(f"t = {t[k]:.2f} s")

        x = xtraj[:, k]
        cart_x = x[0]
        theta = x[1]

        # Cart
        cart = Rectangle(
            (cart_x - cart_w / 2, -cart_h / 2),
            cart_w,
            cart_h,
            fill=False,
            linewidth=2,
        )
        ax.add_patch(cart)

        # Wheels
        ax.add_patch(Circle((cart_x - cart_w / 3, -cart_h / 2 - wheel_r), wheel_r, fill=False))
        ax.add_patch(Circle((cart_x + cart_w / 3, -cart_h / 2 - wheel_r), wheel_r, fill=False))

        # Pole
        pole_x = cart_x + l * np.sin(theta)
        pole_y = -l * np.cos(theta)

        ax.plot([cart_x, pole_x], [0.0, pole_y], linewidth=3)
        ax.plot(pole_x, pole_y, "o", markersize=8)

        ax.axhline(-cart_h / 2 - 2 * wheel_r, linewidth=1)
        plt.pause(0.03)

    plt.show()


if __name__ == "__main__":
    main()

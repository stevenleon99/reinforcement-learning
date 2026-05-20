"""
Rockit-MECO NMPC version with solver-stable soft obstacle penalties.

This version fixes the common IPOPT issue:

    return_status is 'Maximum_Iterations_Exceeded'

Main changes from the earlier version:
1. Removed the inverse-distance repulsion objective.
   That term can make the nonlinear program very stiff and hard to solve.
2. Kept obstacle avoidance as soft constraints with slack variables.
3. Used a moderate obstacle slack penalty by default.
4. Added warm-start from the previous solution.
5. Added IPOPT settings that make repeated NMPC solves more stable.

Install:
    pip install rockit-meco casadi tyro matplotlib numpy

Run:
    python rocket_meco_mpc_example_solver_stable.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
import tyro

try:
    from rockit import Ocp, MultipleShooting
except ImportError as exc:
    raise ImportError(
        "Rockit is not installed. Install it with:\n"
        "    pip install rockit-meco casadi\n"
        "Then rerun this script."
    ) from exc


# -----------------------------
# Command-line arguments
# -----------------------------
@dataclass
class Args:
    # Simulation settings
    max_steps: int = 160
    dt: float = 0.1

    # NMPC horizon
    # A shorter horizon is easier for IPOPT to solve.
    horizon_steps: int = 30

    # Vehicle settings
    wheelbase: float = 2.5
    max_speed: float = 8.0
    max_accel: float = 1.0
    max_steer_deg: float = 30.0
    vehicle_radius: float = 0.7

    # Keep this modest. Large safety margins can make the problem much harder.
    safety_margin: float = 0.20

    # Target behavior
    target_radius: float = 1.2
    target_lock_radius: float = 3.0

    # Objective weights
    position_weight: float = 5.0
    terminal_weight: float = 80.0
    control_weight: float = 0.2
    speed_weight: float = 0.05
    goal_speed_weight: float = 4.0

    # Custom obstacle tuning parameter.
    # This is NOT a Rockit built-in option.
    # It is used as a weight in ocp.add_objective().
    obstacle_slack_weight: float = 15000.0

    # IPOPT options
    ipopt_print_level: int = 0
    print_time: bool = False
    ipopt_max_iter: int = 800


args = tyro.cli(Args)


# -----------------------------
# NumPy transition model for actual simulation rollout
# -----------------------------
def vehicle_step(state, action, dt=0.1, L=2.5, max_speed=8.0, max_steer_deg=30.0):
    """
    state = [x, y, theta, v]
    action = [acceleration, steering_angle]
    """
    x, y, theta, v = state
    a, delta = action

    max_steer = np.deg2rad(max_steer_deg)
    delta = np.clip(delta, -max_steer, max_steer)

    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + (v / L) * np.tan(delta) * dt
    v_next = v + a * dt
    v_next = np.clip(v_next, 0.0, max_speed)

    return np.array([x_next, y_next, theta_next, v_next])


# -----------------------------
# Obstacles
# -----------------------------
obstacles = [
    {"center": np.array([7.0, 3.0]), "radius": 1.5},
    {"center": np.array([12.0, 7.0]), "radius": 1.7},
    {"center": np.array([16.0, 5.0]), "radius": 1.3},
]


def is_collision(state, obstacles, vehicle_radius=0.7):
    position = state[:2]

    for obstacle in obstacles:
        center = obstacle["center"]
        radius = obstacle["radius"]
        distance = np.linalg.norm(position - center)

        if distance <= radius + vehicle_radius:
            return True

    return False


def angle_normalize(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def fallback_goal_action(state, goal, args):
    """
    Emergency fallback only if the solver fails.
    Brake gently and steer toward the goal.
    """
    x, y, theta, v = state
    goal_vec = goal - np.array([x, y])
    desired_heading = np.arctan2(goal_vec[1], goal_vec[0])
    heading_error = angle_normalize(desired_heading - theta)

    steering = np.clip(
        heading_error,
        -np.deg2rad(args.max_steer_deg),
        np.deg2rad(args.max_steer_deg),
    )

    # Brake instead of accelerating during fallback.
    # This reduces the chance of crashing if NMPC fails.
    acceleration = -0.6

    return np.array([acceleration, steering])


# Store previous solution to warm-start the next NMPC solve
previous_solution_guess = {
    "x": None,
    "y": None,
    "theta": None,
    "v": None,
    "a": None,
    "delta": None,
}


# -----------------------------
# Rockit-MECO NMPC planner
# -----------------------------
def choose_action_rockit_mpc(current_state, goal, obstacles, args):
    global previous_solution_guess

    horizon_time = args.horizon_steps * args.dt
    ocp = Ocp(t0=0, T=horizon_time)

    # States
    x = ocp.state()
    y = ocp.state()
    theta = ocp.state()
    v = ocp.state()

    # Controls
    a = ocp.control()
    delta = ocp.control()

    max_steer = np.deg2rad(args.max_steer_deg)

    # -----------------------------
    # Transition model
    # -----------------------------
    ocp.set_der(x, v * np.cos(theta))
    ocp.set_der(y, v * np.sin(theta))
    ocp.set_der(theta, v / args.wheelbase * np.tan(delta))
    ocp.set_der(v, a)

    # -----------------------------
    # Initial condition
    # -----------------------------
    ocp.subject_to(ocp.at_t0(x) == float(current_state[0]))
    ocp.subject_to(ocp.at_t0(y) == float(current_state[1]))
    ocp.subject_to(ocp.at_t0(theta) == float(current_state[2]))
    ocp.subject_to(ocp.at_t0(v) == float(current_state[3]))

    # -----------------------------
    # Bounds
    # -----------------------------
    ocp.subject_to(-args.max_accel <= (a <= args.max_accel))
    ocp.subject_to(-max_steer <= (delta <= max_steer))
    ocp.subject_to(0.0 <= (v <= args.max_speed))

    # -----------------------------
    # Goal objective
    # -----------------------------
    goal_x = float(goal[0])
    goal_y = float(goal[1])

    distance_sq = (x - goal_x) ** 2 + (y - goal_y) ** 2
    control_effort = a**2 + delta**2

    distance_now = np.linalg.norm(current_state[:2] - goal)

    terminal_weight = args.terminal_weight
    goal_speed_weight = args.goal_speed_weight

    if distance_now < args.target_lock_radius:
        terminal_weight *= 5.0
        goal_speed_weight *= 8.0

    ocp.add_objective(
        ocp.integral(
            args.position_weight * distance_sq
            + args.control_weight * control_effort
            + args.speed_weight * v**2
        )
    )

    ocp.add_objective(terminal_weight * ocp.at_tf(distance_sq))
    ocp.add_objective(goal_speed_weight * ocp.at_tf(v**2))

    # -----------------------------
    # Solver-stable soft obstacle avoidance
    # -----------------------------
    # Important:
    # We do NOT use inverse-distance repulsion here because it often causes
    # Maximum_Iterations_Exceeded.
    #
    # Instead:
    #     dist_sq + slack >= safe_radius^2
    #     slack >= 0
    #     minimize integral(slack^2)
    #
    # This makes the problem almost always solvable, while still discouraging
    # obstacle-buffer violation.
    # -----------------------------
    min_clearance_extra = args.vehicle_radius + args.safety_margin

    for obstacle in obstacles:
        cx, cy = obstacle["center"]
        safe_radius = float(obstacle["radius"] + min_clearance_extra)

        slack = ocp.variable()
        ocp.subject_to(slack >= 0)

        dist_sq_to_obstacle = (x - float(cx)) ** 2 + (y - float(cy)) ** 2

        ocp.subject_to(dist_sq_to_obstacle + slack >= safe_radius**2)

        ocp.add_objective(
            args.obstacle_slack_weight * ocp.integral(slack**2)
        )

    # -----------------------------
    # Method and solver
    # -----------------------------
    ocp.method(MultipleShooting(N=args.horizon_steps, M=1, intg="rk"))

    ocp.solver(
        "ipopt",
        {
            "ipopt.print_level": args.ipopt_print_level,
            "print_time": args.print_time,
            "ipopt.sb": "yes",
            "ipopt.max_iter": args.ipopt_max_iter,
            "ipopt.tol": 1e-3,
            "ipopt.acceptable_tol": 1e-2,
            "ipopt.acceptable_iter": 10,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.nlp_scaling_method": "gradient-based",
        },
    )

    # -----------------------------
    # Initial guesses
    # -----------------------------
    # Warm-start from previous solution when available.
    # Otherwise use a curved path guess.
    if previous_solution_guess["x"] is not None:
        ocp.set_initial(x, previous_solution_guess["x"])
        ocp.set_initial(y, previous_solution_guess["y"])
        ocp.set_initial(theta, previous_solution_guess["theta"])
        ocp.set_initial(v, previous_solution_guess["v"])
        ocp.set_initial(a, previous_solution_guess["a"])
        ocp.set_initial(delta, previous_solution_guess["delta"])
    else:
        x_guess = np.linspace(current_state[0], goal_x, args.horizon_steps + 1)
        y_guess = np.linspace(current_state[1], goal_y, args.horizon_steps + 1)

        # Slight curved initial guess to avoid a perfectly straight path through obstacles.
        t_guess = np.linspace(0, np.pi, args.horizon_steps + 1)
        y_guess = y_guess + 2.0 * np.sin(t_guess)

        ocp.set_initial(x, x_guess)
        ocp.set_initial(y, y_guess)
        ocp.set_initial(theta, current_state[2])
        ocp.set_initial(v, max(current_state[3], 1.0))
        ocp.set_initial(a, 0.0)
        ocp.set_initial(delta, 0.0)

    try:
        sol = ocp.solve()
    except Exception as exc:
        print(f"Rockit solver failed; using fallback action. Reason: {exc}")
        action = fallback_goal_action(current_state, goal, args)
        return action, np.array([current_state.copy()]), []

    # Sample controls and states
    _, a_values = sol.sample(a, grid="control")
    _, delta_values = sol.sample(delta, grid="control")

    _, x_values = sol.sample(x, grid="control")
    _, y_values = sol.sample(y, grid="control")
    _, theta_values = sol.sample(theta, grid="control")
    _, v_values = sol.sample(v, grid="control")

    first_action = np.array([float(a_values[0]), float(delta_values[0])])

    best_predicted_trajectory = np.column_stack(
        [x_values, y_values, theta_values, v_values]
    )

    # Save warm-start guess for next iteration.
    # Shift the solution forward by one step and repeat the last value.
    def shift_guess(arr):
        arr = np.asarray(arr).reshape(-1)
        if len(arr) <= 1:
            return arr
        return np.concatenate([arr[1:], arr[-1:]])

    previous_solution_guess = {
        "x": shift_guess(x_values),
        "y": shift_guess(y_values),
        "theta": shift_guess(theta_values),
        "v": shift_guess(v_values),
        "a": shift_guess(a_values),
        "delta": shift_guess(delta_values),
    }

    return first_action, best_predicted_trajectory, []


# -----------------------------
# Drawing helpers
# -----------------------------
def draw_car(ax, state):
    x, y, theta, v = state

    car_length = 1.2
    car_width = 0.6

    front = np.array([
        x + car_length * np.cos(theta),
        y + car_length * np.sin(theta),
    ])

    left = np.array([
        x + car_width * np.cos(theta + 2.5),
        y + car_width * np.sin(theta + 2.5),
    ])

    right = np.array([
        x + car_width * np.cos(theta - 2.5),
        y + car_width * np.sin(theta - 2.5),
    ])

    car_shape = np.array([front, left, right, front])
    ax.plot(car_shape[:, 0], car_shape[:, 1], linewidth=2)


def draw_obstacles(ax, obstacles, vehicle_radius=0.7, safety_margin=0.20):
    for obstacle in obstacles:
        center = obstacle["center"]
        radius = obstacle["radius"]

        circle = plt.Circle(center, radius, alpha=0.35)
        ax.add_patch(circle)

        collision_circle = plt.Circle(
            center,
            radius + vehicle_radius,
            fill=False,
            linestyle="-",
            alpha=0.5,
        )
        ax.add_patch(collision_circle)

        safety_circle = plt.Circle(
            center,
            radius + vehicle_radius + safety_margin,
            fill=False,
            linestyle="--",
            alpha=0.4,
        )
        ax.add_patch(safety_circle)


# -----------------------------
# Simulation setup
# -----------------------------
state = np.array([0.0, 0.0, 0.0, 2.0])
goal = np.array([20.0, 10.0])

actual_trajectory = [state.copy()]
history = []
collision_happened = False

print("Using Rockit-MECO NMPC planner with solver-stable soft obstacle penalties")
print(
    f"obstacle_slack_weight={args.obstacle_slack_weight}, "
    f"safety_margin={args.safety_margin}, "
    f"horizon_steps={args.horizon_steps}, "
    f"ipopt_max_iter={args.ipopt_max_iter}"
)


# -----------------------------
# Main simulation loop
# -----------------------------
for step in range(args.max_steps):
    action, best_predicted_trajectory, candidate_trajectories = choose_action_rockit_mpc(
        state,
        goal,
        obstacles,
        args,
    )

    history.append({
        "state": state.copy(),
        "actual_trajectory": np.array(actual_trajectory),
        "best_predicted_trajectory": best_predicted_trajectory,
        "candidate_trajectories": candidate_trajectories,
        "chosen_action": action.copy(),
        "collision": collision_happened,
    })

    state = vehicle_step(
        state,
        action,
        dt=args.dt,
        L=args.wheelbase,
        max_speed=args.max_speed,
        max_steer_deg=args.max_steer_deg,
    )

    actual_trajectory.append(state.copy())

    if is_collision(state, obstacles, vehicle_radius=args.vehicle_radius):
        print("Collision happened!")
        collision_happened = True
        break

    distance_to_goal = np.linalg.norm(state[:2] - goal)

    if distance_to_goal < args.target_radius:
        print("Target hit!")
        break


# -----------------------------
# Animation
# -----------------------------
fig, ax = plt.subplots(figsize=(9, 7))


def update(frame):
    ax.clear()

    data = history[frame]

    current_state = data["state"]
    actual = data["actual_trajectory"]
    best_pred = data["best_predicted_trajectory"]
    chosen_action = data["chosen_action"]

    draw_obstacles(
        ax,
        obstacles,
        vehicle_radius=args.vehicle_radius,
        safety_margin=args.safety_margin,
    )

    if best_pred is not None and len(best_pred) > 0:
        ax.plot(
            best_pred[:, 0],
            best_pred[:, 1],
            linewidth=3,
            label="Rockit optimized trajectory",
        )

    ax.plot(
        actual[:, 0],
        actual[:, 1],
        marker="o",
        linewidth=2,
        label="Actual trajectory",
    )

    draw_car(ax, current_state)

    target_circle = plt.Circle(
        goal,
        args.target_radius,
        fill=False,
        linestyle="--",
        linewidth=2,
        alpha=0.8,
    )
    ax.add_patch(target_circle)

    ax.scatter(goal[0], goal[1], s=120, marker="*", label="Goal")

    acceleration, steering = chosen_action

    ax.set_title(
        f"Rockit-MECO NMPC Solver-Stable Version - Step {frame}\n"
        f"Chosen action: acceleration={acceleration:.2f}, "
        f"steering={np.rad2deg(steering):.1f} degrees"
    )

    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="upper left")

    ax.set_xlim(-2, 24)
    ax.set_ylim(-4, 15)


ani = FuncAnimation(
    fig,
    update,
    frames=len(history),
    interval=250,
    repeat=False,
)

plt.show()

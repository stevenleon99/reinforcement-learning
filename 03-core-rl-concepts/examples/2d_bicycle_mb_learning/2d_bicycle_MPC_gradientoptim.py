import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
import tyro

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None


# -----------------------------
# Command-line arguments
# -----------------------------
@dataclass
class Args:
    choose_action_mpc: int = 4

    # Simulation settings
    max_steps: int = 150
    num_samples: int = 1500
    horizon: int = 70

    # Gradient optimizer MPC settings for choose_action_mpc_4
    optimizer_horizon: int = 35
    optimizer_max_iter: int = 80
    optimizer_num_restarts: int = 3
    optimizer_collision_penalty: float = 200000.0
    optimizer_obstacle_weight: float = 1500.0
    optimizer_goal_weight: float = 8.0
    optimizer_terminal_weight: float = 80.0
    optimizer_control_weight: float = 0.08
    optimizer_speed_weight: float = 0.05
    optimizer_smoothness_weight: float = 0.15

    # MPC score weights for choose_action_mpc_3
    progress_weight: float = 20.0
    closeness_weight: float = 2.0
    obstacle_weight: float = 300.0
    terminal_weight: float = 10.0
    speed_weight: float = 0.5
    steering_weight: float = 0.1

    # Steering exploration
    steering_noise_deg: float = 50.0

    # Target behavior
    target_radius: float = 1.2
    target_lock_radius: float = 3.0

    # Vehicle / obstacle safety
    vehicle_radius: float = 0.7
    safety_margin: float = 1.5


args = tyro.cli(Args)


# -----------------------------
# Vehicle model: kinematic bicycle
# -----------------------------
def vehicle_step(state, action, dt=0.1, L=2.5):
    """
    state = [x, y, theta, v]
    action = [acceleration, steering_angle]
    """

    x, y, theta, v = state
    a, delta = action

    max_steer = np.deg2rad(30)
    delta = np.clip(delta, -max_steer, max_steer)

    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + (v / L) * np.tan(delta) * dt
    v_next = v + a * dt

    v_next = np.clip(v_next, 0.0, 8.0)

    return np.array([x_next, y_next, theta_next, v_next])


# -----------------------------
# Obstacles
# -----------------------------
obstacles = [
    {"center": np.array([7.0, 3.0]), "radius": 1.5},
    {"center": np.array([12.0, 7.0]), "radius": 1.7},
    {"center": np.array([16.0, 5.0]), "radius": 1.3},
]


def obstacle_penalty(
    state,
    obstacles,
    vehicle_radius=0.7,
    safety_margin=1.5,
):
    """
    Penalize collision or getting too close to obstacles.
    This version considers the vehicle size.
    """

    x, y, theta, v = state
    position = np.array([x, y])

    total_penalty = 0.0

    for obstacle in obstacles:
        center = obstacle["center"]
        radius = obstacle["radius"]

        distance = np.linalg.norm(position - center)

        # Effective clearance includes vehicle radius
        clearance = distance - radius - vehicle_radius

        # Hard collision penalty
        if clearance <= 0:
            total_penalty += 100000.0

        # Soft near-obstacle penalty
        elif clearance < safety_margin:
            total_penalty += 1000.0 * (safety_margin - clearance) ** 2

    return total_penalty


# -----------------------------
# Cost function for MPC 1 and 2
# -----------------------------
def cost_function(state, goal, obstacles):
    x, y, theta, v = state
    goal_x, goal_y = goal

    distance_cost = (x - goal_x) ** 2 + (y - goal_y) ** 2
    speed_cost = 0.2 * v ** 2
    obs_cost = obstacle_penalty(state, obstacles)

    return distance_cost + speed_cost + obs_cost


# -----------------------------
# Helper: normalize angle
# -----------------------------
def angle_normalize(angle):
    """
    Normalize angle to [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


# -----------------------------
# MPC 1: pure random shooting
# -----------------------------
def choose_action_mpc_1(current_state, goal, obstacles, num_samples=300, horizon=30):
    best_cost = float("inf")
    best_first_action = None
    best_trajectory = None
    candidate_trajectories = []

    for _ in range(num_samples):
        state = current_state.copy()
        total_cost = 0.0
        trajectory = [state.copy()]
        action_sequence = []

        for _ in range(horizon):
            acceleration = np.random.uniform(-1.0, 1.0)
            steering = np.random.uniform(-np.deg2rad(30), np.deg2rad(30))

            action = np.array([acceleration, steering])
            action_sequence.append(action)

            state = vehicle_step(state, action)
            trajectory.append(state.copy())

            total_cost += cost_function(state, goal, obstacles)

            if obstacle_penalty(state, obstacles) >= 100000.0:
                total_cost += 1000000.0
                break

        trajectory = np.array(trajectory)
        candidate_trajectories.append(trajectory)

        if total_cost < best_cost:
            best_cost = total_cost
            best_first_action = action_sequence[0]
            best_trajectory = trajectory

    print(f"Best trajectory cost: {best_cost:.2f}")

    return best_first_action, best_trajectory, candidate_trajectories


# -----------------------------
# MPC 2: random shooting with terminal cost
# -----------------------------
def choose_action_mpc_2(current_state, goal, obstacles, num_samples=300, horizon=30):
    best_cost = float("inf")
    best_first_action = None
    best_trajectory = None
    candidate_trajectories = []

    for _ in range(num_samples):
        state = current_state.copy()
        total_cost = 0.0
        trajectory = [state.copy()]
        action_sequence = []

        for _ in range(horizon):
            acceleration = np.random.uniform(-1.0, 1.0)
            steering = np.random.uniform(-np.deg2rad(30), np.deg2rad(30))

            action = np.array([acceleration, steering])
            action_sequence.append(action)

            state = vehicle_step(state, action)
            trajectory.append(state.copy())

            total_cost += 0.1 * cost_function(state, goal, obstacles)

            if obstacle_penalty(state, obstacles) >= 100000.0:
                total_cost += 1000000.0
                break

        total_cost += 5.0 * cost_function(state, goal, obstacles)

        trajectory = np.array(trajectory)
        candidate_trajectories.append(trajectory)

        if total_cost < best_cost:
            best_cost = total_cost
            best_first_action = action_sequence[0]
            best_trajectory = trajectory

    print(f"Best trajectory cost: {best_cost:.2f}")

    return best_first_action, best_trajectory, candidate_trajectories


# -----------------------------
# MPC 3: goal-vector biased MPC with target lock and safety gate
# -----------------------------
def choose_action_mpc_3(
    current_state,
    goal,
    obstacles,
    num_samples=1500,
    horizon=70,
    progress_weight=20.0,
    closeness_weight=2.0,
    obstacle_weight=300.0,
    terminal_weight=10.0,
    speed_weight=0.5,
    steering_weight=0.1,
    steering_noise_deg=25.0,
    target_radius=1.2,
    target_lock_radius=3.0,
    vehicle_radius=0.7,
    safety_margin=1.5,
):
    """
    Goal-biased random shooting MPC with target lock and obstacle safety gate.

    Behavior:
    - Far from target: explore around the goal direction.
    - Close to target: reduce randomness and try to enter the target zone.
    - Unsafe predicted actions are rejected before being added to the candidate path.
    - Target-hit reward is only given when the state is safe.
    """

    best_score = -float("inf")
    best_first_action = None
    best_trajectory = None
    candidate_trajectories = []

    for _ in range(num_samples):
        state = current_state.copy()
        total_score = 0.0
        trajectory = [state.copy()]
        action_sequence = []

        previous_distance_to_goal = np.linalg.norm(state[:2] - goal)

        for _ in range(horizon):
            x, y, theta, v = state

            # -----------------------------
            # Vector from vehicle to goal
            # -----------------------------
            goal_vector = goal - np.array([x, y])
            distance_to_goal = np.linalg.norm(goal_vector)

            desired_heading = np.arctan2(goal_vector[1], goal_vector[0])
            heading_error = angle_normalize(desired_heading - theta)

            # -----------------------------
            # Goal-biased steering with target lock
            # -----------------------------
            max_steer = np.deg2rad(30)
            goal_steering = np.clip(heading_error, -max_steer, max_steer)

            # Close to target: reduce random exploration
            if distance_to_goal < target_lock_radius:
                steering_noise_scale = np.deg2rad(2.0)
            else:
                steering_noise_scale = np.deg2rad(steering_noise_deg)

            steering_noise = np.random.normal(
                loc=0.0,
                scale=steering_noise_scale,
            )

            steering = goal_steering + steering_noise
            steering = np.clip(steering, -max_steer, max_steer)

            # -----------------------------
            # Target-forcing acceleration
            # -----------------------------
            if distance_to_goal > 5.0:
                acceleration = np.random.uniform(0.0, 1.0)

            elif distance_to_goal > target_radius:
                # Near goal but not inside: move forward, but not too aggressively
                acceleration = np.random.uniform(-0.1, 0.4)

            else:
                # Inside target area: slow down
                acceleration = np.random.uniform(-1.0, -0.2)

            action = np.array([acceleration, steering])

            # -----------------------------
            # Safety gate: reject immediately unsafe predicted action
            # -----------------------------
            next_state_candidate = vehicle_step(state, action)

            next_obs_penalty = obstacle_penalty(
                next_state_candidate,
                obstacles,
                vehicle_radius=vehicle_radius,
                safety_margin=safety_margin,
            )

            if next_obs_penalty >= 100000.0:
                total_score -= 10000000.0
                break

            action_sequence.append(action)

            # Accept predicted next state
            state = next_state_candidate
            trajectory.append(state.copy())

            new_distance_to_goal = np.linalg.norm(state[:2] - goal)

            # -----------------------------
            # Reward / penalty terms
            # -----------------------------
            progress_reward = previous_distance_to_goal - new_distance_to_goal
            closeness_reward = -new_distance_to_goal

            obs_penalty = next_obs_penalty
            safe_state = obs_penalty < 100000.0

            # Target-hit reward only if safe
            target_hit_reward = 0.0

            if safe_state and new_distance_to_goal < target_radius:
                target_hit_reward = 20000.0

            if safe_state and new_distance_to_goal < target_radius and state[3] < 1.0:
                target_hit_reward = 50000.0

            # Stronger obstacle weight near target lock zone
            effective_obstacle_weight = obstacle_weight
            if new_distance_to_goal < target_lock_radius:
                effective_obstacle_weight = obstacle_weight * 5.0

            # Stronger speed penalty near target
            if new_distance_to_goal < target_lock_radius:
                speed_penalty = 3.0 * state[3] ** 2
            else:
                speed_penalty = 0.2 * state[3] ** 2

            steering_penalty = steering ** 2

            step_score = (
                progress_weight * progress_reward
                + closeness_weight * closeness_reward
                + target_hit_reward
                - effective_obstacle_weight * obs_penalty
                - speed_weight * speed_penalty
                - steering_weight * steering_penalty
            )

            total_score += step_score
            previous_distance_to_goal = new_distance_to_goal

            # Stop early if predicted trajectory hits target safely
            if safe_state and new_distance_to_goal < target_radius:
                total_score += 10000.0
                break

        # If no action was accepted, skip this candidate
        if len(action_sequence) == 0:
            continue

        # -----------------------------
        # Terminal reward
        # -----------------------------
        terminal_distance = np.linalg.norm(state[:2] - goal)
        terminal_speed = state[3]

        terminal_obs_penalty = obstacle_penalty(
            state,
            obstacles,
            vehicle_radius=vehicle_radius,
            safety_margin=safety_margin,
        )

        safe_terminal = terminal_obs_penalty < 100000.0

        total_score += (
            -terminal_weight * terminal_distance
            -5.0 * terminal_speed ** 2
            -obstacle_weight * terminal_obs_penalty
        )

        if safe_terminal and terminal_distance < target_radius:
            total_score += 50000.0

        if safe_terminal and terminal_distance < target_radius and terminal_speed < 1.0:
            total_score += 100000.0

        trajectory = np.array(trajectory)
        candidate_trajectories.append(trajectory)

        if total_score > best_score:
            best_score = total_score
            best_first_action = action_sequence[0]
            best_trajectory = trajectory

    # Fallback if all sampled trajectories were rejected
    if best_first_action is None:
        print("Warning: all sampled actions were unsafe. Braking.")
        best_first_action = np.array([-1.0, 0.0])
        best_trajectory = np.array([current_state.copy()])
        candidate_trajectories.append(best_trajectory)

    print(f"Best trajectory score: {best_score:.2f}")

    return best_first_action, best_trajectory, candidate_trajectories



# -----------------------------
# MPC 4: gradient-based nonlinear MPC using scipy.optimize
# -----------------------------
def rollout_trajectory(current_state, action_sequence, dt=0.1, L=2.5):
    """
    Simulate the vehicle forward using a full candidate action sequence.

    action_sequence shape = (horizon, 2)
    each action = [acceleration, steering_angle]
    """
    state = current_state.copy()
    trajectory = [state.copy()]

    for action in action_sequence:
        state = vehicle_step(state, action, dt=dt, L=L)
        trajectory.append(state.copy())

    return np.array(trajectory)


def smooth_obstacle_cost(
    state,
    obstacles,
    vehicle_radius=0.7,
    safety_margin=1.5,
    collision_penalty=200000.0,
    obstacle_weight=1500.0,
):
    """
    Differentiable-style obstacle cost for gradient-based optimization.

    It is still evaluated numerically by scipy, but it is smoother than the
    hard reject logic used in random shooting. This helps the optimizer find
    a direction for improvement.
    """
    position = state[:2]
    cost = 0.0

    for obstacle in obstacles:
        center = obstacle["center"]
        radius = obstacle["radius"]

        distance = np.linalg.norm(position - center) + 1e-6
        clearance = distance - radius - vehicle_radius

        # Very large cost if collision happens.
        if clearance <= 0:
            cost += collision_penalty + collision_penalty * (-clearance + 1.0) ** 2

        # Soft buffer cost inside the safety margin.
        elif clearance < safety_margin:
            cost += obstacle_weight * (safety_margin - clearance) ** 2

    return cost


def optimizer_objective(
    flat_actions,
    current_state,
    goal,
    obstacles,
    horizon,
    goal_weight,
    terminal_weight,
    control_weight,
    speed_weight,
    smoothness_weight,
    obstacle_weight,
    collision_penalty,
    vehicle_radius,
    safety_margin,
):
    """
    Objective minimized by gradient-based MPC.

    The decision variables are the entire future action sequence:
        [a_0, delta_0, a_1, delta_1, ..., a_H, delta_H]

    The optimizer changes these actions to minimize:
    - distance to goal
    - terminal distance to goal
    - obstacle cost
    - speed cost
    - control effort
    - action smoothness
    """
    action_sequence = flat_actions.reshape(horizon, 2)
    trajectory = rollout_trajectory(current_state, action_sequence)

    total_cost = 0.0
    previous_action = None

    for t in range(1, len(trajectory)):
        state = trajectory[t]
        action = action_sequence[t - 1]

        distance_to_goal = np.linalg.norm(state[:2] - goal)

        total_cost += goal_weight * distance_to_goal**2
        total_cost += speed_weight * state[3] ** 2
        total_cost += control_weight * (action[0] ** 2 + action[1] ** 2)

        total_cost += smooth_obstacle_cost(
            state,
            obstacles,
            vehicle_radius=vehicle_radius,
            safety_margin=safety_margin,
            collision_penalty=collision_penalty,
            obstacle_weight=obstacle_weight,
        )

        if previous_action is not None:
            action_change = action - previous_action
            total_cost += smoothness_weight * np.sum(action_change**2)

        previous_action = action

    terminal_state = trajectory[-1]
    terminal_distance = np.linalg.norm(terminal_state[:2] - goal)

    # Strong terminal pull toward the goal.
    total_cost += terminal_weight * terminal_distance**2

    # Encourage stopping near the target.
    if terminal_distance < 3.0:
        total_cost += 10.0 * terminal_state[3] ** 2

    return total_cost


def make_initial_action_guess(current_state, goal, horizon):
    """
    Build a reasonable initial action sequence for the optimizer.

    This is important because gradient-based optimization starts from one
    initial guess. Here we aim roughly toward the goal and use moderate
    acceleration.
    """
    x, y, theta, v = current_state
    goal_vector = goal - np.array([x, y])
    desired_heading = np.arctan2(goal_vector[1], goal_vector[0])
    heading_error = angle_normalize(desired_heading - theta)

    max_steer = np.deg2rad(30)
    base_steering = np.clip(heading_error, -max_steer, max_steer)

    distance_to_goal = np.linalg.norm(goal_vector)

    if distance_to_goal > 5.0:
        base_acceleration = 0.4
    else:
        base_acceleration = -0.4

    guess = np.zeros((horizon, 2))
    guess[:, 0] = base_acceleration
    guess[:, 1] = base_steering

    # Gradually reduce steering near the end for smoother behavior.
    guess[:, 1] *= np.linspace(1.0, 0.3, horizon)

    return guess


def choose_action_mpc_4(
    current_state,
    goal,
    obstacles,
    horizon=35,
    max_iter=80,
    num_restarts=3,
    goal_weight=8.0,
    terminal_weight=80.0,
    control_weight=0.08,
    speed_weight=0.05,
    smoothness_weight=0.15,
    obstacle_weight=150000.0,
    collision_penalty=30000000.0,
    vehicle_radius=0.7,
    safety_margin=1.5,
):
    """
    Gradient-based nonlinear MPC.

    Unlike random shooting, this method does not sample many random paths and
    pick the best one. Instead, it treats the future action sequence as an
    optimization variable and uses scipy.optimize.minimize to improve that
    sequence.

    This is conceptually closer to the Rockit/NMPC idea, but implemented in a
    lightweight way using scipy instead of Rockit.
    """
    if minimize is None:
        raise ImportError(
            "scipy is required for choose_action_mpc_4. Install it with: pip install scipy"
        )

    max_steer = np.deg2rad(30)

    # Bounds for every future action.
    bounds = []
    for _ in range(horizon):
        bounds.append((-1.0, 1.0))          # acceleration
        bounds.append((-max_steer, max_steer))  # steering

    base_guess = make_initial_action_guess(current_state, goal, horizon)

    initial_guesses = [base_guess]

    # Add a few small perturbation restarts so the optimizer has a better chance
    # to avoid poor local minima.
    for _ in range(max(0, num_restarts - 1)):
        guess = base_guess.copy()
        guess[:, 0] += np.random.normal(0.0, 0.15, size=horizon)
        guess[:, 1] += np.random.normal(0.0, np.deg2rad(8), size=horizon)
        guess[:, 0] = np.clip(guess[:, 0], -1.0, 1.0)
        guess[:, 1] = np.clip(guess[:, 1], -max_steer, max_steer)
        initial_guesses.append(guess)

    best_result = None
    best_cost = float("inf")
    candidate_trajectories = []

    for guess in initial_guesses:
        result = minimize(
            optimizer_objective,
            guess.reshape(-1),
            args=(
                current_state,
                goal,
                obstacles,
                horizon,
                goal_weight,
                terminal_weight,
                control_weight,
                speed_weight,
                smoothness_weight,
                obstacle_weight,
                collision_penalty,
                vehicle_radius,
                safety_margin,
            ),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": max_iter,
                "ftol": 1e-4,
                "maxls": 20,
            },
        )

        action_sequence = result.x.reshape(horizon, 2)
        trajectory = rollout_trajectory(current_state, action_sequence)
        candidate_trajectories.append(trajectory)

        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result

    if best_result is None:
        print("Warning: gradient optimizer failed. Braking.")
        best_first_action = np.array([-1.0, 0.0])
        best_trajectory = np.array([current_state.copy()])
        candidate_trajectories.append(best_trajectory)
        return best_first_action, best_trajectory, candidate_trajectories

    best_action_sequence = best_result.x.reshape(horizon, 2)
    best_trajectory = rollout_trajectory(current_state, best_action_sequence)
    best_first_action = best_action_sequence[0]

    print(
        f"Gradient MPC cost: {best_cost:.2f}, "
        f"success={best_result.success}, "
        f"message={best_result.message}"
    )

    return best_first_action, best_trajectory, candidate_trajectories



# -----------------------------
# Collision check for actual vehicle
# -----------------------------
def is_collision(state, obstacles, vehicle_radius=0.7):
    x, y, theta, v = state
    position = np.array([x, y])

    for obstacle in obstacles:
        center = obstacle["center"]
        radius = obstacle["radius"]

        distance = np.linalg.norm(position - center)

        if distance <= radius + vehicle_radius:
            return True

    return False


# -----------------------------
# Draw vehicle
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


# -----------------------------
# Draw obstacles
# -----------------------------
def draw_obstacles(ax, obstacles, vehicle_radius=0.7, safety_margin=1.5):
    for obstacle in obstacles:
        center = obstacle["center"]
        radius = obstacle["radius"]

        # Actual obstacle
        circle = plt.Circle(
            center,
            radius,
            alpha=0.35,
        )
        ax.add_patch(circle)

        # Collision boundary considering vehicle size
        collision_circle = plt.Circle(
            center,
            radius + vehicle_radius,
            fill=False,
            linestyle="-",
            alpha=0.5,
        )
        ax.add_patch(collision_circle)

        # Safety margin boundary
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


# -----------------------------
# Select planner
# -----------------------------
if args.choose_action_mpc == 1:
    choose_action_fn = choose_action_mpc_1
elif args.choose_action_mpc == 2:
    choose_action_fn = choose_action_mpc_2
elif args.choose_action_mpc == 3:
    choose_action_fn = choose_action_mpc_3
elif args.choose_action_mpc == 4:
    choose_action_fn = choose_action_mpc_4
else:
    raise ValueError("choose_action_mpc must be 1, 2, 3, or 4")

print(f"Using choose_action_mpc_{args.choose_action_mpc} for planning")


# -----------------------------
# Main simulation loop
# -----------------------------
for step in range(args.max_steps):
    if args.choose_action_mpc == 3:
        action, best_predicted_trajectory, candidate_trajectories = choose_action_fn(
            state,
            goal,
            obstacles,
            num_samples=args.num_samples,
            horizon=args.horizon,
            progress_weight=args.progress_weight,
            closeness_weight=args.closeness_weight,
            obstacle_weight=args.obstacle_weight,
            terminal_weight=args.terminal_weight,
            speed_weight=args.speed_weight,
            steering_weight=args.steering_weight,
            steering_noise_deg=args.steering_noise_deg,
            target_radius=args.target_radius,
            target_lock_radius=args.target_lock_radius,
            vehicle_radius=args.vehicle_radius,
            safety_margin=args.safety_margin,
        )
    elif args.choose_action_mpc == 4:
        action, best_predicted_trajectory, candidate_trajectories = choose_action_fn(
            state,
            goal,
            obstacles,
            horizon=args.optimizer_horizon,
            max_iter=args.optimizer_max_iter,
            num_restarts=args.optimizer_num_restarts,
            goal_weight=args.optimizer_goal_weight,
            terminal_weight=args.optimizer_terminal_weight,
            control_weight=args.optimizer_control_weight,
            speed_weight=args.optimizer_speed_weight,
            smoothness_weight=args.optimizer_smoothness_weight,
            obstacle_weight=args.optimizer_obstacle_weight,
            collision_penalty=args.optimizer_collision_penalty,
            vehicle_radius=args.vehicle_radius,
            safety_margin=args.safety_margin,
        )
    else:
        action, best_predicted_trajectory, candidate_trajectories = choose_action_fn(
            state,
            goal,
            obstacles,
            num_samples=args.num_samples,
            horizon=args.horizon,
        )

    history.append({
        "state": state.copy(),
        "actual_trajectory": np.array(actual_trajectory),
        "best_predicted_trajectory": best_predicted_trajectory,
        "candidate_trajectories": candidate_trajectories,
        "chosen_action": action.copy(),
        "collision": collision_happened,
    })

    # Apply only the first action from the best predicted trajectory
    state = vehicle_step(state, action)
    actual_trajectory.append(state.copy())

    if is_collision(
        state,
        obstacles,
        vehicle_radius=args.vehicle_radius,
    ):
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
    candidates = data["candidate_trajectories"]
    chosen_action = data["chosen_action"]

    draw_obstacles(
        ax,
        obstacles,
        vehicle_radius=args.vehicle_radius,
        safety_margin=args.safety_margin,
    )

    # Plot all candidate predicted trajectories
    for traj in candidates:
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            alpha=0.10,
            linewidth=0.8,
        )

    # Plot best predicted trajectory
    ax.plot(
        best_pred[:, 0],
        best_pred[:, 1],
        linewidth=3,
        label="Best predicted trajectory",
    )

    # Plot actual trajectory so far
    ax.plot(
        actual[:, 0],
        actual[:, 1],
        marker="o",
        linewidth=2,
        label="Actual trajectory",
    )

    draw_car(ax, current_state)

    # Target circle
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
        f"Model-Based RL / MPC - Step {frame}\n"
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
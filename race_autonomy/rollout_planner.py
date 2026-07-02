import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDrive
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path


class BasicMPPIDynamicHorizonPlanner(Node):
    """Lean adaptive-horizon rollout/MPPI-style racing planner.

    Design goal:
    - Keep the planner simple enough to race fast.
    - Use a long horizon only when the visible path is open.
    - Use a short horizon in tight corners / near walls so the best path does
      not swing into a wall.
    - Make the cost function strongly prefer the correct direction and make
      wrong-way rollouts almost never win.

    This is not the huge state-machine version. It has no cost map, no
    overtaking layer, and no sticky recovery mode.
    """

    def __init__(self):
        super().__init__('simple_centerline_racing_planner')

        self.scan_topic = '/scan'
        self.cmd_topic = '/cmd_ackermann'

        # ==========================================================
        # Vehicle model
        # ==========================================================
        self.wheelbase = 0.30
        self.max_steer = 0.50
        self.rollout_steer_rate = 6.8

        # ==========================================================
        # Rollout / MPPI-style samples
        # ==========================================================
        self.num_steers = 111
        self.dt = 0.055

        # Adaptive horizon. The code changes the number of simulated steps
        # every scan. Long open view => long horizon. Tight / near wall => short.
        self.min_horizon_time = 0.95
        self.mid_horizon_time = 1.45
        self.max_horizon_time = 2.35

        self.min_preview_distance = 1.05
        self.max_preview_distance = 4.15

        # ==========================================================
        # LiDAR
        # ==========================================================
        self.max_obstacle_range = 5.0
        self.front_angle_limit_deg = 128.0
        self.collision_radius = 0.14
        self.max_side_distance = 2.5

        # ==========================================================
        # Target / racing line
        # ==========================================================
        self.smoothed_left = self.max_side_distance
        self.smoothed_right = self.max_side_distance
        self.have_sides = False
        self.side_smoothing_alpha = 0.22

        self.smoothed_far_front = self.max_obstacle_range
        self.have_far_front = False
        self.far_front_alpha = 0.30

        self.smoothed_target_y = 0.0
        self.have_target = False

        self.side_push_start = 0.48
        self.side_push_gain = 0.74
        self.max_side_push_y = 0.12

        # Turn signal from open-space / front geometry.
        self.corner_front_start = 3.05
        self.corner_front_full = 0.70
        self.corner_bias_gain = 0.15
        self.max_corner_bias_y = 0.18

        # Short turn lock, but it can flip quickly when the real turn changes.
        self.turn_sign = 0.0
        self.turn_timer = 0.0
        self.turn_hold_time = 0.36
        self.turn_start_strength = 0.26
        self.turn_flip_strength = 0.44

        # Racing line is deliberately small. We are not trying to hug the wall.
        self.racing_wall_buffer = 0.60
        self.racing_max_offset = 0.18
        self.racing_min_offset = 0.035
        self.center_limit = 0.13

        # ==========================================================
        # Speed
        # ==========================================================
        self.max_speed = 3.65
        self.min_speed = 0.25
        self.lateral_accel_limit = 4.55

        self.max_accel_step = 0.38
        self.straight_accel_step = 0.72
        self.max_decel_step = 0.60
        self.emergency_decel_step = 1.05
        self.max_steer_step = 0.42

        self.min_planning_speed = 1.00
        self.max_planning_speed = 3.10
        self.preview_boost = 0.58

        # Race-fix V2: do not let the car enter a blind/tight turn at
        # full speed. The previous dynamic horizon still logged SHORT
        # horizon while speed was above 3 m/s, so the turn was detected
        # too late.
        self.high_speed_horizon_speed = 2.20
        self.min_high_speed_horizon_time = 1.38

        # V5 turn lookahead tune:
        # V4 was finally stable, but the video shows the car starts several
        # turns slightly late because the rollout horizon collapses to SHORT
        # too early. Keep preview speed controlled in turns, but allow a
        # little more horizon time so the selected arc starts bending sooner.
        self.turn_lookahead_min_time = 1.12
        self.turn_lookahead_mid_time = 1.30
        self.turn_lookahead_max_time = 1.55
        self.turn_lookahead_front_gate = 1.75

        self.pre_turn_front_start = 3.05
        self.pre_turn_front_mid = 1.55
        self.pre_turn_front_near = 0.85

        # Overshoot is now treated as a path-quality failure, not just a
        # soft cost. A path far past target_y toward the wall should not win.
        self.max_good_overshoot = 0.11
        self.max_relaxed_overshoot = 0.18
        self.straight_preferred_lat = 0.24
        self.straight_safe_lat = 0.34
        self.straight_relaxed_lat = 0.44
        self.no_opposite_target_strength = 0.55
        self.min_same_turn_target = 0.020


        # Simple emergency escape. No sticky state machine, just one-cycle
        # command when the car is already almost touching the wall.
        # Emergency escape is now a LAST resort. The previous version
        # entered escape too early and forced steering opposite to the
        # desired turn, which made the car scrape along the wall.
        self.escape_front_threshold = 0.20
        self.escape_clearance_threshold = 0.090
        self.escape_speed = 0.30

        self.previous_steer = 0.0
        self.previous_speed = self.min_speed
        self.previous_desired_steer = 0.0
        self.callback_count = 0

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(
            AckermannDrive,
            self.cmd_topic,
            10,
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/yellow_car/rollouts',
            10,
        )

        self.best_path_pub = self.create_publisher(
            Path,
            '/yellow_car/best_path',
            10,
        )

        self.get_logger().info(
            'BASIC MPPI V5 started: slightly longer turn lookahead + cleaner fallback.'
        )

    # ==========================================================
    # Basic helpers
    # ==========================================================

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def sign_or_zero(self, value, deadband=1e-6):
        if value > deadband:
            return 1.0
        if value < -deadband:
            return -1.0
        return 0.0

    def valid_range(self, scan, r):
        if math.isnan(r) or math.isinf(r):
            return False
        if r < scan.range_min or r > scan.range_max:
            return False
        return True

    def sector_values(self, scan, min_deg, max_deg):
        values = []

        for i, r in enumerate(scan.ranges):
            if not self.valid_range(scan, r):
                continue

            angle = scan.angle_min + i * scan.angle_increment
            angle_deg = math.degrees(angle)

            if min_deg <= angle_deg <= max_deg:
                values.append(min(r, self.max_obstacle_range))

        return values

    def sector_percentile(self, scan, min_deg, max_deg, percentile):
        values = self.sector_values(scan, min_deg, max_deg)
        if not values:
            return self.max_obstacle_range

        values.sort()
        idx = int(self.clamp(percentile, 0.0, 1.0) * (len(values) - 1))
        return values[idx]

    def sector_mean_top(self, scan, min_deg, max_deg):
        values = self.sector_values(scan, min_deg, max_deg)
        if not values:
            return self.max_obstacle_range

        values.sort(reverse=True)
        top_n = max(1, len(values) // 3)
        return sum(values[:top_n]) / top_n

    def make_steering_samples(self):
        samples = set()

        for i in range(self.num_steers):
            ratio = i / max(1, self.num_steers - 1)
            steer = -self.max_steer + 2.0 * self.max_steer * ratio
            samples.add(round(steer, 5))

        # Exact useful racing candidates.
        for steer in [
            -0.50, -0.44, -0.38, -0.32, -0.26, -0.20, -0.14, -0.08, -0.04,
            0.0,
            0.04, 0.08, 0.14, 0.20, 0.26, 0.32, 0.38, 0.44, 0.50,
        ]:
            samples.add(round(steer, 5))

        return sorted(samples)

    # ==========================================================
    # LiDAR extraction
    # ==========================================================

    def scan_to_obstacle_points(self, scan):
        obstacles = []

        # Decimate to keep CPU low. This still keeps enough wall geometry.
        for i in range(0, len(scan.ranges), 2):
            r = scan.ranges[i]
            if not self.valid_range(scan, r):
                continue

            r = min(r, self.max_obstacle_range)
            angle = scan.angle_min + i * scan.angle_increment
            angle_deg = math.degrees(angle)

            if abs(angle_deg) > self.front_angle_limit_deg:
                continue

            x = r * math.cos(angle)
            y = r * math.sin(angle)

            if x < -0.10:
                continue

            obstacles.append((x, y))

        return obstacles

    def get_side_distances(self, scan):
        raw_left = self.sector_percentile(scan, 28, 108, 0.20)
        raw_right = self.sector_percentile(scan, -108, -28, 0.20)

        raw_left = min(raw_left, self.max_side_distance)
        raw_right = min(raw_right, self.max_side_distance)

        if not self.have_sides:
            self.smoothed_left = raw_left
            self.smoothed_right = raw_right
            self.have_sides = True
        else:
            a = self.side_smoothing_alpha
            self.smoothed_left = (1.0 - a) * self.smoothed_left + a * raw_left
            self.smoothed_right = (1.0 - a) * self.smoothed_right + a * raw_right

        return self.smoothed_left, self.smoothed_right, raw_left, raw_right

    def get_far_front(self, scan):
        raw_far_front = self.sector_percentile(scan, -22, 22, 0.12)

        if not self.have_far_front:
            self.smoothed_far_front = raw_far_front
            self.have_far_front = True
        else:
            a = self.far_front_alpha
            self.smoothed_far_front = (1.0 - a) * self.smoothed_far_front + a * raw_far_front

        return self.smoothed_far_front

    # ==========================================================
    # Target and turn estimate
    # ==========================================================

    def compute_side_push(self, raw_left, raw_right):
        push_y = 0.0

        # y positive is left.
        if raw_left < self.side_push_start:
            push_y -= self.side_push_gain * (self.side_push_start - raw_left)
        if raw_right < self.side_push_start:
            push_y += self.side_push_gain * (self.side_push_start - raw_right)

        return self.clamp(push_y, -self.max_side_push_y, self.max_side_push_y)

    def update_turn_lock(self, raw_turn_y, front, turn_strength):
        raw_sign = self.sign_or_zero(raw_turn_y, 0.030)
        turn_is_real = raw_sign != 0.0 and turn_strength > self.turn_start_strength and front < 2.85

        if turn_is_real:
            # Flip quickly when the turn really changed.
            if (
                self.turn_sign != 0.0
                and raw_sign != self.turn_sign
                and turn_strength > self.turn_flip_strength
                and abs(raw_turn_y) > 0.060
            ):
                self.turn_sign = raw_sign
                self.turn_timer = self.turn_hold_time
            elif self.turn_sign == 0.0:
                self.turn_sign = raw_sign
                self.turn_timer = self.turn_hold_time
            elif raw_sign == self.turn_sign:
                self.turn_timer = self.turn_hold_time
            else:
                self.turn_timer = max(0.0, self.turn_timer - 2.0 * self.dt)
        else:
            self.turn_timer = max(0.0, self.turn_timer - self.dt)

        if self.turn_timer <= 0.0:
            self.turn_sign = 0.0

        return self.turn_sign

    def compute_target(self, scan):
        left_dist, right_dist, raw_left, raw_right = self.get_side_distances(scan)

        front = self.sector_percentile(scan, -14, 14, 0.10)
        far_front = self.get_far_front(scan)
        left_open = self.sector_mean_top(scan, 22, 100)
        right_open = self.sector_mean_top(scan, -100, -22)

        min_side = min(left_dist, right_dist)
        wall_proximity = self.clamp(1.0 - (min_side - 0.48) / 1.10, 0.0, 1.0)

        # Local center estimate from side distances. Positive y = left.
        center_y = self.clamp(
            0.30 * (left_dist - right_dist) * wall_proximity,
            -self.center_limit,
            self.center_limit,
        )

        side_push_y = self.compute_side_push(raw_left, raw_right)

        corner_strength = self.clamp(
            (self.corner_front_start - front)
            / max(self.corner_front_start - self.corner_front_full, 1e-6),
            0.0,
            1.0,
        )

        open_diff = self.clamp(left_open - right_open, -1.15, 1.15)
        raw_turn_y = self.clamp(
            self.corner_bias_gain * open_diff * corner_strength,
            -self.max_corner_bias_y,
            self.max_corner_bias_y,
        )

        turn_strength = self.clamp(
            max(
                abs(raw_turn_y) / max(self.max_corner_bias_y, 1e-6),
                0.78 * corner_strength,
            ),
            0.0,
            1.0,
        )

        turn_sign = self.update_turn_lock(raw_turn_y, front, turn_strength)
        if turn_sign == 0.0 and turn_strength > 0.40:
            turn_sign = self.sign_or_zero(raw_turn_y, 0.035)

        # Racing line target: small offset toward turn direction only if safe.
        if turn_sign != 0.0 and turn_strength > 0.30:
            side_wall_distance = left_dist if turn_sign > 0.0 else right_dist
            safe_offset = self.clamp(
                side_wall_distance - self.racing_wall_buffer,
                0.0,
                self.racing_max_offset,
            )

            if safe_offset <= self.racing_min_offset:
                racing_y = 0.0
            else:
                racing_y = turn_sign * self.clamp(
                    0.42 * safe_offset,
                    self.racing_min_offset,
                    self.racing_max_offset,
                )

            race_blend = self.clamp((turn_strength - 0.25) / 0.65, 0.0, 1.0)
            center_return_y = self.clamp(center_y + side_push_y, -self.center_limit, self.center_limit)
            target_raw = (1.0 - race_blend) * center_return_y + race_blend * racing_y
            target_raw = self.clamp(target_raw, -self.racing_max_offset, self.racing_max_offset)
        else:
            racing_y = 0.0
            race_blend = 0.0
            target_raw = self.clamp(0.70 * center_y + side_push_y, -self.center_limit, self.center_limit)

        if not self.have_target:
            self.smoothed_target_y = target_raw
            self.have_target = True
        else:
            # Fast enough to respond before the turn, not so fast that target jumps.
            if turn_sign != 0.0 or turn_strength > 0.35:
                alpha = 0.38
            else:
                alpha = 0.45

            # If sign changed and the old target is on the wrong side, snap faster.
            if turn_sign != 0.0 and self.smoothed_target_y * turn_sign < -0.02:
                alpha = 0.90

            self.smoothed_target_y = (1.0 - alpha) * self.smoothed_target_y + alpha * target_raw

        target_y = self.clamp(self.smoothed_target_y, -0.24, 0.24)

        # V4 fix: when a real turn is detected, never keep the target on
        # the opposite side. In the video/log this caused delayed turn entry:
        # turn_sign=-1 while target_y was still slightly positive. If the
        # wall on the turn side is too close, use neutral center instead of
        # forcing an unsafe wall-hugging target.
        if turn_sign != 0.0 and turn_strength > self.no_opposite_target_strength:
            if target_y * turn_sign < 0.0:
                if abs(racing_y) >= self.racing_min_offset:
                    target_y = turn_sign * max(self.min_same_turn_target, 0.50 * abs(racing_y))
                else:
                    target_y = 0.0
                self.smoothed_target_y = target_y
            elif (
                abs(racing_y) >= self.racing_min_offset
                and target_y * turn_sign < self.min_same_turn_target
            ):
                target_y = turn_sign * self.min_same_turn_target
                self.smoothed_target_y = target_y

        return {
            'target_y': target_y,
            'raw_turn_y': raw_turn_y,
            'turn_sign': turn_sign,
            'turn_strength': turn_strength,
            'front': front,
            'far_front': far_front,
            'left_dist': left_dist,
            'right_dist': right_dist,
            'raw_left': raw_left,
            'raw_right': raw_right,
            'left_open': left_open,
            'right_open': right_open,
            'side_push_y': side_push_y,
            'center_y': center_y,
            'racing_y': racing_y,
            'race_blend': race_blend,
            'min_side': min(raw_left, raw_right),
        }

    # ==========================================================
    # Dynamic horizon / rollout simulation
    # ==========================================================

    def compute_dynamic_horizon(self, target_info):
        turn_strength = target_info['turn_strength']
        front = target_info['front']
        far_front = target_info['far_front']
        min_side = target_info['min_side']

        open_front_score = self.clamp((far_front - 0.90) / 2.70, 0.0, 1.0)
        open_side_score = self.clamp((min_side - 0.42) / 0.48, 0.0, 1.0)
        straight_score = self.clamp(1.0 - turn_strength, 0.0, 1.0)

        open_path_score = 0.50 * open_front_score + 0.25 * open_side_score + 0.25 * straight_score

        # Desired preview distance. Long on open straights, short in tight turns.
        desired_distance = (
            self.min_preview_distance
            + open_path_score * (self.max_preview_distance - self.min_preview_distance)
        )

        # In strong turns, do not allow too long a rollout, because it will
        # swing far past the current turn entry.
        if turn_strength > 0.45 or target_info['turn_sign'] != 0.0:
            desired_distance = min(desired_distance, 2.20 - 0.45 * turn_strength)

        # If the front is very short, keep the horizon short and precise.
        if front < 1.10:
            desired_distance = min(desired_distance, max(0.90, front + 0.70))

        planning_speed = self.previous_speed + self.preview_boost
        planning_speed = self.clamp(planning_speed, self.min_planning_speed, self.max_planning_speed)

        # Use the distance to determine horizon time.
        horizon_time = desired_distance / max(planning_speed, 0.30)
        horizon_time = self.clamp(horizon_time, self.min_horizon_time, self.max_horizon_time)

        # Important fix: if the real car is already fast, do not use a tiny
        # horizon just because one side wall is close. That was the bad log:
        # speed > 3 m/s with horizon=SHORT, then the corner appeared too late.
        if self.previous_speed > self.high_speed_horizon_speed and front > 1.35:
            horizon_time = max(horizon_time, self.min_high_speed_horizon_time)

        # In a clean open straight, allow high preview speed and long horizon.
        if open_path_score > 0.75 and turn_strength < 0.25:
            planning_speed = min(planning_speed + 0.25, self.max_planning_speed)
            horizon_time = max(horizon_time, 1.85)

        # In tight / near-wall turns, keep rollout speed limited, but do
        # not make the horizon too short. The current video shows the car is
        # stable but begins some turns late; a slightly longer turn preview
        # lets the green rollout bend earlier without making it wall-hug.
        near_wall = min_side < 0.42
        turn_context = turn_strength > 0.42 or target_info['turn_sign'] != 0.0

        if turn_context:
            if front > self.turn_lookahead_front_gate:
                horizon_time = max(horizon_time, self.turn_lookahead_max_time)
            elif front > 1.05:
                horizon_time = max(horizon_time, self.turn_lookahead_mid_time)
            else:
                horizon_time = max(horizon_time, self.turn_lookahead_min_time)

        if turn_strength > 0.55 or near_wall:
            planning_speed = min(planning_speed, 1.62 - 0.20 * turn_strength)
            planning_speed = max(planning_speed, self.min_planning_speed)

            if self.previous_speed > self.high_speed_horizon_speed and front > 1.35:
                # Fast car approaching turn: keep enough lookahead to start
                # steering early, but cap it so the arc does not overshoot.
                horizon_time = min(max(horizon_time, 1.42), 1.70)
            elif front > 1.10:
                horizon_time = min(max(horizon_time, self.turn_lookahead_mid_time), 1.55)
            else:
                horizon_time = min(max(horizon_time, self.turn_lookahead_min_time), 1.36)

        steps = int(self.clamp(round(horizon_time / self.dt), 14, 48))
        actual_horizon = steps * self.dt
        preview_distance = planning_speed * actual_horizon

        if actual_horizon >= 1.80:
            horizon_mode = 'LONG'
        elif actual_horizon <= 1.18:
            horizon_mode = 'SHORT'
        else:
            horizon_mode = 'MID'

        return planning_speed, steps, actual_horizon, preview_distance, horizon_mode

    def simulate_rollout(self, target_steer, planning_speed, steps):
        x = 0.0
        y = 0.0
        yaw = 0.0
        sim_steer = self.previous_steer
        path = []

        for _ in range(steps):
            delta = target_steer - sim_steer
            max_delta = self.rollout_steer_rate * self.dt
            delta = self.clamp(delta, -max_delta, max_delta)

            sim_steer += delta
            sim_steer = self.clamp(sim_steer, -self.max_steer, self.max_steer)

            x += planning_speed * math.cos(yaw) * self.dt
            y += planning_speed * math.sin(yaw) * self.dt
            yaw += (planning_speed / self.wheelbase) * math.tan(sim_steer) * self.dt

            path.append((x, y, yaw))

        return path

    # ==========================================================
    # Rollout scoring / reward
    # ==========================================================

    def min_clearance_along_path(self, path, obstacles):
        min_clearance = self.max_obstacle_range
        sampled_path = path[::2]

        for px, py, _pyaw in sampled_path:
            for ox, oy in obstacles:
                dx = px - ox
                dy = py - oy
                dist_sq = dx * dx + dy * dy
                if dist_sq < min_clearance * min_clearance:
                    min_clearance = math.sqrt(dist_sq)

        return min_clearance

    def average_path_y(self, path):
        total = 0.0
        weight_sum = 0.0
        n = max(1, len(path) - 1)

        for idx, (_x, y, _yaw) in enumerate(path):
            progress = idx / n
            weight = 1.0 + 0.15 * progress
            total += weight * y
            weight_sum += weight

        return total / max(weight_sum, 1e-6)

    def score_rollout(self, path, steer, obstacles, target_info, planning_speed, preview_distance):
        target_y = target_info['target_y']
        turn_sign = target_info['turn_sign']
        turn_strength = target_info['turn_strength']
        raw_turn_y = target_info['raw_turn_y']

        min_clearance = self.min_clearance_along_path(path, obstacles)
        collision = min_clearance < self.collision_radius

        one_third_y = path[len(path) // 3][1]
        mid_y = path[len(path) // 2][1]
        final_x, final_y, final_yaw = path[-1]
        avg_y = self.average_path_y(path)

        avg_error = avg_y - target_y
        mid_error = mid_y - target_y
        final_error = final_y - target_y

        cost = 0.0
        reward = 0.0
        wrong_way_score = 0.0
        overshoot_score = 0.0

        # 1) Hard safety.
        if collision:
            cost += 18000.0

        if min_clearance < 0.42:
            cost += 44.0 * (0.42 - min_clearance) ** 2 / max(min_clearance, 0.03)
        if min_clearance < 0.29:
            cost += 220.0 * (0.29 - min_clearance) ** 2 / max(min_clearance, 0.03)
        if min_clearance < 0.18:
            cost += 1100.0 * (0.18 - min_clearance) ** 2 / max(min_clearance, 0.03)

        # 2) Track the intended local line. This is the main MPPI reward/cost.
        cost += (18.0 + 24.0 * turn_strength) * avg_error ** 2
        cost += (12.0 + 14.0 * turn_strength) * mid_error ** 2
        cost += (14.0 + 16.0 * turn_strength) * final_error ** 2

        # 3) Straight: center is strongly preferred.
        if turn_strength < 0.32:
            cost += 18.0 * avg_y ** 2 + 12.0 * final_y ** 2

        # 4) Correct direction reward / wrong-way punishment.
        if turn_sign != 0.0 and turn_strength > 0.30:
            wrong_avg = max(0.0, -(avg_y * turn_sign) + 0.018)
            wrong_mid = max(0.0, -(mid_y * turn_sign) + 0.018)
            wrong_early = max(0.0, -(one_third_y * turn_sign) + 0.012)
            wrong_way_score = wrong_avg + 0.70 * wrong_mid + 0.50 * wrong_early

            # This makes wrong direction almost impossible to win.
            cost += (650.0 + 480.0 * turn_strength) * wrong_way_score ** 2

            # Reward the correct corridor, not simply being far on the
            # correct side. The previous reward encouraged wall-side
            # overshoot because any same-side y looked good.
            line_reward = self.clamp(0.14 - abs(avg_error), 0.0, 0.14)
            reward += (0.9 + 1.4 * turn_strength) * line_reward

            target_side_y = max(0.0, target_y * turn_sign)
            allowed_overshoot = 0.20 + 0.07 * (1.0 - turn_strength)
            overshoot_avg = max(0.0, avg_y * turn_sign - target_side_y - allowed_overshoot)
            overshoot_mid = max(0.0, mid_y * turn_sign - target_side_y - allowed_overshoot - 0.04)
            overshoot_final = max(0.0, final_y * turn_sign - target_side_y - allowed_overshoot - 0.12)
            overshoot_score = overshoot_avg + 0.70 * overshoot_mid + 0.40 * overshoot_final
            cost += (240.0 + 230.0 * turn_strength) * overshoot_score ** 2

            # Steering against the real turn is heavily bad unless it is tiny.
            if steer * turn_sign < 0.0 and abs(steer) > 0.045:
                cost += (95.0 + 135.0 * turn_strength) * abs(steer)
                wrong_way_score += abs(steer)

        # 5) Heading points into the detected turn.
        if abs(raw_turn_y) > 0.020:
            desired_final_yaw = self.clamp(1.50 * raw_turn_y, -0.42, 0.42)
            cost += (3.0 + 5.0 * turn_strength) * (final_yaw - desired_final_yaw) ** 2

        # 6) Forward progress. A path that spins sideways is bad.
        min_forward = 0.54 * max(0.20, preview_distance)
        if final_x < min_forward:
            cost += 58.0 * (min_forward - final_x) ** 2

        # 7) Smoothness.
        if self.previous_steer * steer < 0.0:
            if abs(self.previous_steer) > 0.12 and abs(steer) > 0.12:
                cost += 7.0 * (abs(self.previous_steer) + abs(steer))

        cost += 5.0 * (steer - self.previous_steer) ** 2
        cost += 0.10 * steer ** 2

        # Apply reward as negative cost, but keep it modest.
        cost -= self.clamp(reward, 0.0, 0.35)

        return {
            'cost': cost,
            'reward': reward,
            'wrong_way_score': wrong_way_score,
            'overshoot_score': overshoot_score,
            'clearance': min_clearance,
            'collision': collision,
            'avg_y': avg_y,
            'lateral_offset': avg_y - target_y,
            'path': path,
            'steer': steer,
            'final_x': final_x,
        }

    def choose_best_rollout(self, results, target_info):
        turn_strength = target_info['turn_strength']
        turn_sign = target_info['turn_sign']

        if turn_sign == 0.0 and turn_strength < 0.28:
            # Straight/open section: do not accept a rollout drifting half a
            # track-width away from the target just because it has clearance.
            preferred_lat = self.straight_preferred_lat
            safe_lat = self.straight_safe_lat
            relaxed_lat = self.straight_relaxed_lat
        else:
            preferred_lat = self.clamp(0.44 - 0.14 * turn_strength, 0.28, 0.44)
            safe_lat = self.clamp(0.56 - 0.17 * turn_strength, 0.36, 0.56)
            relaxed_lat = self.clamp(0.68 - 0.20 * turn_strength, 0.46, 0.68)

        def good_direction(r):
            if turn_sign == 0.0 or turn_strength < 0.30:
                return (
                    r['overshoot_score'] < 0.18
                    and abs(r['lateral_offset']) <= relaxed_lat
                )
            return (
                r['wrong_way_score'] < 0.045
                and r['overshoot_score'] < self.max_good_overshoot
            )

        preferred = [
            i for i, r in enumerate(results)
            if (not r['collision'])
            and r['clearance'] >= 0.33
            and abs(r['lateral_offset']) <= preferred_lat
            and good_direction(r)
        ]

        safe = [
            i for i, r in enumerate(results)
            if (not r['collision'])
            and r['clearance'] >= 0.25
            and abs(r['lateral_offset']) <= safe_lat
            and good_direction(r)
        ]

        relaxed = [
            i for i, r in enumerate(results)
            if (not r['collision'])
            and r['clearance'] >= 0.18
            and abs(r['lateral_offset']) <= relaxed_lat
            and r['wrong_way_score'] < 0.12
            and r['overshoot_score'] < self.max_relaxed_overshoot
        ]

        if preferred:
            return min(preferred, key=lambda i: results[i]['cost']), 'PREFERRED'
        if safe:
            return min(safe, key=lambda i: results[i]['cost']), 'SAFE'
        if relaxed:
            return min(relaxed, key=lambda i: results[i]['cost']), 'RELAXED'

        # Fallback: V4 sometimes chose MAX_CLEARANCE with a huge lateral
        # drift/overshoot, because clearance alone looked best. For racing,
        # a slightly lower-clearance path that follows the intended line is
        # better than a wide path that turns late or points at the wall.
        fallback = [
            i for i, r in enumerate(results)
            if (not r['collision'])
            and r['clearance'] >= 0.15
            and r['wrong_way_score'] < 0.22
            and r['overshoot_score'] < 0.42
        ]

        if fallback:
            return min(
                fallback,
                key=lambda i: (
                    results[i]['cost']
                    + 10.0 * abs(results[i]['lateral_offset'])
                    + 16.0 * results[i]['overshoot_score']
                    + 18.0 * results[i]['wrong_way_score']
                    - 2.0 * results[i]['clearance']
                    + 1.5 * abs(results[i]['steer'] - self.previous_steer)
                ),
            ), 'FALLBACK_LINE'

        # True emergency fallback: still do not love wrong-way. It can choose
        # it only if there is really no clean path.
        best_i = max(
            range(len(results)),
            key=lambda i: (
                results[i]['clearance']
                - 0.72 * abs(results[i]['lateral_offset'])
                - 0.95 * results[i]['wrong_way_score']
                - 1.05 * results[i]['overshoot_score']
                - 0.10 * abs(results[i]['steer'] - self.previous_steer)
            ),
        )
        return best_i, 'MAX_CLEARANCE'

    # ==========================================================
    # Command policy
    # ==========================================================

    def smooth_steering(self, desired_steer, clearance, turn_strength, turn_sign=0.0, front=5.0):
        """Steering smoothing with fast turn-entry snap.

        The bad video showed this pattern:
          desired_steer was already positive for the turn,
          but the commanded steer stayed negative because smoothing/escape
          was fighting the new direction.

        Here, when the turn signal is strong and the rollout agrees with it,
        we allow a quick sign change. This is important for competition speed:
        braking without turning early just drives into the wall.
        """
        urgency = self.clamp((0.36 - clearance) / 0.24, 0.0, 1.0)
        max_step = self.max_steer_step * (1.0 + 0.55 * urgency + 0.28 * turn_strength)

        if (
            turn_sign != 0.0
            and turn_strength > 0.55
            and front < 1.85
            and desired_steer * turn_sign > 0.035
        ):
            # If we were steering the wrong way or almost straight, snap into
            # the turn direction. This fixes the "desired=+0.24 but steer=-0.03"
            # type of failure.
            min_turn_steer = 0.20 + 0.14 * self.clamp((1.55 - front) / 0.85, 0.0, 1.0)
            desired_steer = turn_sign * max(abs(desired_steer), min_turn_steer)
            max_step = max(max_step, 0.78)

        delta = desired_steer - self.previous_steer
        delta = self.clamp(delta, -max_step, max_step)

        return self.clamp(self.previous_steer + delta, -self.max_steer, self.max_steer)

    def choose_speed(self, steer, best, target_info, mode, horizon_mode):
        clearance = best['clearance']
        collision = best['collision']
        turn_strength = target_info['turn_strength']
        front = target_info['front']
        far_front = target_info['far_front']
        lateral_error = abs(best['lateral_offset'])
        wrong_way_score = best['wrong_way_score']
        overshoot_score = best['overshoot_score']

        abs_steer = abs(steer)
        tan_steer = abs(math.tan(abs_steer))

        if tan_steer < 1e-4:
            curvature_speed = self.max_speed
        else:
            curvature_speed = math.sqrt(self.lateral_accel_limit * self.wheelbase / tan_steer)

        target_speed = min(self.max_speed, curvature_speed)

        # Clearance is the only reason to crawl.
        if clearance <= self.collision_radius:
            clearance_speed = self.min_speed
        elif clearance >= 0.42:
            clearance_speed = self.max_speed
        else:
            ratio = (clearance - self.collision_radius) / max(0.42 - self.collision_radius, 1e-6)
            clearance_speed = self.min_speed + ratio * (self.max_speed - self.min_speed)
        target_speed = min(target_speed, clearance_speed)

        # Front limit. Relax it in turns, because front ray sees the outside wall.
        warning = self.clamp(1.55 - 0.52 * turn_strength, 0.92, 1.55)
        stop = 0.34

        if far_front <= stop:
            front_speed = self.min_speed
        elif far_front >= warning:
            front_speed = self.max_speed
        else:
            ratio = (far_front - stop) / max(warning - stop, 1e-6)
            front_speed = self.min_speed + ratio * (self.max_speed - self.min_speed)

        if turn_strength > 0.35 and clearance > 0.30 and not collision:
            front_speed = max(front_speed, 1.30)
        target_speed = min(target_speed, front_speed)

        # Fast, but do not enter tight turns at full speed.
        if turn_strength > 0.25:
            turn_cap = self.max_speed - 1.30 * turn_strength
            turn_cap = self.clamp(turn_cap, 1.35, self.max_speed)
            target_speed = min(target_speed, turn_cap)

        # Pre-turn braking from front distance. This is the missing piece in
        # the bad log: the car was still >3 m/s when the visible front path
        # was already shortening, so it arrived at the corner too late.
        front_min = min(front, far_front)
        min_side = target_info['min_side']

        # Wall-side speed discipline. If one side wall is already close,
        # the car must not arrive at the next corner at full speed.
        if min_side < 0.36:
            target_speed = min(target_speed, 1.10)
        elif min_side < 0.46 and self.previous_speed > 1.85:
            target_speed = min(target_speed, 1.55)
        elif min_side < 0.54 and self.previous_speed > 2.30:
            target_speed = min(target_speed, 2.05)

        # In a real corner, the front ray often sees the outside wall. Brake
        # before the corner, but do not over-crawl when the chosen rollout has
        # good clearance and low path error.
        stable_turn_path = (
            turn_strength > 0.55
            and clearance > 0.30
            and lateral_error < 0.26
            and wrong_way_score < 0.04
            and overshoot_score < 0.12
            and not collision
        )

        if front_min < self.pre_turn_front_near:
            target_speed = min(target_speed, 1.18 if stable_turn_path else 0.90)
        elif front_min < self.pre_turn_front_mid:
            target_speed = min(target_speed, 1.55 if stable_turn_path else 1.28)
        elif front_min < self.pre_turn_front_start and self.previous_speed > 1.90:
            target_speed = min(target_speed, 2.05 if stable_turn_path else 1.82)

        # If path quality is bad, slow down; if it is good, do not over-limit.
        if lateral_error > 0.70:
            target_speed = min(target_speed, 1.15)
        elif lateral_error > 0.55:
            target_speed = min(target_speed, 1.55)
        elif lateral_error > 0.42:
            target_speed = min(target_speed, 1.95)

        if wrong_way_score > 0.12:
            target_speed = min(target_speed, 0.95)
        elif wrong_way_score > 0.05:
            target_speed = min(target_speed, 1.35)

        if overshoot_score > 0.28:
            target_speed = min(target_speed, 1.05)
        elif overshoot_score > 0.16:
            target_speed = min(target_speed, 1.35)

        if mode == 'MAX_CLEARANCE':
            if clearance < 0.20:
                target_speed = min(target_speed, 0.82)
            else:
                target_speed = min(target_speed, 1.45)

        if horizon_mode == 'SHORT' and clearance < 0.26:
            target_speed = min(target_speed, 1.15)

        if collision:
            target_speed = min(target_speed, self.min_speed)

        emergency = collision or clearance < 0.18 or front < 0.32

        if target_speed > self.previous_speed:
            straight_fast = (
                horizon_mode == 'LONG'
                and turn_strength < 0.20
                and abs_steer < 0.10
                and lateral_error < 0.22
                and clearance > 0.44
                and front > 2.00
            )
            accel_step = self.straight_accel_step if straight_fast else self.max_accel_step
            accel_step *= self.clamp(1.0 - 0.10 * turn_strength, 0.78, 1.0)
            speed = min(target_speed, self.previous_speed + accel_step)
        else:
            decel_step = self.emergency_decel_step if emergency else self.max_decel_step
            speed = max(target_speed, self.previous_speed - decel_step)

        return self.clamp(speed, self.min_speed, self.max_speed)

    def publish_command(self, steer, speed):
        cmd = AckermannDrive()
        cmd.steering_angle = float(steer)
        cmd.speed = float(speed)
        self.cmd_pub.publish(cmd)

    # ==========================================================
    # Visualization
    # ==========================================================

    def create_rollout_marker(self, frame_id, stamp, marker_id, path, is_best, is_collision):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = 'candidate_rollouts'
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.018

        if is_best:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.scale.x = 0.070
        elif is_collision:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.14
        else:
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 0.12

        for x, y, _yaw in path:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.05
            marker.points.append(p)

        return marker

    def publish_visualization(self, scan, results, best_index):
        marker_array = MarkerArray()

        delete_all = Marker()
        delete_all.header.frame_id = scan.header.frame_id
        delete_all.header.stamp = scan.header.stamp
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for i, result in enumerate(results):
            if i % 2 != 0 and i != best_index:
                continue
            marker = self.create_rollout_marker(
                scan.header.frame_id,
                scan.header.stamp,
                i,
                result['path'],
                i == best_index,
                result['collision'],
            )
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

        best_path_msg = Path()
        best_path_msg.header.frame_id = scan.header.frame_id
        best_path_msg.header.stamp = scan.header.stamp

        for x, y, yaw in results[best_index]['path']:
            pose = PoseStamped()
            pose.header.frame_id = scan.header.frame_id
            pose.header.stamp = scan.header.stamp
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.05
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            best_path_msg.poses.append(pose)

        self.best_path_pub.publish(best_path_msg)

    # ==========================================================
    # Main callback
    # ==========================================================

    def scan_callback(self, scan):
        self.callback_count += 1

        obstacles = self.scan_to_obstacle_points(scan)
        target_info = self.compute_target(scan)
        planning_speed, steps, horizon_time, preview_distance, horizon_mode = self.compute_dynamic_horizon(target_info)

        results = []
        for steer_candidate in self.make_steering_samples():
            path = self.simulate_rollout(steer_candidate, planning_speed, steps)
            result = self.score_rollout(
                path,
                steer_candidate,
                obstacles,
                target_info,
                planning_speed,
                preview_distance,
            )
            results.append(result)

        best_index, mode = self.choose_best_rollout(results, target_info)
        best = results[best_index]

        desired_steer = best['steer']

        # Turn-entry boost: if the LiDAR already says this is a strong turn,
        # do not allow a weak rollout command to delay the steering response.
        if (
            target_info['turn_sign'] != 0.0
            and target_info['turn_strength'] > 0.62
            and target_info['front'] < 1.55
            and desired_steer * target_info['turn_sign'] >= -0.02
        ):
            min_entry_steer = 0.20 + 0.14 * self.clamp((1.45 - target_info['front']) / 0.90, 0.0, 1.0)
            desired_steer = target_info['turn_sign'] * max(abs(desired_steer), min_entry_steer)

        steer = self.smooth_steering(
            desired_steer,
            best['clearance'],
            target_info['turn_strength'],
            target_info['turn_sign'],
            target_info['front'],
        )
        speed = self.choose_speed(steer, best, target_info, mode, horizon_mode)

        # Last-resort escape only. The old version activated escape too early
        # and forced -0.45 steering while the correct turn wanted + steering.
        # Here escape follows the detected turn when possible.
        emergency_escape = (
            best['clearance'] < self.escape_clearance_threshold
            or target_info['front'] < self.escape_front_threshold
            or (best['collision'] and best['clearance'] < 0.10)
        )
        if emergency_escape:
            if target_info['turn_sign'] != 0.0:
                escape_sign = target_info['turn_sign']
            elif abs(desired_steer) > 0.05:
                escape_sign = self.sign_or_zero(desired_steer, 0.01)
            else:
                escape_sign = 1.0 if target_info['left_dist'] > target_info['right_dist'] else -1.0
            steer = self.clamp(0.42 * escape_sign, -self.max_steer, self.max_steer)
            speed = self.escape_speed

        self.publish_command(steer, speed)

        # Publish visualization every other callback to keep CPU lighter.
        if self.callback_count % 2 == 0:
            self.publish_visualization(scan, results, best_index)

        self.previous_steer = steer
        self.previous_speed = speed
        self.previous_desired_steer = desired_steer

        if self.callback_count % 8 == 0:
            self.get_logger().info(
                f'mode={mode}, '
                f'horizon={horizon_mode}, '
                f'horizon_t={horizon_time:.2f}, '
                f'preview_d={preview_distance:.2f}, '
                f'desired_steer={desired_steer:.2f}, '
                f'steer={steer:.2f}, '
                f'speed={speed:.2f}, '
                f'planning_speed={planning_speed:.2f}, '
                f'front={target_info["front"]:.2f}, '
                f'far_front={target_info["far_front"]:.2f}, '
                f'left_dist={target_info["left_dist"]:.2f}, '
                f'right_dist={target_info["right_dist"]:.2f}, '
                f'raw_left={target_info["raw_left"]:.2f}, '
                f'raw_right={target_info["raw_right"]:.2f}, '
                f'raw_turn_y={target_info["raw_turn_y"]:.2f}, '
                f'turn_sign={target_info["turn_sign"]:.0f}, '
                f'turn_t={self.turn_timer:.2f}, '
                f'turn_strength={target_info["turn_strength"]:.2f}, '
                f'target_y={target_info["target_y"]:.2f}, '
                f'center_y={target_info["center_y"]:.2f}, '
                f'racing_y={target_info["racing_y"]:.2f}, '
                f'race_blend={target_info["race_blend"]:.2f}, '
                f'avg_path_y={best["avg_y"]:.2f}, '
                f'lateral_offset={best["lateral_offset"]:.2f}, '
                f'clearance={best["clearance"]:.2f}, '
                f'reward={best["reward"]:.2f}, '
                f'wrong_way={best["wrong_way_score"]:.2f}, '
                f'overshoot={best["overshoot_score"]:.2f}, '
                f'collision={best["collision"]}, '
                f'escape={emergency_escape}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = BasicMPPIDynamicHorizonPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
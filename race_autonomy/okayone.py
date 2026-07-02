import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDrive
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path


class RolloutPlanner(Node):
    def __init__(self):
        super().__init__('simple_centerline_racing_planner')

        self.scan_topic = '/yellow_car/scan'
        self.cmd_topic = '/yellow_car/cmd_ackermann'

        # ==========================================================
        # Vehicle
        # ==========================================================
        self.wheelbase = 0.30
        self.max_steer = 0.50

        # ==========================================================
        # Rollout planner
        # ==========================================================
        self.num_steers = 91

        # Keep this around 2.0 m lookahead.
        self.horizon_time = 2.30
        self.dt = 0.05

        # IMPORTANT: rollout_speed is used ONLY to simulate candidate
        # paths (geometry prediction), NOT as a speed cap. It should
        # reflect a "typical" speed for the rollout horizon to be
        # geometrically meaningful. Setting this equal to max_speed
        # (1.35) caused crashes: when actual speed swings down to
        # 0.25-0.3 in tight sections, the rollout still predicts a
        # path as if going 1.35 m/s, so the chosen path doesn't match
        # where the car actually ends up -> collisions.
        self.rollout_speed = 0.85
        self.rollout_steer_rate = 5.5

        # ==========================================================
        # Lidar
        # ==========================================================
        self.max_obstacle_range = 5.0
        self.front_angle_limit_deg = 130.0

        self.collision_radius = 0.14

        self.center_check_forward_min = -0.10
        self.center_check_forward_max = 1.40
        self.max_side_distance = 2.5

        # ==========================================================
        # Centerline target smoothing
        # ==========================================================
        self.center_smoothing_alpha = 0.12

        self.smoothed_left = self.max_side_distance
        self.smoothed_right = self.max_side_distance
        self.have_smoothed = False

        self.target_y_smoothing_alpha = 0.08
        self.turn_target_y_smoothing_alpha = 0.22

        self.smoothed_target_y = 0.0
        self.have_smoothed_target_y = False

        # ==========================================================
        # Far-front smoothing
        # ==========================================================
        self.smoothed_far_front = self.max_obstacle_range
        self.have_smoothed_far_front = False
        self.far_front_smoothing_alpha = 0.25

        # ==========================================================
        # Corner anticipation
        # ==========================================================
        # corner_bias_gain was capping corner_bias_y at ~0.12 regardless
        # of max_corner_bias_y, because corner_bias_y = corner_bias_gain
        # * open_diff * corner_strength, and open_diff rarely exceeds
        # ~1.0. Raise the gain so the signal can actually use the
        # 0.24 ceiling.
        self.corner_bias_gain = 0.24
        self.max_corner_bias_y = 0.24

        # Start anticipating corners earlier - at rollout_speed/max_speed
        # of ~1.0-1.3 m/s, 2.30m was only ~1.8-2.3s of warning, which
        # was not enough time for corner_bias_y to build up before the
        # car needed to commit. 3.20m gives roughly 2.5-3.2s.
        self.corner_front_start = 3.20
        self.corner_front_full = 0.90

        # ==========================================================
        # NEW: Wide-turn (wall curvature) anticipation
        # ==========================================================
        # Detects sweeping turns by comparing near vs far side-wall
        # distance, even when "front" is still clear.
        self.wide_turn_bias_gain = 0.70
        self.max_wide_turn_bias_y = 0.24

        # How much the far-side distance must drop below its recent
        # baseline (in meters) before we start reacting. With the new
        # EMA-based signal this is a much smaller, smoother number
        # than before.
        self.wide_turn_diff_threshold = 0.10

        # Smoothing for the final bias output.
        self.wide_turn_smoothing_alpha = 0.15
        self.smoothed_wide_turn_bias = 0.0
        self.have_smoothed_wide_turn_bias = False

        # EMA state for the wide-turn signal (initialized on first call).
        self.smoothed_left_far = self.max_side_distance
        self.smoothed_right_far = self.max_side_distance
        self.baseline_left_far = self.max_side_distance
        self.baseline_right_far = self.max_side_distance

        # ==========================================================
        # Mild side-wall push
        # ==========================================================
        self.side_push_start = 0.42
        self.side_push_gain = 0.55
        self.max_side_push_y = 0.09

        # ==========================================================
        # Speed schedule
        # ==========================================================
        self.max_speed = 2.50
        self.creep_speed = 0.25

        # Faster acceleration on straights.
        self.max_accel_step = 0.1

        # KEEP THESE - this is what made the car stop crashing.
        # Do not weaken these when raising max_speed; they are what
        # makes the higher top speed safe.
        self.max_decel_step = 0.18
        self.emergency_decel_step = 0.75

        self.max_steer_step = 0.50

        # Pulled back from 2.00 - that, combined with clearance_margin
        # 0.42, allowed too-fast entry into a corner that collapsed
        # clearance from 0.39 to 0.10 in one frame.
        self.lateral_accel_limit = 1.75

        self.previous_steer = 0.0
        self.previous_speed = self.creep_speed
        self.callback_count = 0

        # Track previous total_corner_bias_y to detect rapid changes -
        # a sign the "this looks like a straight" read is about to flip
        # (e.g. wide_turn_bias flipping sign late). Used to cap
        # acceleration before the planner has fully committed.
        self.previous_total_corner_bias_y = 0.0

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            AckermannDrive,
            self.cmd_topic,
            10
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/yellow_car/rollouts',
            10
        )

        self.best_path_pub = self.create_publisher(
            Path,
            '/yellow_car/best_path',
            10
        )

        self.get_logger().info('Wide-turn-aware centerline racing planner started.')

    # ==========================================================
    # Helpers
    # ==========================================================

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def valid_range(self, scan, r):
        if math.isnan(r) or math.isinf(r):
            return False

        if r < scan.range_min or r > scan.range_max:
            return False

        return True

    def make_steering_samples(self):
        samples = []

        for i in range(self.num_steers):
            ratio = i / (self.num_steers - 1)
            steer = -self.max_steer + 2.0 * self.max_steer * ratio
            samples.append(steer)

        return samples

    # ==========================================================
    # Lidar helpers
    # ==========================================================

    def sector_percentile(self, scan, min_deg, max_deg, percentile):
        values = []

        for i, r in enumerate(scan.ranges):
            if not self.valid_range(scan, r):
                continue

            r = min(r, self.max_obstacle_range)
            angle = scan.angle_min + i * scan.angle_increment
            angle_deg = math.degrees(angle)

            if min_deg <= angle_deg <= max_deg:
                values.append(r)

        if not values:
            return self.max_obstacle_range

        values.sort()
        index = int(self.clamp(percentile, 0.0, 1.0) * (len(values) - 1))
        return values[index]

    def sector_min(self, scan, min_deg, max_deg):
        best = self.max_obstacle_range

        for i, r in enumerate(scan.ranges):
            if not self.valid_range(scan, r):
                continue

            r = min(r, self.max_obstacle_range)
            angle = scan.angle_min + i * scan.angle_increment
            angle_deg = math.degrees(angle)

            if min_deg <= angle_deg <= max_deg:
                best = min(best, r)

        return best

    def sector_mean_top(self, scan, min_deg, max_deg):
        values = []

        for i, r in enumerate(scan.ranges):
            if not self.valid_range(scan, r):
                continue

            r = min(r, self.max_obstacle_range)
            angle = scan.angle_min + i * scan.angle_increment
            angle_deg = math.degrees(angle)

            if min_deg <= angle_deg <= max_deg:
                values.append(r)

        if not values:
            return self.max_obstacle_range

        values.sort(reverse=True)
        top_n = max(1, len(values) // 3)

        return sum(values[:top_n]) / top_n

    def find_open_side_info(self, scan):
        """
        front: obstacle distance ahead.
        left_open/right_open: how open each side is.
        """
        front = self.sector_percentile(scan, -14, 14, 0.10)

        left_open = self.sector_mean_top(scan, 25, 100)
        right_open = self.sector_mean_top(scan, -100, -25)

        return front, left_open, right_open

    def get_raw_side_distances(self, scan):
        """
        Faster side-distance estimate used only for mild wall push.
        """
        raw_left = self.sector_percentile(scan, 30, 105, 0.15)
        raw_right = self.sector_percentile(scan, -105, -30, 0.15)

        raw_left = min(raw_left, self.max_side_distance)
        raw_right = min(raw_right, self.max_side_distance)

        return raw_left, raw_right

    def get_wide_turn_signal(self, scan):
        """
        Detect a sweeping/wide turn by tracking how the FAR-FORWARD
        side distances change over time, relative to a slow-moving
        baseline of "normal straight" side distances.

        Rationale for the rewrite: comparing near-cone vs far-cone in
        a single frame was extremely noisy (values jumping by >1m
        frame-to-frame) because a 35deg cone flips between hitting the
        near wall and the far wall depending on tiny heading changes.

        Instead: track each side's far-forward distance with its own
        slow EMA (the "baseline", representing what's normal right
        now). Compare the CURRENT far-forward reading against that
        baseline. If the current reading drops well below the
        baseline, the wall ahead-left (or ahead-right) is closing in
        -> a turn is coming. Since both signals are EMAs, the
        difference is much smoother.
        """
        left_far_raw = self.sector_percentile(scan, 15, 75, 0.30)
        right_far_raw = self.sector_percentile(scan, -75, -15, 0.30)

        left_far_raw = min(left_far_raw, self.max_side_distance)
        right_far_raw = min(right_far_raw, self.max_side_distance)

        if not self.have_smoothed_wide_turn_bias:
            # First call: initialize everything from current readings.
            self.smoothed_left_far = left_far_raw
            self.smoothed_right_far = right_far_raw
            self.baseline_left_far = left_far_raw
            self.baseline_right_far = right_far_raw
            self.smoothed_wide_turn_bias = 0.0
            self.have_smoothed_wide_turn_bias = True
        else:
            # Fast EMA: tracks current far-forward distance, but still
            # filters out frame-to-frame lidar noise. Too fast (close
            # to 1.0) makes this track raw noise, which then leaks
            # into left_closing/right_closing and saturates the bias.
            fast_a = 0.12
            self.smoothed_left_far = (1.0 - fast_a) * self.smoothed_left_far + fast_a * left_far_raw
            self.smoothed_right_far = (1.0 - fast_a) * self.smoothed_right_far + fast_a * right_far_raw

            # Slow EMA: the "baseline" / what's been normal recently.
            slow_a = 0.02
            self.baseline_left_far = (1.0 - slow_a) * self.baseline_left_far + slow_a * left_far_raw
            self.baseline_right_far = (1.0 - slow_a) * self.baseline_right_far + slow_a * right_far_raw

        # Positive "closing" means the far-forward distance on that
        # side has dropped below its recent baseline -> wall curving
        # toward our future path on that side.
        left_closing = self.baseline_left_far - self.smoothed_left_far
        right_closing = self.baseline_right_far - self.smoothed_right_far

        raw_bias = 0.0

        # Left wall closing in ahead -> upcoming right turn -> bias right (negative y).
        if left_closing > self.wide_turn_diff_threshold:
            raw_bias -= self.wide_turn_bias_gain * (left_closing - self.wide_turn_diff_threshold)

        # Right wall closing in ahead -> upcoming left turn -> bias left (positive y).
        if right_closing > self.wide_turn_diff_threshold:
            raw_bias += self.wide_turn_bias_gain * (right_closing - self.wide_turn_diff_threshold)

        raw_bias = self.clamp(raw_bias, -self.max_wide_turn_bias_y, self.max_wide_turn_bias_y)

        # Light smoothing on the final bias - the inputs are already
        # EMAs so this just removes residual jitter.
        a = self.wide_turn_smoothing_alpha
        self.smoothed_wide_turn_bias = (1.0 - a) * self.smoothed_wide_turn_bias + a * raw_bias

        return self.smoothed_wide_turn_bias, left_closing, right_closing

    def scan_to_obstacle_points(self, scan):
        obstacles = []

        for i, r in enumerate(scan.ranges):
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

    def get_smoothed_far_front_distance(self, scan):
        raw_far_front = self.sector_percentile(scan, -15, 15, 0.08)

        if not self.have_smoothed_far_front:
            self.smoothed_far_front = raw_far_front
            self.have_smoothed_far_front = True
        else:
            a = self.far_front_smoothing_alpha
            self.smoothed_far_front = (
                (1.0 - a) * self.smoothed_far_front
                + a * raw_far_front
            )

        return self.smoothed_far_front

    def get_smoothed_side_distances(self, scan):
        """
        Robust current left/right wall distance, smoothed over time.
        This is used to compute the centerline target.
        """
        left_raw = self.sector_percentile(scan, 20, 110, 0.20)
        right_raw = self.sector_percentile(scan, -110, -20, 0.20)

        left_raw = min(left_raw, self.max_side_distance)
        right_raw = min(right_raw, self.max_side_distance)

        if not self.have_smoothed:
            self.smoothed_left = left_raw
            self.smoothed_right = right_raw
            self.have_smoothed = True
        else:
            a = self.center_smoothing_alpha
            self.smoothed_left = (1.0 - a) * self.smoothed_left + a * left_raw
            self.smoothed_right = (1.0 - a) * self.smoothed_right + a * right_raw

        return self.smoothed_left, self.smoothed_right

    def compute_side_push(self, raw_left, raw_right):
        """
        Mild immediate push away from a close side wall.

        y is positive to the left.
        left wall close  -> push right -> negative y.
        right wall close -> push left  -> positive y.
        """
        push_y = 0.0

        if raw_left < self.side_push_start:
            push_y -= self.side_push_gain * (self.side_push_start - raw_left)

        if raw_right < self.side_push_start:
            push_y += self.side_push_gain * (self.side_push_start - raw_right)

        return self.clamp(push_y, -self.max_side_push_y, self.max_side_push_y)

    # ==========================================================
    # Rollout simulation
    # ==========================================================

    def simulate_rollout(self, target_steer):
        x = 0.0
        y = 0.0
        yaw = 0.0

        sim_steer = self.previous_steer
        path = []

        steps = int(self.horizon_time / self.dt)

        for _ in range(steps):
            delta = target_steer - sim_steer
            max_delta = self.rollout_steer_rate * self.dt
            delta = self.clamp(delta, -max_delta, max_delta)

            sim_steer += delta
            sim_steer = self.clamp(sim_steer, -self.max_steer, self.max_steer)

            x += self.rollout_speed * math.cos(yaw) * self.dt
            y += self.rollout_speed * math.sin(yaw) * self.dt
            yaw += (self.rollout_speed / self.wheelbase) * math.tan(sim_steer) * self.dt

            path.append((x, y, yaw))

        return path

    # ==========================================================
    # Rollout analysis
    # ==========================================================

    def min_clearance_along_path(self, path, obstacles):
        """
        Minimum distance from any rollout point to any obstacle point.
        """
        min_clearance = self.max_obstacle_range

        sampled_path = path[::2]

        for (px, py, _pyaw) in sampled_path:
            for ox, oy in obstacles:
                dx = px - ox
                dy = py - oy
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < min_clearance:
                    min_clearance = dist

        return min_clearance

    def average_lateral_offset(self, path, target_y):
        """
        Average signed lateral offset from target_y.
        """
        total = 0.0
        weight_sum = 0.0

        n = max(1, len(path) - 1)

        for idx, (_px, py, _pyaw) in enumerate(path):
            progress = idx / n

            weight = 1.0 - 0.35 * progress

            total += weight * (py - target_y)
            weight_sum += weight

        return total / max(weight_sum, 1e-6)

    # ==========================================================
    # Scoring
    # ==========================================================

    def score_rollout(self, path, steer, obstacles, target_y, corner_bias_y):
        min_clearance = self.min_clearance_along_path(path, obstacles)
        collision = min_clearance < self.collision_radius

        lateral_offset = self.average_lateral_offset(path, target_y)

        mid_y = path[len(path) // 2][1]
        final_x, final_y, final_yaw = path[-1]

        mid_lateral_error = mid_y - target_y
        final_lateral_error = final_y - target_y

        cost = 0.0

        # 1) Hard safety
        if collision:
            cost += 12000.0

        # 2) Stronger clearance cost.
        safety_margin = 0.52

        if min_clearance < safety_margin:
            cost += 28.0 * (safety_margin - min_clearance) ** 2 / max(min_clearance, 0.02)

        if min_clearance < 0.34:
            cost += 120.0 * (0.34 - min_clearance) ** 2 / max(min_clearance, 0.02)

        if min_clearance < 0.24:
            cost += 450.0 * (0.24 - min_clearance) ** 2 / max(min_clearance, 0.02)

        # 3) Centerline tracking
        cost += 5.3 * (lateral_offset ** 2)

        # 4) Prevent going wide then correcting late
        cost += 2.0 * (mid_lateral_error ** 2)
        cost += 4.0 * (final_lateral_error ** 2)

        # 5) Early-turn yaw preference.
        if abs(corner_bias_y) > 0.025:
            desired_final_yaw = self.clamp(2.2 * corner_bias_y, -0.34, 0.34)

            cost += 2.2 * (final_yaw - desired_final_yaw) ** 2

            if corner_bias_y * steer < 0.0:
                cost += 0.8 * abs(corner_bias_y) * abs(steer)

        # 6) Steering smoothness - increased to reduce oscillation
        #    when multiple candidate steers achieve similar
        #    lateral_offset over the (short) rollout horizon.
        cost += 9.0 * (steer - self.previous_steer) ** 2

        # 7) Avoid unnecessary extreme steering
        cost += 0.12 * (steer ** 2)

        return cost, min_clearance, collision, lateral_offset

    # ==========================================================
    # Command
    # ==========================================================

    def smooth_steering(self, desired_steer, min_clearance):
        # Scale the allowed steering rate with how close we are to an
        # obstacle. At clearance >= 0.45, use the normal rate. As
        # clearance drops toward collision_radius, allow up to 2x the
        # rate so the car can react faster when it matters most.
        clearance_ratio = self.clamp(
            (min_clearance - self.collision_radius) / (0.45 - self.collision_radius),
            0.0,
            1.0
        )

        urgency = 1.0 - clearance_ratio
        effective_max_step = self.max_steer_step * (1.0 + urgency)

        delta = desired_steer - self.previous_steer
        delta = self.clamp(delta, -effective_max_step, effective_max_step)

        return self.clamp(
            self.previous_steer + delta,
            -self.max_steer,
            self.max_steer
        )

    def choose_speed(self, steer, min_clearance, collision, far_front_distance, wide_turn_strength, corner_bias_y, corner_bias_change):
        abs_steer = abs(steer)

        # Curvature-based speed limit.
        tan_steer = abs(math.tan(abs_steer))

        if tan_steer < 1e-4:
            curvature_speed = self.max_speed
        else:
            curvature_speed = math.sqrt(
                self.lateral_accel_limit * self.wheelbase / tan_steer
            )

        target_speed = min(self.max_speed, curvature_speed)

        # Clearance-based speed limit. 0.42 was too aggressive - allowed
        # speed=1.16 at clearance=0.39 right before a clearance collapse
        # to 0.10 (collision). 0.50 is a middle ground between the
        # original 0.55 (too slow) and 0.42 (too fast/late).
        clearance_margin = 0.50

        if min_clearance <= self.collision_radius:
            clearance_speed = self.creep_speed
        elif min_clearance >= clearance_margin:
            clearance_speed = self.max_speed
        else:
            ratio = (min_clearance - self.collision_radius) / (
                clearance_margin - self.collision_radius
            )
            clearance_speed = self.creep_speed + ratio * (self.max_speed - self.creep_speed)

        target_speed = min(target_speed, clearance_speed)

        # Far-front braking. Widened warning distance for earlier
        # reaction given the decel-rate findings above.
        warning_distance = 1.70
        stop_distance = 0.50

        if far_front_distance <= stop_distance:
            early_warning_speed = self.creep_speed
        elif far_front_distance >= warning_distance:
            early_warning_speed = self.max_speed
        else:
            ratio = (far_front_distance - stop_distance) / (
                warning_distance - stop_distance
            )
            early_warning_speed = self.creep_speed + ratio * (self.max_speed - self.creep_speed)

        target_speed = min(target_speed, early_warning_speed)

        # NEW: wide-turn braking. If the wide-turn signal is strong
        # (wall curving toward us ahead, even though front is clear),
        # cap speed proportionally. This is what prevents hitting the
        # outer wall of a sweeping turn at full speed.
        # wide-turn braking. Floor pulled back to 0.65 (between the
        # original 0.55 and the too-aggressive 0.80) - the previous
        # 0.80 floor allowed a 1.16 m/s commit into a corner that
        # turned out sharper than wide_turn_strength predicted.
        # Also factor in corner_bias_y, which reacts to the NARROW
        # forward cone and can catch sharper corners that the wide-turn
        # (side-distance) signal underestimates.
        turn_signal_strength = max(
            wide_turn_strength,
            abs(corner_bias_y) / self.max_corner_bias_y
        )

        if turn_signal_strength > 0.0:
            turn_speed = self.max_speed - turn_signal_strength * (self.max_speed - 0.65)
            target_speed = min(target_speed, turn_speed)

        if collision:
            target_speed = min(target_speed, self.creep_speed)

        # NOTE: do NOT cap commanded speed to rollout_speed - that was
        # an artificial coupling. rollout_speed is now a fixed
        # simulation-only parameter (see __init__ comment); the real
        # speed ceiling is max_speed, already enforced via
        # curvature_speed/clearance_speed/early_warning_speed above.

        # Smooth speed.
        emergency = (
            collision
            or min_clearance < 0.30
            or far_front_distance < 0.45
        )

        if target_speed > self.previous_speed:
            # If the corner-anticipation signal is changing rapidly,
            # the "this looks like a straight" read may be about to
            # flip (e.g. wide_turn_bias swinging late). Don't commit
            # to a hard acceleration into that uncertainty - throttle
            # the accel step down toward zero as change rate grows.
            instability = self.clamp(corner_bias_change / 0.15, 0.0, 1.0)
            accel_step = self.max_accel_step * (1.0 - 0.8 * instability)

            speed = min(target_speed, self.previous_speed + accel_step)
        else:
            if emergency:
                decel_step = self.emergency_decel_step
            else:
                decel_step = self.max_decel_step

            speed = max(target_speed, self.previous_speed - decel_step)

        return self.clamp(speed, self.creep_speed, self.max_speed)

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
        marker.scale.x = 0.020

        if is_best:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.scale.x = 0.075

        elif is_collision:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.20

        else:
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 0.18

        for x, y, _ in path:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.05
            marker.points.append(p)

        return marker

    def publish_visualization(self, scan, all_results, best_index):
        marker_array = MarkerArray()

        delete_all = Marker()
        delete_all.header.frame_id = scan.header.frame_id
        delete_all.header.stamp = scan.header.stamp
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for i, result in enumerate(all_results):
            marker = self.create_rollout_marker(
                scan.header.frame_id,
                scan.header.stamp,
                i,
                result['path'],
                i == best_index,
                result['collision']
            )
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

        best_path_msg = Path()
        best_path_msg.header.frame_id = scan.header.frame_id
        best_path_msg.header.stamp = scan.header.stamp

        best_path = all_results[best_index]['path']

        for x, y, yaw in best_path:
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

        left_dist, right_dist = self.get_smoothed_side_distances(scan)
        raw_left, raw_right = self.get_raw_side_distances(scan)

        front, left_open, right_open = self.find_open_side_info(scan)
        far_front_distance = self.get_smoothed_far_front_distance(scan)

        wide_turn_bias, left_closing, right_closing = self.get_wide_turn_signal(scan)

        # Main center target from left/right wall distance.
        # In wide-open areas (both walls far away), left_dist-right_dist
        # can be large and noisy with no nearby reference to center
        # against - dampen the centering target as both distances grow,
        # so the car holds its current line instead of swinging toward
        # a far, noisy "center".
        wall_proximity = self.clamp(
            1.0 - (min(left_dist, right_dist) - 0.5) / 1.0,
            0.0,
            1.0
        )

        base_target_y = 0.27 * (left_dist - right_dist) * wall_proximity

        # Open-side corner hint.
        open_diff = self.clamp(left_open - right_open, -1.0, 1.0)

        corner_strength = self.clamp(
            (self.corner_front_start - front) / (self.corner_front_start - self.corner_front_full),
            0.0,
            1.0
        )

        corner_bias_y = self.clamp(
            self.corner_bias_gain * open_diff * corner_strength,
            -self.max_corner_bias_y,
            self.max_corner_bias_y
        )

        # Small side-wall push.
        side_push_y = self.compute_side_push(raw_left, raw_right)

        # wide_turn_bias is the earliest-firing signal for sweeping
        # turns (fires around front~1.0-1.6, well before corner_bias_y
        # ramps up from the narrow forward cone). Weight it close to
        # equal with corner_bias_y so it can actually pull the target
        # line early on wide turns.
        total_corner_bias_y = self.clamp(
            corner_bias_y + 1.0 * wide_turn_bias,
            -0.40,
            0.40
        )

        # How fast the corner anticipation signal is changing,
        # regardless of its current magnitude. A near-zero bias that
        # is changing rapidly means "this looks like a straight right
        # now, but that read may flip soon" - don't accelerate hard
        # into that uncertainty.
        corner_bias_change = abs(total_corner_bias_y - self.previous_total_corner_bias_y)
        self.previous_total_corner_bias_y = total_corner_bias_y

        target_y_raw = self.clamp(
            base_target_y + total_corner_bias_y + side_push_y,
            -0.45,
            0.45
        )

        wide_turn_strength = self.clamp(
            abs(wide_turn_bias) / self.max_wide_turn_bias_y,
            0.0,
            1.0
        )

        turn_is_coming = (
            corner_strength > 0.15
            or abs(side_push_y) > 0.025
            or wide_turn_strength > 0.10
        )

        if not self.have_smoothed_target_y:
            self.smoothed_target_y = target_y_raw
            self.have_smoothed_target_y = True
        else:
            if turn_is_coming:
                a = self.turn_target_y_smoothing_alpha
            else:
                a = self.target_y_smoothing_alpha

            self.smoothed_target_y = (
                (1.0 - a) * self.smoothed_target_y
                + a * target_y_raw
            )

        target_y = self.smoothed_target_y

        steering_samples = self.make_steering_samples()
        all_results = []

        for steer_candidate in steering_samples:
            path = self.simulate_rollout(steer_candidate)

            cost, min_clearance, collision, lateral_offset = self.score_rollout(
                path,
                steer_candidate,
                obstacles,
                target_y,
                total_corner_bias_y
            )

            all_results.append({
                'steer': steer_candidate,
                'path': path,
                'cost': cost,
                'clearance': min_clearance,
                'collision': collision,
                'lateral_offset': lateral_offset,
            })

        # ======================================================
        # Selection logic
        # ======================================================
        preferred = [
            i for i, r in enumerate(all_results)
            if (not r['collision']) and r['clearance'] >= 0.36
        ]

        safe = [
            i for i, r in enumerate(all_results)
            if (not r['collision']) and r['clearance'] >= 0.30
        ]

        relaxed = [
            i for i, r in enumerate(all_results)
            if (not r['collision']) and r['clearance'] >= 0.24
        ]

        if preferred:
            best_index = min(preferred, key=lambda i: all_results[i]['cost'])
            mode = 'PREFERRED'

        elif safe:
            best_index = min(safe, key=lambda i: all_results[i]['cost'])
            mode = 'SAFE'

        elif relaxed:
            best_index = min(relaxed, key=lambda i: all_results[i]['cost'])
            mode = 'RELAXED'

        else:
            max_clearance = max(r['clearance'] for r in all_results)
            clearance_tolerance = 0.02

            candidates = [
                i for i, r in enumerate(all_results)
                if r['clearance'] >= max_clearance - clearance_tolerance
            ]

            best_index = min(candidates, key=lambda i: all_results[i]['cost'])
            mode = 'MAX_CLEARANCE'

        best = all_results[best_index]

        desired_steer = best['steer']
        steer = self.smooth_steering(desired_steer, best['clearance'])

        speed = self.choose_speed(
            steer,
            best['clearance'],
            best['collision'],
            far_front_distance,
            wide_turn_strength,
            corner_bias_y,
            corner_bias_change
        )

        self.publish_command(steer, speed)
        self.publish_visualization(scan, all_results, best_index)

        self.previous_steer = steer
        self.previous_speed = speed

        if self.callback_count % 8 == 0:
            self.get_logger().info(
                f'mode={mode}, '
                f'desired_steer={desired_steer:.2f}, '
                f'steer={steer:.2f}, '
                f'speed={speed:.2f}, '
                f'front={front:.2f}, '
                f'left_dist={left_dist:.2f}, '
                f'right_dist={right_dist:.2f}, '
                f'left_closing={left_closing:.2f}, '
                f'right_closing={right_closing:.2f}, '
                f'wide_turn_bias={wide_turn_bias:.2f}, '
                f'wide_turn_strength={wide_turn_strength:.2f}, '
                f'corner_bias_y={corner_bias_y:.2f}, '
                f'total_corner_bias_y={total_corner_bias_y:.2f}, '
                f'side_push_y={side_push_y:.2f}, '
                f'target_y={target_y:.2f}, '
                f'lateral_offset={best["lateral_offset"]:.2f}, '
                f'clearance={best["clearance"]:.2f}, '
                f'far_front={far_front_distance:.2f}, '
                f'collision={best["collision"]}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = RolloutPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

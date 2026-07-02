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
        super().__init__('safe_gap_racing_planner_dynamic_speed')

        self.scan_topic = '/yellow_car/scan'
        self.cmd_topic = '/yellow_car/cmd_ackermann'

        # ==========================================================
        # Vehicle
        # ==========================================================
        self.wheelbase = 0.30
        self.max_steer = 0.50

        # ==========================================================
        # Dynamic F1-style speed controller
        # ==========================================================
        # Conservative first. We can increase later after testing.
        self.min_speed = 0.32
        self.corner_speed = 0.42
        self.medium_speed = 0.65
        self.fast_speed = 0.90
        self.max_speed = 1.10

        # Accelerate slowly, brake faster.
        self.max_accel_step = 0.035
        self.max_decel_step = 0.10

        # ==========================================================
        # Rollout parameters
        # ==========================================================
        self.num_steers = 91
        self.horizon_time = 3.40
        self.dt = 0.05
        self.rollout_speed = 0.55
        self.rollout_steer_rate = 5.5

        # ==========================================================
        # Lidar
        # ==========================================================
        self.max_range = 5.0
        self.front_fov_deg = 125.0

        # ==========================================================
        # Safety margin around walls/obstacles
        # ==========================================================
        self.collision_radius = 0.34
        self.safe_clearance = 0.58
        self.warning_clearance = 0.90

        # ==========================================================
        # Gap-following parameters
        # ==========================================================
        self.inflation_radius = 0.55
        self.inflate_obstacles_until = 2.30
        self.min_gap_range = 0.75

        # ==========================================================
        # Target selection
        # ==========================================================
        self.target_distance = 1.45
        self.previous_target_angle = 0.0
        self.target_smoothing = 0.25

        # ==========================================================
        # Command smoothing
        # ==========================================================
        self.max_steer_step = 0.42

        self.previous_steer = 0.0
        self.previous_speed = 0.0
        self.callback_count = 0

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

        self.get_logger().info('Safe gap racing planner with dynamic speed started.')

    # ==========================================================
    # Helpers
    # ==========================================================

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def angle_wrap(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

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
    # Lidar processing
    # ==========================================================

    def scan_to_points(self, scan):
        points = []

        for i, r in enumerate(scan.ranges):
            if not self.valid_range(scan, r):
                continue

            r = min(r, self.max_range)
            angle = scan.angle_min + i * scan.angle_increment
            angle_deg = math.degrees(angle)

            if abs(angle_deg) > self.front_fov_deg:
                continue

            x = r * math.cos(angle)
            y = r * math.sin(angle)

            if x < -0.10:
                continue

            points.append((x, y))

        return points

    def build_front_scan_arrays(self, scan):
        angles = []
        ranges = []

        for i, r in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment
            angle_deg = math.degrees(angle)

            if abs(angle_deg) > self.front_fov_deg:
                continue

            if self.valid_range(scan, r):
                r = min(r, self.max_range)
            else:
                r = self.max_range

            angles.append(angle)
            ranges.append(r)

        return angles, ranges

    def inflate_obstacles(self, angles, ranges):
        """
        Safety trick:
        Make walls/obstacles look bigger than they are.
        This prevents the car from choosing paths very close to walls.
        """
        inflated = list(ranges)
        n = len(angles)

        for i in range(n):
            r = ranges[i]
            angle = angles[i]

            if r > self.inflate_obstacles_until:
                continue

            x = r * math.cos(angle)

            if x < 0.05:
                continue

            bubble_angle = math.atan2(self.inflation_radius, max(r, 0.05))

            for j in range(n):
                if abs(self.angle_wrap(angles[j] - angle)) <= bubble_angle:
                    inflated[j] = 0.0

        return inflated

    def find_gaps(self, inflated_ranges):
        gaps = []
        start = None

        for i, r in enumerate(inflated_ranges):
            free = r > self.min_gap_range

            if free and start is None:
                start = i

            if (not free) and start is not None:
                end = i - 1
                if end > start:
                    gaps.append((start, end))
                start = None

        if start is not None:
            end = len(inflated_ranges) - 1
            if end > start:
                gaps.append((start, end))

        return gaps

    def choose_best_gap(self, angles, ranges, inflated_ranges):
        gaps = self.find_gaps(inflated_ranges)

        if not gaps:
            best_i = max(range(len(ranges)), key=lambda i: ranges[i])
            return best_i, None

        best_gap = None
        best_score = -1e9

        for start, end in gaps:
            width = end - start + 1
            center_i = (start + end) // 2
            center_angle = angles[center_i]

            gap_ranges = [ranges[i] for i in range(start, end + 1)]
            avg_range = sum(gap_ranges) / len(gap_ranges)
            max_range = max(gap_ranges)

            # Prefer wide/open gaps, avoid aiming too much to the side.
            score = (
                2.2 * width
                + 1.0 * avg_range
                + 0.4 * max_range
                - 10.0 * abs(center_angle)
            )

            if score > best_score:
                best_score = score
                best_gap = (start, end)

        start, end = best_gap

        # Aim inside the middle of the safe gap, not at the edge.
        gap_width = end - start + 1
        trim = max(2, int(0.18 * gap_width))

        safe_start = min(end, start + trim)
        safe_end = max(start, end - trim)

        if safe_end <= safe_start:
            target_i = (start + end) // 2
        else:
            target_i = (safe_start + safe_end) // 2

        return target_i, best_gap

    def compute_target_angle(self, scan):
        angles, ranges = self.build_front_scan_arrays(scan)

        if not angles:
            return 0.0, [], [], [], None, 0

        inflated = self.inflate_obstacles(angles, ranges)
        target_i, best_gap = self.choose_best_gap(angles, ranges, inflated)

        raw_target_angle = angles[target_i]

        # Smooth target angle slightly.
        target_angle = (
            self.target_smoothing * self.previous_target_angle
            + (1.0 - self.target_smoothing) * raw_target_angle
        )

        target_angle = self.clamp(
            target_angle,
            -math.radians(85),
            math.radians(85)
        )

        return target_angle, angles, ranges, inflated, best_gap, target_i

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
    # Rollout scoring
    # ==========================================================

    def analyze_clearance(self, path, obstacles):
        min_clearance = self.max_range
        late_clearance = self.max_range
        collision = False

        sampled_path = path[::2]
        n = max(1, len(sampled_path) - 1)

        for idx, (px, py, _) in enumerate(sampled_path):
            progress = idx / n

            for ox, oy in obstacles:
                dx = px - ox
                dy = py - oy
                dist = math.sqrt(dx * dx + dy * dy)

                min_clearance = min(min_clearance, dist)

                if progress > 0.50:
                    late_clearance = min(late_clearance, dist)

                if dist < self.collision_radius:
                    collision = True

        return min_clearance, late_clearance, collision

    def score_rollout(self, path, steer, obstacles, target_angle):
        clearance, late_clearance, collision = self.analyze_clearance(path, obstacles)

        final_x, final_y, final_yaw = path[-1]

        target_x = self.target_distance * math.cos(target_angle)
        target_y = self.target_distance * math.sin(target_angle)

        closest_to_target = min(
            path,
            key=lambda p: (p[0] - target_x) ** 2 + (p[1] - target_y) ** 2
        )

        target_dist_error = math.sqrt(
            (closest_to_target[0] - target_x) ** 2
            + (closest_to_target[1] - target_y) ** 2
        )

        final_path_angle = math.atan2(final_y, max(final_x, 1e-6))
        angle_error = abs(self.angle_wrap(final_path_angle - target_angle))
        yaw_error = abs(self.angle_wrap(final_yaw - target_angle))

        cost = 0.0

        # 1) Hard safety
        if collision:
            cost += 200000.0

        if clearance < self.safe_clearance:
            cost += 4000.0 * (self.safe_clearance - clearance) ** 2

        if late_clearance < self.safe_clearance:
            cost += 6000.0 * (self.safe_clearance - late_clearance) ** 2

        if clearance < self.warning_clearance:
            cost += 500.0 * (self.warning_clearance - clearance) ** 2

        if late_clearance < self.warning_clearance:
            cost += 800.0 * (self.warning_clearance - late_clearance) ** 2

        # 2) Main objective: follow the safe gap target
        cost += 45.0 * target_dist_error
        cost += 28.0 * angle_error
        cost += 8.0 * yaw_error

        # 3) Prefer more wall clearance even among safe paths
        cost += 1.5 / (clearance + 0.05)
        cost += 2.5 / (late_clearance + 0.05)

        # 4) Tiny forward reward only
        cost += -0.03 * final_x

        # 5) Smoothness
        cost += 0.12 * abs(steer - self.previous_steer)

        return cost, clearance, late_clearance, collision, target_dist_error, angle_error, yaw_error

    # ==========================================================
    # Command
    # ==========================================================

    def smooth_steering(self, desired_steer):
        delta = desired_steer - self.previous_steer
        delta = self.clamp(delta, -self.max_steer_step, self.max_steer_step)

        steer = self.previous_steer + delta
        return self.clamp(steer, -self.max_steer, self.max_steer)

    def choose_speed(self, desired_steer, steer, target_angle, clearance, late_clearance, collision, mode):
        """
        F1-style speed controller:
        - fast on straights
        - slower before/inside turns
        - slows if clearance is low
        - never stops completely
        """

        abs_target_angle = abs(target_angle)
        abs_desired_steer = abs(desired_steer)
        abs_steer = abs(steer)

        turn_demand = max(abs_target_angle, abs_desired_steer, abs_steer)

        # 1) Base speed from turn demand
        if turn_demand > 0.42:
            target_speed = self.corner_speed

        elif turn_demand > 0.28:
            target_speed = self.medium_speed

        elif turn_demand > 0.16:
            target_speed = self.fast_speed

        else:
            target_speed = self.max_speed

        # 2) Slow down if the selected future path is near walls
        min_future_clearance = min(clearance, late_clearance)

        if min_future_clearance < 0.45:
            target_speed = min(target_speed, self.min_speed)

        elif min_future_clearance < 0.60:
            target_speed = min(target_speed, self.corner_speed)

        elif min_future_clearance < 0.80:
            target_speed = min(target_speed, self.medium_speed)

        # 3) If planner is not fully safe, stay cautious
        if mode != 'SAFE_GAP':
            target_speed = min(target_speed, self.corner_speed)

        # 4) Never fully stop
        if collision:
            target_speed = self.min_speed

        target_speed = max(self.min_speed, target_speed)

        # 5) Smooth speed: accelerate slowly, brake faster
        if target_speed > self.previous_speed:
            speed = min(target_speed, self.previous_speed + self.max_accel_step)
        else:
            speed = max(target_speed, self.previous_speed - self.max_decel_step)

        return speed

    def publish_command(self, steer, speed):
        cmd = AckermannDrive()
        cmd.steering_angle = float(steer)
        cmd.speed = float(speed)
        self.cmd_pub.publish(cmd)

    # ==========================================================
    # Visualization
    # ==========================================================

    def create_rollout_marker(self, frame_id, stamp, marker_id, path, is_best, is_collision, is_safe):
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
            marker.scale.x = 0.080

        elif is_collision or not is_safe:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.18

        else:
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 0.15

        for x, y, _ in path:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.05
            marker.points.append(p)

        return marker

    def create_target_marker(self, frame_id, stamp, target_angle):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = 'safe_gap_target'
        marker.id = 9999
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.060

        marker.color.r = 0.0
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 1.0

        p0 = Point()
        p0.x = 0.0
        p0.y = 0.0
        p0.z = 0.12
        marker.points.append(p0)

        p1 = Point()
        p1.x = self.target_distance * math.cos(target_angle)
        p1.y = self.target_distance * math.sin(target_angle)
        p1.z = 0.12
        marker.points.append(p1)

        return marker

    def publish_visualization(self, scan, all_results, best_index, target_angle):
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
                result['collision'],
                result['safe']
            )
            marker_array.markers.append(marker)

        marker_array.markers.append(
            self.create_target_marker(scan.header.frame_id, scan.header.stamp, target_angle)
        )

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

        obstacles = self.scan_to_points(scan)

        target_angle, angles, ranges, inflated, best_gap, target_i = self.compute_target_angle(scan)

        steering_samples = self.make_steering_samples()
        all_results = []

        for steer in steering_samples:
            path = self.simulate_rollout(steer)

            cost, clearance, late_clearance, collision, target_error, angle_error, yaw_error = \
                self.score_rollout(path, steer, obstacles, target_angle)

            safe = (
                not collision
                and clearance > self.safe_clearance
                and late_clearance > self.safe_clearance
            )

            all_results.append({
                'steer': steer,
                'path': path,
                'cost': cost,
                'clearance': clearance,
                'late_clearance': late_clearance,
                'collision': collision,
                'target_error': target_error,
                'angle_error': angle_error,
                'yaw_error': yaw_error,
                'safe': safe,
            })

        safe_indices = [
            i for i, r in enumerate(all_results)
            if r['safe']
        ]

        relaxed_indices = [
            i for i, r in enumerate(all_results)
            if (not r['collision'])
            and r['clearance'] > 0.42
            and r['late_clearance'] > 0.40
        ]

        if safe_indices:
            best_index = min(safe_indices, key=lambda i: all_results[i]['cost'])
            mode = 'SAFE_GAP'

        elif relaxed_indices:
            best_index = min(relaxed_indices, key=lambda i: all_results[i]['cost'])
            mode = 'RELAXED'

        else:
            best_index = max(
                range(len(all_results)),
                key=lambda i: min(all_results[i]['clearance'], all_results[i]['late_clearance'])
            )
            mode = 'MAX_CLEARANCE'

        best = all_results[best_index]

        desired_steer = best['steer']
        steer = self.smooth_steering(desired_steer)

        speed = self.choose_speed(
            desired_steer,
            steer,
            target_angle,
            best['clearance'],
            best['late_clearance'],
            best['collision'],
            mode
        )

        self.publish_command(steer, speed)
        self.publish_visualization(scan, all_results, best_index, target_angle)

        self.previous_steer = steer
        self.previous_speed = speed
        self.previous_target_angle = target_angle

        if self.callback_count % 8 == 0:
            self.get_logger().info(
                f'mode={mode}, '
                f'desired_steer={desired_steer:.2f}, '
                f'steer={steer:.2f}, '
                f'speed={speed:.2f}, '
                f'target_angle={math.degrees(target_angle):.1f}deg, '
                f'clearance={best["clearance"]:.2f}, '
                f'late_clearance={best["late_clearance"]:.2f}, '
                f'target_error={best["target_error"]:.2f}, '
                f'angle_error={best["angle_error"]:.2f}, '
                f'yaw_error={best["yaw_error"]:.2f}, '
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
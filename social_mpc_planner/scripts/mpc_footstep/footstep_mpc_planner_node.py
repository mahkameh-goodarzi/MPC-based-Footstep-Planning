#!/usr/bin/env python3

import rospy
import math
import casadi as ca
import json
import os
import subprocess
from collections import deque
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Point
from social_mpc_planner.msg import CircularObstacleArray
from social_mpc_planner.msg import PredictedPeople
from social_mpc_planner.msg import PredictedPeopleUncertainty
from gait_planner.srv import Trajectory, TrajectoryRequest
from std_msgs.msg import Float32, Int32, Bool, String


class FootstepMPCPlanner:
    def __init__(self):
        rospy.init_node("footstep_mpc_planner_node")

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.odom_received = False

        self.gx = 0.0
        self.gy = 0.0
        self.gtheta = 0.0
        self.goal_received = False

        # Initial value; it is updated adaptively before each MPC solve.
        self.N = 20

        self.step_width = 0.11
        self.step_width_tol = 0.02

        # lateral distance between opposite feet
        self.min_cross_width = 0.2
        self.max_cross_width = 0.22
        self.nominal_cross_width = 2.0 * self.step_width

        self.min_step_x = -0.15
        self.max_step_x = 0.18

        self.max_yaw_step = 0.07

        self.Q_goal = 80.0
        # Soft terminal weight for the desired final heading.
        # It is intentionally smaller than Q_goal so position tracking remains dominant.
        self.Q_goal_yaw = 10.0
        self.Q_progress = 3.0
        self.Q_nominal = 30.0
        self.Q_yaw = 0.2
        self.R_smooth = 7.0
        self.R_yaw_smooth = 7.0

        self.next_foot_is_left = True

        # Static obstacles received from /static_obstacles.
        # all_static_obstacles keeps the full YAML/Gazebo obstacle set, while
        # self.obstacles is refreshed before each MPC solve with only the nearest
        # obstacles to reduce the number of CasADi/IPOPT constraints.
        self.all_static_obstacles = []
        self.obstacles = []
        self.max_obstacles = rospy.get_param("~max_static_obstacles", 10)
        self.obstacle_selection_radius = rospy.get_param("~obstacle_selection_radius", 3.0)
        self.foot_clearance = 0.7
        self.enable_static_obstacles = True
        self.enable_dynamic_obstacles = True
        self.W_backward = 300.0
        self.W_obs_slack = 150000.0
        self.W_human_slack = 100000.0

        self.static_obstacles_sub = rospy.Subscriber(
            "/static_obstacles",
            CircularObstacleArray,
            self.static_obstacles_callback
        )

        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback)

        self.marker_pub = rospy.Publisher("/footstep_markers", MarkerArray, queue_size=10)
        self.chance_direction_marker_pub = rospy.Publisher(
            "/chance_direction_markers",
            MarkerArray,
            queue_size=10
        )

        # Publish the optimized/generated footsteps in the global odom frame so
        # the experiment logger can store them for offline plots and videos.
        # These are the full MPC horizon footsteps, not only the executed batch.
        self.generated_footsteps_world_pub = rospy.Publisher(
            "/evaluation/generated_footsteps_world",
            MarkerArray,
            queue_size=10
        )

        # Experiment/evaluation topics. These are intentionally simple std_msgs
        # so the logger can be used without defining new custom messages.
        self.mpc_solve_time_pub = rospy.Publisher("/evaluation/mpc_solve_time", Float32, queue_size=10)
        self.replan_cycle_time_pub = rospy.Publisher("/evaluation/replan_cycle_time", Float32, queue_size=10)
        self.executed_footsteps_pub = rospy.Publisher("/evaluation/executed_footsteps_count", Int32, queue_size=10)
        self.generated_footsteps_pub = rospy.Publisher("/evaluation/generated_footsteps_count", Int32, queue_size=10)
        self.prediction_horizon_pub = rospy.Publisher("/evaluation/prediction_horizon", Int32, queue_size=10)
        self.execution_horizon_pub = rospy.Publisher("/evaluation/execution_horizon", Int32, queue_size=10)
        self.goal_reached_pub = rospy.Publisher("/evaluation/goal_reached", Bool, queue_size=10)
        self.planner_status_pub = rospy.Publisher("/evaluation/planner_status", String, queue_size=10)
        self.replan_progress_pub = rospy.Publisher("/evaluation/replan_progress", Float32, queue_size=10)
        self.stagnation_detected_pub = rospy.Publisher("/evaluation/stagnation_detected", Bool, queue_size=10)

        rospy.loginfo("footstep_mpc_planner_node started.")

        self.walk_config_path = (
    "/home/linux/mpc_planner/src/MPC-based-Footstep-Planning/"
    "gait_planner/config/walk_config.json"
)

        self.step_width_config = 0.10
        self.default_walk_config = {
            "ankle_height": 0.025,
            "COM_height": 0.68,
            "com_offset": 0.01,
            "t_step": 1.0,
            "t_double_support": 0.1,
            "dt": 0.005,
            "alpha": 0.44,
            "hand_swing_angle": 0.0,
        }

        self.step_time = 1.0
        self.human_prediction_dt = 0.5

        # Fixed per-human/per-footstep risk level for the first chance-constraint
        # implementation. Global risk allocation will be added later.
        self.chance_epsilon = 0.05
        self.chance_beta = 1.6448536269514722
        self.chance_robot_radius = 0.20
        self.use_dummy_human_covariance = True
        self.dummy_sigma_x = 0.20
        self.dummy_sigma_y = 0.10
        self.dummy_rho = 0.0

        # Absolute human-position covariance sequences received from the predictor.
        # Their correctness and calibration will be handled in a later version.
        self.predicted_people_covariance = {}
        self.human_uncertainty_dt = 0.5

        self.max_people = 3

        # Base deterministic human avoidance margin [m].
        # This is the margin used at the beginning of the prediction horizon.
        self.human_safety_margin = 0.45

        # Uncertainty-aware prediction margin parameters.
        # The predicted human position is less reliable at later prediction times.
        # Therefore, the safety distance is inflated as k increases.
        #
        # Model used in the MPC constraint:
        #   uncertainty_margin(k) = min(
        #       human_uncertainty_max_margin,
        #       human_uncertainty_initial_margin + human_uncertainty_growth_per_step * k^p
        #   )
        #
        # With the default values, the first predicted points are close to the
        # original planner, while the final predicted points get a larger margin.
        self.enable_human_uncertainty_margin = rospy.get_param("~enable_human_uncertainty_margin", True)
        self.human_uncertainty_initial_margin = rospy.get_param("~human_uncertainty_initial_margin", 0.00)
        self.human_uncertainty_growth_per_step = rospy.get_param("~human_uncertainty_growth_per_step", 0.05)
        self.human_uncertainty_power = rospy.get_param("~human_uncertainty_power", 1.0)
        self.human_uncertainty_max_margin = rospy.get_param("~human_uncertainty_max_margin", 0.2)

        self.default_human_radius = 0.25
        self.foot_radius = 0.4

        self.predicted_people = [
            {
                "id": -1,
                "radius": self.default_human_radius,
                "trajectory": [(100.0, 100.0) for _ in range(self.N + 1)]
            }
            for _ in range(self.max_people)
        ]

        self.predicted_people_topic = rospy.get_param("~predicted_people_topic", "/predicted_people")
        self.predicted_people_sub = rospy.Subscriber(
            self.predicted_people_topic,
            PredictedPeople,
            self.predicted_people_callback
        )
        self.predicted_people_uncertainty_sub = rospy.Subscriber(
            "/predicted_people_uncertainty",
            PredictedPeopleUncertainty,
            self.predicted_people_uncertainty_callback
        )
        rospy.loginfo("MPC dynamic obstacle prediction topic: %s", self.predicted_people_topic)

        # Adaptive MPC horizon parameters
        # N is the prediction / optimization horizon and is updated at every replanning cycle.
        self.min_prediction_steps = 16
        self.max_prediction_steps = 16
        self.nominal_step_length = 0.16
        self.prediction_margin_steps = 3
        self.N = self.max_prediction_steps

        # Execution horizon parameters
        # The robot executes only a short batch from the optimized horizon.
        # This value counts the commanded footsteps written to the gait planner,
        # including the optional final support footstep added in make_execution_batch().
        #
        # Normal mode:
        #   execute a short batch to preserve receding-horizon behavior.
        # Stagnation mode:
        #   temporarily execute a longer batch to escape narrow passages or
        #   local repeated replanning behavior.
        self.min_execution_steps = 6
        self.preferred_execution_steps = 6
        self.stagnation_execution_steps = 6
        self.max_execution_steps = 6
        self.execution_steps = self.preferred_execution_steps

        # Stagnation detection parameters. Progress is measured as the
        # projection of the robot displacement onto the direction from the
        # previous replanning pose toward the goal. If the recent progress is
        # consistently small, the execution horizon is temporarily increased.
        self.progress_history = deque(maxlen=3)
        self.last_replan_start_x = None
        self.last_replan_start_y = None
        self.stagnation_progress_threshold = 0.05  # [m] over one executed batch
        self.stagnation_window = 3
        self.stagnation_detected = False
        self.near_obstacle_distance_threshold = 0.70  # [m], center-to-obstacle clearance

        self.use_receding_horizon = True

        self.goal_tolerance = 0.1
        self.goal_yaw_tolerance = math.radians(10.0)
        self.max_replan_cycles = 100
        self.retry_after_mpc_failure = rospy.get_param("~retry_after_mpc_failure", True)
        self.max_consecutive_mpc_failures = int(rospy.get_param("~max_consecutive_mpc_failures", 3))
        self.consecutive_mpc_failures = 0
        self.replan_count = 0
        self.is_walking = False
        self.active_goal = False

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        _, _, self.theta = euler_from_quaternion(quat)

        self.odom_received = True

    def goal_callback(self, msg):
        self.gx = msg.pose.position.x
        self.gy = msg.pose.position.y

        q = msg.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        _, _, self.gtheta = euler_from_quaternion(quat)

        self.goal_received = True
        self.active_goal = True
        self.replan_count = 0
        self.consecutive_mpc_failures = 0
        self.progress_history.clear()
        self.last_replan_start_x = None
        self.last_replan_start_y = None
        self.stagnation_detected = False
        self.stagnation_detected_pub.publish(Bool(False))

        rospy.loginfo(
            "New footstep goal: x=%.2f y=%.2f theta=%.2f",
            self.gx, self.gy, self.gtheta
        )
        self.goal_reached_pub.publish(Bool(False))
        self.planner_status_pub.publish(String("new_goal"))

        # Do not fix N here. The prediction horizon must be recomputed
        # from the current odometry at every replanning cycle.
        if self.odom_received:
            self.plan_footsteps()

    def distance_to_goal(self):
        dx = self.gx - self.x
        dy = self.gy - self.y
        return math.sqrt(dx * dx + dy * dy)

    def heading_error_to_goal(self):
        return abs(self.wrap_angle(self.theta - self.gtheta))

    def goal_pose_reached(self):
        return (
            self.distance_to_goal() <= self.goal_tolerance and
            self.heading_error_to_goal() <= self.goal_yaw_tolerance
        )

    def update_adaptive_horizon(self):
        """
        Update the MPC prediction horizon and the execution batch size.

        self.N is adaptive and can become long for distant goals.
        self.execution_steps is short in normal operation, but it is
        temporarily increased when stagnation is detected. This prevents the
        robot from repeatedly executing only very small initial footsteps near
        narrow passages.
        """
        dist = self.distance_to_goal()

        # Prediction horizon: adaptive with distance.
        N = int(math.ceil(dist / self.nominal_step_length)) + self.prediction_margin_steps
        self.N = max(self.min_prediction_steps, min(N, self.max_prediction_steps))

        # Execution horizon: bounded and mostly independent from N.
        if dist < 0.35:
            exec_steps = self.min_execution_steps
        elif self.should_use_stagnation_escape():
            exec_steps = self.stagnation_execution_steps
        else:
            exec_steps = self.preferred_execution_steps

        self.execution_steps = max(
            self.min_execution_steps,
            min(exec_steps, self.max_execution_steps, self.N)
        )

        rospy.loginfo(
            "Adaptive horizons | dist=%.3f m | N=%d | execution_steps=%d | stagnation=%s | recent_progress=%s",
            dist,
            self.N,
            self.execution_steps,
            str(self.stagnation_detected),
            ["%.3f" % p for p in list(self.progress_history)]
        )

    def should_use_stagnation_escape(self):
        """
        Return True if the robot has shown little forward progress during the
        last replanning cycles. The test is intentionally simple and robust:
        if all recent progress values are below the threshold, the planner
        sends a longer execution batch.
        """
        if len(self.progress_history) < self.stagnation_window:
            self.stagnation_detected = False
            self.stagnation_detected_pub.publish(Bool(False))
            return False

        max_recent_progress = max(self.progress_history)
        mean_recent_progress = sum(self.progress_history) / float(len(self.progress_history))

        stagnating = (
            max_recent_progress < self.stagnation_progress_threshold and
            mean_recent_progress < self.stagnation_progress_threshold
        )

        # Prefer activating escape behavior near obstacles. If no obstacle data
        # is available, stagnation alone is still sufficient.
        if stagnating and len(self.obstacles) > 0:
            min_clearance = self.min_obstacle_clearance()
            stagnating = min_clearance < self.near_obstacle_distance_threshold

        self.stagnation_detected = bool(stagnating)
        self.stagnation_detected_pub.publish(Bool(self.stagnation_detected))

        if self.stagnation_detected:
            rospy.logwarn(
                "Stagnation detected. Increasing execution horizon to %d.",
                self.stagnation_execution_steps
            )
            self.planner_status_pub.publish(String("stagnation_escape"))

        return self.stagnation_detected

    def min_obstacle_clearance(self):
        """
        Minimum clearance from the robot center to static circular obstacles.
        This is used only for deciding whether stagnation escape should be
        activated near obstacles.
        """
        if len(self.obstacles) == 0:
            return float("inf")

        clearances = []
        for ox, oy, r in self.obstacles:
            d = math.sqrt((self.x - ox) ** 2 + (self.y - oy) ** 2) - r
            clearances.append(d)

        return min(clearances)

    def update_replan_progress(self, start_x, start_y, end_x, end_y):
        """
        Compute forward progress along the direction from the replanning start
        pose to the goal. Negative progress is clipped to zero so small
        oscillations do not cancel forward motion in the history.
        """
        goal_dx = self.gx - start_x
        goal_dy = self.gy - start_y
        goal_norm = math.sqrt(goal_dx * goal_dx + goal_dy * goal_dy)

        if goal_norm < 1e-6:
            progress = 0.0
        else:
            ux = goal_dx / goal_norm
            uy = goal_dy / goal_norm

            dx = end_x - start_x
            dy = end_y - start_y

            progress = dx * ux + dy * uy
            progress = max(0.0, progress)

        self.progress_history.append(progress)
        self.replan_progress_pub.publish(Float32(progress))

        rospy.loginfo(
            "Replan progress: %.3f m | history=%s",
            progress,
            ["%.3f" % p for p in list(self.progress_history)]
        )

    def normalize_angle_expr(self, angle):
        return ca.atan2(ca.sin(angle), ca.cos(angle))

    def world_to_local(self, xw, yw, x0, y0, th0):
        dx = xw - x0
        dy = yw - y0

        xl = math.cos(th0) * dx + math.sin(th0) * dy
        yl = -math.sin(th0) * dx + math.cos(th0) * dy

        return xl, yl


    def local_to_world(self, xl, yl, x0, y0, th0):
        xw = x0 + math.cos(th0) * xl - math.sin(th0) * yl
        yw = y0 + math.sin(th0) * xl + math.cos(th0) * yl

        return xw, yw

    def human_uncertainty_margin_at_step(self, k):
        """
        Return an additional safety margin [m] for predicted humans at MPC step k.

        The farther a point is in the prediction horizon, the less certain the
        predicted human position is. This function inflates the collision
        avoidance radius for later predicted points.
        """
        if not self.enable_human_uncertainty_margin:
            return 0.0

        k_float = float(max(0, k))
        margin = (
            float(self.human_uncertainty_initial_margin) +
            float(self.human_uncertainty_growth_per_step) *
            (k_float ** float(self.human_uncertainty_power))
        )

        margin = max(0.0, margin)
        margin = min(float(self.human_uncertainty_max_margin), margin)
        return margin
        
    def predicted_people_callback(self, msg):
        if msg.dt > 1e-6:
            self.human_prediction_dt = msg.dt

        people = []

        for person in msg.people[:self.max_people]:
            traj = []

            for pose in person.trajectory:
                traj.append((pose.x, pose.y))

            if len(traj) == 0:
                traj = [(100.0, 100.0)]

            people.append({
                "id": person.id,
                "radius": person.radius,
                "trajectory": traj
            })

        while len(people) < self.max_people:
            people.append({
                "id": -1,
                "radius": self.default_human_radius,
                "trajectory": [(100.0, 100.0)]
            })

        self.predicted_people = people

    def predicted_people_uncertainty_callback(self, msg):
        """Store absolute-position covariance samples indexed by person ID."""
        if msg.dt > 1e-6:
            self.human_uncertainty_dt = msg.dt

        for person in msg.people:
            covariance_sequence = []

            for sigma_x, sigma_y, rho in zip(person.sigma_x, person.sigma_y, person.rho):
                sigma_x = max(0.0, float(sigma_x))
                sigma_y = max(0.0, float(sigma_y))
                rho = max(-0.999, min(0.999, float(rho)))

                sigma_xx = sigma_x * sigma_x
                sigma_xy = rho * sigma_x * sigma_y
                sigma_yy = sigma_y * sigma_y
                covariance_sequence.append((sigma_xx, sigma_xy, sigma_yy))

            self.predicted_people_covariance[person.id] = covariance_sequence


    def plan_footsteps(self):
        if self.is_walking:
            rospy.logwarn("Robot is walking; skip planning.")
            return

        if not self.active_goal:
            return

        replan_cycle_start_time = rospy.Time.now()

        if self.goal_pose_reached():
            pos_err = self.distance_to_goal()
            yaw_err = self.heading_error_to_goal()
            rospy.loginfo(
                "Goal pose reached before planning. pos_err=%.3f m yaw_err=%.2f deg",
                pos_err,
                math.degrees(yaw_err)
            )
            self.active_goal = False
            self.goal_reached_pub.publish(Bool(True))
            self.planner_status_pub.publish(String("goal_pose_reached"))
            return

        self.update_adaptive_horizon()
        self.prediction_horizon_pub.publish(Int32(self.N))
        self.execution_horizon_pub.publish(Int32(self.execution_steps))
        self.planner_status_pub.publish(String("planning"))

        if self.enable_dynamic_obstacles:
            rospy.loginfo(
                "Human chance constraints | epsilon=%.3f | beta=%.6f | robot radius=%.3f m",
                self.chance_epsilon,
                self.chance_beta,
                self.chance_robot_radius
            )

        N = self.N
        x_start = self.x
        y_start = self.y
        th_start = self.theta

        gx_l, gy_l = self.world_to_local(
            self.gx, self.gy,
            x_start, y_start, th_start
        )

        # Desired final heading expressed in the local MPC frame.
        # This is the actual goal orientation from PoseStamped, not only the direction to the goal point.
        goal_theta_final_l = self.wrap_angle(self.gtheta - th_start)

        # Heading of the line from the current robot pose to the goal position.
        # This is used as a weak intermediate reference so the robot generally faces the path.
        goal_theta_path = math.atan2(gy_l, gx_l)

        # Fixed nominal footsteps define each chance constraint's separating
        # direction and are reused as the IPOPT initial guess. Thus n remains
        # constant throughout one MPC solve and never depends on X or Y.
        nominal_footsteps = []
        for k in range(N):
            alpha = float(k + 1) / float(N)
            is_left = self.next_foot_is_left if k % 2 == 0 else not self.next_foot_is_left
            lateral = self.step_width if is_left else -self.step_width
            x_ref = alpha * gx_l
            y_ref = alpha * gy_l
            yaw_nom = self.wrap_angle(
                (1.0 - alpha) * goal_theta_path + alpha * goal_theta_final_l
            )
            x_nom = x_ref - lateral * math.sin(yaw_nom)
            y_nom = y_ref + lateral * math.cos(yaw_nom)
            nominal_footsteps.append((x_nom, y_nom, yaw_nom))

        # Select only the nearest static obstacles to the current robot pose.
        # This avoids adding hundreds of circular-obstacle constraints to the MPC.
        self.select_nearest_static_obstacles()

        local_obstacles = []

        if self.enable_static_obstacles:
            for ox, oy, r in self.obstacles:
                ox_l, oy_l = self.world_to_local(
                    ox, oy,
                    x_start, y_start, th_start
                )
                local_obstacles.append((ox_l, oy_l, r))

        local_people = []

        if self.enable_dynamic_obstacles:
            for person in self.predicted_people[:self.max_people]:
                traj_l = []
                cov_l = []
                traj = person["trajectory"]
                covariance_sequence = self.predicted_people_covariance.get(person["id"], [])

                cos_start = math.cos(th_start)
                sin_start = math.sin(th_start)

                for k in range(N):
                    t_k = (k + 1) * self.step_time
                    human_index = int(round(t_k / self.human_prediction_dt))

                    if human_index >= len(traj):
                        human_index = len(traj) - 1

                    hx, hy = traj[human_index]

                    hx_l, hy_l = self.world_to_local(
                        hx, hy,
                        x_start, y_start, th_start
                    )

                    traj_l.append((hx_l, hy_l))

                    if covariance_sequence:
                        cov_index = int(round(t_k / self.human_uncertainty_dt))
                        cov_index = max(0, min(cov_index, len(covariance_sequence) - 1))
                        sigma_xx, sigma_xy, sigma_yy = covariance_sequence[cov_index]

                        # Rotate absolute world-frame covariance with the same
                        # world-to-local rotation used for the human mean.
                        sigma_xx_l = (
                            cos_start * cos_start * sigma_xx
                            + 2.0 * cos_start * sin_start * sigma_xy
                            + sin_start * sin_start * sigma_yy
                        )
                        sigma_xy_l = (
                            cos_start * sin_start * (sigma_yy - sigma_xx)
                            + (cos_start * cos_start - sin_start * sin_start) * sigma_xy
                        )
                        sigma_yy_l = (
                            sin_start * sin_start * sigma_xx
                            - 2.0 * cos_start * sin_start * sigma_xy
                            + cos_start * cos_start * sigma_yy
                        )
                        cov_l.append((sigma_xx_l, sigma_xy_l, sigma_yy_l))
                    else:
                        cov_l.append((0.0, 0.0, 0.0))



                local_people.append({
                    "id": person["id"],
                    "radius": person["radius"],
                    "trajectory": traj_l,
                    "covariance": cov_l
                })

        opti = ca.Opti()

        X = opti.variable(N)
        Y = opti.variable(N)
        TH = opti.variable(N)
        num_obs = len(local_obstacles)
        S_obs = opti.variable(num_obs, N) if num_obs > 0 else None

        cost = 0.0

        prev_x = 0.0
        prev_y = 0.0
        prev_th = 0.0
        chance_direction_debug = []

        for k in range(N):
            xk = X[k]
            yk = Y[k]
            thk = TH[k]

            dx = xk - prev_x
            dy = yk - prev_y

            c = ca.cos(prev_th)
            s = ca.sin(prev_th)

            dx_body = c * dx + s * dy
            dy_body = -s * dx + c * dy

            backward_amount = 0.5 * (ca.sqrt(dx_body * dx_body + 1e-6) - dx_body)
            cost += self.W_backward * backward_amount * backward_amount

            is_left = self.next_foot_is_left if k % 2 == 0 else not self.next_foot_is_left
            if k == 0:
                # first planned step is relative to body center / odom pose
                desired_y = self.step_width if is_left else -self.step_width
            else:
                # later steps are relative to previous foot
                desired_y = 2.0 * self.step_width if is_left else -2.0 * self.step_width

            # step reachability
            opti.subject_to(dx_body >= self.min_step_x)
            opti.subject_to(dx_body <= self.max_step_x)

            if k == 0:
                desired_y = self.step_width if is_left else -self.step_width

                opti.subject_to(dy_body >= desired_y - self.step_width_tol)
                opti.subject_to(dy_body <= desired_y + self.step_width_tol)

            else:
                desired_y = 2.0 * self.step_width if is_left else -2.0 * self.step_width

                if is_left:
                    opti.subject_to(dy_body >= self.min_cross_width)
                    opti.subject_to(dy_body <= self.max_cross_width)

                else:
                    opti.subject_to(dy_body <= -self.min_cross_width)
                    opti.subject_to(dy_body >= -self.max_cross_width)


            for i, (ox, oy, r) in enumerate(local_obstacles):
                safe_dist = r + self.foot_clearance

                dx_obs = xk - ox
                dy_obs = yk - oy

                dist_sq = dx_obs * dx_obs + dy_obs * dy_obs

                opti.subject_to(S_obs[i, k] >= 0.0)
                opti.subject_to(dist_sq + S_obs[i, k] >= safe_dist * safe_dist)

                cost += self.W_obs_slack * S_obs[i, k] * S_obs[i, k]



            human_constraint_horizon = 6

            if self.enable_dynamic_obstacles and k < human_constraint_horizon:
                for h, person in enumerate(local_people):
                     
                    human_state = person["trajectory"][k]

                    if human_state is None:
                        continue

                    hx, hy = human_state
                    hr = person["radius"]

                    if self.use_dummy_human_covariance:
                        sigma_xx = self.dummy_sigma_x ** 2
                        sigma_yy = self.dummy_sigma_y ** 2
                        sigma_xy = (
                            self.dummy_rho
                            * self.dummy_sigma_x
                            * self.dummy_sigma_y
                        )
                    else:
                        sigma_xx, sigma_xy, sigma_yy = person["covariance"][k]
                    x_nom, y_nom, _ = nominal_footsteps[k]

                    dx_nom = x_nom - hx
                    dy_nom = y_nom - hy
                    norm_nom = math.hypot(dx_nom, dy_nom)

                    if norm_nom > 1e-6:
                        nx = dx_nom / norm_nom
                        ny = dy_nom / norm_nom
                    else:
                        nx = math.cos(goal_theta_path)
                        ny = math.sin(goal_theta_path)

                    if k == 0:
                        chance_direction_debug.append({
                            "human_id": person["id"],
                            "hx": hx,
                            "hy": hy,
                            "x_nom": x_nom,
                            "y_nom": y_nom,
                            "nx": nx,
                            "ny": ny,
                        })

                    # Conservative fixed-halfspace approximation of the circular
                    # Gaussian collision chance constraint (not an exact
                    # reformulation). The directional variance is n^T Sigma n.
                    sigma_n_sq = (
                        nx * nx * sigma_xx
                        + 2.0 * nx * ny * sigma_xy
                        + ny * ny * sigma_yy
                    )
                    sigma_n = math.sqrt(max(0.0, sigma_n_sq))
                    collision_radius = self.chance_robot_radius + hr
                    required_separation = (
                        collision_radius
                        + self.chance_beta * sigma_n
                    )

                    opti.subject_to(
                        nx * (xk - hx)
                        + ny * (yk - hy)
                        >= required_separation
                    )

            # yaw constraint
            dtheta = self.normalize_angle_expr(thk - prev_th)
            opti.subject_to(dtheta >= -self.max_yaw_step)
            opti.subject_to(dtheta <= self.max_yaw_step)

            # cost
            dist_goal_sq = (xk - gx_l) ** 2 + (yk - gy_l) ** 2

            cost += self.Q_progress * dist_goal_sq
            cost += self.Q_nominal * (dy_body - desired_y) ** 2
            cost += self.Q_yaw * self.normalize_angle_expr(thk - goal_theta_path) ** 2

            if k > 0:
                dstep_x = X[k] - X[k - 1]
                dstep_y = Y[k] - Y[k - 1]
                cost += self.R_smooth * (dstep_x ** 2 + dstep_y ** 2)

                dyaw_smooth = self.normalize_angle_expr(TH[k] - TH[k - 1])
                cost += self.R_yaw_smooth * dyaw_smooth ** 2

            prev_x = xk
            prev_y = yk
            prev_th = thk

        self.publish_chance_direction_markers(chance_direction_debug)

        terminal_position_cost = (X[N - 1] - gx_l) ** 2 + (Y[N - 1] - gy_l) ** 2
        terminal_yaw_error = self.normalize_angle_expr(TH[N - 1] - goal_theta_final_l)

        # Soft terminal pose cost:
        # Position remains dominant, while final yaw is encouraged without imposing a hard constraint.
        cost += self.Q_goal * terminal_position_cost
        cost += self.Q_goal_yaw * terminal_yaw_error ** 2

        opti.minimize(cost)

        for k in range(N):
            x_init, y_init, yaw_init = nominal_footsteps[k]

            opti.set_initial(X[k], x_init)
            opti.set_initial(Y[k], y_init)
            opti.set_initial(TH[k], yaw_init)

        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 250,
            "ipopt.tol": 1e-1,
            "ipopt.acceptable_tol": 2e-1,
            "ipopt.acceptable_iter": 10,
        }

        opti.solver("ipopt", opts)

        try:
            t0 = rospy.Time.now()
            sol = opti.solve()
            solve_time = (rospy.Time.now() - t0).to_sec()
            rospy.loginfo("MPC solve time: %.3f", solve_time)
            self.consecutive_mpc_failures = 0
            self.mpc_solve_time_pub.publish(Float32(solve_time))

            footsteps = []

            for k in range(N):
                is_left = self.next_foot_is_left if k % 2 == 0 else not self.next_foot_is_left

                xk = float(sol.value(X[k]))
                yk = float(sol.value(Y[k]))
                thk = float(sol.value(TH[k]))

                footsteps.append({
                    "x": xk,
                    "y": yk,
                    "theta": thk,
                    "foot": "L" if is_left else "R"
                })


            self.generated_footsteps_pub.publish(Int32(len(footsteps)))

            # 1) add final double support in LOCAL frame
            # if len(footsteps) >= 1:
            #     last = footsteps[-1]
            #     last_theta = last["theta"]

            #     last_is_left = (last["foot"] == "L")
            #     lateral_last = self.step_width if last_is_left else -self.step_width

            #     other_is_left = not last_is_left
            #     lateral_other = self.step_width if other_is_left else -self.step_width

            #     # center between feet
            #     cx = last["x"] + lateral_last * math.sin(last_theta)
            #     cy = last["y"] - lateral_last * math.cos(last_theta)

            #     # opposite foot
            #     x_final = cx - lateral_other * math.sin(last_theta)
            #     y_final = cy + lateral_other * math.cos(last_theta)

            #     footsteps.append({
            #         "x": x_final,
            #         "y": y_final,
            #         "theta": last_theta,
            #         "foot": "L" if other_is_left else "R"
            #     })


            # 2) convert LOCAL footsteps to ODOM only for visualization
            world_footsteps = []

            for fs in footsteps:
                xw, yw = self.local_to_world(
                    fs["x"], fs["y"],
                    x_start, y_start, th_start
                )

                world_footsteps.append({
                    "x": xw,
                    "y": yw,
                    "theta": self.wrap_angle(th_start + fs["theta"]),
                    "foot": fs["foot"]
                })


            # 3) local footsteps go to config, world footsteps go to RViz
            if self.use_receding_horizon:
                exec_footsteps = self.make_execution_batch(footsteps)
            else:
                exec_footsteps = footsteps

            self.print_footsteps(exec_footsteps)
            self.publish_markers(world_footsteps)
            self.publish_generated_footsteps_world(world_footsteps)

            self.executed_footsteps_pub.publish(Int32(len(exec_footsteps)))
            replan_cycle_time = (rospy.Time.now() - replan_cycle_start_time).to_sec()
            self.replan_cycle_time_pub.publish(Float32(replan_cycle_time))
            self.planner_status_pub.publish(String("commanding_walk"))

            self.write_walk_config(exec_footsteps)
            success = self.call_walk_service()

            if not success:
                return

            rospy.sleep(0.1)  # allow odom to update

            self.update_replan_progress(
                x_start,
                y_start,
                self.x,
                self.y
            )

            dist = self.distance_to_goal()
            yaw_err = self.heading_error_to_goal()
            rospy.loginfo(
                "Goal pose error after batch: position=%.3f m yaw=%.2f deg",
                dist,
                math.degrees(yaw_err)
            )

            if dist <= self.goal_tolerance and yaw_err <= self.goal_yaw_tolerance:
                rospy.loginfo(
                    "Goal pose reached. Stopping receding horizon. position=%.3f m yaw=%.2f deg",
                    dist,
                    math.degrees(yaw_err)
                )
                self.active_goal = False
                self.goal_reached_pub.publish(Bool(True))
                self.planner_status_pub.publish(String("goal_pose_reached"))
                return

            self.replan_count += 1

            if self.replan_count >= self.max_replan_cycles:
                rospy.logwarn("Max replan cycles reached. Stopping.")
                self.active_goal = False
                self.planner_status_pub.publish(String("max_replan_cycles_reached"))
                return

            if self.active_goal and not rospy.is_shutdown():
                rospy.loginfo("Replanning next batch...")
                self.plan_footsteps()

        except RuntimeError as e:
            rospy.logwarn("Footstep MPC failed: %s", str(e))
            self.planner_status_pub.publish(String("mpc_failed"))
            self.consecutive_mpc_failures += 1

            if (
                self.retry_after_mpc_failure and
                self.active_goal and
                self.consecutive_mpc_failures <= self.max_consecutive_mpc_failures and
                not rospy.is_shutdown()
            ):
                rospy.logwarn(
                    "Retrying MPC after failure (%d/%d)...",
                    self.consecutive_mpc_failures,
                    self.max_consecutive_mpc_failures
                )
                rospy.sleep(0.5)
                self.plan_footsteps()
            elif self.consecutive_mpc_failures > self.max_consecutive_mpc_failures:
                rospy.logwarn("Maximum consecutive MPC failures reached. Stopping active goal.")
                self.active_goal = False
                self.planner_status_pub.publish(String("mpc_failed_max_retries"))

    def print_footsteps(self, footsteps):
        rospy.loginfo("Generated footsteps:")
        for i, fs in enumerate(footsteps):
            rospy.loginfo(
                "%02d | %s | x=%.2f y=%.2f theta=%.2f",
                i,
                fs["foot"],
                fs["x"],
                fs["y"],
                fs["theta"]
            )
    
    def static_obstacles_callback(self, msg):
        """
        Store the full static-obstacle list.

        The planner should not simply use the first N obstacles from the message,
        because the YAML/world publisher may contain hundreds of circles and their
        order is unrelated to the robot position. The nearest subset is selected
        immediately before each MPC solve using the latest odometry.
        """
        obstacles = []

        for obs in msg.obstacles:
            obstacles.append((obs.x, obs.y, obs.radius))

        self.all_static_obstacles = obstacles

        # If odometry is not available yet, keep a small fallback subset so other
        # helper functions can still operate safely. Once odom is received, the
        # subset is updated by select_nearest_static_obstacles().
        if not self.odom_received:
            self.obstacles = obstacles[:self.max_obstacles]


    def select_nearest_static_obstacles(self):
        """
        Select the closest static obstacles to the current robot pose.

        Distance is computed as center distance minus obstacle radius, so larger
        obstacles are prioritized correctly. The optional selection radius removes
        far-away obstacles before limiting the result to max_obstacles. If no
        obstacle falls inside the radius, the globally closest obstacles are used
        as a fallback.
        """
        if len(self.all_static_obstacles) == 0:
            self.obstacles = []
            return

        scored = []
        fallback_scored = []

        for ox, oy, r in self.all_static_obstacles:
            clearance = math.sqrt((self.x - ox) ** 2 + (self.y - oy) ** 2) - r
            item = (clearance, ox, oy, r)
            fallback_scored.append(item)

            if clearance <= self.obstacle_selection_radius:
                scored.append(item)

        if len(scored) == 0:
            scored = fallback_scored

        scored.sort(key=lambda item: item[0])
        selected = scored[:self.max_obstacles]

        self.obstacles = [
            (ox, oy, r)
            for _, ox, oy, r in selected
        ]

        if len(selected) > 0:
            nearest_clearance = selected[0][0]
        else:
            nearest_clearance = float("inf")

        rospy.loginfo_throttle(
            1.0,
            "Selected %d/%d nearest static obstacles for MPC | nearest_clearance=%.3f m | radius=%.2f m",
            len(self.obstacles),
            len(self.all_static_obstacles),
            nearest_clearance,
            self.obstacle_selection_radius
        )

    def publish_chance_direction_markers(self, debug_data):
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        now = rospy.Time.now()
        marker_id = 0
        arrow_length = 0.5

        for data in debug_data:
            hx = data["hx"]
            hy = data["hy"]
            x_nom = data["x_nom"]
            y_nom = data["y_nom"]
            nx = data["nx"]
            ny = data["ny"]

            # All of the above coordinates are in the MPC local frame.
            # Convert them back to world/odom coordinates for RViz.

            human_world_x, human_world_y = self.local_to_world(
                hx, hy, self.x, self.y, self.theta
            )

            arrow_end_local_x = hx + arrow_length * nx
            arrow_end_local_y = hy + arrow_length * ny

            arrow_end_world_x, arrow_end_world_y = self.local_to_world(
                arrow_end_local_x,
                arrow_end_local_y,
                self.x,
                self.y,
                self.theta
            )

            nominal_world_x, nominal_world_y = self.local_to_world(
                x_nom,
                y_nom,
                self.x,
                self.y,
                self.theta
            )

            # Arrow showing direction n
            arrow = Marker()
            arrow.header.frame_id = "odom"
            arrow.header.stamp = now
            arrow.ns = "chance_direction_n"
            arrow.id = marker_id
            marker_id += 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD

            start = Point()
            start.x = human_world_x
            start.y = human_world_y
            start.z = 0.25

            end = Point()
            end.x = arrow_end_world_x
            end.y = arrow_end_world_y
            end.z = 0.25

            arrow.points = [start, end]

            arrow.scale.x = 0.03
            arrow.scale.y = 0.07
            arrow.scale.z = 0.10

            arrow.color.r = 0.0
            arrow.color.g = 1.0
            arrow.color.b = 0.0
            arrow.color.a = 1.0

            marker_array.markers.append(arrow)

            # Nominal foot position used to construct n
            nominal_marker = Marker()
            nominal_marker.header.frame_id = "odom"
            nominal_marker.header.stamp = now
            nominal_marker.ns = "chance_nominal_foot"
            nominal_marker.id = marker_id
            marker_id += 1
            nominal_marker.type = Marker.SPHERE
            nominal_marker.action = Marker.ADD

            nominal_marker.pose.position.x = nominal_world_x
            nominal_marker.pose.position.y = nominal_world_y
            nominal_marker.pose.position.z = 0.25
            nominal_marker.pose.orientation.w = 1.0

            nominal_marker.scale.x = 0.12
            nominal_marker.scale.y = 0.12
            nominal_marker.scale.z = 0.12

            nominal_marker.color.r = 1.0
            nominal_marker.color.g = 0.0
            nominal_marker.color.b = 1.0
            nominal_marker.color.a = 1.0

            marker_array.markers.append(nominal_marker)

        self.chance_direction_marker_pub.publish(marker_array)

    def publish_generated_footsteps_world(self, footsteps):
        """
        Publish generated MPC footsteps in the global odom frame.

        The gait planner still receives local-frame footsteps through
        write_walk_config(). This publisher is only for logging, plotting and
        video rendering. Each marker contains:
          - pose.position.x/y: footstep position in odom/world frame
          - pose.orientation: footstep yaw in odom/world frame
          - id: step index inside the optimized horizon
          - text: foot side ("L" or "R")
        """
        markers = MarkerArray()
        now = rospy.Time.now()

        for i, fs in enumerate(footsteps):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = now
            marker.ns = "generated_footsteps_world"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = float(fs["x"])
            marker.pose.position.y = float(fs["y"])
            marker.pose.position.z = 0.02

            q = quaternion_from_euler(0.0, 0.0, float(fs["theta"]))
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]

            marker.scale.x = 0.22
            marker.scale.y = 0.10
            marker.scale.z = 0.02

            if fs["foot"] == "L":
                marker.color.r = 0.0
                marker.color.g = 0.2
                marker.color.b = 1.0
            else:
                marker.color.r = 1.0
                marker.color.g = 0.2
                marker.color.b = 0.0

            marker.color.a = 0.9
            marker.text = fs["foot"]
            markers.markers.append(marker)

        self.generated_footsteps_world_pub.publish(markers)

        rospy.loginfo(
            "Published %d generated footsteps in odom frame for logging.",
            len(footsteps)
        )

    def publish_markers(self, footsteps):
        markers = MarkerArray()

        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        markers.markers.append(delete_marker)

        now = rospy.Time.now()

        # Start marker
        start_marker = Marker()
        start_marker.header.frame_id = "odom"
        start_marker.header.stamp = now
        start_marker.ns = "footstep_start"
        start_marker.id = 1000
        start_marker.type = Marker.SPHERE
        start_marker.action = Marker.ADD
        start_marker.pose.position.x = self.x
        start_marker.pose.position.y = self.y
        start_marker.pose.position.z = 0.08
        start_marker.pose.orientation.w = 1.0
        start_marker.scale.x = 0.18
        start_marker.scale.y = 0.18
        start_marker.scale.z = 0.18
        start_marker.color.r = 0.0
        start_marker.color.g = 1.0
        start_marker.color.b = 0.0
        start_marker.color.a = 1.0
        markers.markers.append(start_marker)

        # Goal marker
        goal_marker = Marker()
        goal_marker.header.frame_id = "odom"
        goal_marker.header.stamp = now
        goal_marker.ns = "footstep_goal"
        goal_marker.id = 1001
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = self.gx
        goal_marker.pose.position.y = self.gy
        goal_marker.pose.position.z = 0.08
        q_goal = quaternion_from_euler(0.0, 0.0, self.gtheta)
        goal_marker.pose.orientation.x = q_goal[0]
        goal_marker.pose.orientation.y = q_goal[1]
        goal_marker.pose.orientation.z = q_goal[2]
        goal_marker.pose.orientation.w = q_goal[3]
        goal_marker.scale.x = 0.22
        goal_marker.scale.y = 0.22
        goal_marker.scale.z = 0.22
        goal_marker.color.r = 1.0
        goal_marker.color.g = 0.0
        goal_marker.color.b = 1.0
        goal_marker.color.a = 1.0
        markers.markers.append(goal_marker)

        # Path line
        line_marker = Marker()
        line_marker.header.frame_id = "odom"
        line_marker.header.stamp = now
        line_marker.ns = "footstep_line"
        line_marker.id = 1002
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.03
        line_marker.color.r = 1.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0

        from geometry_msgs.msg import Point

        p0 = Point()
        p0.x = self.x
        p0.y = self.y
        p0.z = 0.05
        line_marker.points.append(p0)

        for fs in footsteps:
            p = Point()
            p.x = fs["x"]
            p.y = fs["y"]
            p.z = 0.05
            line_marker.points.append(p)

        markers.markers.append(line_marker)

        # Footstep markers
        for i, fs in enumerate(footsteps):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = now

            marker.ns = "footsteps"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = fs["x"]
            marker.pose.position.y = fs["y"]
            marker.pose.position.z = 0.02

            q = quaternion_from_euler(0.0, 0.0, fs["theta"])
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]

            marker.scale.x = 0.22
            marker.scale.y = 0.10
            marker.scale.z = 0.02

            if fs["foot"] == "L":
                marker.color.r = 0.0
                marker.color.g = 0.2
                marker.color.b = 1.0
            else:
                marker.color.r = 1.0
                marker.color.g = 0.2
                marker.color.b = 0.0

            marker.color.a = 0.9
            markers.markers.append(marker)

        self.marker_pub.publish(markers)

        if len(footsteps) > 0:
            last = footsteps[-1]
            dist_to_goal = math.sqrt((last["x"] - self.gx)**2 + (last["y"] - self.gy)**2)
            rospy.loginfo(
                "Last footstep to goal distance: %.3f m",
                dist_to_goal
            )

    def wrap_angle(self, a):
        return math.atan2(math.sin(a), math.cos(a))


    def write_walk_config(self, footsteps):
        config_steps = []

        config_steps.append([0.0, self.step_width_config, 0.0, 0.0])
        config_steps.append([0.0, -self.step_width_config, 0.0, 0.0])

        for fs in footsteps:
            config_steps.append([
                float(fs["x"]),
                float(fs["y"]),
                0.0,
                float(fs["theta"])
            ])

        walk_config = dict(self.default_walk_config)
        walk_config["footsteps"] = config_steps
        walk_config["do_start_transition"] = (self.replan_count == 0)

        tmp_path = self.walk_config_path + ".tmp"

        with open(tmp_path, "w") as f:
            json.dump(walk_config, f, indent=4)

        os.replace(tmp_path, self.walk_config_path)

        rospy.loginfo(
            "Wrote walk config with %d footsteps to %s",
            len(config_steps),
            self.walk_config_path
        )

    def make_execution_batch(self, footsteps):
        """
        Select only a short execution batch from the optimized footsteps.

        Important:
        - self.N is the prediction horizon.
        - self.execution_steps is the maximum number of commanded footsteps
          sent to the gait planner for this batch.
        - If a final support footstep is added, it is counted inside
          self.execution_steps. Therefore, the robot will not execute more
          than self.execution_steps footsteps from this batch.
        """
        if len(footsteps) == 0:
            return []

        max_commanded_steps = min(
            self.execution_steps,
            self.max_execution_steps,
            len(footsteps) + 1  # +1 because we may add final support
        )

        # Near the goal, do not force an extra support step if it would create
        # unnecessary motion. Otherwise reserve one slot for final support.
        add_final_support = not self.goal_pose_reached()

        if add_final_support and max_commanded_steps >= 2:
            K = min(max_commanded_steps - 1, len(footsteps))
        else:
            K = min(max_commanded_steps, len(footsteps))

        K = max(1, K)
        batch = footsteps[:K]

        if not add_final_support:
            rospy.loginfo(
                "Execution batch selected: %d footsteps, no final support step.",
                len(batch)
            )
            return batch

        last = batch[-1]
        last_theta = last["theta"]
        final_cross_width = self.min_cross_width

        if last["foot"] == "L":
            # Place right foot to the right of the last left foot.
            final_x = last["x"] + final_cross_width * math.sin(last_theta)
            final_y = last["y"] - final_cross_width * math.cos(last_theta)
            final_foot = "R"
        else:
            # Place left foot to the left of the last right foot.
            final_x = last["x"] - final_cross_width * math.sin(last_theta)
            final_y = last["y"] + final_cross_width * math.cos(last_theta)
            final_foot = "L"

        batch.append({
            "x": final_x,
            "y": final_y,
            "theta": last_theta,
            "foot": final_foot
        })

        rospy.loginfo(
            "Execution batch selected: %d commanded footsteps out of N=%d.",
            len(batch),
            self.N
        )

        return batch

    def call_walk_service(self):
        try:
            self.is_walking = True

            rospy.wait_for_service("/walk_service", timeout=5.0)
            walk_srv = rospy.ServiceProxy("/walk_service", Trajectory)
            req = TrajectoryRequest()

            req.is_config = True
            res = walk_srv(req)

            if not res.result:
                rospy.logwarn("Walk service returned false.")
                self.active_goal = False
                self.planner_status_pub.publish(String("walk_service_returned_false"))
                return False

            return True

        except Exception as e:
            rospy.logwarn("Walk service failed: %s", str(e))
            self.active_goal = False
            self.planner_status_pub.publish(String("walk_service_failed"))
            return False

        finally:
            self.is_walking = False

if __name__ == "__main__":
    node = FootstepMPCPlanner()
    rospy.spin()

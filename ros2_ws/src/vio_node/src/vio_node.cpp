#if 0
// ─────────────────────────────────────────────────────────────────────────────
//  vio_node.cpp  —  ASCEND IROC-U 2026
//  VIO Frontend: VPI KLT tracking + IMU-coupled ESKF
//  Publishes: /odom (200 Hz, SensorDataQoS) + /mavros/vision_pose/pose
// ─────────────────────────────────────────────────────────────────────────────
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/header.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

// VPI headers
#include <vpi/VPI.h>
#include <vpi/OpenCVInterop.hpp>

// Pull in the VPI tracker class defined in the header-style .cpp above
// (CMakeLists will compile both files; here we include as a header)
#include "vpi_tracker.cpp"

#include <deque>
#include <mutex>
#include <cmath>
#include <array>
#include <algorithm>
#include <chrono>

using namespace std::placeholders;

// ─────────────────────────────────────────────────────────────────────────────
//  ESKF state — 15-DOF (position, velocity, orientation Euler, acc bias, gyro bias)
//  Orientation stored as roll/pitch/yaw in radians (small-angle approx valid for
//  indoor hover flights).  For competition-level drift <0.5m/40m path.
// ─────────────────────────────────────────────────────────────────────────────
struct ESKFState {
    // World-frame
    std::array<double, 3> pos   = {0,0,0};  // x,y,z (m)
    std::array<double, 3> vel   = {0,0,0};  // vx,vy,vz (m/s)
    std::array<double, 3> rpy   = {0,0,0};  // roll, pitch, yaw (rad)
    // Biases (estimated online)
    std::array<double, 3> b_acc = {0,0,0};  // accelerometer bias (m/s²)
    std::array<double, 3> b_gyr = {0,0,0};  // gyroscope bias (rad/s)
};

// ─────────────────────────────────────────────────────────────────────────────
//  IMU measurement struct
// ─────────────────────────────────────────────────────────────────────────────
struct ImuMeasurement {
    double stamp;
    double ax, ay, az;   // m/s²  (raw, body frame)
    double gx, gy, gz;   // rad/s (raw, body frame)
};

// ─────────────────────────────────────────────────────────────────────────────
//  Tiny 3×3 rotation matrix from Euler angles (ZYX / yaw-pitch-roll)
// ─────────────────────────────────────────────────────────────────────────────
static void euler_to_R(double r, double p, double y,
                        std::array<std::array<double,3>,3>& R) {
    double cr=std::cos(r), sr=std::sin(r);
    double cp=std::cos(p), sp=std::sin(p);
    double cy=std::cos(y), sy=std::sin(y);
    R[0]={cy*cp,  cy*sp*sr-sy*cr,  cy*sp*cr+sy*sr};
    R[1]={sy*cp,  sy*sp*sr+cy*cr,  sy*sp*cr-cy*sr};
    R[2]={-sp,    cp*sr,            cp*cr};
}

static std::array<double,3> mat_vec(
        const std::array<std::array<double,3>,3>& R,
        const std::array<double,3>& v) {
    return {
        R[0][0]*v[0]+R[0][1]*v[1]+R[0][2]*v[2],
        R[1][0]*v[0]+R[1][1]*v[1]+R[1][2]*v[2],
        R[2][0]*v[0]+R[2][1]*v[1]+R[2][2]*v[2]
    };
}

// ─────────────────────────────────────────────────────────────────────────────
//  VIONode
// ─────────────────────────────────────────────────────────────────────────────
class VIONode : public rclcpp::Node {
public:
    VIONode() : Node("vio_node"),
                tracker_(std::make_unique<VPIFeatureTracker>()),
                tf_broadcaster_(this)
    {
        RCLCPP_INFO(get_logger(), "Initializing ASCEND VIO Pipeline (VPI + ESKF)...");

        // ── Camera subscribers (synchronized stereo pair) ─────────────────
        auto image_qos = rclcpp::SensorDataQoS();
        left_sub_.subscribe(this,  "/oakd/left/image_raw", image_qos.get_rmw_qos_profile());
        right_sub_.subscribe(this, "/oakd/right/image_raw", image_qos.get_rmw_qos_profile());

        using SyncPolicy = message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), left_sub_, right_sub_);
        sync_->registerCallback(
            std::bind(&VIONode::stereo_callback, this, _1, _2));

        left_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/oakd/left/camera_info", rclcpp::QoS(10).best_effort(),
            std::bind(&VIONode::left_camera_info_callback, this, _1));
        right_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/oakd/right/camera_info", rclcpp::QoS(10).best_effort(),
            std::bind(&VIONode::right_camera_info_callback, this, _1));

        // ── IMU subscriber (high frequency, best effort) ──────────────────
        auto imu_qos = rclcpp::SensorDataQoS();
        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            "/oakd/imu", imu_qos,
            std::bind(&VIONode::imu_callback, this, _1));

        // External altitude reference from MAVROS local position.
        // This helps clamp vertical drift from pure IMU integration.
        mavros_local_pos_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            "/mavros/local_position/pose", rclcpp::QoS(10).best_effort(),
            std::bind(&VIONode::mavros_local_position_callback, this, _1));

        // ── Publishers ────────────────────────────────────────────────────
        // /odom  — consumed by Nav2 and BT; use SensorDataQoS (Best Effort,
        //          Volatile) to prevent queue buildup as specified in VIO spec §8
        auto sensor_qos = rclcpp::SensorDataQoS();
        odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/odom", sensor_qos);

        // /mavros/vision_pose/pose — closes the loop into ArduPilot EKF2
        vision_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
            "/mavros/vision_pose/pose", rclcpp::QoS(1).best_effort());

        // /mavros/vision_pose/pose_cov — publish pose with conservative covariance
        // Use best-effort, depth=1 to minimize latency and avoid blocking.
        vision_pose_cov_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/mavros/vision_pose/pose_cov", rclcpp::QoS(1).best_effort());

        RCLCPP_INFO(get_logger(), "VIO node ready. Waiting for stereo + IMU...");

        // Prediction timer: publish a predicted vision pose at ~30 Hz to
        // reduce apparent latency for the EKF during maneuvers.
        predict_timer_ = create_wall_timer(
            std::chrono::milliseconds(33),
            std::bind(&VIONode::publish_predicted_pose, this));

        // IMU processing timer (200 Hz) to keep /odom fresh and time-aligned.
        imu_timer_ = create_wall_timer(
            std::chrono::milliseconds(5),
            std::bind(&VIONode::process_imu_timer, this));

        // Heartbeat log so position is visible even if other ROS logs are noisy.
        state_log_timer_ = create_wall_timer(
            std::chrono::milliseconds(1000),
            std::bind(&VIONode::log_state_heartbeat, this));

        stereo_sgbm_ = cv::StereoSGBM::create(0, 96, 7);
        stereo_sgbm_->setP1(8 * 7 * 7);
        stereo_sgbm_->setP2(32 * 7 * 7);
        stereo_sgbm_->setUniquenessRatio(5);
        stereo_sgbm_->setSpeckleWindowSize(50);
        stereo_sgbm_->setSpeckleRange(2);
    }

    ~VIONode() = default;

private:
    // ── Stereo callback: runs VPI tracker, then ESKF visual update ────────
    void stereo_callback(
            const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
            const sensor_msgs::msg::Image::ConstSharedPtr& right_msg)
    {
        try {
            const auto stamp = left_msg->header.stamp;
            const double stamp_sec = rclcpp::Time(stamp).seconds();
            process_imu_queue(stamp_sec);

            cv::Mat left_cv = cv_bridge::toCvShare(left_msg, "bgr8")->image;
            cv::Mat right_cv = cv_bridge::toCvShare(right_msg, "bgr8")->image;

            double dt_vis = 0.0;
            if (last_image_stamp_ > 0.0) {
                dt_vis = stamp_sec - last_image_stamp_;
            }
            last_image_stamp_ = stamp_sec;

            // ── VPI KLT tracking (GPU) ────────────────────────────────────
            auto tracks = tracker_->track(left_cv);

            // Count valid tracks for diagnostics
            int valid = 0;
            double mean_flow = 0.0;
            for (auto& t : tracks) {
                if (!t.valid) continue;
                ++valid;
                double dx = t.curr_x - t.prev_x;
                double dy = t.curr_y - t.prev_y;
                mean_flow += std::sqrt(dx*dx + dy*dy);
            }
            if (valid > 0) mean_flow /= valid;

            if (dt_vis <= 0.0 || dt_vis > 0.2) {
                _publish_state(stamp);
                return;
            }

            double gyro_norm = 0.0;
            {
                std::lock_guard<std::mutex> lk(state_mutex_);
                gyro_norm = std::sqrt(
                    latest_gyro_[0] * latest_gyro_[0] +
                    latest_gyro_[1] * latest_gyro_[1] +
                    latest_gyro_[2] * latest_gyro_[2]);
            }

            if (gyro_norm > 0.8) {
                _publish_state(stamp);
                return;
            }

            cv::Mat left_gray, right_gray;
            cv::cvtColor(left_cv, left_gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(right_cv, right_gray, cv::COLOR_BGR2GRAY);

            cv::Mat disp16;
            stereo_sgbm_->compute(left_gray, right_gray, disp16);
            cv::Mat disp;
            disp16.convertTo(disp, CV_32F, 1.0 / 16.0);

            double fx = 460.0, baseline = 0.075;
            {
                std::lock_guard<std::mutex> lk(cam_info_mutex_);
                if (stereo_calib_ready_) {
                    fx = fx_;
                    baseline = baseline_;
                }
            }

            // ── ESKF visual update ────────────────────────────────────────
            // Use stereo depth + optical flow to estimate planar velocity.
            int used = 0;
            double sum_vx = 0.0;
            double sum_vy = 0.0;
            for (const auto& t : tracks) {
                if (!t.valid) continue;

                int u0 = static_cast<int>(std::lround(t.prev_x));
                int v0 = static_cast<int>(std::lround(t.prev_y));
                int u1 = static_cast<int>(std::lround(t.curr_x));
                int v1 = static_cast<int>(std::lround(t.curr_y));

                if (u0 < 0 || v0 < 0 || u1 < 0 || v1 < 0 ||
                    u0 >= disp.cols || u1 >= disp.cols ||
                    v0 >= disp.rows || v1 >= disp.rows) {
                    continue;
                }

                float d0 = disp.at<float>(v0, u0);
                float d1 = disp.at<float>(v1, u1);
                if (d0 <= 1.0f || d1 <= 1.0f) continue;

                double z0 = (fx * baseline) / d0;
                double z1 = (fx * baseline) / d1;

                double v_cam_x = ((u1 - u0) / fx) * z1 / dt_vis; // right
                double v_cam_z = (z1 - z0) / dt_vis;             // forward

                sum_vx += v_cam_z; // map camera forward → body x
                sum_vy += v_cam_x; // map camera right   → body y
                ++used;
            }

            {
                std::lock_guard<std::mutex> lk(state_mutex_);
                constexpr double K_VIS = 0.4;

                if (used >= 10) {
                    double meas_vx = sum_vx / used;
                    double meas_vy = sum_vy / used;
                    state_.vel[0] = (1.0 - K_VIS) * state_.vel[0] + K_VIS * meas_vx;
                    state_.vel[1] = (1.0 - K_VIS) * state_.vel[1] + K_VIS * meas_vy;
                }

                // Zero-velocity update when hovering.
                if (mean_flow < 0.2 && gyro_norm < 0.05) {
                    state_.vel[0] = 0.0;
                    state_.vel[1] = 0.0;
                }
            }

            // ── Publish ───────────────────────────────────────────────────
            _publish_state(stamp);

            static int log_counter = 0;
            if (++log_counter % 30 == 0) {
                std::lock_guard<std::mutex> lk(state_mutex_);
                RCLCPP_INFO(get_logger(),
                    "VIO | tracks:%d flow:%.1fpx  pos=(%.2f,%.2f,%.2f)m",
                    valid, mean_flow,
                    state_.pos[0], state_.pos[1], state_.pos[2]);
            }

        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "stereo_callback: %s", e.what());
        }
    }

    // ── IMU callback: runs ESKF prediction at 200 Hz ──────────────────────
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        ImuMeasurement imu;
        imu.stamp = rclcpp::Time(msg->header.stamp).seconds();
        imu.ax = msg->linear_acceleration.x;
        imu.ay = msg->linear_acceleration.y;
        imu.az = msg->linear_acceleration.z;
        imu.gx = msg->angular_velocity.x;
        imu.gy = msg->angular_velocity.y;
        imu.gz = msg->angular_velocity.z;

        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            latest_gyro_[0] = imu.gx;
            latest_gyro_[1] = imu.gy;
            latest_gyro_[2] = imu.gz;
        }

        std::lock_guard<std::mutex> qlk(imu_mutex_);
        imu_queue_.push_back(imu);
        if (imu_queue_.size() > 1000) {
            imu_queue_.pop_front();
        }
    }

    void process_imu_timer() {
        const double target = this->now().seconds();
        if (process_imu_queue(target) && last_imu_stamp_ > 0.0) {
            auto stamp = rclcpp::Time(static_cast<int64_t>(last_imu_stamp_ * 1e9));
            _publish_state(stamp);
        }
    }

    bool process_imu_queue(double target_stamp) {
        std::deque<ImuMeasurement> batch;
        {
            std::lock_guard<std::mutex> qlk(imu_mutex_);
            while (!imu_queue_.empty() && imu_queue_.front().stamp <= target_stamp) {
                batch.push_back(imu_queue_.front());
                imu_queue_.pop_front();
            }
        }

        if (batch.empty()) return false;

        std::lock_guard<std::mutex> lk(state_mutex_);
        for (const auto& imu : batch) {
            if (last_imu_stamp_ < 0.0) {
                last_imu_stamp_ = imu.stamp;
                continue;
            }

            double dt = imu.stamp - last_imu_stamp_;
            if (dt <= 0.0 || dt > 0.05) {
                last_imu_stamp_ = imu.stamp;
                continue;
            }

            latest_gyro_[0] = imu.gx;
            latest_gyro_[1] = imu.gy;
            latest_gyro_[2] = imu.gz;
            _eskf_propagate(imu, dt);
            last_imu_stamp_ = imu.stamp;
        }

        return true;
    }

    void left_camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(cam_info_mutex_);
        left_info_ = *msg;
        have_left_info_ = true;
        update_stereo_calib_locked();
    }

    void right_camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(cam_info_mutex_);
        right_info_ = *msg;
        have_right_info_ = true;
        update_stereo_calib_locked();
    }

    void update_stereo_calib_locked() {
        if (!have_left_info_ || !have_right_info_) return;

        fx_ = left_info_.k[0];
        fy_ = left_info_.k[4];
        cx_ = left_info_.k[2];
        cy_ = left_info_.k[5];

        if (right_info_.p[0] != 0.0) {
            baseline_ = -right_info_.p[3] / right_info_.p[0];
        }

        if (baseline_ <= 0.0) {
            baseline_ = 0.075; // fallback: 2 * 0.0375 from model
        }

        if (fx_ > 0.0 && fy_ > 0.0 && baseline_ > 0.0) {
            stereo_calib_ready_ = true;
        }
    }

    void mavros_local_position_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(state_mutex_);

        const double z_ref = msg->pose.position.z;

        // Hard clamp z to FCU local altitude to avoid startup and long-horizon drift.
        state_.pos[2] = z_ref;
        state_.vel[2] = 0.0;

        external_altitude_m_ = z_ref;
        have_external_altitude_ = true;
    }

    // ── ESKF propagation — integrates IMU to predict state ───────────────
    void _eskf_propagate(const ImuMeasurement& imu, double dt) {
        // Remove estimated biases
        double ax = imu.ax - state_.b_acc[0];
        double ay = imu.ay - state_.b_acc[1];
        double az = imu.az - state_.b_acc[2];
        double gx = imu.gx - state_.b_gyr[0];
        double gy = imu.gy - state_.b_gyr[1];
        double gz = imu.gz - state_.b_gyr[2];

        // ── Orientation integration (Euler) ──────────────────────────────
        //  ṙpy ≈ J(rpy) * omega_body,  J = simple Euler angle Jacobian
        double r = state_.rpy[0], p = state_.rpy[1];
        double cr = std::cos(r), sr = std::sin(r);
        double cp = std::cos(p), tp = std::tan(p);

        state_.rpy[0] += dt * (gx + gy*sr*tp + gz*cr*tp);
        state_.rpy[1] += dt * (gy*cr - gz*sr);
        state_.rpy[2] += dt * (gy*sr/cp + gz*cr/cp);

        // ── Rotate body-frame accelerations to world frame ────────────────
        std::array<std::array<double,3>,3> R{};
        euler_to_R(state_.rpy[0], state_.rpy[1], state_.rpy[2], R);
        auto a_world = mat_vec(R, {ax, ay, az});

        // Remove gravity (z-down convention: gravity = +9.81 in world z)
        static constexpr double G = 9.81;
        a_world[2] -= G;

        // ── Velocity & position update ────────────────────────────────────
        for (int i = 0; i < 2; ++i) {
            state_.pos[i] += state_.vel[i] * dt + 0.5 * a_world[i] * dt * dt;
            state_.vel[i] += a_world[i] * dt;
        }

        // Z is sourced from FCU altitude whenever available.
        if (have_external_altitude_) {
            state_.pos[2] = external_altitude_m_;
            state_.vel[2] = 0.0;
        } else {
            state_.pos[2] += state_.vel[2] * dt + 0.5 * a_world[2] * dt * dt;
            state_.vel[2] += a_world[2] * dt;
        }

        // ── Very simple bias decay (prevents run-away bias) ───────────────
        constexpr double BIAS_DECAY = 0.9999;
        for (int i=0;i<3;++i){
            state_.b_acc[i] *= BIAS_DECAY;
            state_.b_gyr[i] *= BIAS_DECAY;
        }
    }

    // ── Publish odometry + vision_pose + TF ──────────────────────────────
    void _publish_state(const rclcpp::Time& stamp) {
        std::lock_guard<std::mutex> lk(state_mutex_);

        // ── Convert RPY to quaternion ─────────────────────────────────────
        double r = state_.rpy[0], p = state_.rpy[1], y = state_.rpy[2];
        double cr = std::cos(r/2), sr = std::sin(r/2);
        double cp = std::cos(p/2), sp = std::sin(p/2);
        double cy = std::cos(y/2), sy = std::sin(y/2);
        double qw = cr*cp*cy + sr*sp*sy;
        double qx = sr*cp*cy - cr*sp*sy;
        double qy = cr*sp*cy + sr*cp*sy;
        double qz = cr*cp*sy - sr*sp*cy;
        double n  = std::sqrt(qw*qw+qx*qx+qy*qy+qz*qz);
        if (n > 1e-9) { qw/=n; qx/=n; qy/=n; qz/=n; }

        // ── /odom ─────────────────────────────────────────────────────────
        auto odom = nav_msgs::msg::Odometry();
        odom.header.stamp    = stamp;
        odom.header.frame_id = "odom";
        odom.child_frame_id  = "base_link";

        odom.pose.pose.position.x    = state_.pos[0];
        odom.pose.pose.position.y    = state_.pos[1];
        odom.pose.pose.position.z    = state_.pos[2];
        odom.pose.pose.orientation.w = qw;
        odom.pose.pose.orientation.x = qx;
        odom.pose.pose.orientation.y = qy;
        odom.pose.pose.orientation.z = qz;

        odom.twist.twist.linear.x  = state_.vel[0];
        odom.twist.twist.linear.y  = state_.vel[1];
        odom.twist.twist.linear.z  = state_.vel[2];

        odom_pub_->publish(odom);

        publish_vision_pose(state_, stamp);
        publish_vision_pose_cov(state_, stamp);

        // ── odom → base_link TF ───────────────────────────────────────────
        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp    = stamp;
        tf.header.frame_id = "odom";
        tf.child_frame_id  = "base_link";
        tf.transform.translation.x = state_.pos[0];
        tf.transform.translation.y = state_.pos[1];
        tf.transform.translation.z = state_.pos[2];
        tf.transform.rotation.w = qw;
        tf.transform.rotation.x = qx;
        tf.transform.rotation.y = qy;
        tf.transform.rotation.z = qz;
        tf_broadcaster_.sendTransform(tf);
    }

    void publish_vision_pose(const ESKFState& state, const rclcpp::Time& stamp) {
        // 1. Create a quaternion from the ESKF state in ENU.
        tf2::Quaternion q_enu;
        q_enu.setRPY(state.rpy[0], state.rpy[1], state.rpy[2]);

        // 2. Populate the PoseStamped message directly in ENU.
        geometry_msgs::msg::PoseStamped msg;
        msg.header.stamp = stamp;
        msg.header.frame_id = "odom";
        msg.pose.orientation.x = q_enu.x();
        msg.pose.orientation.y = q_enu.y();
        msg.pose.orientation.z = q_enu.z();
        msg.pose.orientation.w = q_enu.w();
        msg.pose.position.x = state.pos[0];
        msg.pose.position.y = state.pos[1];
        msg.pose.position.z = state.pos[2];

        // vision_pose_pub_->publish(msg);
    }

    void publish_vision_pose_cov(const ESKFState& state, const rclcpp::Time& stamp) {
        geometry_msgs::msg::PoseWithCovarianceStamped msg;
        msg.header.stamp = stamp;
        msg.header.frame_id = "odom";

        tf2::Quaternion q_enu;
        q_enu.setRPY(state.rpy[0], state.rpy[1], state.rpy[2]);

        msg.pose.pose.orientation.x = q_enu.x();
        msg.pose.pose.orientation.y = q_enu.y();
        msg.pose.pose.orientation.z = q_enu.z();
        msg.pose.pose.orientation.w = q_enu.w();
        msg.pose.pose.position.x = state.pos[0];
        msg.pose.pose.position.y = state.pos[1];
        msg.pose.pose.position.z = state.pos[2];

        for (double &v : msg.pose.covariance) v = 0.0;
        msg.pose.covariance[0]  = 0.25;  // x
        msg.pose.covariance[7]  = 0.25;  // y
        msg.pose.covariance[14] = 0.50;  // z
        msg.pose.covariance[21] = 0.05;  // roll
        msg.pose.covariance[28] = 0.05;  // pitch
        msg.pose.covariance[35] = 0.10;  // yaw

        // vision_pose_cov_pub_->publish(msg);
    }

    void publish_predicted_pose() {
        std::lock_guard<std::mutex> lock(state_mutex_);
        rclcpp::Time now = this->now();
        double dt = 0.0;
        if (last_imu_stamp_ > 0.0) {
            dt = now.seconds() - last_imu_stamp_;
            if (dt < 0.0) dt = 0.0;
            if (dt > 0.5) dt = 0.5; // clamp unreasonable intervals
        }

        ESKFState pred = state_;
        pred.pos[0] += pred.vel[0] * dt;
        pred.pos[1] += pred.vel[1] * dt;
        if (have_external_altitude_) {
            pred.pos[2] = external_altitude_m_;
            pred.vel[2] = 0.0;
        } else {
            pred.pos[2] += pred.vel[2] * dt;
        }

        publish_vision_pose(pred, now);
        publish_vision_pose_cov(pred, now);
    }

    void log_state_heartbeat() {
        std::lock_guard<std::mutex> lk(state_mutex_);
        RCLCPP_INFO(
            get_logger(),
            "VIO HB | pos=(%.2f, %.2f, %.2f) vel=(%.2f, %.2f, %.2f) ext_alt=%s(%.2f)",
            state_.pos[0], state_.pos[1], state_.pos[2],
            state_.vel[0], state_.vel[1], state_.vel[2],
            have_external_altitude_ ? "yes" : "no", external_altitude_m_);
    }

    // ── ROS 2 comms ───────────────────────────────────────────────────────
    message_filters::Subscriber<sensor_msgs::msg::Image> left_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> right_sub_;
    std::shared_ptr<message_filters::Synchronizer<
        message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::Image, sensor_msgs::msg::Image>>> sync_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr right_info_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr mavros_local_pos_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr         odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr vision_pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr vision_pose_cov_pub_;
    rclcpp::TimerBase::SharedPtr predict_timer_;
    rclcpp::TimerBase::SharedPtr imu_timer_;
    rclcpp::TimerBase::SharedPtr state_log_timer_;
    std::array<double, 3> latest_gyro_ = {0.0, 0.0, 0.0};
    bool have_external_altitude_ = false;
    double external_altitude_m_ = 0.0;
    // ── VPI frontend ──────────────────────────────────────────────────────
    std::unique_ptr<VPIFeatureTracker> tracker_;

    tf2_ros::TransformBroadcaster tf_broadcaster_;

    // ── ESKF state (protected by mutex — IMU at 200Hz, vision at 30Hz) ───
    std::mutex  state_mutex_;
    ESKFState   state_{};
    double      last_imu_stamp_ = -1.0;
    double      last_image_stamp_ = -1.0;

    std::mutex cam_info_mutex_;
    sensor_msgs::msg::CameraInfo left_info_;
    sensor_msgs::msg::CameraInfo right_info_;
    bool have_left_info_ = false;
    bool have_right_info_ = false;
    double fx_ = 460.0;
    double fy_ = 460.0;
    double cx_ = 320.0;
    double cy_ = 240.0;
    double baseline_ = 0.075;
    bool stereo_calib_ready_ = false;

    std::mutex imu_mutex_;
    std::deque<ImuMeasurement> imu_queue_;

    cv::Ptr<cv::StereoSGBM> stereo_sgbm_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VIONode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
#endif

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <cv_bridge/cv_bridge.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Dense>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "vpi_tracker.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std::placeholders;

namespace {
constexpr double kGravity = 9.80665;
constexpr int kErrDim = 21;
constexpr int kIdxP = 0;
constexpr int kIdxV = 3;
constexpr int kIdxTheta = 6;
constexpr int kIdxBa = 9;
constexpr int kIdxBg = 12;
constexpr int kIdxPbc = 15;
constexpr int kIdxQbc = 18;

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Vec21 = Eigen::Matrix<double, kErrDim, 1>;
using Mat21 = Eigen::Matrix<double, kErrDim, kErrDim>;

struct ImuMeasurement {
    double stamp = 0.0;
    Vec3 acc = Vec3::Zero();
    Vec3 gyr = Vec3::Zero();
};

struct VioNominalState {
    Vec3 p_WB = Vec3::Zero();
    Vec3 v_WB = Vec3::Zero();
    Eigen::Quaterniond q_WB = Eigen::Quaterniond::Identity();
    Vec3 b_a = Vec3::Zero();
    Vec3 b_g = Vec3::Zero();
    Vec3 p_BC = Vec3(0.1, 0.0375, 0.05);
    Eigen::Quaterniond q_BC = Eigen::Quaterniond::Identity();
};

Mat3 skew(const Vec3& v) {
    Mat3 S;
    S << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
        -v.y(), v.x(), 0.0;
    return S;
}

Eigen::Quaterniond delta_q(const Vec3& omega_dt) {
    const double theta = omega_dt.norm();
    if (theta < 1e-12) {
        return Eigen::Quaterniond(1.0, 0.5 * omega_dt.x(), 0.5 * omega_dt.y(), 0.5 * omega_dt.z()).normalized();
    }
    return Eigen::Quaterniond(Eigen::AngleAxisd(theta, omega_dt / theta));
}

double median(std::vector<double>& values) {
    if (values.empty()) return 0.0;
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    return values[mid];
}

cv::Mat image_to_gray(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    auto cv_ptr = cv_bridge::toCvShare(msg);
    const cv::Mat& src = cv_ptr->image;
    if (src.type() == CV_8UC1) return src.clone();
    cv::Mat gray;
    if (src.channels() == 3) {
        if (msg->encoding == "rgb8") {
            cv::cvtColor(src, gray, cv::COLOR_RGB2GRAY);
        } else {
            cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        }
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        src.convertTo(gray, CV_8U);
    }
    return gray;
}

Vec3 back_project(double u, double v, double z, double fx, double fy, double cx, double cy) {
    return Vec3((u - cx) * z / fx, (v - cy) * z / fy, z);
}

bool depth_at(const cv::Mat& depth_m, int u, int v, double* z) {
    if (depth_m.empty() || u < 0 || v < 0 || u >= depth_m.cols || v >= depth_m.rows) return false;
    float d = depth_m.at<float>(v, u);
    if (!std::isfinite(d) || d < 0.08f || d > 50.0f) return false;
    *z = static_cast<double>(d);
    return true;
}

geometry_msgs::msg::Quaternion to_msg(const Eigen::Quaterniond& q_in) {
    Eigen::Quaterniond q = q_in.normalized();
    geometry_msgs::msg::Quaternion msg;
    msg.w = q.w();
    msg.x = q.x();
    msg.y = q.y();
    msg.z = q.z();
    return msg;
}
}

class CublasCovarianceUpdater {
public:
    CublasCovarianceUpdater() {
        if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS) {
            handle_ = nullptr;
        }
    }

    ~CublasCovarianceUpdater() {
        if (handle_) cublasDestroy(handle_);
    }

    bool available() const { return handle_ != nullptr; }

    Mat21 joseph_left_multiply(const Mat21& A, const Mat21& P) {
        if (!handle_) return A * P;

        double* d_A = nullptr;
        double* d_P = nullptr;
        double* d_out = nullptr;
        const size_t bytes = sizeof(double) * kErrDim * kErrDim;
        if (cudaMalloc(&d_A, bytes) != cudaSuccess ||
            cudaMalloc(&d_P, bytes) != cudaSuccess ||
            cudaMalloc(&d_out, bytes) != cudaSuccess) {
            if (d_A) cudaFree(d_A);
            if (d_P) cudaFree(d_P);
            if (d_out) cudaFree(d_out);
            return A * P;
        }

        cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_P, P.data(), bytes, cudaMemcpyHostToDevice);
        const double alpha = 1.0;
        const double beta = 0.0;
        const auto status = cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                        kErrDim, kErrDim, kErrDim,
                                        &alpha, d_A, kErrDim, d_P, kErrDim,
                                        &beta, d_out, kErrDim);

        Mat21 out = A * P;
        if (status == CUBLAS_STATUS_SUCCESS) {
            cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost);
        }
        cudaFree(d_A);
        cudaFree(d_P);
        cudaFree(d_out);
        return out;
    }

private:
    cublasHandle_t handle_ = nullptr;
};

class ErrorStateKalmanFilter {
public:
    ErrorStateKalmanFilter() {
        P_.setIdentity();
        P_.diagonal().segment<3>(kIdxP).setConstant(0.05);
        P_.diagonal().segment<3>(kIdxV).setConstant(0.10);
        P_.diagonal().segment<3>(kIdxTheta).setConstant(0.02);
        P_.diagonal().segment<3>(kIdxBa).setConstant(0.05);
        P_.diagonal().segment<3>(kIdxBg).setConstant(0.01);
        P_.diagonal().segment<3>(kIdxPbc).setConstant(1e-4);
        P_.diagonal().segment<3>(kIdxQbc).setConstant(1e-4);
    }

    const VioNominalState& state() const { return state_; }
    const Mat21& covariance() const { return P_; }

    void set_external_altitude(double z) {
        state_.p_WB.z() = z;
        state_.v_WB.z() = 0.0;
    }

    void propagate(const ImuMeasurement& imu, double dt) {
        const Vec3 omega = imu.gyr - state_.b_g;
        const Vec3 acc = imu.acc - state_.b_a;
        state_.q_WB = (state_.q_WB * delta_q(omega * dt)).normalized();

        const Mat3 R = state_.q_WB.toRotationMatrix();
        const Vec3 a_world = R * acc - Vec3(0.0, 0.0, kGravity);
        state_.p_WB += state_.v_WB * dt + 0.5 * a_world * dt * dt;
        state_.v_WB += a_world * dt;

        Mat21 F = Mat21::Zero();
        F.block<3, 3>(kIdxP, kIdxV).setIdentity();
        F.block<3, 3>(kIdxV, kIdxTheta) = -R * skew(acc);
        F.block<3, 3>(kIdxV, kIdxBa) = -R;
        F.block<3, 3>(kIdxTheta, kIdxBg) = -Mat3::Identity();

        Mat21 Phi = Mat21::Identity() + F * dt;
        Mat21 Q = Mat21::Zero();
        Q.diagonal().segment<3>(kIdxP).setConstant(1e-6 * dt);
        Q.diagonal().segment<3>(kIdxV).setConstant(acc_noise_ * acc_noise_ * dt);
        Q.diagonal().segment<3>(kIdxTheta).setConstant(gyr_noise_ * gyr_noise_ * dt);
        Q.diagonal().segment<3>(kIdxBa).setConstant(acc_bias_rw_ * acc_bias_rw_ * dt);
        Q.diagonal().segment<3>(kIdxBg).setConstant(gyr_bias_rw_ * gyr_bias_rw_ * dt);
        Q.diagonal().segment<3>(kIdxPbc).setConstant(extrinsic_frozen_ ? 0.0 : 1e-10 * dt);
        Q.diagonal().segment<3>(kIdxQbc).setConstant(extrinsic_frozen_ ? 0.0 : 1e-10 * dt);
        P_ = Phi * P_ * Phi.transpose() + Q;
        symmetrize();
    }

    bool update_velocity_body(const Vec3& v_body_meas, const Mat3& R_meas,
                              CublasCovarianceUpdater& cublas) {
        const Mat3 R_WB = state_.q_WB.toRotationMatrix();
        const Vec3 v_pred_body = R_WB.transpose() * state_.v_WB;
        const Vec3 r = v_body_meas - v_pred_body;
        if (!r.allFinite() || r.norm() > 8.0) return false;

        Eigen::Matrix<double, 3, kErrDim> H = Eigen::Matrix<double, 3, kErrDim>::Zero();
        H.block<3, 3>(0, kIdxV) = R_WB.transpose();
        H.block<3, 3>(0, kIdxTheta) = -R_WB.transpose() * skew(state_.v_WB);

        const Eigen::Matrix3d S = H * P_ * H.transpose() + R_meas;
        const Eigen::Matrix<double, kErrDim, 3> K = P_ * H.transpose() * S.inverse();
        const Vec21 dx = K * r;
        inject(dx);

        const Mat21 I = Mat21::Identity();
        const Mat21 A = I - K * H;
        P_ = cublas.joseph_left_multiply(A, P_) * A.transpose() + K * R_meas * K.transpose();
        symmetrize();
        return true;
    }

    void freeze_extrinsics(bool freeze) { extrinsic_frozen_ = freeze; }

private:
    void inject(const Vec21& dx) {
        state_.p_WB += dx.segment<3>(kIdxP);
        state_.v_WB += dx.segment<3>(kIdxV);
        state_.q_WB = (state_.q_WB * delta_q(dx.segment<3>(kIdxTheta))).normalized();
        state_.b_a += dx.segment<3>(kIdxBa);
        state_.b_g += dx.segment<3>(kIdxBg);
        if (!extrinsic_frozen_) {
            state_.p_BC += dx.segment<3>(kIdxPbc);
            state_.q_BC = (state_.q_BC * delta_q(dx.segment<3>(kIdxQbc))).normalized();
        }
    }

    void symmetrize() {
        P_ = 0.5 * (P_ + P_.transpose());
        for (int i = 0; i < kErrDim; ++i) {
            if (P_(i, i) < 1e-12) P_(i, i) = 1e-12;
        }
    }

    VioNominalState state_;
    Mat21 P_ = Mat21::Identity();
    double acc_noise_ = 0.10;
    double gyr_noise_ = 0.015;
    double acc_bias_rw_ = 0.001;
    double gyr_bias_rw_ = 0.0002;
    bool extrinsic_frozen_ = false;
};

class VIONode : public rclcpp::Node {
public:
    VIONode()
        : Node("vio_node"),
          tracker_(std::make_unique<VPIFeatureTracker>()),
          tf_broadcaster_(this) {
        declare_parameter("publish_mavros_vision", false);
        declare_parameter("freeze_extrinsics", false);
        publish_mavros_vision_ = get_parameter("publish_mavros_vision").as_bool();
        eskf_.freeze_extrinsics(get_parameter("freeze_extrinsics").as_bool());

        auto image_qos = rclcpp::SensorDataQoS();
        left_sub_.subscribe(this, "/oakd/left/image_raw", image_qos.get_rmw_qos_profile());
        right_sub_.subscribe(this, "/oakd/right/image_raw", image_qos.get_rmw_qos_profile());
        using SyncPolicy = message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), left_sub_, right_sub_);
        sync_->registerCallback(std::bind(&VIONode::stereo_callback, this, _1, _2));

        left_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/oakd/left/camera_info", rclcpp::QoS(10).best_effort(),
            std::bind(&VIONode::left_camera_info_callback, this, _1));
        right_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/oakd/right/camera_info", rclcpp::QoS(10).best_effort(),
            std::bind(&VIONode::right_camera_info_callback, this, _1));
        depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/oakd/depth/image_raw", image_qos,
            std::bind(&VIONode::depth_callback, this, _1));
        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            "/oakd/imu", image_qos, std::bind(&VIONode::imu_callback, this, _1));
        mavros_local_pos_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            "/mavros/local_position/pose", rclcpp::QoS(10).best_effort(),
            std::bind(&VIONode::mavros_local_position_callback, this, _1));

        odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/odom", image_qos);
        vision_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
            "/mavros/vision_pose/pose", rclcpp::QoS(1).best_effort());
        vision_pose_cov_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/mavros/vision_pose/pose_cov", rclcpp::QoS(1).best_effort());
        tracks_pub_ = create_publisher<sensor_msgs::msg::Image>("/vio/tracks_image", rclcpp::QoS(1));
        depth_debug_pub_ = create_publisher<sensor_msgs::msg::Image>("/vio/depth_debug", rclcpp::QoS(1));

        imu_timer_ = create_wall_timer(std::chrono::milliseconds(5),
                                       std::bind(&VIONode::process_imu_timer, this));
        heartbeat_timer_ = create_wall_timer(std::chrono::seconds(1),
                                             std::bind(&VIONode::log_heartbeat, this));

        stereo_sgbm_ = cv::StereoSGBM::create(0, 128, 5);
        stereo_sgbm_->setP1(8 * 5 * 5);
        stereo_sgbm_->setP2(32 * 5 * 5);
        stereo_sgbm_->setUniquenessRatio(8);
        stereo_sgbm_->setSpeckleWindowSize(50);
        stereo_sgbm_->setSpeckleRange(2);

        RCLCPP_INFO(get_logger(), "PDF-aligned VIO frontend ready | VPI Harris+LK | cuBLAS=%s",
                    cublas_.available() ? "yes" : "no");
    }

private:
    void stereo_callback(const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
                         const sensor_msgs::msg::Image::ConstSharedPtr& right_msg) {
        try {
            const rclcpp::Time stamp = left_msg->header.stamp;
            const double stamp_sec = stamp.seconds();
            process_imu_queue(stamp_sec);

            cv::Mat left_gray = image_to_gray(left_msg);
            cv::Mat right_gray = image_to_gray(right_msg);
            auto tracks = tracker_->track(left_gray);

            double dt_vis = last_image_stamp_ > 0.0 ? stamp_sec - last_image_stamp_ : 0.0;
            last_image_stamp_ = stamp_sec;

            cv::Mat depth_m;
            std::string depth_source;
            const bool got_depth = get_depth_for_stamp(stamp_sec, depth_m);
            if (got_depth) {
                depth_source = "gazebo_depth";
            } else {
                depth_m = stereo_depth(left_gray, right_gray);
                depth_source = "stereo_fallback";
            }

            int valid_tracks = 0;
            double mean_flow = 0.0;
            for (const auto& t : tracks) {
                if (!t.valid) continue;
                valid_tracks++;
                const double dx = t.curr_x - t.prev_x;
                const double dy = t.curr_y - t.prev_y;
                mean_flow += std::sqrt(dx * dx + dy * dy);
            }
            if (valid_tracks > 0) mean_flow /= valid_tracks;

            bool accepted = false;
            int depth_used = 0;
            if (dt_vis > 0.0 && dt_vis < 0.2 && !prev_depth_m_.empty()) {
                double gyro_norm = latest_gyro_norm();
                if (gyro_norm < 1.2 && valid_tracks >= 15) {
                    Vec3 v_body;
                    if (estimate_visual_velocity(tracks, prev_depth_m_, depth_m, dt_vis, &v_body, &depth_used)) {
                        Mat3 R_meas = Mat3::Identity() * std::max(0.02, 0.20 / std::max(1, depth_used));
                        std::lock_guard<std::mutex> lk(state_mutex_);
                        accepted = eskf_.update_velocity_body(v_body, R_meas, cublas_);
                    }
                }
            }

            if (mean_flow < 0.20 && latest_gyro_norm() < 0.05) {
                Vec3 zero = Vec3::Zero();
                Mat3 R_meas = Mat3::Identity() * 0.005;
                std::lock_guard<std::mutex> lk(state_mutex_);
                eskf_.update_velocity_body(zero, R_meas, cublas_);
            }

            prev_depth_m_ = depth_m.clone();
            publish_state(stamp);
            publish_debug(left_gray, depth_m, tracks, stamp);

            if (++log_counter_ % 30 == 0) {
                const auto& s = eskf_.state();
                RCLCPP_INFO(get_logger(),
                    "VIO | tracks=%d depth_used=%d source=%s update=%s pos=(%.2f %.2f %.2f) vel=(%.2f %.2f %.2f)",
                    valid_tracks, depth_used, depth_source.c_str(), accepted ? "accepted" : "rejected",
                    s.p_WB.x(), s.p_WB.y(), s.p_WB.z(), s.v_WB.x(), s.v_WB.y(), s.v_WB.z());
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "stereo_callback: %s", e.what());
        }
    }

    void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            auto cv_ptr = cv_bridge::toCvShare(msg);
            cv::Mat depth;
            if (cv_ptr->image.type() == CV_32FC1) {
                depth = cv_ptr->image.clone();
            } else if (cv_ptr->image.type() == CV_16UC1) {
                cv_ptr->image.convertTo(depth, CV_32F, 0.001);
            } else {
                cv_ptr->image.convertTo(depth, CV_32F);
            }
            std::lock_guard<std::mutex> lk(depth_mutex_);
            latest_depth_m_ = depth;
            latest_depth_stamp_ = rclcpp::Time(msg->header.stamp).seconds();
        } catch (const std::exception& e) {
            RCLCPP_WARN(get_logger(), "depth_callback: %s", e.what());
        }
    }

    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        ImuMeasurement imu;
        imu.stamp = rclcpp::Time(msg->header.stamp).seconds();
        imu.acc = Vec3(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
        imu.gyr = Vec3(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            latest_gyro_ = imu.gyr;
        }
        std::lock_guard<std::mutex> qlk(imu_mutex_);
        imu_queue_.push_back(imu);
        while (imu_queue_.size() > 1000) imu_queue_.pop_front();
    }

    void process_imu_timer() {
        if (process_imu_queue(now().seconds()) && last_imu_stamp_ > 0.0) {
            publish_state(rclcpp::Time(static_cast<int64_t>(last_imu_stamp_ * 1e9)));
        }
    }

    bool process_imu_queue(double target_stamp) {
        std::deque<ImuMeasurement> batch;
        {
            std::lock_guard<std::mutex> lk(imu_mutex_);
            while (!imu_queue_.empty() && imu_queue_.front().stamp <= target_stamp) {
                batch.push_back(imu_queue_.front());
                imu_queue_.pop_front();
            }
        }
        if (batch.empty()) return false;

        std::lock_guard<std::mutex> lk(state_mutex_);
        for (const auto& imu : batch) {
            if (last_imu_stamp_ < 0.0) {
                last_imu_stamp_ = imu.stamp;
                continue;
            }
            const double dt = imu.stamp - last_imu_stamp_;
            last_imu_stamp_ = imu.stamp;
            if (dt <= 0.0 || dt > 0.05) continue;
            eskf_.propagate(imu, dt);
            if (have_external_altitude_) eskf_.set_external_altitude(external_altitude_m_);
        }
        return true;
    }

    void left_camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(cam_info_mutex_);
        left_info_ = *msg;
        have_left_info_ = true;
        update_stereo_calib_locked();
    }

    void right_camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(cam_info_mutex_);
        right_info_ = *msg;
        have_right_info_ = true;
        update_stereo_calib_locked();
    }

    void update_stereo_calib_locked() {
        if (!have_left_info_) return;
        fx_ = left_info_.k[0] > 0.0 ? left_info_.k[0] : fx_;
        fy_ = left_info_.k[4] > 0.0 ? left_info_.k[4] : fy_;
        cx_ = left_info_.k[2] > 0.0 ? left_info_.k[2] : cx_;
        cy_ = left_info_.k[5] > 0.0 ? left_info_.k[5] : cy_;
        if (have_right_info_ && right_info_.p[0] != 0.0) {
            const double candidate = -right_info_.p[3] / right_info_.p[0];
            if (candidate > 0.01) baseline_ = candidate;
        }
        stereo_calib_ready_ = fx_ > 0.0 && fy_ > 0.0 && baseline_ > 0.0;
    }

    void mavros_local_position_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        external_altitude_m_ = msg->pose.position.z;
        have_external_altitude_ = true;
        std::lock_guard<std::mutex> lk(state_mutex_);
        eskf_.set_external_altitude(external_altitude_m_);
    }

    bool get_depth_for_stamp(double stamp, cv::Mat& depth) {
        std::lock_guard<std::mutex> lk(depth_mutex_);
        if (latest_depth_m_.empty() || std::abs(stamp - latest_depth_stamp_) > 0.20) return false;
        depth = latest_depth_m_.clone();
        return true;
    }

    cv::Mat stereo_depth(const cv::Mat& left_gray, const cv::Mat& right_gray) {
        cv::Mat disp16;
        stereo_sgbm_->compute(left_gray, right_gray, disp16);
        cv::Mat disp;
        disp16.convertTo(disp, CV_32F, 1.0 / 16.0);
        cv::Mat depth(disp.size(), CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
        const double fb = fx_ * baseline_;
        for (int y = 0; y < disp.rows; ++y) {
            const float* drow = disp.ptr<float>(y);
            float* zrow = depth.ptr<float>(y);
            for (int x = 0; x < disp.cols; ++x) {
                if (drow[x] > 1.0f) zrow[x] = static_cast<float>(fb / drow[x]);
            }
        }
        return depth;
    }

    bool estimate_visual_velocity(const std::vector<TrackedPoint>& tracks,
                                  const cv::Mat& prev_depth,
                                  const cv::Mat& curr_depth,
                                  double dt,
                                  Vec3* v_body,
                                  int* used) {
        std::vector<double> vx;
        std::vector<double> vy;
        std::vector<double> vz;
        for (const auto& t : tracks) {
            if (!t.valid) continue;
            const int u0 = static_cast<int>(std::lround(t.prev_x));
            const int v0 = static_cast<int>(std::lround(t.prev_y));
            const int u1 = static_cast<int>(std::lround(t.curr_x));
            const int v1 = static_cast<int>(std::lround(t.curr_y));
            double z0 = 0.0, z1 = 0.0;
            if (!depth_at(prev_depth, u0, v0, &z0) || !depth_at(curr_depth, u1, v1, &z1)) continue;
            const Vec3 P0 = back_project(t.prev_x, t.prev_y, z0, fx_, fy_, cx_, cy_);
            const Vec3 P1 = back_project(t.curr_x, t.curr_y, z1, fx_, fy_, cx_, cy_);
            const Vec3 vc = (P1 - P0) / dt;

            vx.push_back(vc.z()); // camera forward -> body x
            vy.push_back(-vc.x()); // camera right -> body y sign for ENU-ish body
            vz.push_back(-vc.y()); // camera down/up image axis -> body z
        }
        *used = static_cast<int>(vx.size());
        if (*used < 12) return false;
        *v_body = Vec3(median(vx), median(vy), median(vz));
        return v_body->allFinite() && v_body->norm() < 6.0;
    }

    double latest_gyro_norm() const {
        std::lock_guard<std::mutex> lk(state_mutex_);
        return latest_gyro_.norm();
    }

    void publish_state(const rclcpp::Time& stamp) {
        VioNominalState state;
        Mat21 cov;
        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            state = eskf_.state();
            cov = eskf_.covariance();
        }

        nav_msgs::msg::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = "odom";
        odom.child_frame_id = "base_link";
        odom.pose.pose.position.x = state.p_WB.x();
        odom.pose.pose.position.y = state.p_WB.y();
        odom.pose.pose.position.z = state.p_WB.z();
        odom.pose.pose.orientation = to_msg(state.q_WB);
        odom.twist.twist.linear.x = state.v_WB.x();
        odom.twist.twist.linear.y = state.v_WB.y();
        odom.twist.twist.linear.z = state.v_WB.z();
        for (double& v : odom.pose.covariance) v = 0.0;
        odom.pose.covariance[0] = cov(kIdxP, kIdxP);
        odom.pose.covariance[7] = cov(kIdxP + 1, kIdxP + 1);
        odom.pose.covariance[14] = cov(kIdxP + 2, kIdxP + 2);
        odom.pose.covariance[21] = cov(kIdxTheta, kIdxTheta);
        odom.pose.covariance[28] = cov(kIdxTheta + 1, kIdxTheta + 1);
        odom.pose.covariance[35] = cov(kIdxTheta + 2, kIdxTheta + 2);
        odom_pub_->publish(odom);

        geometry_msgs::msg::TransformStamped tf;
        tf.header = odom.header;
        tf.child_frame_id = "base_link";
        tf.transform.translation.x = state.p_WB.x();
        tf.transform.translation.y = state.p_WB.y();
        tf.transform.translation.z = state.p_WB.z();
        tf.transform.rotation = to_msg(state.q_WB);
        tf_broadcaster_.sendTransform(tf);

        if (publish_mavros_vision_) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header = odom.header;
            pose.pose = odom.pose.pose;
            vision_pose_pub_->publish(pose);
        }
    }

    void publish_debug(const cv::Mat& gray, const cv::Mat& depth,
                       const std::vector<TrackedPoint>& tracks,
                       const rclcpp::Time& stamp) {
        if (tracks_pub_->get_subscription_count() > 0) {
            cv::Mat vis;
            cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
            for (const auto& t : tracks) {
                if (!t.valid) continue;
                cv::line(vis, cv::Point2f(t.prev_x, t.prev_y), cv::Point2f(t.curr_x, t.curr_y),
                         cv::Scalar(0, 255, 0), 1);
                cv::circle(vis, cv::Point2f(t.curr_x, t.curr_y), 2, cv::Scalar(0, 0, 255), -1);
            }
            auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", vis).toImageMsg();
            msg->header.stamp = stamp;
            msg->header.frame_id = "oakd_left";
            tracks_pub_->publish(*msg);
        }
        if (depth_debug_pub_->get_subscription_count() > 0 && !depth.empty()) {
            cv::Mat norm, u8;
            cv::patchNaNs(depth, 0.0);
            cv::normalize(depth, norm, 0, 255, cv::NORM_MINMAX);
            norm.convertTo(u8, CV_8U);
            auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", u8).toImageMsg();
            msg->header.stamp = stamp;
            msg->header.frame_id = "oakd_depth";
            depth_debug_pub_->publish(*msg);
        }
    }

    void log_heartbeat() {
        const auto& s = eskf_.state();
        RCLCPP_INFO(get_logger(),
            "VIO HB | p=(%.2f %.2f %.2f) v=(%.2f %.2f %.2f) features=%d cuBLAS=%s",
            s.p_WB.x(), s.p_WB.y(), s.p_WB.z(),
            s.v_WB.x(), s.v_WB.y(), s.v_WB.z(),
            tracker_->num_features(), cublas_.available() ? "yes" : "no");
    }

    message_filters::Subscriber<sensor_msgs::msg::Image> left_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> right_sub_;
    std::shared_ptr<message_filters::Synchronizer<
        message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::Image, sensor_msgs::msg::Image>>> sync_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr right_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr mavros_local_pos_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr vision_pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr vision_pose_cov_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr tracks_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_debug_pub_;
    rclcpp::TimerBase::SharedPtr imu_timer_;
    rclcpp::TimerBase::SharedPtr heartbeat_timer_;

    std::unique_ptr<VPIFeatureTracker> tracker_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    CublasCovarianceUpdater cublas_;
    ErrorStateKalmanFilter eskf_;

    mutable std::mutex state_mutex_;
    Vec3 latest_gyro_ = Vec3::Zero();
    double last_imu_stamp_ = -1.0;
    double last_image_stamp_ = -1.0;
    bool have_external_altitude_ = false;
    double external_altitude_m_ = 0.0;
    bool publish_mavros_vision_ = false;

    std::mutex imu_mutex_;
    std::deque<ImuMeasurement> imu_queue_;

    std::mutex depth_mutex_;
    cv::Mat latest_depth_m_;
    double latest_depth_stamp_ = -1.0;
    cv::Mat prev_depth_m_;

    std::mutex cam_info_mutex_;
    sensor_msgs::msg::CameraInfo left_info_;
    sensor_msgs::msg::CameraInfo right_info_;
    bool have_left_info_ = false;
    bool have_right_info_ = false;
    double fx_ = 460.0;
    double fy_ = 460.0;
    double cx_ = 320.0;
    double cy_ = 240.0;
    double baseline_ = 0.075;
    bool stereo_calib_ready_ = false;

    cv::Ptr<cv::StereoSGBM> stereo_sgbm_;
    int log_counter_ = 0;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VIONode>());
    rclcpp::shutdown();
    return 0;
}

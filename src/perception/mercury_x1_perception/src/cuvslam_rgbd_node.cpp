// cuvslam_rgbd_node: GPU visual odometry (NVIDIA cuVSLAM, RGB-D mode) from the Orbbec DaBai.
//
// The DaBai has no hardware colour/depth sync (colour is a separate UVC device), so each depth
// frame (15 Hz) is paired with the nearest colour frame (30 Hz); the pair is decided once a newer
// colour frame arrived. cuVSLAM tracks the colour optical frame; the base pose is
// camera pose * TF(camera <- base) at the image time, so jaw / neck / waist motion is handled.
// Frame-to-frame deltas are projected onto the floor (x, y, yaw) and integrated (planar robot).
//
// Out: /vslam/odom (nav_msgs/Odometry, frame vslam_odom -> base_footprint), /vslam/path, and a static
// TF align_frame -> vslam_odom (the align_frame pose of the base at the first tracked frame), so the
// trajectory overlays the lidar odometry for comparison. It does not drive the TF tree.
#include <chrono>
#include <cmath>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "cuvslam/cuvslam2.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tf2_eigen/tf2_eigen.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"

using sensor_msgs::msg::Image;

namespace {

int64_t to_ns(const builtin_interfaces::msg::Time& t) { return int64_t(t.sec) * 1000000000LL + t.nanosec; }

Eigen::Isometry3d to_iso(const cuvslam::Pose& p) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = Eigen::Quaterniond(p.rotation[3], p.rotation[0], p.rotation[1], p.rotation[2]).normalized().toRotationMatrix();
  T.translation() = Eigen::Vector3d(p.translation[0], p.translation[1], p.translation[2]);
  return T;
}

// x, y, yaw of a transform whose z axis is (nearly) vertical
Eigen::Vector3d planar(const Eigen::Isometry3d& T) {
  const Eigen::Vector3d x = T.linear().col(0);
  return {T.translation().x(), T.translation().y(), std::atan2(x.y(), x.x())};
}

Eigen::Isometry3d from_planar(double x, double y, double yaw) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  T.translation() = Eigen::Vector3d(x, y, 0.0);
  return T;
}

}  // namespace

class CuvslamRgbdNode : public rclcpp::Node {
 public:
  CuvslamRgbdNode() : Node("cuvslam_rgbd") {
    odom_frame_ = declare_parameter("odom_frame", "vslam_odom");
    base_frame_ = declare_parameter("base_frame", "base_footprint");
    align_frame_ = declare_parameter("align_frame", "odom");
    depth_offset_ns_ = int64_t(declare_parameter("depth_time_offset_ms", 0.0) * 1e6);
    max_pair_dt_ns_ = int64_t(declare_parameter("max_pair_dt_ms", 25.0) * 1e6);
    depth_scale_ = declare_parameter("depth_scale", 1000.0);  // uint16 mm -> m
    border_ = declare_parameter("border_px", 8);

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    static_tf_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("vslam/odom", 10);
    path_pub_ = create_publisher<nav_msgs::msg::Path>("vslam/path", 1);

    const auto qos = rclcpp::SensorDataQoS().keep_last(10);
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        "camera_info", qos, [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) { if (!tracker_) init(*m); });
    color_sub_ = create_subscription<Image>("color", qos, [this](Image::ConstSharedPtr m) {
      ++color_rx_;
      colors_.push_back(m);
      while (colors_.size() > 12) colors_.pop_front();
      process();
    });
    depth_sub_ = create_subscription<Image>("depth", qos, [this](Image::ConstSharedPtr m) {
      ++depth_rx_;
      depths_.push_back(m);
      while (depths_.size() > 4) depths_.pop_front();
      process();
    });
    stats_timer_ = create_wall_timer(std::chrono::seconds(10), [this] { report(); });
    path_.header.frame_id = odom_frame_;
  }

 private:
  void init(const sensor_msgs::msg::CameraInfo& ci) {
    cuvslam::Camera cam;
    cam.size = {int32_t(ci.width), int32_t(ci.height)};
    cam.focal = {float(ci.k[0]), float(ci.k[4])};
    cam.principal = {float(ci.k[2]), float(ci.k[5])};
    cam.border_top = cam.border_bottom = cam.border_left = cam.border_right = border_;
    const auto& d = ci.d;
    if (ci.distortion_model == "rational_polynomial" && d.size() >= 8) {
      cam.distortion.model = cuvslam::Distortion::Model::Polynomial;  // OpenCV order k1 k2 p1 p2 k3 k4 k5 k6
      cam.distortion.parameters.assign(d.begin(), d.begin() + 8);
    } else if (ci.distortion_model == "plumb_bob" && d.size() >= 5) {
      cam.distortion.model = cuvslam::Distortion::Model::Brown;  // cuVSLAM order k1 k2 k3 p1 p2
      cam.distortion.parameters = {float(d[0]), float(d[1]), float(d[4]), float(d[2]), float(d[3])};
    }
    cuvslam::Rig rig;
    rig.cameras = {cam};  // rig frame = colour optical frame

    cuvslam::Odometry::Config cfg;
    cfg.odometry_mode = cuvslam::Odometry::OdometryMode::RGBD;
    cfg.rgbd_settings.depth_scale_factor = float(depth_scale_);
    cfg.rgbd_settings.depth_camera_id = 0;
    cfg.rgbd_settings.enable_depth_stereo_tracking = false;
    cfg.async_sba = true;
    cfg.use_motion_model = true;
    cfg.max_frame_delta_s = 1.0F;
    cuvslam::WarmUpGPU();
    tracker_ = std::make_unique<cuvslam::Odometry>(rig, cfg);
    camera_frame_ = ci.header.frame_id;
    width_ = ci.width;
    height_ = ci.height;
    int32_t ma, mi, pa;
    const auto ver = cuvslam::GetVersion(&ma, &mi, &pa);
    RCLCPP_INFO(get_logger(), "cuVSLAM %.*s RGB-D on %s %ux%u f=%.1f distortion=%s", int(ver.size()), ver.data(),
                camera_frame_.c_str(), width_, height_, ci.k[0], ci.distortion_model.c_str());
  }

  // pair the oldest pending depth with the nearest colour once a newer colour exists (or give up)
  void process() {
    if (!tracker_) return;
    while (!depths_.empty() && !colors_.empty()) {
      const auto depth = depths_.front();
      const int64_t td = to_ns(depth->header.stamp) + depth_offset_ns_;
      if (to_ns(colors_.back()->header.stamp) < td && depths_.size() < 3) return;  // wait for a later colour
      Image::ConstSharedPtr best;
      int64_t best_dt = INT64_MAX;
      for (const auto& c : colors_) {
        const int64_t dt = std::llabs(to_ns(c->header.stamp) - td);
        if (dt < best_dt) { best_dt = dt; best = c; }
      }
      depths_.pop_front();
      signed_dt_sum_ += double(to_ns(best->header.stamp) - td) * 1e-6;
      ++depth_count_;
      if (best_dt > max_pair_dt_ns_) { ++unpaired_; continue; }
      pair_dt_sum_ += double(best_dt) * 1e-6;
      track(*best, *depth);
    }
  }

  void track(const Image& color, const Image& depth) {
    if (color.encoding != "rgb8" || depth.encoding != "16UC1" || color.width != width_ || depth.width != width_ ||
        color.height != height_ || depth.height != height_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "unexpected images: %s %ux%u / %s %ux%u",
                           color.encoding.c_str(), color.width, color.height, depth.encoding.c_str(), depth.width, depth.height);
      return;
    }
    const int64_t ts = to_ns(color.header.stamp);
    if (ts <= last_ts_) return;  // cuVSLAM rejects non-increasing timestamps
    last_ts_ = ts;

    cuvslam::Image img{};
    img.pixels = color.data.data();
    img.width = int32_t(color.width);
    img.height = int32_t(color.height);
    img.pitch = int32_t(color.step);
    img.encoding = cuvslam::ImageData::Encoding::RGB;
    img.data_type = cuvslam::ImageData::DataType::UINT8;
    img.is_gpu_mem = false;
    img.timestamp_ns = ts;
    img.camera_index = 0;
    cuvslam::Image dep = img;
    dep.pixels = depth.data.data();
    dep.pitch = int32_t(depth.step);
    dep.encoding = cuvslam::ImageData::Encoding::MONO;
    dep.data_type = cuvslam::ImageData::DataType::UINT16;

    const auto t0 = std::chrono::steady_clock::now();
    cuvslam::PoseEstimate est;
    try {
      est = tracker_->Track({img}, {}, {dep});
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Track failed: %s", e.what());
      return;
    }
    track_ms_sum_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ++frames_;
    if (!est.world_from_rig) {  // lost: the next valid pose starts a new cuVSLAM world
      ++lost_;
      prev_world_base_.reset();
      return;
    }

    // camera <- base at the image time (head joints move the camera)
    Eigen::Isometry3d cam_base;
    try {
      cam_base = tf2::transformToEigen(
          tf_buffer_->lookupTransform(camera_frame_, base_frame_, color.header.stamp, rclcpp::Duration::from_seconds(0.05)));
    } catch (const tf2::TransformException& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "no TF %s <- %s: %s", camera_frame_.c_str(),
                           base_frame_.c_str(), e.what());
      prev_world_base_.reset();
      return;
    }
    const Eigen::Isometry3d world_base = to_iso(est.world_from_rig->pose) * cam_base;

    if (!aligned_ && align_tries_ < 100) aligned_ = align(color.header.stamp);
    if (prev_world_base_) {
      // base-frame delta; the base's z axis is vertical, so keep only x, y, yaw of it
      const Eigen::Vector3d d = planar(prev_world_base_->inverse() * world_base);
      const double dt = double(ts - prev_ts_) * 1e-9;
      pose_ = pose_ * from_planar(d.x(), d.y(), d.z());
      if (dt > 1e-3) twist_ = d / dt;
    }
    prev_world_base_ = world_base;
    prev_ts_ = ts;
    publish(color.header.stamp);
  }

  // align_frame -> vslam_odom = align_frame pose of the base, minus what vslam integrated so far
  bool align(const builtin_interfaces::msg::Time& stamp) {
    if (align_frame_.empty()) return true;
    ++align_tries_;
    geometry_msgs::msg::TransformStamped t;
    try {
      t = tf_buffer_->lookupTransform(align_frame_, base_frame_, stamp, rclcpp::Duration::from_seconds(0.05));
    } catch (const tf2::TransformException& e) {
      if (align_tries_ == 100) RCLCPP_WARN(get_logger(), "vslam_odom left unaligned: %s", e.what());
      return false;
    }
    const Eigen::Vector3d p = planar(tf2::transformToEigen(t) * pose_.inverse());
    auto st = tf2::eigenToTransform(from_planar(p.x(), p.y(), p.z()));
    st.header.stamp = stamp;
    st.header.frame_id = align_frame_;
    st.child_frame_id = odom_frame_;
    static_tf_->sendTransform(st);
    RCLCPP_INFO(get_logger(), "aligned %s -> %s", align_frame_.c_str(), odom_frame_.c_str());
    return true;
  }

  void publish(const builtin_interfaces::msg::Time& stamp) {
    nav_msgs::msg::Odometry o;
    o.header.stamp = stamp;
    o.header.frame_id = odom_frame_;
    o.child_frame_id = base_frame_;
    o.pose.pose = tf2::toMsg(pose_);
    o.twist.twist.linear.x = twist_.x();
    o.twist.twist.linear.y = twist_.y();
    o.twist.twist.angular.z = twist_.z();
    odom_pub_->publish(o);

    const Eigen::Vector3d p = planar(pose_);
    if (path_.poses.empty() || (p.head<2>() - last_path_xy_).norm() > 0.02 ||
        std::abs(p.z() - last_path_yaw_) > 0.05) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header = o.header;
      ps.pose = o.pose.pose;
      path_.poses.push_back(ps);
      if (path_.poses.size() > 3000) path_.poses.erase(path_.poses.begin());
      path_.header.stamp = stamp;
      path_pub_->publish(path_);
      last_path_xy_ = p.head<2>();
      last_path_yaw_ = p.z();
    }
  }

  void report() {
    if (!tracker_) {
      RCLCPP_INFO(get_logger(), "waiting for camera_info");
      return;
    }
    const Eigen::Vector3d p = planar(pose_);
    RCLCPP_INFO(get_logger(),
                "10 s: rx colour %d depth %d, %d depth, %d tracked (%.1f Hz), lost %d, unpaired %d, track %.1f ms, |dt| %.1f ms, "
                "colour-depth %.1f ms | x %.3f y %.3f yaw %.1f deg",
                color_rx_, depth_rx_, depth_count_, frames_, frames_ / 10.0, lost_, unpaired_, frames_ ? track_ms_sum_ / frames_ : 0.0,
                frames_ ? pair_dt_sum_ / frames_ : 0.0, depth_count_ ? signed_dt_sum_ / depth_count_ : 0.0, p.x(),
                p.y(), p.z() * 180.0 / M_PI);
    frames_ = lost_ = unpaired_ = depth_count_ = color_rx_ = depth_rx_ = 0;
    track_ms_sum_ = pair_dt_sum_ = signed_dt_sum_ = 0.0;
  }

  std::string odom_frame_, base_frame_, align_frame_, camera_frame_;
  int64_t depth_offset_ns_, max_pair_dt_ns_;
  double depth_scale_;
  int border_;
  uint32_t width_ = 0, height_ = 0;
  std::unique_ptr<cuvslam::Odometry> tracker_;
  std::deque<Image::ConstSharedPtr> colors_, depths_;
  int64_t last_ts_ = 0, prev_ts_ = 0;
  bool aligned_ = false;
  int align_tries_ = 0;
  std::optional<Eigen::Isometry3d> prev_world_base_;
  Eigen::Isometry3d pose_ = Eigen::Isometry3d::Identity();
  Eigen::Vector3d twist_ = Eigen::Vector3d::Zero();
  Eigen::Vector2d last_path_xy_ = Eigen::Vector2d::Zero();
  double last_path_yaw_ = 0.0;
  nav_msgs::msg::Path path_;
  int frames_ = 0, lost_ = 0, unpaired_ = 0, depth_count_ = 0, color_rx_ = 0, depth_rx_ = 0;
  double track_ms_sum_ = 0.0, pair_dt_sum_ = 0.0, signed_dt_sum_ = 0.0;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Subscription<Image>::SharedPtr color_sub_, depth_sub_;
  rclcpp::TimerBase::SharedPtr stats_timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CuvslamRgbdNode>());
  rclcpp::shutdown();
  return 0;
}

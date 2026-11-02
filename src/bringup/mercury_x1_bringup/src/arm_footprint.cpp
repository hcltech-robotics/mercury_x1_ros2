// Arm-aware Nav2 footprint: the base outline grown to cover whatever part of the arms sticks out.
//
// Every tick the arm links' hull points (config/arm_hulls.yaml, from the URDF meshes) plus a tool
// point in front of each flange are put through TF into base_footprint and flattened to the floor.
// Points that stay inside the base outline are ignored (nav2.yaml's footprint already covers the arms
// in the carry pose); if any point is further out than `tolerance`, the footprint becomes the convex
// hull of the base outline and those points, padded by `padding`.
//
// The result goes to <costmap>/footprint (geometry_msgs/Polygon), which Nav2's costmaps take over
// their `footprint` parameter: MPPI (consider_footprint) and the behaviours check collisions with it,
// the costmaps' inflation follows it. It is published only when it changes (a footprint change makes
// the costmaps re-inflate) and again whenever a costmap (re)subscribes.

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "geometry_msgs/msg/polygon.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Transform.h"
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "yaml-cpp/yaml.h"

namespace
{
using P2 = std::array<double, 2>;
using P3 = std::array<double, 3>;
using Poly = std::vector<P2>;

double cross(const P2 & o, const P2 & a, const P2 & b)
{
  return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
}

// 2D convex hull (Andrew's monotone chain), counter-clockwise
Poly convexHull(Poly p)
{
  std::sort(p.begin(), p.end());
  p.erase(std::unique(p.begin(), p.end()), p.end());
  if (p.size() < 3) {return p;}
  Poly h(2 * p.size());
  size_t k = 0;
  for (size_t i = 0; i < p.size(); ++i) {                      // lower
    while (k >= 2 && cross(h[k - 2], h[k - 1], p[i]) <= 0) {--k;}
    h[k++] = p[i];
  }
  for (size_t i = p.size() - 1, t = k + 1; i-- > 0;) {         // upper
    while (k >= t && cross(h[k - 2], h[k - 1], p[i]) <= 0) {--k;}
    h[k++] = p[i];
  }
  h.resize(k - 1);
  return h;
}

// distance of a point outside a convex counter-clockwise polygon (<= 0 inside)
double outsideDistance(const Poly & poly, const P2 & q)
{
  double d = -1e9;
  for (size_t i = 0; i < poly.size(); ++i) {
    const P2 & a = poly[i];
    const P2 & b = poly[(i + 1) % poly.size()];
    const double ex = b[0] - a[0], ey = b[1] - a[1], n = std::hypot(ex, ey);
    if (n < 1e-12) {continue;}
    d = std::max(d, ((q[0] - a[0]) * ey - (q[1] - a[1]) * ex) / n);   // outward normal (ey, -ex)
  }
  return d;
}

// points grown by a disc of radius r (each hull vertex -> a ring of points)
Poly padPoints(const Poly & pts, double r, int segments = 8)
{
  if (r <= 0) {return pts;}
  const Poly hull = pts.size() >= 3 ? convexHull(pts) : pts;
  Poly out;
  for (const auto & v : hull) {
    for (int s = 0; s < segments; ++s) {
      const double a = 2 * M_PI * s / segments;
      out.push_back({v[0] + r * std::cos(a), v[1] + r * std::sin(a)});
    }
  }
  return out;
}
}  // namespace

class ArmFootprint : public rclcpp::Node
{
public:
  ArmFootprint()
  : Node("arm_footprint")
  {
    const auto base = YAML::Load(declare_parameter("base_footprint",
        std::string("[[0.22, 0.24], [0.22, -0.24], [-0.39, -0.24], [-0.39, 0.24]]")));
    Poly b;
    for (const auto & p : base) {b.push_back({p[0].as<double>(), p[1].as<double>()});}
    base_ = convexHull(b);
    frame_ = declare_parameter("robot_base_frame", std::string("base_footprint"));
    tolerance_ = declare_parameter("tolerance", 0.05);       // ignore arm parts this little outside the base
    padding_ = declare_parameter("padding", 0.04);           // clearance added around protruding arm parts
    const double tool_length = declare_parameter("tool_length", 0.08);   // hand / gripper beyond the flange (+z of link7)
    min_change_ = declare_parameter("min_change", 0.02);
    max_height_ = declare_parameter("max_height", 2.0);      // arm parts above this cannot hit anything
    const auto costmaps = declare_parameter("costmaps", std::vector<std::string>{"local_costmap", "global_costmap"});
    const auto hull_file = declare_parameter("hulls_file",
        ament_index_cpp::get_package_share_directory("mercury_x1_bringup") + "/config/arm_hulls.yaml");
    for (const auto & kv : YAML::LoadFile(hull_file)) {
      std::vector<P3> pts;
      for (const auto & p : kv.second) {pts.push_back({p[0].as<double>(), p[1].as<double>(), p[2].as<double>()});}
      hulls_[kv.first.as<std::string>()] = pts;
    }
    for (const std::string side : {"L", "R"}) {
      auto it = hulls_.find("link7_" + side);
      if (it != hulls_.end() && tool_length > 0) {it->second.push_back({0.0, 0.0, tool_length});}
    }

    tf_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_);
    for (const auto & c : costmaps) {
      pubs_.push_back({create_publisher<geometry_msgs::msg::Polygon>("/" + c + "/footprint", 1), 0});
    }
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / declare_parameter("rate", 5.0)), [this] {tick();});
    std::string names;
    for (const auto & c : costmaps) {names += (names.empty() ? "" : ", ") + c;}
    RCLCPP_INFO(get_logger(), "arm-aware footprint for [%s] (%zu links, tool %.2f m)", names.c_str(), hulls_.size(),
                tool_length);
  }

private:
  // arm hull points in the base frame, flattened; false if any link's TF is missing (arms off / driver down)
  bool armPoints(Poly & out)
  {
    out.clear();
    for (const auto & [link, hull] : hulls_) {
      geometry_msgs::msg::TransformStamped ts;
      try {
        ts = tf_->lookupTransform(frame_, link, tf2::TimePointZero);
      } catch (const tf2::TransformException &) {
        return false;
      }
      const auto & r = ts.transform.rotation;
      const auto & t = ts.transform.translation;
      const tf2::Transform tr(tf2::Quaternion(r.x, r.y, r.z, r.w), tf2::Vector3(t.x, t.y, t.z));
      for (const auto & p : hull) {
        const tf2::Vector3 w = tr * tf2::Vector3(p[0], p[1], p[2]);
        if (w.z() <= max_height_) {out.push_back({w.x(), w.y()});}
      }
    }
    return true;
  }

  Poly footprint(bool & expanded)
  {
    expanded = false;
    Poly arm;
    if (!armPoints(arm)) {return base_;}
    Poly outside;
    for (const auto & p : arm) {
      if (outsideDistance(base_, p) > tolerance_) {outside.push_back(p);}
    }
    if (outside.empty()) {return base_;}
    Poly all = base_;
    const Poly grown = padPoints(outside, padding_);
    all.insert(all.end(), grown.begin(), grown.end());
    expanded = true;
    return convexHull(all);
  }

  void tick()
  {
    bool expanded;
    const Poly fp = footprint(expanded);
    bool changed = !have_ || fp.size() != current_.size();
    for (size_t i = 0; !changed && i < fp.size(); ++i) {
      changed = std::abs(fp[i][0] - current_[i][0]) > min_change_ || std::abs(fp[i][1] - current_[i][1]) > min_change_;
    }
    if (changed) {
      current_ = fp;
      have_ = true;
      if (expanded != expanded_ || expanded) {
        if (!expanded) {
          RCLCPP_INFO(get_logger(), "footprint: base outline (arms inside)");
        } else {
          double x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
          for (const auto & p : fp) {
            x0 = std::min(x0, p[0]); x1 = std::max(x1, p[0]); y0 = std::min(y0, p[1]); y1 = std::max(y1, p[1]);
          }
          RCLCPP_INFO(get_logger(), "footprint: grown for the arms, x %+.2f..%+.2f y %+.2f..%+.2f m", x0, x1, y0, y1);
        }
      }
      expanded_ = expanded;
    }
    for (auto & [pub, seen] : pubs_) {
      const size_t n = pub->get_subscription_count();
      if (changed || n > seen) {         // a costmap (re)started: give it the current footprint
        geometry_msgs::msg::Polygon msg;
        for (const auto & p : current_) {
          geometry_msgs::msg::Point32 q;
          q.x = static_cast<float>(p[0]);
          q.y = static_cast<float>(p[1]);
          msg.points.push_back(q);
        }
        pub->publish(msg);
      }
      seen = n;
    }
  }

  Poly base_, current_;
  bool have_{false}, expanded_{false};
  std::string frame_;
  double tolerance_, padding_, min_change_, max_height_;
  std::map<std::string, std::vector<P3>> hulls_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::vector<std::pair<rclcpp::Publisher<geometry_msgs::msg::Polygon>::SharedPtr, size_t>> pubs_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ArmFootprint>());
  rclcpp::shutdown();
  return 0;
}

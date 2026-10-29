// Mercury X1 upper-body "goalkeeper": owns /dev/left_arm and /dev/right_arm.
//
// One worker thread per serial port polls joint angles + flange coords in a single
// transaction (0xD5) and runs commands between polls. Neck, jaw and waist hang off
// the right-arm MCU.
//
// Publishes  joint_states, left_arm/state, right_arm/state      (mercury_x1_msgs/ArmState)
// Subscribes upper_body/joint_command     (mercury_x1_msgs/JointCommand)
//            upper_body/cartesian_command (mercury_x1_msgs/CartesianCommand)
// Services   left_arm/power, right_arm/power (std_srvs/SetBool)
//            left_arm/stop, right_arm/stop, upper_body/stop (std_srvs/Trigger)
//
// Firmware behaviour relied on (measured, see the devlog):
//  * motion commands queue inside the MCU rather than replacing the running motion, so a
//    preempting command is sent as STOP (no payload) followed by the new target;
//  * only power/status/single-joint reads answer while an arm is powered off.

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <optional>
#include <thread>

#include "mercury_x1_driver/arm_protocol.hpp"
#include "mercury_x1_msgs/msg/arm_state.hpp"
#include "mercury_x1_msgs/msg/cartesian_command.hpp"
#include "mercury_x1_msgs/msg/joint_command.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "std_srvs/srv/trigger.hpp"

using namespace std::chrono_literals;
namespace arm = mercury_x1::arm;

namespace
{
// firmware joint limits [deg] (pymycobot robot_info "Mercury")
constexpr double kArmLimits[7][2] = {
  {-165, 165}, {-50, 120}, {-165, 165}, {-165, 1}, {-165, 165}, {-75, 255}, {-165, 165}};
const std::map<int, std::pair<double, double>> kExtraLimits = {{11, {-55, 0}}, {12, {-70, 245}}, {13, {-160, 160}}};

double deg2rad(double d) {return d * M_PI / 180.0;}
double rad2deg(double r) {return r * 180.0 / M_PI;}
double steadyNow()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

struct Motion
{
  enum Kind { kJoints, kCartesian } kind;
  int speed;
  std::map<int, double> arm;     // joint index 0..6 -> deg
  std::map<int, double> extra;   // firmware id 11..13 -> deg
  arm::Coords coords{};
  bool body_frame{false};
  bool linear{true};
};

struct ArmSnapshot
{
  bool connected{false}, powered{false}, moving{false}, have_q{false};
  arm::Joints q_deg{};
  std::array<double, 7> qd{};
  arm::Coords coords{};
  arm::Coords body{};
  arm::Bytes status;
  std::map<int, double> extra;
  double rate_hz{0};
};

class ArmWorker
{
public:
  using Job = std::function<void(arm::ArmLink &)>;

  ArmWorker(rclcpp::Logger log, std::string side, std::string port, std::vector<int> extra_ids,
            bool power_on_start, bool poll_body)
  : log_(log), side_(std::move(side)), port_(std::move(port)), extra_ids_(std::move(extra_ids)),
    power_on_start_(power_on_start), poll_body_(poll_body)
  {
    snap_.coords.fill(NAN);
    snap_.body.fill(NAN);
    for (int id : extra_ids_) {snap_.extra[id] = NAN;}
    thread_ = std::thread([this] {run();});
  }

  ~ArmWorker()
  {
    running_ = false;
    if (thread_.joinable()) {thread_.join();}
  }

  ArmSnapshot snapshot()
  {
    std::lock_guard<std::mutex> lk(mtx_);
    return snap_;
  }

  // Runs fn on the worker thread and waits for it.
  template<class R>
  std::optional<R> call(std::function<R(arm::ArmLink &)> fn, std::chrono::milliseconds timeout)
  {
    auto prom = std::make_shared<std::promise<R>>();
    auto fut = prom->get_future();
    {
      std::lock_guard<std::mutex> lk(mtx_);
      jobs_.push_back([fn, prom](arm::ArmLink & l) {
          try {prom->set_value(fn(l));} catch (...) {prom->set_exception(std::current_exception());}
        });
    }
    if (fut.wait_for(timeout) != std::future_status::ready) {return std::nullopt;}
    return fut.get();
  }

  void setTarget(Motion m, bool queued)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    if (queued) {
      jobs_.push_back([this, m](arm::ArmLink & l) {execute(l, m, false);});
    } else {
      target_ = std::move(m);
    }
  }

  void clearTarget()
  {
    std::lock_guard<std::mutex> lk(mtx_);
    target_.reset();
  }

  void markPowered(bool p)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    snap_.powered = p;
  }

private:
  template<class F>
  void update(F && f)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    f(snap_);
  }

  void run()
  {
    std::unique_ptr<arm::ArmLink> link;
    uint64_t k = 0;
    double last_t = -1, rate_t = steadyNow();
    int rate_n = 0;
    arm::Joints last_q{};
    std::array<double, 7> qd{};
    while (running_) {
      if (!link) {
        try {
          link = std::make_unique<arm::ArmLink>(port_);
          auto pw = link->isPowerOn();
          RCLCPP_INFO(log_, "%s arm: opened %s (powered=%d)", side_.c_str(), port_.c_str(), pw.value_or(false));
          bool powered = pw.value_or(false);
          if (power_on_start_ && pw.has_value() && !powered) {
            RCLCPP_INFO(log_, "%s arm: powering on", side_.c_str());
            powered = link->power(true);
          }
          update([&](ArmSnapshot & s) {s.connected = pw.has_value(); s.powered = powered;});
        } catch (const std::exception & e) {
          RCLCPP_ERROR(log_, "%s arm: %s", side_.c_str(), e.what());
          link.reset();
          std::this_thread::sleep_for(2s);
          continue;
        }
      }
      try {
        // 1) jobs (services, queued motions)
        while (true) {
          Job job;
          {
            std::lock_guard<std::mutex> lk(mtx_);
            if (jobs_.empty()) {break;}
            job = std::move(jobs_.front());
            jobs_.pop_front();
          }
          job(*link);
        }
        // 2) latest-wins target
        std::optional<Motion> tgt;
        {
          std::lock_guard<std::mutex> lk(mtx_);
          tgt.swap(target_);
        }
        if (tgt) {execute(*link, *tgt, true);}

        // 3) polling
        const bool powered = snapshot().powered;
        if (powered) {
          if (auto r = link->anglesCoords()) {
            const double t = steadyNow();
            if (last_t > 0 && t - last_t > 1e-3) {
              for (int i = 0; i < 7; ++i) {
                qd[i] = 0.6 * qd[i] + 0.4 * deg2rad(r->first[i] - last_q[i]) / (t - last_t);
              }
            }
            last_t = t;
            last_q = r->first;
            ++rate_n;
            update([&](ArmSnapshot & s) {
                s.q_deg = r->first; s.coords = r->second; s.qd = qd; s.have_q = true; s.connected = true;
              });
          } else {
            auto pw = link->isPowerOn();
            update([&](ArmSnapshot & s) {s.connected = pw.has_value(); s.powered = pw.value_or(false);});
          }
          if (k % 10 == 0) {
            if (auto m = link->isMoving()) {update([&](ArmSnapshot & s) {s.moving = *m;});}
          }
          if (poll_body_ && k % 10 == 5) {
            if (auto b = link->bodyCoords()) {update([&](ArmSnapshot & s) {s.body = *b;});}
          }
          if (k % 25 == 0) {
            if (auto st = link->status()) {update([&](ArmSnapshot & s) {s.status = *st;});}
          }
        } else {
          if (k % 20 == 0) {
            auto pw = link->isPowerOn();
            auto st = link->status();
            update([&](ArmSnapshot & s) {
                s.connected = pw.has_value(); s.powered = pw.value_or(false); s.moving = false;
                if (st) {s.status = *st;}
              });
          }
          std::this_thread::sleep_for(20ms);
        }
        if (!extra_ids_.empty() && k % 3 == 0) {
          const int id = extra_ids_[(k / 3) % extra_ids_.size()];
          if (auto v = link->joint(id)) {update([&](ArmSnapshot & s) {s.extra[id] = *v;});}
        }
        ++k;
        const double t = steadyNow();
        if (t - rate_t >= 1.0) {
          const double hz = rate_n / (t - rate_t);
          update([&](ArmSnapshot & s) {s.rate_hz = hz;});
          rate_n = 0;
          rate_t = t;
        }
      } catch (const std::exception & e) {
        RCLCPP_ERROR(log_, "%s arm: %s; reopening", side_.c_str(), e.what());
        update([](ArmSnapshot & s) {s.connected = false;});
        link.reset();
        std::this_thread::sleep_for(1s);
      }
    }
  }

  void execute(arm::ArmLink & link, const Motion & m, bool preempt)
  {
    const bool moves_arm = m.kind == Motion::kCartesian || !m.arm.empty();
    if (moves_arm && preempt && link.isMoving().value_or(false)) {link.stop();}
    if (m.kind == Motion::kJoints) {
      if (m.arm.size() == 1) {
        link.sendAngle(m.arm.begin()->first + 1, m.arm.begin()->second, m.speed);
      } else if (!m.arm.empty()) {
        arm::Joints q = snapshot().q_deg;
        for (auto & [i, d] : m.arm) {q[i] = d;}
        link.sendAngles(q, m.speed);
      }
      for (auto & [id, d] : m.extra) {link.sendAngle(id, d, m.speed);}
    } else {
      if (movement_linear_ != m.linear && link.setMovementType(m.linear)) {movement_linear_ = m.linear;}
      link.sendCoords(m.coords, m.speed, m.body_frame);
    }
    if (moves_arm) {update([](ArmSnapshot & s) {s.moving = true;});}
  }

  rclcpp::Logger log_;
  std::string side_, port_;
  std::vector<int> extra_ids_;
  bool power_on_start_, poll_body_;
  std::optional<bool> movement_linear_;
  std::mutex mtx_;
  ArmSnapshot snap_;
  std::deque<Job> jobs_;
  std::optional<Motion> target_;
  std::atomic<bool> running_{true};
  std::thread thread_;
};

class UpperBodyNode : public rclcpp::Node
{
public:
  UpperBodyNode()
  : Node("mercury_upper_body")
  {
    const auto left_port = declare_parameter("left_port", "/dev/left_arm");
    const auto right_port = declare_parameter("right_port", "/dev/right_arm");
    const bool power_on_start = declare_parameter("power_on_start", true);
    const bool poll_body = declare_parameter("poll_body_coords", true);
    default_speed_ = declare_parameter("default_speed", 30);
    max_speed_ = declare_parameter("max_speed", 100);
    const double rate = declare_parameter("publish_rate", 50.0);
    // neck/jaw/waist: URDF joint name -> firmware id, with sign/offset into the URDF convention
    extra_names_ = declare_parameter("extra_joint_names", std::vector<std::string>{"eye", "head", "body"});
    const auto ids = declare_parameter("extra_joint_ids", std::vector<int64_t>{11, 12, 13});
    extra_sign_ = declare_parameter("extra_joint_signs", std::vector<double>{-1.0, 1.0, 1.0});
    extra_offset_ = declare_parameter("extra_joint_offsets_deg", std::vector<double>{0.0, 0.0, 0.0});
    if (ids.size() != extra_names_.size() || extra_sign_.size() != ids.size() || extra_offset_.size() != ids.size()) {
      throw std::runtime_error("extra_joint_* parameter lists must have equal length");
    }
    std::vector<int> extra_ids;
    for (size_t i = 0; i < ids.size(); ++i) {
      extra_ids.push_back(static_cast<int>(ids[i]));
      extra_index_[extra_names_[i]] = i;
    }
    for (const std::string side : {"left", "right"}) {
      const char s = side == "left" ? 'L' : 'R';
      for (int i = 1; i <= 7; ++i) {arm_names_[side].push_back("joint" + std::to_string(i) + "_" + s);}
    }
    workers_["left"] = std::make_unique<ArmWorker>(get_logger(), "left", left_port, std::vector<int>{},
        power_on_start, poll_body);
    workers_["right"] = std::make_unique<ArmWorker>(get_logger(), "right", right_port, extra_ids,
        power_on_start, poll_body);
    extra_ids_ = extra_ids;

    pub_js_ = create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
    for (auto & [side, _] : workers_) {
      pub_arm_[side] = create_publisher<mercury_x1_msgs::msg::ArmState>(side + "_arm/state", 10);
    }
    sub_joint_ = create_subscription<mercury_x1_msgs::msg::JointCommand>("upper_body/joint_command", 10,
        [this](mercury_x1_msgs::msg::JointCommand::ConstSharedPtr m) {onJointCmd(*m);});
    sub_cart_ = create_subscription<mercury_x1_msgs::msg::CartesianCommand>("upper_body/cartesian_command", 10,
        [this](mercury_x1_msgs::msg::CartesianCommand::ConstSharedPtr m) {onCartCmd(*m);});

    cb_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    for (auto & [side, _] : workers_) {
      const std::string s = side;
      srv_power_.push_back(create_service<std_srvs::srv::SetBool>(s + "_arm/power",
          [this, s](std_srvs::srv::SetBool::Request::ConstSharedPtr req,
          std_srvs::srv::SetBool::Response::SharedPtr res) {srvPower(s, req->data, *res);},
          rmw_qos_profile_services_default, cb_));
      srv_stop_.push_back(create_service<std_srvs::srv::Trigger>(s + "_arm/stop",
          [this, s](std_srvs::srv::Trigger::Request::ConstSharedPtr,
          std_srvs::srv::Trigger::Response::SharedPtr res) {srvStop({s}, *res);},
          rmw_qos_profile_services_default, cb_));
    }
    srv_stop_.push_back(create_service<std_srvs::srv::Trigger>("upper_body/stop",
        [this](std_srvs::srv::Trigger::Request::ConstSharedPtr, std_srvs::srv::Trigger::Response::SharedPtr res) {
          srvStop({"left", "right"}, *res);
        }, rmw_qos_profile_services_default, cb_));
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate), [this] {publish();});
    RCLCPP_INFO(get_logger(), "upper body driver up");
  }

private:
  int speed(int s) const
  {
    return std::clamp<int>(s > 0 ? s : static_cast<int>(default_speed_), 1, static_cast<int>(max_speed_));
  }

  double extraToUrdf(size_t i, double deg) const {return deg2rad(extra_sign_[i] * deg + extra_offset_[i]);}
  double extraFromUrdf(size_t i, double rad) const {return (rad2deg(rad) - extra_offset_[i]) / extra_sign_[i];}

  void onJointCmd(const mercury_x1_msgs::msg::JointCommand & msg)
  {
    if (msg.name.size() != msg.position.size()) {
      RCLCPP_ERROR(get_logger(), "JointCommand: name/position length mismatch");
      return;
    }
    std::map<std::string, Motion> per;
    for (auto & [side, _] : workers_) {per[side] = Motion{Motion::kJoints, speed(msg.speed), {}, {}, {}, false, true};}
    for (size_t n = 0; n < msg.name.size(); ++n) {
      const auto & name = msg.name[n];
      if (auto it = extra_index_.find(name); it != extra_index_.end()) {
        const int id = extra_ids_[it->second];
        auto [lo, hi] = kExtraLimits.count(id) ? kExtraLimits.at(id) : std::make_pair(-180.0, 180.0);
        per["right"].extra[id] = std::clamp(extraFromUrdf(it->second, msg.position[n]), lo, hi);
        continue;
      }
      bool found = false;
      for (auto & [side, names] : arm_names_) {
        auto it = std::find(names.begin(), names.end(), name);
        if (it == names.end()) {continue;}
        const int i = static_cast<int>(it - names.begin());
        const double deg = rad2deg(msg.position[n]);
        const double lo = kArmLimits[i][0], hi = kArmLimits[i][1];
        if (deg < lo || deg > hi) {
          RCLCPP_WARN(get_logger(), "%s: %.1f deg clamped to [%.0f, %.0f]", name.c_str(), deg, lo, hi);
        }
        per[side].arm[i] = std::clamp(deg, lo, hi);
        found = true;
        break;
      }
      if (!found) {RCLCPP_WARN(get_logger(), "JointCommand: unknown joint '%s'", name.c_str());}
    }
    for (auto & [side, m] : per) {
      if (m.arm.empty() && m.extra.empty()) {continue;}
      if (!m.arm.empty() && !workers_[side]->snapshot().powered) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "%s arm is not powered; call /%s_arm/power",
            side.c_str(), side.c_str());
      }
      workers_[side]->setTarget(m, msg.queue);
    }
  }

  void onCartCmd(const mercury_x1_msgs::msg::CartesianCommand & msg)
  {
    std::string side = msg.arm;
    std::transform(side.begin(), side.end(), side.begin(), ::tolower);
    if (auto p = side.find("_arm"); p != std::string::npos) {side.erase(p);}
    if (!workers_.count(side)) {
      RCLCPP_ERROR(get_logger(), "CartesianCommand: unknown arm '%s'", msg.arm.c_str());
      return;
    }
    Motion m{Motion::kCartesian, speed(msg.speed), {}, {}, {}, false, !msg.joint_interpolated};
    std::copy(msg.coords.begin(), msg.coords.end(), m.coords.begin());
    m.body_frame = msg.frame == mercury_x1_msgs::msg::CartesianCommand::FRAME_BODY;
    workers_[side]->setTarget(m, msg.queue);
  }

  void srvPower(const std::string & side, bool on, std_srvs::srv::SetBool::Response & res)
  {
    auto & w = *workers_[side];
    auto ok = w.call<bool>([on](arm::ArmLink & l) {return l.power(on);}, 15s);
    auto pw = w.call<std::optional<bool>>([](arm::ArmLink & l) {return l.isPowerOn();}, 2s);
    const bool powered = pw && *pw && **pw;
    w.markPowered(powered);
    res.success = ok.value_or(false) && powered == on;
    res.message = side + " arm powered=" + (powered ? "true" : "false");
  }

  void srvStop(const std::vector<std::string> & sides, std_srvs::srv::Trigger::Response & res)
  {
    res.success = true;
    for (auto & s : sides) {
      auto & w = *workers_[s];
      w.clearTarget();
      auto ok = w.call<bool>([](arm::ArmLink & l) {return l.stop();}, 3s);
      res.success = res.success && ok.value_or(false);
      res.message += s + (ok.value_or(false) ? ": stopped " : ": no ack ");
    }
  }

  void publish()
  {
    const auto stamp = now();
    sensor_msgs::msg::JointState js;
    js.header.stamp = stamp;
    ArmSnapshot right_snap;
    for (auto & [side, w] : workers_) {
      const ArmSnapshot s = w->snapshot();
      if (side == "right") {right_snap = s;}
      mercury_x1_msgs::msg::ArmState st;
      st.header.stamp = stamp;
      st.header.frame_id = side + "_arm_base";
      st.arm = side;
      st.connected = s.connected;
      st.powered = s.powered;
      st.moving = s.moving;
      for (int i = 0; i < 7; ++i) {
        st.position[i] = deg2rad(s.q_deg[i]);
        st.velocity[i] = s.qd[i];
      }
      std::copy(s.coords.begin(), s.coords.end(), st.coords.begin());
      std::copy(s.body.begin(), s.body.end(), st.body_coords.begin());
      st.status_raw = s.status;
      st.state_rate_hz = static_cast<float>(s.rate_hz);
      pub_arm_[side]->publish(st);
      if (!s.have_q) {continue;}  // never publish made-up arm joint positions
      for (int i = 0; i < 7; ++i) {
        js.name.push_back(arm_names_[side][i]);
        js.position.push_back(st.position[i]);
        js.velocity.push_back(st.velocity[i]);
      }
    }
    for (size_t i = 0; i < extra_names_.size(); ++i) {
      const double deg = right_snap.extra.count(extra_ids_[i]) ? right_snap.extra.at(extra_ids_[i]) : NAN;
      if (std::isnan(deg)) {continue;}
      js.name.push_back(extra_names_[i]);
      js.position.push_back(extraToUrdf(i, deg));
      js.velocity.push_back(0.0);
    }
    if (!js.name.empty()) {pub_js_->publish(js);}
  }

  int64_t default_speed_, max_speed_;
  std::vector<std::string> extra_names_;
  std::vector<int> extra_ids_;
  std::vector<double> extra_sign_, extra_offset_;
  std::map<std::string, size_t> extra_index_;
  std::map<std::string, std::vector<std::string>> arm_names_;
  std::map<std::string, std::unique_ptr<ArmWorker>> workers_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_js_;
  std::map<std::string, rclcpp::Publisher<mercury_x1_msgs::msg::ArmState>::SharedPtr> pub_arm_;
  rclcpp::Subscription<mercury_x1_msgs::msg::JointCommand>::SharedPtr sub_joint_;
  rclcpp::Subscription<mercury_x1_msgs::msg::CartesianCommand>::SharedPtr sub_cart_;
  rclcpp::CallbackGroup::SharedPtr cb_;
  std::vector<rclcpp::ServiceBase::SharedPtr> srv_power_, srv_stop_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<UpperBodyNode>();
  rclcpp::executors::MultiThreadedExecutor ex(rclcpp::ExecutorOptions(), 4);
  ex.add_node(node);
  ex.spin();
  node.reset();
  rclcpp::shutdown();
  return 0;
}

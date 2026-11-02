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
// Actions    {left_arm,right_arm,waist}_controller/follow_joint_trajectory (control_msgs/FollowJointTrajectory)
//            for MoveIt: the trajectory is thinned to waypoints ~trajectory_step_deg apart (straight joint-space
//            moves between them stay close to the planned, collision-checked path), each segment's firmware
//            speed is taken from the trajectory timing, and all waypoints go to the firmware motion queue at
//            once (the first one preempts). Done when joint_states reach the last point.
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

#include "control_msgs/action/follow_joint_trajectory.hpp"
#include "mercury_x1_driver/arm_protocol.hpp"
#include "mercury_x1_msgs/msg/arm_state.hpp"
#include "mercury_x1_msgs/msg/cartesian_command.hpp"
#include "mercury_x1_msgs/msg/joint_command.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
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
  bool connected{false}, powered{false}, moving{false}, have_q{false}, freedrive{false};
  arm::Joints q_deg{};
  std::array<double, 7> qd{};
  arm::Coords coords{};
  arm::Coords body{};
  arm::Bytes status;
  std::map<int, double> extra;
  int fw{-1}, atom{-1}, board{-1};   // version reads (raw: fw 11 = 1.1)
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

  // returns false (and drops the motion) while the arm is in freedrive
  bool setTarget(Motion m, bool queued)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    const bool moves_arm = m.kind == Motion::kCartesian || !m.arm.empty();
    if (snap_.freedrive && moves_arm) {return false;}
    if (queued) {
      jobs_.push_back([this, m](arm::ArmLink & l) {execute(l, m, false);});
    } else {
      target_ = std::move(m);
    }
    return true;
  }

  // Freedrive on/off on the worker thread. Stops and clears any motion first.
  bool setFreedrive(bool on, bool drag_teach)
  {
    if (on && !snapshot().powered) {return false;}
    {
      std::lock_guard<std::mutex> lk(mtx_);
      target_.reset();
      if (on) {snap_.freedrive = true;}   // block new targets right away
    }
    auto ok = call<bool>([on, drag_teach](arm::ArmLink & l) {
          if (on) {l.stop();}
          return l.freedrive(on, drag_teach);
        }, 3s);
    std::lock_guard<std::mutex> lk(mtx_);
    snap_.freedrive = on;
    return ok.value_or(false);
  }

  // A trajectory: replaces the pending target; the moves go to the firmware queue back to back,
  // the first one preempting what the arm is doing. false while in freedrive.
  bool runSequence(const std::vector<Motion> & ms)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    bool moves_arm = false;
    for (auto & m : ms) {moves_arm = moves_arm || !m.arm.empty();}
    if (snap_.freedrive && moves_arm) {return false;}
    target_.reset();
    for (size_t i = 0; i < ms.size(); ++i) {
      jobs_.push_back([this, m = ms[i], first = i == 0](arm::ArmLink & l) {execute(l, m, first);});
    }
    return true;
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
    if (!p) {snap_.freedrive = false;}   // power on again = joints hold
  }

private:
  template<class F>
  void update(F && f)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    f(snap_);
  }

  // firmware / end board / board versions: on every (re)open, and every 30 s while one is missing
  void readVersions(arm::ArmLink & link)
  {
    auto fw = link.version(arm::op::kGetFirmwareVersion);
    auto atom = link.version(arm::op::kGetAtomVersion);
    auto board = link.version(arm::op::kGetBoardVersion);
    update([&](ArmSnapshot & s) {
      s.fw = fw.value_or(s.fw); s.atom = atom.value_or(s.atom); s.board = board.value_or(s.board);
    });
    if (fw) {
      RCLCPP_INFO(log_, "%s arm: firmware %.1f, end board %.1f, board %d", side_.c_str(), *fw / 10.0,
                  atom.value_or(-10) / 10.0, board.value_or(-1));
    }
  }

  void run()
  {
    std::unique_ptr<arm::ArmLink> link;
    double last_version_try = 0;
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
          readVersions(*link);
          last_version_try = steadyNow();
        } catch (const std::exception & e) {
          RCLCPP_ERROR(log_, "%s arm: %s", side_.c_str(), e.what());
          link.reset();
          std::this_thread::sleep_for(2s);
          continue;
        }
      }
      try {
        if (snapshot().fw < 0 && steadyNow() - last_version_try > 30.0) {
          readVersions(*link);
          last_version_try = steadyNow();
        }
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

using FJT = control_msgs::action::FollowJointTrajectory;
using FJTGoal = rclcpp_action::ServerGoalHandle<FJT>;

// one FollowJointTrajectory server: an arm (7 joints) or the waist (firmware "extra" joint on the right MCU)
struct TrajController
{
  std::string name, worker;
  std::vector<std::string> joints;     // URDF names, controller order
  bool extra{false};
  std::mutex mtx;
  std::shared_ptr<FJTGoal> active;
  uint64_t gen{0};
  rclcpp_action::Server<FJT>::SharedPtr server;
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
    // drag_teach: vendor drag-teaching mode (gravity-compensated); release: torque off, damping
    freedrive_drag_teach_ = declare_parameter("freedrive_mode", std::string("drag_teach")) != "release";
    max_speed_ = declare_parameter("max_speed", 100);
    const double rate = declare_parameter("publish_rate", 50.0);
    // neck/jaw/waist: URDF joint name -> firmware id, with sign/offset into the URDF convention
    extra_names_ = declare_parameter("extra_joint_names", std::vector<std::string>{"eye", "head", "body"});
    const auto ids = declare_parameter("extra_joint_ids", std::vector<int64_t>{11, 12, 13});
    extra_sign_ = declare_parameter("extra_joint_signs", std::vector<double>{1.0, 1.0, -1.0});
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
      srv_power_.push_back(create_service<std_srvs::srv::SetBool>(s + "_arm/freedrive",
          [this, s](std_srvs::srv::SetBool::Request::ConstSharedPtr req,
          std_srvs::srv::SetBool::Response::SharedPtr res) {srvFreedrive({s}, req->data, *res);},
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
    srv_power_.push_back(create_service<std_srvs::srv::SetBool>("upper_body/freedrive",
        [this](std_srvs::srv::SetBool::Request::ConstSharedPtr req, std_srvs::srv::SetBool::Response::SharedPtr res) {
          srvFreedrive({"left", "right"}, req->data, *res);
        }, rmw_qos_profile_services_default, cb_));
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate), [this] {publish();});

    // MoveIt trajectory execution (see the header). The firmware speed scale is not documented:
    // speed_deg_s_at_100 = joint speed at firmware speed 100 (too high -> moves slower than planned, safe side).
    traj_step_deg_ = declare_parameter("trajectory_step_deg", 8.0);
    speed_deg_s_at_100_ = declare_parameter("speed_deg_s_at_100", 180.0);
    goal_tol_deg_ = declare_parameter("goal_tolerance_deg", 2.0);
    traj_timeout_scale_ = declare_parameter("trajectory_timeout_scale", 3.0);
    if (declare_parameter("trajectory_controllers", true)) {
      addController("left_arm_controller", "left", arm_names_["left"], false);
      addController("right_arm_controller", "right", arm_names_["right"], false);
      if (extra_index_.count("body")) {addController("waist_controller", "right", {"body"}, true);}
    }
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
      if (!workers_[side]->setTarget(m, msg.queue)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "%s arm is in freedrive: motion ignored", side.c_str());
      }
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
    if (!workers_[side]->setTarget(m, msg.queue)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "%s arm is in freedrive: motion ignored", side.c_str());
    }
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

  void srvFreedrive(const std::vector<std::string> & sides, bool on, std_srvs::srv::SetBool::Response & res)
  {
    res.success = true;
    for (auto & s : sides) {
      const bool ok = workers_[s]->setFreedrive(on, freedrive_drag_teach_);
      res.success = res.success && ok;
      res.message += s + (ok ? (on ? ": free " : ": holding ") : (on ? ": not powered / no answer " : ": no answer "));
      RCLCPP_INFO(get_logger(), "%s arm freedrive %s (%s)%s", s.c_str(), on ? "ON" : "off",
          freedrive_drag_teach_ ? "drag teach" : "release", ok ? "" : " - no answer from the MCU");
    }
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

  // ---- FollowJointTrajectory --------------------------------------------------------------------------
  void addController(const std::string & name, const std::string & worker, const std::vector<std::string> & joints,
                     bool extra)
  {
    auto c = std::make_shared<TrajController>();
    c->name = name; c->worker = worker; c->joints = joints; c->extra = extra;
    controllers_.push_back(c);
    TrajController * cp = c.get();
    c->server = rclcpp_action::create_server<FJT>(this, name + "/follow_joint_trajectory",
        [this, cp](const rclcpp_action::GoalUUID &, std::shared_ptr<const FJT::Goal> g) {return onGoal(*cp, *g);},
        [](std::shared_ptr<FJTGoal>) {return rclcpp_action::CancelResponse::ACCEPT;},
        [this, cp](std::shared_ptr<FJTGoal> gh) {onAccepted(*cp, gh);},
        rcl_action_server_get_default_options(), cb_);
  }

  rclcpp_action::GoalResponse onGoal(TrajController & c, const FJT::Goal & g)
  {
    const auto & tr = g.trajectory;
    for (auto & j : c.joints) {
      if (std::find(tr.joint_names.begin(), tr.joint_names.end(), j) == tr.joint_names.end()) {
        RCLCPP_ERROR(get_logger(), "%s: trajectory lacks joint %s", c.name.c_str(), j.c_str());
        return rclcpp_action::GoalResponse::REJECT;
      }
    }
    const auto s = workers_[c.worker]->snapshot();
    if (tr.points.empty() || !s.connected || !s.powered || s.freedrive || (!c.extra && !s.have_q)) {
      RCLCPP_ERROR(get_logger(), "%s: rejected (%s)", c.name.c_str(), tr.points.empty() ? "no points" :
          !s.connected ? "arm not connected" : !s.powered ? "arm not powered" : s.freedrive ? "freedrive on" :
          "no joint reading yet");
      return rclcpp_action::GoalResponse::REJECT;
    }
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  // controller joint positions now [rad], empty if unknown
  std::vector<double> current(const TrajController & c)
  {
    const auto s = workers_[c.worker]->snapshot();
    std::vector<double> q;
    for (auto & j : c.joints) {
      if (c.extra) {
        const size_t i = extra_index_.at(j);
        const double deg = s.extra.count(extra_ids_[i]) ? s.extra.at(extra_ids_[i]) : NAN;
        if (std::isnan(deg)) {return {};}
        q.push_back(extraToUrdf(i, deg));
      } else {
        if (!s.have_q) {return {};}
        const auto & names = arm_names_.at(c.worker);
        q.push_back(deg2rad(s.q_deg[std::find(names.begin(), names.end(), j) - names.begin()]));
      }
    }
    return q;
  }

  void onAccepted(TrajController & c, std::shared_ptr<FJTGoal> gh)
  {
    uint64_t gen;
    {
      std::lock_guard<std::mutex> lk(c.mtx);
      if (c.active && c.active->is_active()) {
        auto r = std::make_shared<FJT::Result>();
        r->error_code = FJT::Result::INVALID_GOAL;
        r->error_string = "preempted by a new trajectory";
        c.active->abort(r);
      }
      c.active = gh;
      gen = ++c.gen;
    }
    std::thread([this, &c, gh, gen] {runTrajectory(c, gh, gen);}).detach();
  }

  // ends the goal unless a newer one took over the controller
  void finish(TrajController & c, const std::shared_ptr<FJTGoal> & gh, uint64_t gen, int code, const std::string & why,
              bool canceled = false)
  {
    std::lock_guard<std::mutex> lk(c.mtx);
    if (gen != c.gen || !gh->is_active()) {return;}
    auto r = std::make_shared<FJT::Result>();
    r->error_code = code;
    r->error_string = why;
    if (canceled) {gh->canceled(r);} else if (code == FJT::Result::SUCCESSFUL) {gh->succeed(r);} else {gh->abort(r);}
    if (code != FJT::Result::SUCCESSFUL || canceled) {
      RCLCPP_WARN(get_logger(), "%s: %s", c.name.c_str(), why.c_str());
    }
  }

  void stopWorker(const std::string & side)
  {
    auto & w = *workers_[side];
    w.clearTarget();
    w.call<bool>([](arm::ArmLink & l) {return l.stop();}, 3s);
  }

  void runTrajectory(TrajController & c, std::shared_ptr<FJTGoal> gh, uint64_t gen)
  {
    const auto & tr = gh->get_goal()->trajectory;
    std::vector<size_t> col;      // controller joint -> trajectory column
    for (auto & j : c.joints) {
      col.push_back(std::find(tr.joint_names.begin(), tr.joint_names.end(), j) - tr.joint_names.begin());
    }
    std::vector<double> prev = current(c);
    if (prev.empty()) {finish(c, gh, gen, FJT::Result::INVALID_GOAL, "no joint reading"); return;}
    // 1) thin to waypoints >= step apart (largest joint change), always keeping the last point
    const double step = deg2rad(traj_step_deg_);
    struct Wp {double t; std::vector<double> q;};
    std::vector<Wp> wps;
    std::vector<double> kept = prev;
    for (size_t i = 0; i < tr.points.size(); ++i) {
      const auto & p = tr.points[i];
      std::vector<double> q;
      for (size_t k : col) {q.push_back(p.positions.at(k));}
      double d = 0;
      for (size_t k = 0; k < q.size(); ++k) {d = std::max(d, std::abs(q[k] - kept[k]));}
      if (d >= step || (i + 1 == tr.points.size() && d > 1e-4)) {
        wps.push_back({rclcpp::Duration(p.time_from_start).seconds(), q});
        kept = q;
      }
    }
    if (wps.empty()) {finish(c, gh, gen, FJT::Result::SUCCESSFUL, "already there"); return;}
    // 2) firmware moves, speed from each segment's timing
    std::vector<Motion> ms;
    double t_prev = 0;
    for (auto & w : wps) {
      double d_deg = 0;
      for (size_t k = 0; k < w.q.size(); ++k) {d_deg = std::max(d_deg, rad2deg(std::abs(w.q[k] - prev[k])));}
      const double dt = std::max(w.t - t_prev, 0.05);
      Motion m{Motion::kJoints, std::clamp(static_cast<int>(std::ceil(100.0 * d_deg / dt / speed_deg_s_at_100_)), 1,
          static_cast<int>(max_speed_)), {}, {}, {}, false, true};
      for (size_t k = 0; k < c.joints.size(); ++k) {
        if (c.extra) {
          const size_t i = extra_index_.at(c.joints[k]);
          const int id = extra_ids_[i];
          auto [lo, hi] = kExtraLimits.count(id) ? kExtraLimits.at(id) : std::make_pair(-180.0, 180.0);
          m.extra[id] = std::clamp(extraFromUrdf(i, w.q[k]), lo, hi);
        } else {
          m.arm[static_cast<int>(k)] = std::clamp(rad2deg(w.q[k]), kArmLimits[k][0], kArmLimits[k][1]);
        }
      }
      ms.push_back(m);
      t_prev = w.t;
      prev = w.q;
    }
    if (!workers_[c.worker]->runSequence(ms)) {
      finish(c, gh, gen, FJT::Result::INVALID_GOAL, "arm is in freedrive");
      return;
    }
    RCLCPP_INFO(get_logger(), "%s: %zu points -> %zu moves over %.1f s", c.name.c_str(), tr.points.size(), ms.size(),
        wps.back().t);
    // 3) follow until the last point is reached
    const double t0 = steadyNow(), t_end = wps.back().t, deadline = t0 + t_end * traj_timeout_scale_ + 5.0;
    const double tol = deg2rad(goal_tol_deg_);
    auto fb = std::make_shared<FJT::Feedback>();
    fb->joint_names = c.joints;
    while (rclcpp::ok()) {
      std::this_thread::sleep_for(50ms);
      {
        std::lock_guard<std::mutex> lk(c.mtx);
        if (gen != c.gen || !gh->is_active()) {return;}      // a newer trajectory took over (it preempts)
      }
      if (gh->is_canceling()) {
        stopWorker(c.worker);
        finish(c, gh, gen, FJT::Result::SUCCESSFUL, "canceled: arm stopped", true);
        return;
      }
      const auto q = current(c);
      if (q.empty()) {continue;}
      double err = 0;
      for (size_t k = 0; k < q.size(); ++k) {err = std::max(err, std::abs(q[k] - wps.back().q[k]));}
      fb->actual.positions = q;
      fb->desired.positions = wps.back().q;
      gh->publish_feedback(fb);
      if (err <= tol) {finish(c, gh, gen, FJT::Result::SUCCESSFUL, ""); return;}
      if (steadyNow() > deadline) {
        stopWorker(c.worker);
        finish(c, gh, gen, FJT::Result::GOAL_TOLERANCE_VIOLATED,
               "not at the goal after " + std::to_string(static_cast<int>(steadyNow() - t0)) + " s (off by " +
               std::to_string(rad2deg(err)) + " deg): stopped");
        return;
      }
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
      st.freedrive = s.freedrive;
      for (int i = 0; i < 7; ++i) {
        st.position[i] = deg2rad(s.q_deg[i]);
        st.velocity[i] = s.qd[i];
      }
      std::copy(s.coords.begin(), s.coords.end(), st.coords.begin());
      std::copy(s.body.begin(), s.body.end(), st.body_coords.begin());
      st.status_raw = s.status;
      st.state_rate_hz = static_cast<float>(s.rate_hz);
      auto ver = [](int v) {return v < 0 ? std::string() : std::to_string(v / 10) + "." + std::to_string(v % 10);};
      st.firmware_version = ver(s.fw);
      st.atom_version = ver(s.atom);
      st.board_version = static_cast<int16_t>(s.board);
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
  double traj_step_deg_{8}, speed_deg_s_at_100_{180}, goal_tol_deg_{2}, traj_timeout_scale_{3};
  std::vector<std::shared_ptr<TrajController>> controllers_;
  bool freedrive_drag_teach_{true};
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

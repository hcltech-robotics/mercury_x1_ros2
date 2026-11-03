// Mercury X1 base "goalkeeper": owns /dev/wheeltec_controller.
//
// Publishes  odom (+ TF odom->base_footprint), imu/data_raw, battery,
//            ultrasonic/<name> (sensor_msgs/Range), base/state
// Subscribes cmd_vel (geometry_msgs/Twist), base/led (std_msgs/ColorRGBA, 0..1)
//
// Safety: velocity limits, a cmd_vel watchdog (the MCU on its own only stops after ~1 s),
// and an optional ultrasonic guard that blocks reversing when the rear sensor sees something close
// (the rangers sit on the sides and the back; the lidar covers the front).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <limits>
#include <string>
#include <mutex>
#include <thread>
#include <vector>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "mercury_x1_driver/base_protocol.hpp"
#include "mercury_x1_driver/serial_port.hpp"
#include "mercury_x1_msgs/msg/base_state.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/battery_state.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/range.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "tf2_ros/transform_broadcaster.h"

using namespace std::chrono_literals;
using mercury_x1::SerialPort;
namespace proto = mercury_x1::base;

class BaseNode : public rclcpp::Node
{
public:
  BaseNode()
  : Node("mercury_base")
  {
    port_ = declare_parameter("port", "/dev/wheeltec_controller");
    baud_ = declare_parameter("baudrate", 115200);
    odom_frame_ = declare_parameter("odom_frame", "odom");
    base_frame_ = declare_parameter("base_frame", "base_footprint");
    imu_frame_ = declare_parameter("imu_frame", "imu_link");
    publish_tf_ = declare_parameter("publish_tf", true);
    cmd_timeout_ = declare_parameter("cmd_timeout", 0.5);
    max_lin_ = declare_parameter("max_linear", 0.5);
    max_ang_ = declare_parameter("max_angular", 1.0);
    lin_scale_ = declare_parameter("linear_scale", 1.0);
    ang_scale_ = declare_parameter("angular_scale", 1.0);
    yaw_from_gyro_ = declare_parameter("yaw_source", "gyro") == "gyro";
    // MCU channel -> mounting position (verified on the robot: ch0 back, ch1 right, ch2 left)
    us_names_ = declare_parameter("ultrasonic_names",
        std::vector<std::string>{"back", "right", "left"});
    us_max_ = declare_parameter("ultrasonic_max_range", 2.9);
    us_min_ = declare_parameter("ultrasonic_min_range", 0.03);
    us_fov_ = declare_parameter("ultrasonic_fov", 0.5);
    guard_dist_ = declare_parameter("ultrasonic_guard_distance", 0.0);
    // reversing guard: by sensor name, so a channel remap cannot silently guard the wrong side
    for (const auto & g : declare_parameter("ultrasonic_guard_names", std::vector<std::string>{"back"})) {
      const auto it = std::find(us_names_.begin(), us_names_.end(), g);
      if (it == us_names_.end()) {
        RCLCPP_WARN(get_logger(), "ultrasonic_guard_names: no sensor named '%s'", g.c_str());
      } else {
        guard_ch_.push_back(it - us_names_.begin());
      }
    }
    // Mercury X1 pack: 7S LiFePO4 (rests at 23.7 V when full = 3.38 V/cell; factory test read 20.8 V)
    battery_cells_ = declare_parameter("battery_cells", 7);
    battery_lfp_ = declare_parameter("battery_chemistry", std::string("lifepo4")) != "li-ion";
    // per-cell voltage only a charger produces (LiFePO4 rests <= ~3.5 V/cell even freshly charged)
    charge_cell_v_ = declare_parameter("charging_cell_voltage", battery_lfp_ ? 3.55 : 4.25);
    battery_log_ = declare_parameter("battery_log", std::string("~/.ros/mercury_battery.csv"));
    if (battery_log_.rfind("~/", 0) == 0 && std::getenv("HOME")) {battery_log_ = std::getenv("HOME") + battery_log_.substr(1);}

    pub_odom_ = create_publisher<nav_msgs::msg::Odometry>("odom", 10);
    pub_imu_ = create_publisher<sensor_msgs::msg::Imu>("imu/data_raw", rclcpp::SensorDataQoS());
    pub_batt_ = create_publisher<sensor_msgs::msg::BatteryState>("battery", 10);
    pub_state_ = create_publisher<mercury_x1_msgs::msg::BaseState>("base/state", 10);
    for (auto & n : us_names_) {
      pub_us_.push_back(create_publisher<sensor_msgs::msg::Range>("ultrasonic/" + n, rclcpp::SensorDataQoS()));
    }
    if (publish_tf_) {tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);}

    sub_cmd_ = create_subscription<geometry_msgs::msg::Twist>("cmd_vel", 10,
        [this](geometry_msgs::msg::Twist::ConstSharedPtr m) {
          std::lock_guard<std::mutex> lk(cmd_mtx_);
          cmd_ = {m->linear.x, m->linear.y, m->angular.z};
          cmd_stamp_ = now_s();
        });
    sub_led_ = create_subscription<std_msgs::msg::ColorRGBA>("base/led", 10,
        [this](std_msgs::msg::ColorRGBA::ConstSharedPtr m) {
          const double a = m->a > 0 ? m->a : 1.0;
          send(proto::ledFrame(int(m->r * a * 255), int(m->g * a * 255), int(m->b * a * 255)));
        });

    us_m_.fill(std::numeric_limits<double>::infinity());
    openPort();
    rx_thread_ = std::thread([this] {rxLoop();});
    tx_timer_ = create_wall_timer(50ms, [this] {txTick();});   // MCU needs a frame < 1 s apart
    slow_timer_ = create_wall_timer(1s, [this] {slowTick();});
    RCLCPP_INFO(get_logger(), "base driver on %s (yaw from %s)", port_.c_str(), yaw_from_gyro_ ? "gyro" : "wheels");
  }

  ~BaseNode() override
  {
    running_ = false;
    try {send(proto::velocityFrame(0, 0, 0));} catch (...) {}
    if (rx_thread_.joinable()) {rx_thread_.join();}
  }

private:
  static double now_s()
  {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  void openPort()
  {
    try {
      std::lock_guard<std::mutex> lk(ser_mtx_);
      ser_.open(port_, static_cast<int>(baud_), SerialPort::Lines::kDtrOnRtsOn);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "%s", e.what());
    }
  }

  void send(const proto::Bytes & f)
  {
    std::lock_guard<std::mutex> lk(ser_mtx_);
    if (!ser_.isOpen()) {return;}
    try {
      ser_.write(f);
    } catch (const std::exception & e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "%s", e.what());
    }
  }

  void txTick()
  {
    std::array<double, 3> c{0, 0, 0};
    double age;
    {
      std::lock_guard<std::mutex> lk(cmd_mtx_);
      age = now_s() - cmd_stamp_;
      if (age < cmd_timeout_) {c = cmd_;}
    }
    c[0] = std::clamp(c[0], -max_lin_, max_lin_);
    c[1] = std::clamp(c[1], -max_lin_, max_lin_);
    c[2] = std::clamp(c[2], -max_ang_, max_ang_);
    bool hold = false;
    if (guard_dist_ > 0 && c[0] < 0) {
      double nearest = std::numeric_limits<double>::infinity();
      std::lock_guard<std::mutex> lk(state_mtx_);
      for (auto ch : guard_ch_) {
        if (ch >= 0 && ch < 6) {nearest = std::min(nearest, us_m_[ch]);}
      }
      if (nearest < guard_dist_) {c[0] = 0; hold = true;}
    }
    {
      std::lock_guard<std::mutex> lk(state_mtx_);
      sent_ = c;
      obstacle_hold_ = hold;
      timed_out_ = age >= cmd_timeout_;
    }
    send(proto::velocityFrame(c[0], c[1], c[2]));
  }

  void rxLoop()
  {
    proto::StreamParser parser;
    proto::Bytes chunk;
    while (running_ && rclcpp::ok()) {
      chunk.clear();
      try {
        bool open;
        {
          std::lock_guard<std::mutex> lk(ser_mtx_);
          open = ser_.isOpen();
        }
        if (!open) {
          std::this_thread::sleep_for(1s);
          openPort();
          continue;
        }
        ser_.read(chunk, 50);  // read() and write() on one fd are safe concurrently
      } catch (const std::exception & e) {
        RCLCPP_ERROR(get_logger(), "%s; reopening", e.what());
        {
          std::lock_guard<std::mutex> lk(ser_mtx_);
          ser_.close();
        }
        continue;
      }
      if (chunk.empty()) {continue;}
      for (auto & m : parser.feed(chunk)) {
        last_rx_ = now_s();
        if (auto * s = std::get_if<proto::StateFrame>(&m)) {onState(*s);} else {
          onUltrasonic(std::get<proto::UltrasonicFrame>(m));
        }
      }
    }
  }

  void onState(const proto::StateFrame & f)
  {
    const double t = now_s();
    const double dt = last_state_t_ < 0 ? 0.0 : std::min(t - last_state_t_, 0.2);
    last_state_t_ = t;
    std::array<double, 3> sent;
    {
      std::lock_guard<std::mutex> lk(state_mtx_);
      firmware_ = f.firmware;
      stop_flag_ = f.stop_flag;
      voltage_ = f.voltage;
      sent = sent_;
    }
    const double vx = f.vx * lin_scale_, vy = f.vy * lin_scale_, wz_wheel = f.wz * ang_scale_;
    const double gz = f.gyro[2];
    // learn the gyro bias while commanded still and the wheels report still
    if (sent == std::array<double, 3>{0, 0, 0} && std::abs(f.vx) < 1e-3 && std::abs(f.wz) < 1e-3) {
      bias_n_ = std::min(bias_n_ + 1, 200);
      gyro_bias_ += (gz - gyro_bias_) / bias_n_;
    }
    double wz = yaw_from_gyro_ ? gz - gyro_bias_ : wz_wheel;
    if (yaw_from_gyro_ && std::abs(wz) < 0.004 && std::abs(wz_wheel) < 1e-3) {wz = 0.0;}

    yaw_ = std::atan2(std::sin(yaw_ + wz * dt), std::cos(yaw_ + wz * dt));
    const double c = std::cos(yaw_), s = std::sin(yaw_);
    x_ += (vx * c - vy * s) * dt;
    y_ += (vx * s + vy * c) * dt;

    const auto stamp = now();
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odom_frame_;
    odom.child_frame_id = base_frame_;
    odom.pose.pose.position.x = x_;
    odom.pose.pose.position.y = y_;
    odom.pose.pose.orientation.z = std::sin(yaw_ / 2);
    odom.pose.pose.orientation.w = std::cos(yaw_ / 2);
    odom.twist.twist.linear.x = vx;
    odom.twist.twist.linear.y = vy;
    odom.twist.twist.angular.z = wz;
    const std::pair<int, double> cov[] = {{0, 1e-3}, {7, 1e-3}, {14, 1e6}, {21, 1e6}, {28, 1e6}, {35, 3e-3}};
    for (auto [i, v] : cov) {
      odom.pose.covariance[i] = v;
      odom.twist.covariance[i] = v;
    }
    pub_odom_->publish(odom);
    {
      std::lock_guard<std::mutex> lk(state_mtx_);
      measured_ = odom.twist.twist;
    }

    if (tf_) {
      geometry_msgs::msg::TransformStamped tf;
      tf.header = odom.header;
      tf.child_frame_id = base_frame_;
      tf.transform.translation.x = x_;
      tf.transform.translation.y = y_;
      tf.transform.rotation = odom.pose.pose.orientation;
      tf_->sendTransform(tf);
    }

    sensor_msgs::msg::Imu imu;
    imu.header.stamp = stamp;
    imu.header.frame_id = imu_frame_;
    imu.orientation_covariance[0] = -1.0;   // no orientation estimate
    imu.angular_velocity.x = f.gyro[0];
    imu.angular_velocity.y = f.gyro[1];
    imu.angular_velocity.z = gz - gyro_bias_;
    imu.linear_acceleration.x = f.accel[0];
    imu.linear_acceleration.y = f.accel[1];
    imu.linear_acceleration.z = f.accel[2];
    for (int i : {0, 4, 8}) {
      imu.angular_velocity_covariance[i] = 1e-4;
      imu.linear_acceleration_covariance[i] = 1e-2;
    }
    pub_imu_->publish(imu);
  }

  void onUltrasonic(const proto::UltrasonicFrame & f)
  {
    const auto stamp = now();
    {
      std::lock_guard<std::mutex> lk(state_mtx_);
      for (int i = 0; i < 6; ++i) {us_m_[i] = f.mm[i] / 1000.0;}
      us_aux_ = f.aux;
    }
    for (size_t i = 0; i < pub_us_.size() && i < 6; ++i) {
      sensor_msgs::msg::Range r;
      r.header.stamp = stamp;
      r.header.frame_id = "ultrasonic_" + us_names_[i];
      r.radiation_type = sensor_msgs::msg::Range::ULTRASOUND;
      r.field_of_view = static_cast<float>(us_fov_);
      r.min_range = static_cast<float>(us_min_);
      r.max_range = static_cast<float>(us_max_);
      const double d = f.mm[i] / 1000.0;
      r.range = d >= us_max_ ? std::numeric_limits<float>::infinity() : static_cast<float>(d);
      pub_us_[i]->publish(r);
    }
  }

  // State of charge from a cell's resting-voltage curve (rough under load). LiFePO4 is flat between
  // ~20 and ~90 %, so mid-range values are only good to about +-15 %; the ends are reliable.
  static double interp(const double (*pts)[2], size_t n, double v)
  {
    if (v <= pts[0][0]) {return pts[0][1];}
    for (size_t i = 1; i < n; ++i) {
      if (v <= pts[i][0]) {
        const double t = (v - pts[i - 1][0]) / (pts[i][0] - pts[i - 1][0]);
        return pts[i - 1][1] + t * (pts[i][1] - pts[i - 1][1]);
      }
    }
    return pts[n - 1][1];
  }

  double socFromCellVoltage(double v) const
  {
    static const double lfp[][2] = {{2.80, 0.00}, {3.00, 0.05}, {3.10, 0.10}, {3.20, 0.20}, {3.22, 0.30},
      {3.25, 0.40}, {3.26, 0.50}, {3.28, 0.60}, {3.30, 0.70}, {3.32, 0.80}, {3.33, 0.90}, {3.36, 0.97},
      {3.38, 1.00}};
    static const double lion[][2] = {{3.30, 0.00}, {3.50, 0.05}, {3.60, 0.12}, {3.70, 0.30},
      {3.80, 0.50}, {3.90, 0.66}, {4.00, 0.80}, {4.10, 0.92}, {4.20, 1.00}};
    return battery_lfp_ ? interp(lfp, sizeof(lfp) / sizeof(lfp[0]), v) :
           interp(lion, sizeof(lion) / sizeof(lion[0]), v);
  }

  // The MCU reports no charging state (fw v1.2: nothing in either frame changes), so estimate it from the
  // slow-filtered voltage: charging when it is above what a resting LiFePO4 pack can show (charger in its
  // constant-voltage phase), or when it rose in each of the last 3 minutes (a load drop is a single step,
  // a charger keeps pushing it up). Unplugging shows within ~1 min as the voltage starts relaxing.
  uint8_t chargeStatus(bool connected, double pct)
  {
    using BS = sensor_msgs::msg::BatteryState;
    if (!connected || v_filt_ <= 1.0 || battery_cells_ <= 0) {return BS::POWER_SUPPLY_STATUS_UNKNOWN;}
    if (++trend_tick_ >= 60) {
      trend_tick_ = 0;
      trend_.push_back(v_filt_);
      if (trend_.size() > 4) {trend_.erase(trend_.begin());}
    }
    bool rising = trend_.size() == 4;
    for (size_t i = 1; rising && i < trend_.size(); ++i) {rising = trend_[i] - trend_[i - 1] > 0.005;}
    rising = rising && trend_.back() - trend_.front() >= 0.04;
    if (v_filt_ / static_cast<double>(battery_cells_) >= charge_cell_v_ || rising) {
      return BS::POWER_SUPPLY_STATUS_CHARGING;
    }
    return pct >= 0.99 ? BS::POWER_SUPPLY_STATUS_FULL : BS::POWER_SUPPLY_STATUS_DISCHARGING;
  }

  // one line a minute: time, raw voltage, filtered voltage, percent (to find the LED's low-battery point)
  void logBattery(double v, double pct)
  {
    if (battery_log_.empty() || ++battery_log_tick_ < 60) {return;}
    battery_log_tick_ = 0;
    std::ofstream f(battery_log_, std::ios::app);
    if (!f) {return;}
    const std::time_t t = std::time(nullptr);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    f << ts << "," << v << "," << v_filt_ << "," << pct * 100.0 << "\n";
  }

  void slowTick()
  {
    const bool connected = now_s() - last_rx_ < 0.5;
    mercury_x1_msgs::msg::BaseState st;
    st.header.stamp = now();
    st.connected = connected;
    {
      std::lock_guard<std::mutex> lk(state_mtx_);
      st.firmware_version = firmware_;
      st.stop_flag = stop_flag_;
      st.voltage = static_cast<float>(voltage_);
      for (size_t i = 0; i < us_m_.size(); ++i) {
        st.ultrasonic_range.push_back(static_cast<float>(us_m_[i]));
        st.ultrasonic_names.push_back(i < us_names_.size() ? us_names_[i] : "");
      }
      st.mcu_aux.assign(us_aux_.begin(), us_aux_.end());
      st.measured = measured_;
      st.commanded.linear.x = sent_[0];
      st.commanded.linear.y = sent_[1];
      st.commanded.angular.z = sent_[2];
      st.cmd_timeout = timed_out_;
      st.obstacle_hold = obstacle_hold_;
    }

    sensor_msgs::msg::BatteryState b;
    b.header.stamp = st.header.stamp;
    b.voltage = st.voltage;
    b.present = connected;
    // the pack voltage swings with arm/base load: estimate charge from a ~30 s low-passed voltage
    if (connected && st.voltage > 1.0f) {
      v_filt_ = v_filt_ <= 0 ? st.voltage : v_filt_ + (st.voltage - v_filt_) / 30.0;
    }
    b.percentage = (connected && battery_cells_ > 0 && v_filt_ > 1.0) ?
      static_cast<float>(socFromCellVoltage(v_filt_ / static_cast<double>(battery_cells_))) :
      std::numeric_limits<float>::quiet_NaN();
    b.cell_voltage.assign(static_cast<size_t>(std::max<int64_t>(battery_cells_, 0)),
        battery_cells_ > 0 ? st.voltage / battery_cells_ : 0.0f);
    b.power_supply_status = chargeStatus(connected, std::isfinite(b.percentage) ? b.percentage : 0.0);
    b.power_supply_technology = battery_lfp_ ? sensor_msgs::msg::BatteryState::POWER_SUPPLY_TECHNOLOGY_LIFE :
      sensor_msgs::msg::BatteryState::POWER_SUPPLY_TECHNOLOGY_LION;
    pub_batt_->publish(b);
    st.charge_status = b.power_supply_status;
    pub_state_->publish(st);
    if (connected && std::isfinite(b.percentage)) {logBattery(st.voltage, b.percentage);}
    if (!connected) {RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "no frames from base MCU");}
  }

  // parameters
  std::string port_, odom_frame_, base_frame_, imu_frame_;
  int64_t baud_, battery_cells_;
  bool battery_lfp_{true};
  double charge_cell_v_{3.55};
  std::string battery_log_;
  int battery_log_tick_{59};
  bool publish_tf_, yaw_from_gyro_;
  double cmd_timeout_, max_lin_, max_ang_, lin_scale_, ang_scale_, us_max_, us_min_, us_fov_, guard_dist_;
  std::vector<std::string> us_names_;
  std::vector<int64_t> guard_ch_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_imu_;
  rclcpp::Publisher<sensor_msgs::msg::BatteryState>::SharedPtr pub_batt_;
  rclcpp::Publisher<mercury_x1_msgs::msg::BaseState>::SharedPtr pub_state_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::Range>::SharedPtr> pub_us_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_cmd_;
  rclcpp::Subscription<std_msgs::msg::ColorRGBA>::SharedPtr sub_led_;
  rclcpp::TimerBase::SharedPtr tx_timer_, slow_timer_;

  SerialPort ser_;
  std::mutex ser_mtx_, cmd_mtx_, state_mtx_;
  std::thread rx_thread_;
  std::atomic<bool> running_{true};
  std::atomic<double> last_rx_{0};

  std::array<double, 3> cmd_{0, 0, 0}, sent_{0, 0, 0};
  double cmd_stamp_{0};
  bool obstacle_hold_{false}, timed_out_{true};
  std::string firmware_;
  uint8_t stop_flag_{0};
  double voltage_{0};
  double v_filt_{0};   // slow-filtered voltage for the state-of-charge estimate (1 Hz updates)
  std::array<double, 6> us_m_{};
  std::array<uint16_t, 2> us_aux_{};
  std::vector<double> trend_;   // filtered voltage once a minute, last 4 (charging trend)
  int trend_tick_{59};
  geometry_msgs::msg::Twist measured_;
  // odometry (rx thread only)
  double x_{0}, y_{0}, yaw_{0}, last_state_t_{-1}, gyro_bias_{0};
  int bias_n_{0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<BaseNode>();
  rclcpp::spin(node);
  node.reset();
  rclcpp::shutdown();
  return 0;
}

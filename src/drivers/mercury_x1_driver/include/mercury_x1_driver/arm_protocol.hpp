// Mercury X1 arm-MCU serial protocol (ESP32 boards, firmware 1.1).
//
// Frame: FE FE LEN CMD DATA... CRC_HI CRC_LO, LEN = len(DATA) + 3,
//        CRC = CRC-16/MODBUS over every preceding byte.
//
// Only opcodes verified on the robot are listed. pymycobot's ProtocolCode table has
// duplicate numbers; probing unverified "get" opcodes powered an arm off once.
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "mercury_x1_driver/serial_port.hpp"

namespace mercury_x1::arm
{

namespace op
{
constexpr uint8_t kPowerOn = 0x10;          // data 0x01; ack ff 01 after ~2.5 s
constexpr uint8_t kPowerOff = 0x11;
constexpr uint8_t kIsPowerOn = 0x12;
constexpr uint8_t kGetAngles = 0x20;        // 7 x int32 millideg (silent while unpowered)
constexpr uint8_t kSendAngle = 0x21;        // id, int16 deg*100, speed
constexpr uint8_t kSendAngles = 0x22;       // 7 x int16 deg*100, speed
constexpr uint8_t kGetCoords = 0x23;
constexpr uint8_t kSendCoords = 0x25;       // 3 x int16 mm*10, 3 x int16 deg*100, speed
constexpr uint8_t kStop = 0x29;             // no data = hard stop; next target runs immediately
constexpr uint8_t kIsMoving = 0x2B;
constexpr uint8_t kGetJoint = 0x1C;         // id 1..7 / 11 jaw / 12 neck / 13 waist -> int32 deg*100
constexpr uint8_t kInPositionEvent = 0x5B;  // unsolicited when a motion completes
constexpr uint8_t kSetMovementType = 0x87;  // 0 moveJ, 1 moveL
constexpr uint8_t kGetMovementType = 0x88;
constexpr uint8_t kRobotStatus = 0xA2;      // all zero = healthy
constexpr uint8_t kGetAnglesCoords = 0xD5;  // 7 x int32 deg*100 + 6 x int16 coords
constexpr uint8_t kGetBaseCoords = 0xF0;    // torso-frame coords
constexpr uint8_t kSendBaseCoords = 0xF2;

// Freedrive (move the arm by hand). From Elephant's Mercury API docs (pymycobot close_loop), not
// probed: drag teaching = gravity-compensated drag while "recording" (vendor: <= 90 s per recording);
// release = joints torque-off in damping mode (the arm sinks under gravity), focus = torque on again.
constexpr uint8_t kReleaseAllServos = 0x13;
constexpr uint8_t kFocusAllServos = 0x18;
constexpr uint8_t kDragTeachSave = 0x70;    // start recording -> arm can be dragged
constexpr uint8_t kDragTeachPause = 0x72;   // stop recording -> joints hold again
}  // namespace op

using Bytes = std::vector<uint8_t>;
using Joints = std::array<double, 7>;   // degrees
using Coords = std::array<double, 6>;   // mm, mm, mm, deg, deg, deg

uint16_t crc16(const uint8_t * data, size_t n);
Bytes encode(uint8_t cmd, const Bytes & data = {});

Bytes encodeAngles(const Joints & deg, int speed);
Bytes encodeAngle(int joint_id, double deg, int speed);
Bytes encodeCoords(const Coords & c, int speed);

std::optional<Coords> decodeCoords(const Bytes & p);
std::optional<Joints> decodeAngles(const Bytes & p);
std::optional<std::pair<Joints, Coords>> decodeAnglesCoords(const Bytes & p);
std::optional<double> decodeJoint(const Bytes & p);

struct Frame
{
  uint8_t cmd;
  Bytes payload;
};

class FrameParser
{
public:
  std::vector<Frame> feed(const Bytes & chunk);
  size_t crcErrors() const {return crc_errors_;}

private:
  Bytes buf_;
  size_t crc_errors_{0};
};

// Request/response over one arm port. Not thread-safe: use from a single owner thread.
class ArmLink
{
public:
  explicit ArmLink(const std::string & port, int baud = 115200);

  std::optional<Bytes> request(uint8_t cmd, const Bytes & data = {},
                               std::chrono::milliseconds timeout = std::chrono::milliseconds(250));

  std::optional<bool> isPowerOn();
  bool power(bool on);
  std::optional<std::pair<Joints, Coords>> anglesCoords();
  std::optional<Coords> bodyCoords();
  std::optional<double> joint(int id);
  std::optional<bool> isMoving();
  std::optional<Bytes> status();
  bool stop();
  bool setMovementType(bool linear);
  bool sendAngles(const Joints & deg, int speed);
  bool sendAngle(int id, double deg, int speed);
  bool sendCoords(const Coords & c, int speed, bool body_frame);
  // drag_teach: kDragTeachSave / kDragTeachPause; otherwise release / focus all servos.
  // Returns whether the MCU answered (both are fire-and-forget in pymycobot).
  bool freedrive(bool on, bool drag_teach);

  double lastRttMs() const {return last_rtt_ms_;}
  size_t timeouts() const {return timeouts_;}
  std::function<void()> on_in_position;

private:
  SerialPort ser_;
  FrameParser parser_;
  double last_rtt_ms_{0};
  size_t timeouts_{0};
};

}  // namespace mercury_x1::arm

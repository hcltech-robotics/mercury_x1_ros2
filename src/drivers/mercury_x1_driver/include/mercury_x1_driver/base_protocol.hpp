// Mercury X1 base MCU (Wheeltec-style STM32F407 board) serial protocol.
//
// MCU -> host at 20 Hz, two frames back to back:
//   state      26 B: 7B flag vx vy wz ax ay az gx gy gz mV fw_major fw_minor XOR 7D
//                    (int16 big-endian; mm/s, mrad/s)
//   ultrasonic 19 B: FA d0..d5 u6 u7 XOR FC  (uint16 BE, mm; 3000 = no echo / not fitted)
// host -> MCU, 11 B: 7B mode b2 b3..b8 XOR 7D
//   mode 0: velocity, b3..b8 = vx vy wz int16 (mm/s, mrad/s). The MCU ramps acceleration and
//           stops on its own ~1 s after the last frame.
//   mode 4: RGB light strip, b2 = on/off, b3..b5 = r g b.
// XOR = xor of every preceding byte.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace mercury_x1::base
{

using Bytes = std::vector<uint8_t>;

constexpr size_t kStateLen = 26;
constexpr size_t kUltraLen = 19;
constexpr double kGyroRadPerLsb = 1.0 / 65.5 / 57.29578;  // +-500 dps
constexpr double kAccelPerLsb = 1.0 / 1671.84;             // +-2 g -> m/s^2

struct StateFrame
{
  uint8_t stop_flag;
  double vx, vy, wz;          // m/s, rad/s
  std::array<double, 3> accel;
  std::array<double, 3> gyro;
  double voltage;
  std::string firmware;
};

struct UltrasonicFrame
{
  std::array<uint16_t, 6> mm;
};

using Message = std::variant<StateFrame, UltrasonicFrame>;

uint8_t xorSum(const uint8_t * d, size_t n);

class StreamParser
{
public:
  std::vector<Message> feed(const Bytes & chunk);
  size_t badFrames() const {return bad_;}

private:
  Bytes buf_;
  size_t bad_{0};
};

Bytes velocityFrame(double vx, double vy, double wz);
Bytes ledFrame(int r, int g, int b);

}  // namespace mercury_x1::base

#include "mercury_x1_driver/arm_protocol.hpp"

#include <algorithm>
#include <cmath>

namespace mercury_x1::arm
{

namespace
{
const Bytes kAck{0xFF, 0x01};

void putI16(Bytes & b, double v)
{
  long x = std::lround(v);
  x = std::clamp<long>(x, -32768, 32767);
  auto u = static_cast<uint16_t>(static_cast<int16_t>(x));
  b.push_back(static_cast<uint8_t>(u >> 8));
  b.push_back(static_cast<uint8_t>(u & 0xFF));
}

int16_t getI16(const uint8_t * p) {return static_cast<int16_t>((p[0] << 8) | p[1]);}

int32_t getI32(const uint8_t * p)
{
  return static_cast<int32_t>((uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
         (uint32_t(p[2]) << 8) | uint32_t(p[3]));
}

uint8_t speedByte(int speed) {return static_cast<uint8_t>(std::clamp(speed, 1, 100));}
}  // namespace

uint16_t crc16(const uint8_t * data, size_t n)
{
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < n; ++i) {
    crc ^= data[i];
    for (int k = 0; k < 8; ++k) {
      crc = (crc & 1) ? static_cast<uint16_t>((crc >> 1) ^ 0xA001) : static_cast<uint16_t>(crc >> 1);
    }
  }
  return crc;
}

Bytes encode(uint8_t cmd, const Bytes & data)
{
  Bytes f;
  f.reserve(data.size() + 6);
  for (uint8_t b : {uint8_t(0xFE), uint8_t(0xFE), static_cast<uint8_t>(data.size() + 3), cmd}) {f.push_back(b);}
  for (uint8_t b : data) {f.push_back(b);}
  uint16_t c = crc16(f.data(), f.size());
  f.push_back(static_cast<uint8_t>(c >> 8));
  f.push_back(static_cast<uint8_t>(c & 0xFF));
  return f;
}

Bytes encodeAngles(const Joints & deg, int speed)
{
  Bytes b;
  for (double a : deg) {putI16(b, a * 100.0);}
  b.push_back(speedByte(speed));
  return b;
}

Bytes encodeAngle(int joint_id, double deg, int speed)
{
  Bytes b{static_cast<uint8_t>(joint_id)};
  putI16(b, deg * 100.0);
  b.push_back(speedByte(speed));
  return b;
}

Bytes encodeCoords(const Coords & c, int speed)
{
  Bytes b;
  for (int i = 0; i < 3; ++i) {putI16(b, c[i] * 10.0);}
  for (int i = 3; i < 6; ++i) {putI16(b, c[i] * 100.0);}
  b.push_back(speedByte(speed));
  return b;
}

std::optional<Coords> decodeCoords(const Bytes & p)
{
  if (p.size() != 12) {return std::nullopt;}
  Coords c{};
  for (int i = 0; i < 6; ++i) {c[i] = getI16(&p[2 * i]) / (i < 3 ? 10.0 : 100.0);}
  return c;
}

std::optional<Joints> decodeAngles(const Bytes & p)
{
  if (p.size() != 28) {return std::nullopt;}
  Joints j{};
  for (int i = 0; i < 7; ++i) {j[i] = getI32(&p[4 * i]) / 1000.0;}
  return j;
}

std::optional<std::pair<Joints, Coords>> decodeAnglesCoords(const Bytes & p)
{
  if (p.size() != 40) {return std::nullopt;}
  Joints j{};
  for (int i = 0; i < 7; ++i) {j[i] = getI32(&p[4 * i]) / 100.0;}
  auto c = decodeCoords(Bytes(p.begin() + 28, p.end()));
  return std::make_pair(j, *c);
}

std::optional<double> decodeJoint(const Bytes & p)
{
  if (p.size() != 4) {return std::nullopt;}
  return getI32(p.data()) / 100.0;
}

std::vector<Frame> FrameParser::feed(const Bytes & chunk)
{
  buf_.insert(buf_.end(), chunk.begin(), chunk.end());
  std::vector<Frame> out;
  size_t i = 0;
  while (true) {
    // find header
    while (i + 1 < buf_.size() && !(buf_[i] == 0xFE && buf_[i + 1] == 0xFE)) {++i;}
    if (i + 4 > buf_.size()) {break;}
    size_t n = buf_[i + 2] + 3u;
    if (n < 6) {i += 2; continue;}
    if (i + n > buf_.size()) {break;}
    const uint8_t * f = &buf_[i];
    uint16_t want = static_cast<uint16_t>((f[n - 2] << 8) | f[n - 1]);
    if (crc16(f, n - 2) == want) {
      out.push_back(Frame{f[3], Bytes(f + 4, f + n - 2)});
      i += n;
    } else {
      ++crc_errors_;
      i += 2;
    }
  }
  buf_.erase(buf_.begin(), buf_.begin() + static_cast<long>(std::min(i, buf_.size())));
  return out;
}

ArmLink::ArmLink(const std::string & port, int baud)
{
  ser_.open(port, baud, SerialPort::Lines::kDtrOnRtsOff);
}

std::optional<Bytes> ArmLink::request(uint8_t cmd, const Bytes & data, std::chrono::milliseconds timeout)
{
  using clock = std::chrono::steady_clock;
  ser_.write(encode(cmd, data));
  const auto t0 = clock::now();
  Bytes chunk;
  while (true) {
    auto left = std::chrono::duration_cast<std::chrono::milliseconds>(timeout - (clock::now() - t0));
    if (left.count() <= 0) {break;}
    chunk.clear();
    ser_.read(chunk, static_cast<int>(std::min<long>(left.count(), 20)));
    if (chunk.empty()) {continue;}
    for (auto & fr : parser_.feed(chunk)) {
      if (fr.cmd == cmd) {
        last_rtt_ms_ = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
        return std::move(fr.payload);
      }
      if (fr.cmd == op::kInPositionEvent && on_in_position) {on_in_position();}
    }
  }
  ++timeouts_;
  return std::nullopt;
}

std::optional<bool> ArmLink::isPowerOn()
{
  auto r = request(op::kIsPowerOn);
  if (!r || r->empty()) {return std::nullopt;}
  return (*r)[0] == 1;
}

bool ArmLink::power(bool on)
{
  if (on) {return request(op::kPowerOn, {0x01}, std::chrono::seconds(10)) == kAck;}
  return request(op::kPowerOff, {}, std::chrono::seconds(5)).has_value();
}

std::optional<std::pair<Joints, Coords>> ArmLink::anglesCoords()
{
  auto r = request(op::kGetAnglesCoords);
  return r ? decodeAnglesCoords(*r) : std::nullopt;
}

std::optional<Coords> ArmLink::bodyCoords()
{
  auto r = request(op::kGetBaseCoords);
  return r ? decodeCoords(*r) : std::nullopt;
}

std::optional<double> ArmLink::joint(int id)
{
  auto r = request(op::kGetJoint, {static_cast<uint8_t>(id)});
  return r ? decodeJoint(*r) : std::nullopt;
}

std::optional<bool> ArmLink::isMoving()
{
  auto r = request(op::kIsMoving);
  if (!r || r->empty()) {return std::nullopt;}
  return (*r)[0] == 1;
}

std::optional<Bytes> ArmLink::status() {return request(op::kRobotStatus);}

bool ArmLink::stop() {return request(op::kStop, {}, std::chrono::seconds(1)).has_value();}

bool ArmLink::setMovementType(bool linear)
{
  return request(op::kSetMovementType, {static_cast<uint8_t>(linear ? 1 : 0)}) == kAck;
}

bool ArmLink::sendAngles(const Joints & deg, int speed)
{
  return request(op::kSendAngles, encodeAngles(deg, speed)) == kAck;
}

bool ArmLink::sendAngle(int id, double deg, int speed)
{
  return request(op::kSendAngle, encodeAngle(id, deg, speed)) == kAck;
}

bool ArmLink::sendCoords(const Coords & c, int speed, bool body_frame)
{
  return request(body_frame ? op::kSendBaseCoords : op::kSendCoords, encodeCoords(c, speed)) == kAck;
}

}  // namespace mercury_x1::arm

#include "mercury_x1_driver/base_protocol.hpp"

#include <algorithm>
#include <cmath>

namespace mercury_x1::base
{

namespace
{
int16_t i16(const uint8_t * p) {return static_cast<int16_t>((p[0] << 8) | p[1]);}
uint16_t u16(const uint8_t * p) {return static_cast<uint16_t>((p[0] << 8) | p[1]);}

StateFrame parseState(const uint8_t * f)
{
  StateFrame s{};
  s.stop_flag = f[1];
  s.vx = i16(f + 2) / 1000.0;
  s.vy = i16(f + 4) / 1000.0;
  s.wz = i16(f + 6) / 1000.0;
  for (int k = 0; k < 3; ++k) {
    s.accel[k] = i16(f + 8 + 2 * k) * kAccelPerLsb;
    s.gyro[k] = i16(f + 14 + 2 * k) * kGyroRadPerLsb;
  }
  s.voltage = u16(f + 20) / 1000.0;
  s.firmware = "v" + std::to_string(f[22]) + "." + std::to_string(f[23]);
  return s;
}

Bytes frame(uint8_t mode, uint8_t b2, const std::array<uint8_t, 6> & p)
{
  Bytes b;
  b.reserve(11);
  b.push_back(0x7B);
  b.push_back(mode);
  b.push_back(b2);
  for (uint8_t v : p) {b.push_back(v);}
  b.push_back(xorSum(b.data(), b.size()));
  b.push_back(0x7D);
  return b;
}
}  // namespace

uint8_t xorSum(const uint8_t * d, size_t n)
{
  uint8_t x = 0;
  for (size_t i = 0; i < n; ++i) {x ^= d[i];}
  return x;
}

std::vector<Message> StreamParser::feed(const Bytes & chunk)
{
  buf_.insert(buf_.end(), chunk.begin(), chunk.end());
  std::vector<Message> out;
  size_t i = 0;
  while (i < buf_.size()) {
    const uint8_t b = buf_[i];
    if (b == 0x7B) {
      if (buf_.size() - i < kStateLen) {break;}
      const uint8_t * f = &buf_[i];
      if (f[kStateLen - 1] == 0x7D && xorSum(f, kStateLen - 2) == f[kStateLen - 2]) {
        out.emplace_back(parseState(f));
        i += kStateLen;
        continue;
      }
      ++bad_;
    } else if (b == 0xFA) {
      if (buf_.size() - i < kUltraLen) {break;}
      const uint8_t * f = &buf_[i];
      if (f[kUltraLen - 1] == 0xFC && xorSum(f, kUltraLen - 2) == f[kUltraLen - 2]) {
        UltrasonicFrame u{};
        for (int k = 0; k < 6; ++k) {u.mm[k] = u16(f + 1 + 2 * k);}
        for (int k = 0; k < 2; ++k) {u.aux[k] = u16(f + 13 + 2 * k);}
        out.emplace_back(u);
        i += kUltraLen;
        continue;
      }
      ++bad_;
    }
    ++i;
  }
  buf_.erase(buf_.begin(), buf_.begin() + static_cast<long>(i));
  return out;
}

Bytes velocityFrame(double vx, double vy, double wz)
{
  std::array<uint8_t, 6> p{};
  const double v[3] = {vx, vy, wz};
  for (int k = 0; k < 3; ++k) {
    long x = std::clamp<long>(std::lround(v[k] * 1000.0), -32768, 32767);
    auto u = static_cast<uint16_t>(static_cast<int16_t>(x));
    p[2 * k] = static_cast<uint8_t>(u >> 8);
    p[2 * k + 1] = static_cast<uint8_t>(u & 0xFF);
  }
  return frame(0, 0, p);
}

Bytes ledFrame(int r, int g, int b)
{
  r = std::clamp(r, 0, 255);
  g = std::clamp(g, 0, 255);
  b = std::clamp(b, 0, 255);
  const bool on = r || g || b;
  return frame(4, on ? 1 : 0, {static_cast<uint8_t>(r), static_cast<uint8_t>(g), static_cast<uint8_t>(b), 0, 0, 0});
}

}  // namespace mercury_x1::base

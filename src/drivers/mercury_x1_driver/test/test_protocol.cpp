// Protocol codec tests against byte sequences captured from the robot.
#include <gtest/gtest.h>

#include "mercury_x1_driver/arm_protocol.hpp"
#include "mercury_x1_driver/base_protocol.hpp"

namespace arm = mercury_x1::arm;
namespace base = mercury_x1::base;

static std::vector<uint8_t> hex(const std::string & s)
{
  std::vector<uint8_t> out;
  for (size_t i = 0; i + 1 < s.size(); ) {
    if (s[i] == ' ') {++i; continue;}
    out.push_back(static_cast<uint8_t>(std::stoi(s.substr(i, 2), nullptr, 16)));
    i += 2;
  }
  return out;
}

TEST(ArmProtocol, EncodeMatchesCapturedFrames)
{
  EXPECT_EQ(arm::encode(0x10), hex("fe fe 03 10 00 51"));
  EXPECT_EQ(arm::encode(0x20), hex("fe fe 03 20 14 51"));
  EXPECT_EQ(arm::encode(0x02), hex("fe fe 03 02 0d d1"));
}

TEST(ArmProtocol, ParsesRepliesAcrossChunksAndNoise)
{
  arm::FrameParser p;
  auto a = hex("00 13 fe fe 04 12 01");
  auto first = p.feed(a);
  EXPECT_TRUE(first.empty());
  // reply to is_power_on(0) captured: fe fe 04 12 00 5d 71 ; build a valid one for value 1
  auto f = arm::encode(0x12, {0x01});
  auto frames = p.feed(std::vector<uint8_t>(f.begin(), f.end()));
  ASSERT_EQ(frames.size(), 1u);
  EXPECT_EQ(frames[0].cmd, 0x12);
  EXPECT_EQ(frames[0].payload, std::vector<uint8_t>{0x01});
  auto cap = p.feed(hex("fe fe 04 12 00 5d 71"));
  ASSERT_EQ(cap.size(), 1u);
  EXPECT_EQ(cap[0].payload, std::vector<uint8_t>{0x00});
}

TEST(ArmProtocol, DecodesAnglesCoordsCaptured)
{
  auto p = hex("00003febfffffd52ffffcf4bffffcfe4000001010000161500001bb005dc04a404ec2b770a04ee10");
  auto r = arm::decodeAnglesCoords(p);
  ASSERT_TRUE(r.has_value());
  EXPECT_NEAR(r->first[0], 163.63, 1e-9);
  EXPECT_NEAR(r->first[6], 70.88, 1e-9);
  EXPECT_NEAR(r->second[0], 150.0, 1e-9);
  EXPECT_NEAR(r->second[2], 126.0, 1e-9);
  EXPECT_NEAR(r->second[5], -45.92, 1e-9);
}

TEST(ArmProtocol, EncodesAngleCommand)
{
  // send_angle(7, 75.88, 10)
  EXPECT_EQ(arm::encodeAngle(7, 75.88, 10), hex("07 1d a4 0a"));
  EXPECT_EQ(arm::encodeAngle(11, -2.5, 300), hex("0b ff 06 64"));  // speed clamped to 100
}

TEST(BaseProtocol, ParsesCapturedStream)
{
  base::StreamParser p;
  auto s = hex(
    "7b 00 00 00 00 00 00 00 00 08 00 65 3f 8a ff d2 ff fd ff f6 60 c4 01 02 22 7d "
    "fa 02 38 04 5d 02 98 0b b8 0b b8 0b b8 00 04 00 00 b4 fc 7b 00 00");
  auto msgs = p.feed(s);
  ASSERT_EQ(msgs.size(), 2u);
  auto st = std::get<base::StateFrame>(msgs[0]);
  EXPECT_NEAR(st.voltage, 24.772, 1e-9);
  EXPECT_EQ(st.firmware, "v1.2");
  EXPECT_NEAR(st.accel[2], 16266 / 1671.84, 1e-9);
  auto us = std::get<base::UltrasonicFrame>(msgs[1]);
  EXPECT_EQ(us.mm[0], 568);
  EXPECT_EQ(us.mm[1], 1117);
  EXPECT_EQ(us.mm[2], 664);
  EXPECT_EQ(us.mm[3], 3000);
}

TEST(BaseProtocol, CommandFrames)
{
  auto v = base::velocityFrame(0, 0, 0.3);
  ASSERT_EQ(v.size(), 11u);
  EXPECT_EQ(v[0], 0x7B);
  EXPECT_EQ(v[7], 0x01);
  EXPECT_EQ(v[8], 0x2C);
  EXPECT_EQ(v[9], base::xorSum(v.data(), 9));
  EXPECT_EQ(v[10], 0x7D);
  auto l = base::ledFrame(255, 0, 10);
  EXPECT_EQ(l[1], 4);
  EXPECT_EQ(l[2], 1);
  EXPECT_EQ(l[3], 255);
  EXPECT_EQ(l[5], 10);
  EXPECT_EQ(base::ledFrame(0, 0, 0)[2], 0);
}

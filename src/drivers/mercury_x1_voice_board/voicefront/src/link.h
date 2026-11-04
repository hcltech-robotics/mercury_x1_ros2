// Host link: one JSON object per line (newline-terminated, ASCII) in both directions.
// On the voice board this is UART2 (/dev/ttyS2 -> CH343 -> the Orin's /dev/ttyACM3). Without a device the
// events go to stderr and no commands are read (development on the Orin).
#pragma once

#include <mutex>
#include <string>
#include <vector>

class Link {
 public:
  ~Link();
  bool open(const std::string& path, int baud, std::string& err);
  // Writes one line. Never blocks: when the device cannot take it, the line is dropped and counted.
  void send(const std::string& line);
  // Complete lines received since the last call (non-blocking).
  std::vector<std::string> poll();
  long dropped() const { return dropped_; }
  bool is_device() const { return fd_ >= 0; }

 private:
  int fd_ = -1;
  std::string rx_;
  std::mutex mu_;
  long dropped_ = 0;
};

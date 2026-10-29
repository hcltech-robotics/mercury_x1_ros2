// Minimal raw termios serial port (no external dependency).
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace mercury_x1
{

class SerialPort
{
public:
  enum class Lines { kDefault, kDtrOnRtsOff, kDtrOnRtsOn };

  SerialPort() = default;
  ~SerialPort();
  SerialPort(const SerialPort &) = delete;
  SerialPort & operator=(const SerialPort &) = delete;

  // Throws std::runtime_error on failure.
  void open(const std::string & path, int baud, Lines lines = Lines::kDefault);
  void close();
  bool isOpen() const {return fd_ >= 0;}

  // Writes everything or throws.
  void write(const std::vector<uint8_t> & data);
  // Waits up to timeout_ms for data, appends whatever is available to out. Returns bytes read.
  size_t read(std::vector<uint8_t> & out, int timeout_ms);
  void flushInput();

private:
  int fd_{-1};
  std::string path_;
};

}  // namespace mercury_x1

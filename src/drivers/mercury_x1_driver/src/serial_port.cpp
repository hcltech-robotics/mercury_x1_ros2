#include "mercury_x1_driver/serial_port.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace mercury_x1
{

namespace
{
speed_t toSpeed(int baud)
{
  switch (baud) {
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
    case 230400: return B230400;
    case 460800: return B460800;
    case 921600: return B921600;
    default: throw std::runtime_error("unsupported baud rate " + std::to_string(baud));
  }
}
}  // namespace

SerialPort::~SerialPort() {close();}

void SerialPort::open(const std::string & path, int baud, Lines lines)
{
  close();
  fd_ = ::open(path.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd_ < 0) {
    throw std::runtime_error("open " + path + ": " + std::strerror(errno));
  }
  path_ = path;
  termios tio{};
  if (tcgetattr(fd_, &tio) != 0) {
    close();
    throw std::runtime_error("tcgetattr " + path + ": " + std::strerror(errno));
  }
  cfmakeraw(&tio);
  tio.c_cflag |= CLOCAL | CREAD;
  tio.c_cflag &= ~CRTSCTS;
  tio.c_cc[VMIN] = 0;
  tio.c_cc[VTIME] = 0;
  cfsetispeed(&tio, toSpeed(baud));
  cfsetospeed(&tio, toSpeed(baud));
  if (tcsetattr(fd_, TCSANOW, &tio) != 0) {
    close();
    throw std::runtime_error("tcsetattr " + path + ": " + std::strerror(errno));
  }
  // The right-arm CH9102 bridge has its RTS/DTR wired to the ESP32 EN/GPIO0 lines.
  // Match what pymycobot leaves them at (DTR asserted, RTS released) so the MCU is not reset.
  int bits = 0;
  if (lines != Lines::kDefault && ioctl(fd_, TIOCMGET, &bits) == 0) {
    bits |= TIOCM_DTR;
    if (lines == Lines::kDtrOnRtsOn) {bits |= TIOCM_RTS;} else {bits &= ~TIOCM_RTS;}
    ioctl(fd_, TIOCMSET, &bits);
  }
  flushInput();
}

void SerialPort::close()
{
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

void SerialPort::write(const std::vector<uint8_t> & data)
{
  size_t off = 0;
  while (off < data.size()) {
    ssize_t n = ::write(fd_, data.data() + off, data.size() - off);
    if (n < 0) {
      if (errno == EAGAIN || errno == EINTR) {
        pollfd p{fd_, POLLOUT, 0};
        ::poll(&p, 1, 10);
        continue;
      }
      throw std::runtime_error("write " + path_ + ": " + std::strerror(errno));
    }
    off += static_cast<size_t>(n);
  }
}

size_t SerialPort::read(std::vector<uint8_t> & out, int timeout_ms)
{
  pollfd p{fd_, POLLIN, 0};
  int r = ::poll(&p, 1, timeout_ms);
  if (r < 0) {
    if (errno == EINTR) {return 0;}
    throw std::runtime_error("poll " + path_ + ": " + std::strerror(errno));
  }
  if (r == 0) {return 0;}
  if (p.revents & (POLLERR | POLLHUP | POLLNVAL)) {
    throw std::runtime_error("serial device " + path_ + " lost");
  }
  uint8_t buf[512];
  ssize_t n = ::read(fd_, buf, sizeof(buf));
  if (n < 0) {
    if (errno == EAGAIN || errno == EINTR) {return 0;}
    throw std::runtime_error("read " + path_ + ": " + std::strerror(errno));
  }
  out.insert(out.end(), buf, buf + n);
  return static_cast<size_t>(n);
}

void SerialPort::flushInput()
{
  if (fd_ >= 0) {tcflush(fd_, TCIFLUSH);}
}

}  // namespace mercury_x1

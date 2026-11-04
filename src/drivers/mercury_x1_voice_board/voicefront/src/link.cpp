#include "link.h"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>

namespace {

speed_t baud_const(int baud) {
  switch (baud) {
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 230400: return B230400;
    case 460800: return B460800;
    case 921600: return B921600;
    default: return B115200;
  }
}

}  // namespace

Link::~Link() {
  if (fd_ >= 0) ::close(fd_);
}

bool Link::open(const std::string& path, int baud, std::string& err) {
  if (path.empty() || path == "-") return true;
  fd_ = ::open(path.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd_ < 0) {
    err = path + ": " + std::strerror(errno);
    return false;
  }
  termios t{};
  if (tcgetattr(fd_, &t) == 0) {  // a tty (or pty): raw 8N1, no flow control
    cfmakeraw(&t);
    t.c_cflag |= CLOCAL | CREAD;
    t.c_cflag &= ~(CRTSCTS | CSTOPB | PARENB);
    cfsetispeed(&t, baud_const(baud));
    cfsetospeed(&t, baud_const(baud));
    t.c_cc[VMIN] = 0;
    t.c_cc[VTIME] = 0;
    tcsetattr(fd_, TCSANOW, &t);
    tcflush(fd_, TCIOFLUSH);
  }
  return true;
}

void Link::send(const std::string& line) {
  std::lock_guard<std::mutex> lk(mu_);
  const std::string s = line + "\n";
  if (fd_ < 0) {
    std::fputs(s.c_str(), stderr);
    std::fflush(stderr);
    return;
  }
  size_t off = 0;
  while (off < s.size()) {
    const ssize_t n = ::write(fd_, s.data() + off, s.size() - off);
    if (n > 0) {
      off += static_cast<size_t>(n);
    } else if (n < 0 && errno == EINTR) {
      continue;
    } else {
      ++dropped_;  // EAGAIN: the UART buffer is full; a partial line is ended so the next one parses
      if (off > 0) (void)!::write(fd_, "\n", 1);
      return;
    }
  }
}

std::vector<std::string> Link::poll() {
  std::vector<std::string> lines;
  if (fd_ < 0) return lines;
  char buf[512];
  while (true) {
    const ssize_t n = ::read(fd_, buf, sizeof(buf));
    if (n <= 0) break;
    rx_.append(buf, static_cast<size_t>(n));
  }
  size_t pos;
  while ((pos = rx_.find('\n')) != std::string::npos) {
    std::string l = rx_.substr(0, pos);
    rx_.erase(0, pos + 1);
    if (!l.empty() && l.back() == '\r') l.pop_back();
    if (!l.empty()) lines.push_back(l);
  }
  if (rx_.size() > 4096) rx_.clear();  // garbage without newlines
  return lines;
}

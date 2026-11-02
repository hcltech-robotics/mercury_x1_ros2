# State of the head voice board link, published periodically by mercury_x1_voice_board.
std_msgs/Header header
string port
bool port_open                   # serial device opened
bool alive                       # a frame from the board within the last alive_timeout seconds
string firmware_version          # reply to {"type":"version"}, empty until known
uint32 handshakes                # handshakes confirmed since start (board resends until confirmed)
uint32 wake_count                # wake events since start
float32 last_wake_angle_deg      # NaN until the first wake
uint32 dropped_bytes             # bytes discarded by the frame parser (noise / bad checksum)
string last_error

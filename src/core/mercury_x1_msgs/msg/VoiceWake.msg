# Wake-word event from the head voice board (iFlytek AIUI, mercury_x1_voice_board).
std_msgs/Header header           # stamp = time the event arrived on the host; frame_id = array frame
string keyword                   # keyword as the board reports it, pinyin with tones, e.g. "xiao3 wei1 xiao3 wei1"
float32 angle_deg                # direction of arrival from the board; linear 4-mic array: 0..180 (NaN if absent)
float32 yaw                      # same direction in frame_id [rad, + = left], from the node's angle_front_deg /
                                 # angle_sign parameters; a linear array cannot tell front from back
int16 beam                       # beam the board locked onto (mic4: 0..2), -1 if not reported
float32 score                    # wake confidence as reported by the board (NaN if absent)
int64 start_ms                   # board timestamp of the utterance start, -1 if absent (used to drop repeats)
string raw                       # the complete event JSON

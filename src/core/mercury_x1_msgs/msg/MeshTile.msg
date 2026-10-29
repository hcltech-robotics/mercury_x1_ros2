# One spatial tile of a reconstructed mesh (flat arrays, cheap to read from Python with numpy).
# An empty `triangles` array means the tile was removed.
std_msgs/Header header      # frame of the vertices
string id                   # tile key, e.g. "3_-2"
float32[] positions         # x y z per vertex
float32[] normals           # nx ny nz per vertex (may be empty)
uint8[] colors              # r g b per vertex (may be empty)
uint32[] triangles          # 3 vertex indices per face

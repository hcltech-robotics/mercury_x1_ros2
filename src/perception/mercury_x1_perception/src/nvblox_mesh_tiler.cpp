// nvblox_mesh_tiler: turns nvblox's incremental block mesh (/nvblox_node/mesh, nvblox_msgs/Mesh)
// into a few large MeshTile messages (one per tile_size x tile_size column), re-sent only when a
// block inside the tile changed. Flat float/uint8 arrays let the Python Rerun bridge read them
// with numpy instead of deserializing one Python object per vertex.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "mercury_x1_msgs/msg/mesh_tile.hpp"
#include "nvblox_msgs/msg/mesh.hpp"
#include "rclcpp/rclcpp.hpp"

namespace {

using BlockKey = std::tuple<int32_t, int32_t, int32_t>;
using TileKey = std::pair<int32_t, int32_t>;

struct Block {
  std::vector<float> pos, nrm;
  std::vector<uint8_t> rgb;
  std::vector<uint32_t> tri;
};

uint8_t to_u8(float c) { return static_cast<uint8_t>(std::clamp(c, 0.0F, 1.0F) * 255.0F + 0.5F); }

}  // namespace

class NvbloxMeshTiler : public rclcpp::Node {
 public:
  NvbloxMeshTiler() : Node("nvblox_mesh_tiler") {
    tile_size_ = declare_parameter("tile_size", 2.0);
    const double period = declare_parameter("publish_period", 1.0);
    max_tiles_per_tick_ = declare_parameter("max_tiles_per_tick", 8);

    pub_ = create_publisher<mercury_x1_msgs::msg::MeshTile>("mesh_tiles", rclcpp::QoS(64).reliable());
    sub_ = create_subscription<nvblox_msgs::msg::Mesh>(
        "mesh", rclcpp::QoS(10).reliable(),
        [this](nvblox_msgs::msg::Mesh::ConstSharedPtr m) { on_mesh(*m); });
    timer_ = create_wall_timer(std::chrono::duration<double>(period), [this] { flush(); });
  }

 private:
  TileKey tile_of(const nvblox_msgs::msg::Index3D& i, float block_size) const {
    // tile from the block centre so a block never straddles two tiles
    const double cx = (i.x + 0.5) * block_size, cy = (i.y + 0.5) * block_size;
    return {static_cast<int32_t>(std::floor(cx / tile_size_)), static_cast<int32_t>(std::floor(cy / tile_size_))};
  }

  void on_mesh(const nvblox_msgs::msg::Mesh& m) {
    frame_id_ = m.header.frame_id;
    stamp_ = m.header.stamp;
    if (m.clear) {
      for (auto& [key, blocks] : tiles_) {
        blocks.clear();
        dirty_.insert(key);
      }
    }
    const size_t n = std::min(m.block_indices.size(), m.blocks.size());
    for (size_t b = 0; b < n; ++b) {
      const auto& idx = m.block_indices[b];
      const auto& src = m.blocks[b];
      const TileKey tk = tile_of(idx, m.block_size_m);
      const BlockKey bk{idx.x, idx.y, idx.z};
      auto& tile = tiles_[tk];
      dirty_.insert(tk);
      if (src.vertices.empty() || src.triangles.empty()) {  // empty block = deleted
        tile.erase(bk);
        continue;
      }
      Block& dst = tile[bk];
      const size_t nv = src.vertices.size();
      dst.pos.resize(nv * 3);
      for (size_t v = 0; v < nv; ++v) {
        dst.pos[3 * v] = src.vertices[v].x;
        dst.pos[3 * v + 1] = src.vertices[v].y;
        dst.pos[3 * v + 2] = src.vertices[v].z;
      }
      dst.nrm.clear();
      if (src.normals.size() == nv) {
        dst.nrm.resize(nv * 3);
        for (size_t v = 0; v < nv; ++v) {
          dst.nrm[3 * v] = src.normals[v].x;
          dst.nrm[3 * v + 1] = src.normals[v].y;
          dst.nrm[3 * v + 2] = src.normals[v].z;
        }
      }
      dst.rgb.clear();
      if (src.colors.size() == nv) {
        dst.rgb.resize(nv * 3);
        for (size_t v = 0; v < nv; ++v) {
          dst.rgb[3 * v] = to_u8(src.colors[v].r);
          dst.rgb[3 * v + 1] = to_u8(src.colors[v].g);
          dst.rgb[3 * v + 2] = to_u8(src.colors[v].b);
        }
      }
      dst.tri.clear();
      dst.tri.reserve(src.triangles.size());
      for (int32_t t : src.triangles) {
        if (t >= 0 && static_cast<size_t>(t) < nv) dst.tri.push_back(static_cast<uint32_t>(t));
      }
      dst.tri.resize(dst.tri.size() / 3 * 3);
    }
  }

  void flush() {
    // a (re)started viewer bridge gets the whole mesh again
    const size_t subs = pub_->get_subscription_count();
    if (subs > last_subs_) {
      for (const auto& kv : tiles_) dirty_.insert(kv.first);
    }
    last_subs_ = subs;
    if (subs == 0 || frame_id_.empty()) return;

    int sent = 0;
    for (auto it = dirty_.begin(); it != dirty_.end() && sent < max_tiles_per_tick_;) {
      publish_tile(*it);
      auto t = tiles_.find(*it);
      if (t != tiles_.end() && t->second.empty()) tiles_.erase(t);
      it = dirty_.erase(it);
      ++sent;
    }
  }

  void publish_tile(const TileKey& key) {
    mercury_x1_msgs::msg::MeshTile out;
    out.header.frame_id = frame_id_;
    out.header.stamp = stamp_;
    out.id = std::to_string(key.first) + "_" + std::to_string(key.second);
    auto it = tiles_.find(key);
    if (it != tiles_.end()) {
      size_t nv = 0, nt = 0;
      bool normals = true, colors = true;
      for (const auto& [bk, b] : it->second) {
        nv += b.pos.size() / 3;
        nt += b.tri.size();
        normals &= !b.nrm.empty();
        colors &= !b.rgb.empty();
      }
      out.positions.reserve(nv * 3);
      out.triangles.reserve(nt);
      if (normals) out.normals.reserve(nv * 3);
      if (colors) out.colors.reserve(nv * 3);
      for (const auto& [bk, b] : it->second) {
        const auto base = static_cast<uint32_t>(out.positions.size() / 3);
        out.positions.insert(out.positions.end(), b.pos.begin(), b.pos.end());
        if (normals) out.normals.insert(out.normals.end(), b.nrm.begin(), b.nrm.end());
        if (colors) out.colors.insert(out.colors.end(), b.rgb.begin(), b.rgb.end());
        for (uint32_t t : b.tri) out.triangles.push_back(base + t);
      }
    }
    pub_->publish(out);
  }

  double tile_size_;
  int max_tiles_per_tick_;
  std::string frame_id_;
  builtin_interfaces::msg::Time stamp_;
  std::map<TileKey, std::map<BlockKey, Block>> tiles_;
  std::set<TileKey> dirty_;
  size_t last_subs_ = 0;
  rclcpp::Publisher<mercury_x1_msgs::msg::MeshTile>::SharedPtr pub_;
  rclcpp::Subscription<nvblox_msgs::msg::Mesh>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<NvbloxMeshTiler>());
  rclcpp::shutdown();
  return 0;
}

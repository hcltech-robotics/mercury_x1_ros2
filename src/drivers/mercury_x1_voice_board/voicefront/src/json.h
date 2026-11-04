// Minimal JSON reader (for the short command lines of the serial protocol) and string escaping.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

struct Json {
  enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
  bool b = false;
  double num = 0.0;
  std::string str;
  std::vector<Json> arr;
  std::vector<std::pair<std::string, Json>> obj;

  const Json* get(const std::string& key) const {
    for (const auto& kv : obj)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
  std::string get_str(const std::string& key, const std::string& def = "") const {
    const Json* v = get(key);
    return v && v->type == Str ? v->str : def;
  }
  double get_num(const std::string& key, double def) const {
    const Json* v = get(key);
    return v && v->type == Num ? v->num : def;
  }
  bool has_num(const std::string& key) const {
    const Json* v = get(key);
    return v && v->type == Num;
  }
  bool get_bool(const std::string& key, bool def) const {
    const Json* v = get(key);
    return v && v->type == Bool ? v->b : def;
  }

  static bool parse(const std::string& s, Json& out) {
    size_t i = 0;
    if (!value(s, i, out, 0)) return false;
    ws(s, i);
    return i == s.size();
  }

 private:
  static void ws(const std::string& s, size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) ++i;
  }
  static bool lit(const std::string& s, size_t& i, const char* w) {
    size_t n = 0;
    while (w[n]) ++n;
    if (s.compare(i, n, w) != 0) return false;
    i += n;
    return true;
  }
  static bool string(const std::string& s, size_t& i, std::string& out) {
    if (i >= s.size() || s[i] != '"') return false;
    ++i;
    out.clear();
    while (i < s.size() && s[i] != '"') {
      char c = s[i++];
      if (c == '\\') {
        if (i >= s.size()) return false;
        const char e = s[i++];
        switch (e) {
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case 'r': out += '\r'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'u': {
            if (i + 4 > s.size()) return false;
            const unsigned cp = static_cast<unsigned>(std::strtoul(s.substr(i, 4).c_str(), nullptr, 16));
            i += 4;
            if (cp < 0x80) {
              out += static_cast<char>(cp);
            } else if (cp < 0x800) {
              out += static_cast<char>(0xC0 | (cp >> 6));
              out += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
              out += static_cast<char>(0xE0 | (cp >> 12));
              out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
              out += static_cast<char>(0x80 | (cp & 0x3F));
            }
            break;
          }
          default: out += e;
        }
      } else {
        out += c;
      }
    }
    if (i >= s.size()) return false;
    ++i;
    return true;
  }
  static bool value(const std::string& s, size_t& i, Json& out, int depth) {
    if (depth > 16) return false;
    ws(s, i);
    if (i >= s.size()) return false;
    const char c = s[i];
    if (c == '{') {
      out.type = Obj;
      ++i;
      ws(s, i);
      if (i < s.size() && s[i] == '}') return ++i, true;
      while (true) {
        ws(s, i);
        std::string key;
        if (!string(s, i, key)) return false;
        ws(s, i);
        if (i >= s.size() || s[i] != ':') return false;
        ++i;
        Json v;
        if (!value(s, i, v, depth + 1)) return false;
        out.obj.emplace_back(std::move(key), std::move(v));
        ws(s, i);
        if (i < s.size() && s[i] == ',') {
          ++i;
          continue;
        }
        if (i < s.size() && s[i] == '}') return ++i, true;
        return false;
      }
    }
    if (c == '[') {
      out.type = Arr;
      ++i;
      ws(s, i);
      if (i < s.size() && s[i] == ']') return ++i, true;
      while (true) {
        Json v;
        if (!value(s, i, v, depth + 1)) return false;
        out.arr.push_back(std::move(v));
        ws(s, i);
        if (i < s.size() && s[i] == ',') {
          ++i;
          continue;
        }
        if (i < s.size() && s[i] == ']') return ++i, true;
        return false;
      }
    }
    if (c == '"') {
      out.type = Str;
      return string(s, i, out.str);
    }
    if (lit(s, i, "true")) return out.type = Bool, out.b = true, true;
    if (lit(s, i, "false")) return out.type = Bool, out.b = false, true;
    if (lit(s, i, "null")) return out.type = Null, true;
    const char* begin = s.c_str() + i;
    char* end = nullptr;
    out.num = std::strtod(begin, &end);
    if (end == begin) return false;
    out.type = Num;
    i += static_cast<size_t>(end - begin);
    return true;
  }
};

inline std::string json_str(const std::string& s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') {
      o += '\\';
      o += static_cast<char>(c);
    } else if (c == '\n') {
      o += "\\n";
    } else if (c < 0x20) {
      char b[8];
      std::snprintf(b, sizeof(b), "\\u%04x", c);
      o += b;
    } else {
      o += static_cast<char>(c);
    }
  }
  return o + "\"";
}

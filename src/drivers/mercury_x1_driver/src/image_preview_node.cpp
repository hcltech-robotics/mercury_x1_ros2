// Throttled JPEG previews of camera streams for the web UI / Rerun (keeps Python bridges from
// deserialising every raw frame).
//   in:  image (sensor_msgs/Image, rgb8/bgr8)      out: image/preview (sensor_msgs/CompressedImage)
#include <chrono>
#include <vector>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"

class ImagePreview : public rclcpp::Node
{
public:
  ImagePreview()
  : Node("image_preview")
  {
    rate_ = declare_parameter("rate", 4.0);
    quality_ = declare_parameter("jpeg_quality", 70);
    max_width_ = declare_parameter("max_width", 640);
    pub_ = create_publisher<sensor_msgs::msg::CompressedImage>("image/preview", rclcpp::SensorDataQoS());
    sub_ = create_subscription<sensor_msgs::msg::Image>("image", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::Image::ConstSharedPtr m) {onImage(m);});
  }

private:
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr & m)
  {
    const auto now = std::chrono::steady_clock::now();
    if (rate_ > 0 && now - last_ < std::chrono::duration<double>(1.0 / rate_)) {return;}
    if (pub_->get_subscription_count() == 0) {return;}
    last_ = now;
    // wrap the message buffer (no cv_bridge: it links a different OpenCV than JetPack's)
    const bool rgb = m->encoding == "rgb8", bgr = m->encoding == "bgr8";
    if (!rgb && !bgr) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "unsupported encoding %s", m->encoding.c_str());
      return;
    }
    cv::Mat wrapped(static_cast<int>(m->height), static_cast<int>(m->width), CV_8UC3,
      const_cast<uint8_t *>(m->data.data()), m->step);
    cv::Mat img;
    if (rgb) {cv::cvtColor(wrapped, img, cv::COLOR_RGB2BGR);} else {img = wrapped;}
    if (max_width_ > 0 && img.cols > max_width_) {
      cv::resize(img, img, cv::Size(), double(max_width_) / img.cols, double(max_width_) / img.cols, cv::INTER_AREA);
    }
    sensor_msgs::msg::CompressedImage out;
    out.header = m->header;
    out.format = "jpeg";
    cv::imencode(".jpg", img, out.data, {cv::IMWRITE_JPEG_QUALITY, static_cast<int>(quality_)});
    pub_->publish(out);
  }

  double rate_;
  int64_t quality_, max_width_;
  std::chrono::steady_clock::time_point last_{};
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImagePreview>());
  rclcpp::shutdown();
  return 0;
}

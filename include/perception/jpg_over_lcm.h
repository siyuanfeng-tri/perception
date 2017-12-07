#pragma once

#include "lcm/lcm-cpp.hpp"
#include <atomic>
#include <bot_core/raw_t.hpp>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <unordered_map>

namespace perception {

class JpgOverLcm {
public:
  JpgOverLcm(const std::vector<std::string> &channels,
             const std::vector<int> &width, const std::vector<int> &height);
  ~JpgOverLcm();

  void Start();
  void Stop();

  bool CopyImage(const std::string &name, cv::Mat *img, int *ctr) const;

private:
  struct Channel {
    Channel(const std::string &channel, int w, int h)
        : name(channel), width(w), height(h) {}

    const std::string name;
    const int width;
    const int height;

    std::vector<uint8_t> bytes;
    int ctr{0};
  };

  void LoopThread();

  int find_channel(const std::string &name) const;

  void handle(const lcm::ReceiveBuffer *, const std::string &,
              const bot_core::raw_t *msg);
  void CopyData(const std::string &channel, std::vector<uint8_t> *buf,
                int *ctr) const;

  lcm::LCM lcm_;

  std::atomic<bool> run_{false};
  std::thread thread_;

  std::unordered_map<std::string, int> look_up_;
  mutable std::vector<std::mutex> locks_;
  std::vector<Channel> channels_;
};

} // namespace perception

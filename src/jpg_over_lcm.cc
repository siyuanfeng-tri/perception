#include "perception/jpg_over_lcm.h"

namespace perception {

JpgOverLcm::JpgOverLcm(const std::vector<std::string> &channels,
                       const std::vector<int> &width,
                       const std::vector<int> &height) {
  if (channels.size() != width.size() || channels.size() != height.size()) {
    throw std::logic_error("JpgOverLcm dimension mismatch");
  }

  int ctr = 0;
  for (size_t i = 0; i < channels.size(); i++) {
    if (look_up_.find(channels[i]) != look_up_.end()) {
      std::cout << "Duplicate channel: " << channels[i] << "\n";
      continue;
    }
    look_up_[channels[i]] = ctr++;
    channels_.push_back(Channel(channels[i], width[i], height[i]));
    auto sub = lcm_.subscribe(channels[i], &JpgOverLcm::handle, this);
    sub->setQueueCapacity(1);
  }
  locks_ = std::vector<std::mutex>(ctr);
}

JpgOverLcm::~JpgOverLcm() { Stop(); }

bool JpgOverLcm::CopyImage(const std::string &name, cv::Mat *img,
                           int *ctr) const {
  std::vector<uint8_t> raw;
  int tmp;
  CopyData(name, &raw, &tmp);
  if (tmp == 0)
    return false;

  int idx = find_channel(name);
  cv::Mat imgbuf(channels_.at(idx).height, channels_.at(idx).width, CV_8UC3,
                 raw.data());
  *img = cv::imdecode(imgbuf, CV_LOAD_IMAGE_COLOR);
  // No encoding
  //*img = imgbuf;
  *ctr = tmp;
  return true;
}

int JpgOverLcm::find_channel(const std::string &name) const {
  auto it = look_up_.find(name);
  if (it == look_up_.end()) {
    throw std::logic_error("No such channel");
  }
  return it->second;
}

void JpgOverLcm::handle(const lcm::ReceiveBuffer *, const std::string &name,
                        const bot_core::raw_t *msg) {
  int idx = find_channel(name);

  std::unique_lock<std::mutex> lock(locks_.at(idx));
  channels_.at(idx).bytes = msg->data;
  channels_.at(idx).ctr++;
}

void JpgOverLcm::CopyData(const std::string &name, std::vector<uint8_t> *buf,
                          int *ctr) const {
  int idx = find_channel(name);

  std::unique_lock<std::mutex> lock(locks_.at(idx));
  *ctr = channels_.at(idx).ctr;
  *buf = channels_.at(idx).bytes;
}

void JpgOverLcm::Start() {
  run_ = true;
  thread_ = std::thread(&JpgOverLcm::LoopThread, this);
}

void JpgOverLcm::Stop() {
  run_ = false;
  thread_.join();
}

void JpgOverLcm::LoopThread() {
  while (run_) {
    lcm_.handleTimeout(10);
  }
}

} // namespace perception

#pragma once

#include <functional>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <mutex>

namespace perception {

class PointCloudFusion {
public:
  PointCloudFusion(
      std::function<Eigen::Vector2f(const Eigen::Vector3f &)> proj_func,
      float resolution);

  void Init();

  bool ProcFrame(const Eigen::Isometry3f &X_WC,
                 const pcl::PointCloud<pcl::PointXYZRGB> &raw_cloud,
                 const cv::Mat &raw_depth);

  void GetLatestFusedPointCloudAndWorldFrameUpdate(
      pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr *cloud,
      Eigen::Isometry3f *tf) const {
    std::unique_lock<std::mutex> lock1(lock_);
    if (cloud)
      *cloud = fused_;
    if (tf)
      *tf = world_update_;
  }

private:
  bool is_uv_in_image(const Eigen::Vector2f& uv, const cv::Mat& img);
  uint16_t get_interp_val(const Eigen::Vector2f& uv, const cv::Mat& img);

  const std::function<Eigen::Vector2f(const Eigen::Vector3f &)> proj_func_;
  float resolution_;

  mutable std::mutex lock_;
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr fused_;
  Eigen::Isometry3f world_update_;
};

} // namespace perception

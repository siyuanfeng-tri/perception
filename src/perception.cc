#include "perception/perception.h"

#include <bot_core/raw_t.hpp>

namespace perception {

template void LoadPCDFile<pcl::PointXYZRGB>(
    const std::string &file_name,
    typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);

template pcl::PointCloud<pcl::PointXYZRGB>::Ptr DownSample<pcl::PointXYZRGB>(
    const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud, double leaf_size);
template pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr
DownSample<pcl::PointXYZRGBNormal>(
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr &cloud,
    double leaf_size);

template pcl::PointCloud<pcl::PointXYZRGB>::Ptr
CutWithWorkSpaceConstraints<pcl::PointXYZRGB>(
    const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud,
    const Eigen::Vector3f &min_range, const Eigen::Vector3f &max_range,
    const Eigen::Isometry3f& transform);
template pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr
CutWithWorkSpaceConstraints<pcl::PointXYZRGBNormal>(
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr &cloud,
    const Eigen::Vector3f &min_range, const Eigen::Vector3f &max_range,
    const Eigen::Isometry3f& transform);

template void FilterPointsWithEmptyNormals<pcl::PointXYZRGBNormal>(
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cloud);

template Eigen::Vector4d FindPlane<pcl::PointXYZRGB>(
    const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud,
    pcl::PointIndices::Ptr &inliers, double dist_threshold);
template Eigen::Vector4d FindPlane<pcl::PointXYZRGBNormal>(
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr &cloud,
    pcl::PointIndices::Ptr &inliers, double dist_threshold);

template
pcl::PointCloud<pcl::PointXYZRGB>::Ptr SubtractTable<pcl::PointXYZRGB>(
    const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud,
    double thickness);
template
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr SubtractTable<pcl::PointXYZRGBNormal>(
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr &cloud,
    double thickness);

template std::vector<int>
SelectNearCentroidPoints(const pcl::PointCloud<pcl::PointXYZRGB> &cloud,
                         double cover_ratio);
template std::vector<int>
SelectNearCentroidPoints(const pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud,
                         double cover_ratio);

template void FilterPointsBasedOnScatterness<pcl::PointXYZRGB>(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, double cover_ratio);
template void FilterPointsBasedOnScatterness<pcl::PointXYZRGBNormal>(
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cloud, double cover_ratio);

template void FindBoundingBox<pcl::PointXYZRGB>(
    const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud,
    Eigen::Vector3f *center, Eigen::Vector3f *top_right_corner,
    Eigen::Vector3f *lower_left_corner, Eigen::Matrix3f *orientation,
    double cover_ratio);
template void FindBoundingBox<pcl::PointXYZRGBNormal>(
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr &cloud,
    Eigen::Vector3f *center, Eigen::Vector3f *top_right_corner,
    Eigen::Vector3f *lower_left_corner, Eigen::Matrix3f *orientation,
    double cover_ratio);

template void VisualizePointCloud<pcl::PointXYZRGB>(
    const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud,
    const Eigen::Affine3f &tf);
template void VisualizePointCloud<pcl::PointXYZRGBNormal>(
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr &cloud,
    const Eigen::Affine3f &tf);

void VisualizePointCloudAndNormal(
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr &cloud,
    const Eigen::Affine3f &tf) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr points =
      boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  pcl::PointCloud<pcl::Normal>::Ptr normal =
      boost::make_shared<pcl::PointCloud<pcl::Normal>>();
  SeparatePointsAndNormals<pcl::PointXYZRGB, pcl::PointXYZRGBNormal>(
      *cloud, *points, *normal);
  VisualizePointCloudAndNormal(points, normal, tf);
}

void VisualizePointCloudAndNormal(
    const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud,
    const pcl::PointCloud<pcl::Normal>::ConstPtr &normals,
    const Eigen::Affine3f &tf) {
  pcl::visualization::PCLVisualizer viewer("Point Cloud Visualization");
  viewer.addCoordinateSystem(0.2, tf);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(
      cloud);
  viewer.addPointCloud(cloud, rgb, "Cloud");

  viewer.addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals, 3);
  // Display the point cloud until 'q' key is pressed.
  while (!viewer.wasStopped()) {
    viewer.spinOnce();
  }
}

void SendImageAsJpg(const cv::Mat &img, const std::string &channel,
                    lcm::LCM *lcm, int quality) {
  bot_core::raw_t message{};

  std::vector<int> param(2);
  param[0] = cv::IMWRITE_JPEG_QUALITY;
  param[1] = quality;
  cv::imencode(".jpg", img, message.data, param);
  // No encoding
  // message.data.resize(img.total() * img.elemSize());
  // memcpy(message.data.data(), img.data, message.data.size());

  message.length = static_cast<int>(message.data.size());

  std::vector<uint8_t> bytes(message.getEncodedSize());
  message.encode(bytes.data(), 0, bytes.size());

  lcm->publish(channel, bytes.data(), bytes.size());
}

void ScaleDepthImage(const cv::Mat &img, cv::Mat *result, double max) {
  if (img.type() != CV_16UC1) {
    std::cout << "not depth image type, expecting CV_16UC1\n";
    return;
  }

  img.convertTo(*result, CV_8UC1, -255. / max, 255);
  for (int i = 0; i < result->rows; i++) {
    for (int j = 0; j < result->cols; j++) {
      if (result->at<uint8_t>(i, j) == 255) {
        result->at<uint8_t>(i, j) = 0;
      }
    }
  }
}

template void
VisualizePointCloudDrake(const pcl::PointCloud<pcl::PointXYZRGB> &cloud,
                         lcm::LCM *lcm, const Eigen::Isometry3d &X_WC,
                         const std::string &suffix);
template void
VisualizePointCloudDrake(const pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud,
                         lcm::LCM *lcm, const Eigen::Isometry3d &X_WC,
                         const std::string &suffix);

template void
SeparatePointsAndNormals<pcl::PointXYZRGB, pcl::PointXYZRGBNormal>(
    const pcl::PointCloud<pcl::PointXYZRGBNormal> &points_and_normal,
    pcl::PointCloud<pcl::PointXYZRGB> &points,
    pcl::PointCloud<pcl::Normal> &normals);

template void ApplyTransformToPointCloud<pcl::PointXYZRGB>(
    const Eigen::Affine3f &tf, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);
template void ApplyTransformToPointCloud<pcl::PointXYZRGBNormal>(
    const Eigen::Affine3f &tf,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cloud);

template pcl::PointCloud<pcl::PointXYZRGB>::Ptr
SOROutlierRemoval<pcl::PointXYZRGB>(
    const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud, int num_neighbor,
    double std_dev_threshold);
template pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr
SOROutlierRemoval<pcl::PointXYZRGBNormal>(
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr &cloud,
    int num_neighbor, double std_dev_threshold);

template pcl::PointCloud<pcl::PointXYZRGB>::Ptr
RADOutlierRemoval<pcl::PointXYZRGB>(
    const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud, int num_neighbor,
    double radius);
template pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr
RADOutlierRemoval<pcl::PointXYZRGBNormal>(
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr &cloud,
    int num_neighbor, double radius);

template void EstimateNormal<pcl::PointXYZRGB>(
    const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud,
    pcl::PointCloud<pcl::Normal>::Ptr &normals,
    const Eigen::Vector3f &view_point, double radius);

template cv::Mat ProjectCloudToImage<pcl::PointXYZRGB>(
    const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const Camera &camera,
    int fill_size);

template cv::Mat ProjectCloudToImage<pcl::PointXYZRGBNormal>(
    const pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud, const Camera &camera,
    int fill_size);

cv::Mat ProjectColoredPointCloudToCameraImagePlane(
    const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const Camera &camera,
    int stride) {
  // OpenCV is row-major allocation.
  cv::Mat projected_cv_img(camera.img_height, camera.img_width, CV_8UC3);
  int num_points = cloud.size();
  // Image coordinate (inferred from pcl openni2_grabber.cpp
  // convertToXYZRGBPointCloud):
  // top left corner is the origin. +x is along the increasing column direction.
  // +y is along the increasing row direction.
  // Todo (Jiaji): it seems to be different from the openni kinect convention.
  // Check what's really going on.

  // Get the image frame center.
  int K = camera.img_width * camera.img_height;
  bool mark[K];
  for (int i = 0; i < camera.img_width * camera.img_height; ++i) {
    mark[i] = false;
  }
  for (int i = 0; i < num_points; ++i) {
    int row_id =
        round(camera.ppy + cloud.points[i].y / cloud.points[i].z * camera.fy);
    int col_id =
        round(camera.ppx + cloud.points[i].x / cloud.points[i].z * camera.fx);
    if (row_id > 0 && row_id < camera.img_height && col_id > 0 &&
        col_id < camera.img_width) {
      // Assuming opencv is using bgr convention.
      // std::cout << row_id << "," << col_id << std::endl;
      int flag_id = row_id * camera.img_width + col_id;
      if (!mark[flag_id]) {
        projected_cv_img.at<cv::Vec3b>(row_id, col_id) =
            cv::Vec3b(cloud.points[i].b, cloud.points[i].g, cloud.points[i].r);
        mark[flag_id] = true;
      }
    }
  }

  // Complete the holes due to missing point clouds in the object.
  for (int i = stride / 2; i < camera.img_height - stride / 2; ++i) {
    for (int j = stride / 2; j < camera.img_width - stride / 2; ++j) {
      int flag_id = i * camera.img_width + j;
      if (!mark[flag_id]) {
        // std::cout << "r,c unmarked" << i << "," << j << std::endl;
        // std::cout << projected_cv_img.at<cv::Vec3b>(i,j) << std::endl;
        Eigen::Vector3d mean_color = Eigen::Vector3d::Zero();
        int num_averages = 0;
        for (int k1 = -stride / 2; k1 <= stride / 2; ++k1) {
          for (int k2 = -stride / 2; k2 <= stride / 2; ++k2) {
            int flag_neighbor_id = (i + k1) * camera.img_width + j + k2;
            if (((k1 != 0) || (k2 != 0)) && mark[flag_neighbor_id]) {
              cv::Vec3b neighbor_color =
                  projected_cv_img.at<cv::Vec3b>(i + k1, j + k2);
              for (unsigned c = 0; c < 3; ++c) {
                mean_color(c) = mean_color(c) + neighbor_color[c];
              }
              ++num_averages;
            }
          }
        }
        if (num_averages > 0) {
          // std::cout << mean_color << std::endl;
          mean_color = mean_color / num_averages;
          // std::cout << mean_color << std::endl;
          projected_cv_img.at<cv::Vec3b>(i, j) = cv::Vec3b(
              round(mean_color(0)), round(mean_color(1)), round(mean_color(2)));
        }
      }
    }
  }

  // cv::namedWindow("test cv");
  // //cv::Mat output_img;
  // //cv::GaussianBlur(projected_cv_img, output_img, cv::Size(2,2), 0, 0);
  // cv::imshow("cv_projected_image", projected_cv_img);
  // cv::waitKey(0);
  return projected_cv_img;
}

/*
void StreamImagesAsJpgLoop(const sensor_bridge::RGBDSensor *driver,
                           const sensor_bridge::ImageType rgb_type,
                           const sensor_bridge::ImageType depth_type,
                           const std::string rgb_name,
                           const std::string depth_name, lcm::LCM *lcm) {
  uint64_t last_rgb_timestamp = 0;
  uint64_t last_depth_timestamp = 0;
  uint64_t timestamp = 0;
  cv::Mat scaled_depth;
  while (true) {
    auto rgb = driver->GetLatestImage(rgb_type, &timestamp);
    if (rgb && timestamp != last_rgb_timestamp) {
      SendImageAsJpg(*rgb, rgb_name, lcm);
      last_rgb_timestamp = timestamp;
    }

    auto depth = driver->GetLatestImage(depth_type, &timestamp);
    if (depth && timestamp != last_depth_timestamp) {
      ScaleDepthImage(*depth, &scaled_depth);
      SendImageAsJpg(scaled_depth, depth_name, lcm);
      last_depth_timestamp = timestamp;
    }
  }
}
*/

template double AlignCloud<pcl::PointXYZRGBNormal>(
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr &src,
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr &dest,
    double max_correspondence_dist,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &aligned_src,
    Eigen::Isometry3f *src_to_dest);

} // namespace perception

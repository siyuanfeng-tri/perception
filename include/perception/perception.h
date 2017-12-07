#pragma once

#include <Eigen/Geometry>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
//#include <pcl/segmentation/lccp_segmentation.h>
#include <pcl/registration/icp.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include "lcm/lcm-cpp.hpp"
//#include "point_cloud_registration.h"
#include <bot_core/pointcloud_t.hpp>

//#include "sensor_bridge/rgbd_bridge.h"

namespace perception {

struct Camera {
  Camera(Eigen::Affine3f _pose, double _fx, double _fy, double _ppx,
         double _ppy, int _img_height, int _img_width)
      : pose(_pose), fx(_fx), fy(_fy), ppx(_ppx), ppy(_ppy),
        img_height(_img_height), img_width(_img_width){};

  Eigen::Affine3f pose{Eigen::Affine3f::Identity()};
  double fx{}; // focal length (in pixel)
  double fy{};
  double ppx{}; // principle point (in pixel)
  double ppy{};
  int img_height{}; // number of rows.
  int img_width{};  // number of cols.
};

template <typename T>
void LoadPCDFile(const std::string &file_name,
                 typename pcl::PointCloud<T>::Ptr &cloud) {
  pcl::PCDReader reader;
  reader.read<T>(file_name, *cloud);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
}

template <typename T>
typename pcl::PointCloud<T>::Ptr
DownSample(const typename pcl::PointCloud<T>::ConstPtr &cloud,
           double leaf_size = 0.002) {
  pcl::VoxelGrid<T> grid;
  grid.setLeafSize(leaf_size, leaf_size, leaf_size);
  grid.setInputCloud(cloud);

  typename pcl::PointCloud<T>::Ptr tmp =
      boost::make_shared<typename pcl::PointCloud<T>>();
  grid.filter(*tmp);
  return tmp;
}

template <typename T>
typename pcl::PointCloud<T>::Ptr
SOROutlierRemoval(const typename pcl::PointCloud<T>::ConstPtr &cloud,
                  int num_neighbor = 50, double std_dev_threshold = 1.0) {
  pcl::StatisticalOutlierRemoval<T> sor;
  sor.setInputCloud(cloud);
  sor.setMeanK(num_neighbor);
  sor.setStddevMulThresh(std_dev_threshold);
  typename pcl::PointCloud<T>::Ptr tmp =
      boost::make_shared<typename pcl::PointCloud<T>>();
  sor.filter(*tmp);
  return tmp;
}

template <typename T>
typename pcl::PointCloud<T>::Ptr
RADOutlierRemoval(const typename pcl::PointCloud<T>::ConstPtr &cloud,
                  int num_neighbor, double radius) {
  pcl::RadiusOutlierRemoval<T> rad;
  rad.setInputCloud(cloud);
  rad.setRadiusSearch(radius);
  rad.setMinNeighborsInRadius(num_neighbor);
  typename pcl::PointCloud<T>::Ptr tmp =
      boost::make_shared<typename pcl::PointCloud<T>>();
  rad.filter(*tmp);
  return tmp;
}

template <typename T>
typename pcl::PointCloud<T>::Ptr
CutWithWorkSpaceConstraints(const typename pcl::PointCloud<T>::ConstPtr &cloud,
                            const Eigen::Vector3f &min_range,
                            const Eigen::Vector3f &max_range,
                            const Eigen::Isometry3f& transform = Eigen::Isometry3f::Identity()) {
  typename pcl::PointCloud<T>::Ptr ret =
      boost::make_shared<typename pcl::PointCloud<T>>();
  Eigen::Vector3f pt;
  Eigen::Vector3f local_pt;
  Eigen::Isometry3f inv = transform.inverse();

  for (size_t i = 0; i < cloud->size(); ++i) {
    pt(0) = cloud->points[i].x;
    pt(1) = cloud->points[i].y;
    pt(2) = cloud->points[i].z;
    local_pt = inv * pt;

    if ((local_pt(0) >= min_range(0)) &&
        (local_pt(0) <= max_range(0)) &&
        (local_pt(1) >= min_range(1)) &&
        (local_pt(1) <= max_range(1)) &&
        (local_pt(2) >= min_range(2)) &&
        (local_pt(2) <= max_range(2))) {
      ret->points.push_back(cloud->points[i]);
    }
  }
  ret->width = 1;
  ret->height = ret->points.size();
  return ret;
}

template <typename TN>
void FilterPointsWithEmptyNormals(typename pcl::PointCloud<TN>::Ptr &cloud) {
  int num_points = cloud->size();
  pcl::PointIndices::Ptr inliers = boost::make_shared<pcl::PointIndices>();
  for (int i = 0; i < num_points; ++i) {
    bool has_normal = true;
    for (int j = 0; j < 3; ++j) {
      // std::cout << cloud->points[i].normal[j] << ",";
      // Elastic fusion can sometimes return normals having compoenent slightly
      // larger than 1.0.
      if (!((cloud->points[i].normal[j] >= -2 &&
             cloud->points[i].normal[j] <= 2))) {
        has_normal = false;
      }
    }
    // std::cout << std::endl;
    if (has_normal) {
      inliers->indices.push_back(i);
    }
  }
  pcl::ExtractIndices<TN> extract;
  extract.setInputCloud(cloud);
  extract.setIndices(inliers);
  // Keep the inlier points;
  extract.setNegative(false);

  typename pcl::PointCloud<TN>::Ptr tmp =
      boost::make_shared<typename pcl::PointCloud<TN>>();
  extract.filter(*tmp);
  cloud.swap(tmp);
}

// Get the plane coeffcients. ax + by + cz + d = 0, returned in vector4d.
template <typename T>
Eigen::Vector4d FindPlane(const typename pcl::PointCloud<T>::ConstPtr &cloud,
                          pcl::PointIndices::Ptr &inliers,
                          double dist_threshold = 0.02) {
  pcl::ModelCoefficients coefficients;
  // Create the segmentation object
  pcl::SACSegmentation<T> seg;
  // Optional
  seg.setOptimizeCoefficients(true);
  // Mandatory
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(dist_threshold);

  seg.setInputCloud(cloud);
  seg.segment(*inliers, coefficients);

  Eigen::Vector4d plane_coefficients;
  for (size_t i = 0; i < coefficients.values.size(); ++i) {
    plane_coefficients(i) = coefficients.values[i];
  }
  return plane_coefficients;
}

template <typename T>
typename pcl::PointCloud<T>::Ptr SubtractTable(
    const typename pcl::PointCloud<T>::ConstPtr &cloud,
    double thickness = 0.01) {
  // Get rid of the table.
  pcl::PointIndices::Ptr inliers = boost::make_shared<pcl::PointIndices>();
  Eigen::Vector4d param = FindPlane<T>(cloud, inliers, thickness);
  std::cout << param.transpose() << "\n";
  if (std::fabs(param[2]) < 0.95) {
    inliers->indices.clear();
  }
  pcl::ExtractIndices<T> extract;
  extract.setInputCloud(cloud);
  extract.setIndices(inliers);
  extract.setNegative(true);

  typename pcl::PointCloud<T>::Ptr result =
      boost::make_shared<typename pcl::PointCloud<T>>();
  extract.filter(*result);
  return result;
}

template <typename T>
typename pcl::PointCloud<T>::Ptr SubtractPointsByColor(
    const typename pcl::PointCloud<T>::ConstPtr &cloud,
    uint8_t max_r, uint8_t max_g, uint8_t max_b) {
  // Get rid of the table.
  pcl::PointIndices::Ptr inliers = boost::make_shared<pcl::PointIndices>();
  for (size_t i = 0; i < cloud->size(); i++) {
    if (cloud->points[i].r < max_r &&
        cloud->points[i].g < max_g &&
        cloud->points[i].b < max_b) {
      inliers->indices.push_back(i);
    }
  }

  pcl::ExtractIndices<T> extract;
  extract.setInputCloud(cloud);
  extract.setIndices(inliers);
  extract.setNegative(true);

  typename pcl::PointCloud<T>::Ptr result =
      boost::make_shared<typename pcl::PointCloud<T>>();
  extract.filter(*result);
  return result;
}

template <typename T>
std::vector<int> SelectNearCentroidPoints(const pcl::PointCloud<T> &cloud,
                                          double cover_ratio) {
  const int num_pts = cloud.size();
  Eigen::Vector4f mid_pt;
  pcl::compute3DCentroid(cloud, mid_pt);

  // Compute average distance to the center.
  double avg_dist = 0;
  std::vector<double> all_dists(num_pts);
  std::vector<double> all_dists_cp(num_pts);
  for (int i = 0; i < num_pts; ++i) {
    Eigen::Vector4f pt;
    pt[0] = float(cloud.points[i].x);
    pt[1] = float(cloud.points[i].y);
    pt[2] = float(cloud.points[i].z);
    pt[3] = 0;
    all_dists[i] = (pt - mid_pt).norm();
    all_dists_cp[i] = (pt - mid_pt).norm();
    avg_dist = avg_dist + all_dists[i];
  }
  avg_dist = avg_dist / cloud.size();
  // Sort the distances and only keep the first 95% for computation of initial
  // bounding box. Update the middle point.
  std::sort(all_dists.begin(), all_dists.end());
  int index_threshold =
      std::min(int(floor(num_pts * cover_ratio)), num_pts - 1);
  double dist_threshold = all_dists.at(index_threshold);
  std::vector<int> indices;
  for (int i = 0; i < num_pts; ++i) {
    if (all_dists_cp[i] < dist_threshold) {
      indices.push_back(i);
    }
  }
  return indices;
}

template <typename T>
void FilterPointsBasedOnScatterness(typename pcl::PointCloud<T>::Ptr &cloud,
                                    double cover_ratio) {
  pcl::PointIndices::Ptr inliers = boost::make_shared<pcl::PointIndices>();
  inliers->indices = SelectNearCentroidPoints(*cloud, cover_ratio);
  pcl::ExtractIndices<T> extract;
  extract.setInputCloud(cloud);
  extract.setIndices(inliers);
  extract.setNegative(false);
  typename pcl::PointCloud<T>::Ptr tmp =
      boost::make_shared<typename pcl::PointCloud<T>>();
  extract.filter(*tmp);
  cloud.swap(tmp);
}

// Find a bounding box to the current point cloud with binary search.
// The reason for cover_ratio not equals to 1 is to be robust to outliers.
// Top left corner is (xmax, ymax, zmax), lower right corner is (xmin, ymin,
// zmin).
template <typename T>
void FindBoundingBox(const typename pcl::PointCloud<T>::ConstPtr &cloud,
                     Eigen::Vector3f *center, Eigen::Vector3f *top_right_corner,
                     Eigen::Vector3f *lower_left_corner,
                     Eigen::Matrix3f *orientation, double cover_ratio = 0.95) {
  // Eigen uses 4f for its interface.
  Eigen::Vector4f min_pt, max_pt;
  Eigen::Vector4f mid_pt;
  std::vector<int> remaining_index =
      SelectNearCentroidPoints(*cloud, cover_ratio);

  pcl::compute3DCentroid(*cloud, remaining_index, mid_pt);
  pcl::PCA<T> pca;
  pca.setInputCloud(cloud);
  Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();
  *orientation = eigen_vectors;
  eigen_vectors.col(2) = eigen_vectors.col(0).cross(eigen_vectors.col(1));

  std::cout << "PCAs:" << std::endl;
  std::cout << eigen_vectors << std::endl;
  pcl::PointCloud<T> cloud_projected;
  pca.project(*cloud, cloud_projected);

  pcl::getMinMax3D(cloud_projected, remaining_index, min_pt, max_pt);
  T pt_projected_min;
  T pt_projected_max;
  pt_projected_min.x = min_pt(0);
  pt_projected_min.y = min_pt(1);
  pt_projected_min.z = min_pt(2);
  pt_projected_max.x = max_pt(0);
  pt_projected_max.y = max_pt(1);
  pt_projected_max.z = max_pt(2);

  std::cout << pt_projected_min << std::endl;
  std::cout << pt_projected_max << std::endl;
  std::cout << "----------------" << std::endl;
  T pt_min, pt_max;
  pca.reconstruct(pt_projected_min, pt_min);
  pca.reconstruct(pt_projected_max, pt_max);
  std::cout << pt_min << std::endl;
  std::cout << pt_max << std::endl;

  *top_right_corner = Eigen::Vector3f(pt_max.x, pt_max.y, pt_max.z);
  *lower_left_corner = Eigen::Vector3f(pt_min.x, pt_min.y, pt_min.z);
  //*center = Eigen::Vector3f(mid_pt(0),mid_pt(1),mid_pt(2));
  *center = (*top_right_corner + *lower_left_corner) / 2.0;
}

template <typename T>
void VisualizePointCloud(
    const typename pcl::PointCloud<T>::ConstPtr &cloud,
    const Eigen::Affine3f &tf = Eigen::Affine3f::Identity()) {
  std::cout << "open up viewer" << std::endl;
  pcl::visualization::PCLVisualizer viewer("Point Cloud Visualization");
  viewer.addCoordinateSystem(0.2, tf);
  pcl::visualization::PointCloudColorHandlerRGBField<T> rgb(cloud);
  std::cout << "!!" << std::endl;
  viewer.addPointCloud(cloud, rgb, "Cloud");
  std::cout << "to display" << std::endl;
  // Display the point cloud until 'q' key is pressed.
  while (!viewer.wasStopped()) {
    viewer.spinOnce();
  }
}

template <typename T>
void VisualizePointCloudDrake(
    const pcl::PointCloud<T> &cloud, lcm::LCM *lcm,
    const Eigen::Isometry3d &X_WC = Eigen::Isometry3d::Identity(),
    const std::string &channel = "DRAKE_POINTCLOUD_RGBD") {
  bot_core::pointcloud_t message{};
  message.points.clear();
  message.frame_id = "world";
  message.n_points = cloud.points.size();
  message.points.resize(message.n_points);
  // See: director.drakevisualizer, DrakeVisualier.onPointCloud
  message.n_channels = 3;
  message.channel_names = {"r", "g", "b"};
  message.channels.resize(3, std::vector<float>(message.n_points));
  for (int i = 0; i < message.n_points; ++i) {
    const auto &point = cloud.points[i];
    message.channels[0][i] = point.r / 255.0;
    message.channels[1][i] = point.g / 255.0;
    message.channels[2][i] = point.b / 255.0;
    Eigen::Vector3f pt_W =
        (X_WC * Eigen::Vector3d(point.x, point.y, point.z)).cast<float>();
    message.points[i] = {pt_W[0], pt_W[1], pt_W[2]};
  }
  message.n_points = message.points.size();
  std::vector<uint8_t> bytes(message.getEncodedSize());
  message.encode(bytes.data(), 0, bytes.size());
  lcm->publish(channel, bytes.data(), bytes.size());
}

// Encodes the @p img as a .jpg, and sends it as a bot_core::raw_t message over
// @p channel.
void SendImageAsJpg(const cv::Mat &img, const std::string &channel,
                    lcm::LCM *lcm, int quality = 90);

// Depth image should be in units of mm, so default 1000 is 1m.
void ScaleDepthImage(const cv::Mat &img, cv::Mat *result, double max = 1000);

void VisualizePointCloudAndNormal(
    const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud,
    const pcl::PointCloud<pcl::Normal>::ConstPtr &normals,
    const Eigen::Affine3f &tf = Eigen::Affine3f::Identity());

void VisualizePointCloudAndNormal(
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr &cloud,
    const Eigen::Affine3f &tf = Eigen::Affine3f::Identity());

template <typename T, typename TN>
void SeparatePointsAndNormals(const pcl::PointCloud<TN> &points_and_normal,
                              pcl::PointCloud<T> &points,
                              pcl::PointCloud<pcl::Normal> &normals) {
  pcl::copyPointCloud(points_and_normal, points);
  pcl::copyPointCloud(points_and_normal, normals);
}

template <typename T>
void ApplyTransformToPointCloud(const Eigen::Affine3f &tf,
                                typename pcl::PointCloud<T>::Ptr &cloud) {
  typename pcl::PointCloud<T>::Ptr tmp =
      boost::make_shared<typename pcl::PointCloud<T>>();
  pcl::transformPointCloud(*cloud, *tmp, tf);
  cloud.swap(tmp);
}

// Points normals estimation. Assume the camera view is at world origin.
// If the points are organized, we use integral image for speeding up.
// Otherwise we use plain covariance matrix estimation with omp multi threading.
template <typename T>
void EstimateNormal(const typename pcl::PointCloud<T>::ConstPtr &cloud,
                    pcl::PointCloud<pcl::Normal>::Ptr &normals,
                    const Eigen::Vector3f &view_point = Eigen::Vector3f::Zero(),
                    double radius = 0.02) {
  // Assume the camera is at the world origin.
  // if (cloud->isOrganized()) {
  // 	std::cout << "Use integral image normal esimation" << std::endl;
  //   	pcl::IntegralImageNormalEstimation<T, pcl::Normal> ne;
  //   	ne.setViewPoint(0.0, 0.0, 0.0);
  //    ne.setNormalEstimationMethod(
  //    	pcl::IntegralImageNormalEstimation<T,
  //    pcl::Normal>::COVARIANCE_MATRIX);
  //    ne.setInputCloud(cloud);
  //    ne.setRectSize(10, 10);
  //    ne.compute(*normals);
  // } else {
  // Todo(jiaji): try out gpu based normal estimation.
  pcl::NormalEstimationOMP<T, pcl::Normal> ne;
  ne.setViewPoint(view_point(0), view_point(1), view_point(2));
  ne.setInputCloud(cloud);
  // boost::shared_ptr<pcl::KdTreeFLANN<T>> tree(new pcl::KdTreeFLANN<T>());
  typename pcl::search::KdTree<T>::Ptr tree =
      boost::make_shared<pcl::search::KdTree<T>>();
  ne.setSearchMethod(tree);
  // Set only one between the number of nearest neighbor and radius.
  // ne.setKSearch(10);
  ne.setRadiusSearch(radius);
  ne.compute(*normals);
  //}
  // Compute the right direction of normals assuming camera is at the world
  // origin and is viewing towards the positive z direction.
  /*
  for (size_t i = 0; i < cloud->size(); ++i) {
    double x, y, z, nx, ny, nz;
    x = cloud->points[i].x;
    y = cloud->points[i].y;
    z = cloud->points[i].z;
    nx = normals->points[i].normal_x;
    ny = normals->points[i].normal_y;
    nz = normals->points[i].normal_z;
    if ((x - view_point(0)) * nx + (y - view_point(1)) * ny +
            (z - view_point(2)) * nz >
        0) {
      normals->points[i].normal_x = -nx;
      normals->points[i].normal_y = -ny;
      normals->points[i].normal_z = -nz;
    }
  }
  */
}

template <typename PixelType, typename AccumulatePixelType>
PixelType get_avg_pixel(const cv::Mat &img, int radius, int px, int py) {
  PixelType zero{};
  AccumulatePixelType avg{};
  int ctr = 0;

  for (int x = -radius; x <= radius; x++) {
    for (int y = -radius; y <= radius; y++) {
      if (x * x + y + y > radius * radius)
        continue;

      const int x_idx = x + px;
      const int y_idx = y + py;
      if (x_idx > 0 && x_idx < img.cols && y_idx > 0 && y_idx < img.rows) {
        if (img.at<PixelType>(y_idx, x_idx) != zero) {
          avg += img.at<PixelType>(y_idx, x_idx);
          ctr++;
        }
      }
    }
  }
  if (ctr != 0)
    return avg / ctr;
  return avg;
}

template <typename PixelType>
cv::Mat FillImgHoles(const cv::Mat &img, int fill_size) {
  cv::Mat ret = img;

  for (int x = 0; x < img.cols; x++) {
    for (int y = 0; y < img.rows; y++) {
      ret.at<cv::Vec3b>(y, x) =
          get_avg_pixel<cv::Vec3b, cv::Vec3i>(img, fill_size, x, y);
    }
  }

  return ret;
}

template <typename PointType>
cv::Mat ProjectCloudToImage(const pcl::PointCloud<PointType> &cloud,
                            const Camera &camera, int fill_size = 0) {
  const cv::Vec3b zero(0, 0, 0);
  // OpenCV is row-major allocation.
  cv::Mat projected_cv_img(camera.img_height, camera.img_width, CV_8UC3);
  projected_cv_img.setTo(zero);

  const int num_points = cloud.size();

  for (int i = 0; i < num_points; ++i) {
    Eigen::Vector3f point_in_camera(cloud.points[i].x, cloud.points[i].y,
                                    cloud.points[i].z);
    point_in_camera = camera.pose.inverse() * point_in_camera;

    int row_id =
        round(camera.ppy + point_in_camera(1) / point_in_camera(2) * camera.fy);
    int col_id =
        round(camera.ppx + point_in_camera(0) / point_in_camera(2) * camera.fx);

    if (row_id > 0 && row_id < camera.img_height && col_id > 0 &&
        col_id < camera.img_width) {
      projected_cv_img.at<cv::Vec3b>(row_id, col_id) =
          cv::Vec3b(cloud.points[i].b, cloud.points[i].g, cloud.points[i].r);
    }
  }

  cv::Mat filled_img = FillImgHoles<cv::Vec3b>(projected_cv_img, fill_size);
  return filled_img;
}

template <typename PointType>
double AlignCloud(const typename pcl::PointCloud<PointType>::ConstPtr &src,
                  const typename pcl::PointCloud<PointType>::ConstPtr &dest,
                  double max_correspondence_dist,
                  typename pcl::PointCloud<PointType>::Ptr &aligned_src,
                  Eigen::Isometry3f *src_to_dest) {
  pcl::IterativeClosestPoint<PointType, PointType> icp;
  icp.setInputSource(src);
  icp.setInputTarget(dest);

  icp.setMaxCorrespondenceDistance(max_correspondence_dist);
  icp.setMaximumIterations(50);
  icp.setTransformationEpsilon(1e-8);
  // Magic number need to set it to something to stop icp from going crazy and
  // "diverge".
  icp.setEuclideanFitnessEpsilon(1);
  icp.align(*aligned_src);
  src_to_dest->matrix() = icp.getFinalTransformation();

  if (!icp.hasConverged())
    return -1;
  return icp.getFitnessScore();
}

/*
void StreamImagesAsJpgLoop(const sensor_bridge::RGBDSensor *driver,
                           const sensor_bridge::ImageType rgb_type,
                           const sensor_bridge::ImageType depth_type,
                           const std::string rgb_name,
                           const std::string depth_name, lcm::LCM *lcm);
*/
} // namespace perception

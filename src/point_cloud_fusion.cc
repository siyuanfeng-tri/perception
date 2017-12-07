#include "perception/point_cloud_fusion.h"
#include "perception/perception.h"

namespace perception {

template <typename T>
T bilinear_interp(const T x, const T y,
    const T x1, const T x2, const T y1, const T y2,
    const T Q11, const T Q12, const T Q21, const T Q22) {
  Eigen::Matrix<T, 2, 2> Q;
  Eigen::Matrix<T, 1, 2> X;
  Eigen::Matrix<T, 2, 1> Y;

  Q << Q11, Q12, Q21, Q22;
  X << x2 - x, x - x1;
  Y << y2 - y, y - y1;

  return (X * Q * Y)(0, 0) / (x2 - x1) / (y2 - y1);
}

PointCloudFusion::PointCloudFusion(
    std::function<Eigen::Vector2f(const Eigen::Vector3f &)> func,
    float resolution)
    : proj_func_(func), resolution_(resolution) {
  fused_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();
  Init();
}

void PointCloudFusion::Init() {
  std::unique_lock<std::mutex> lock1(lock_);
  fused_->clear();
  world_update_.setIdentity();
}

bool PointCloudFusion::is_uv_in_image(
    const Eigen::Vector2f& uv, const cv::Mat& img) {
  Eigen::Vector2i uv_low(std::floor(uv(0)), std::floor(uv(1)));
  Eigen::Vector2i uv_high(std::ceil(uv(0)), std::ceil(uv(1)));
  return uv_low(0) >= 0 && uv_low(0) < img.cols &&
         uv_low(1) >= 0 && uv_low(1) < img.rows &&
         uv_high(0) >= 0 && uv_high(0) < img.cols &&
         uv_high(1) >= 0 && uv_high(1) < img.rows;
}

uint16_t PointCloudFusion::get_interp_val(
    const Eigen::Vector2f& uv, const cv::Mat& img) {
  Eigen::Vector2i uv_low(std::floor(uv(0)), std::floor(uv(1)));
  Eigen::Vector2i uv_high(std::ceil(uv(0)), std::ceil(uv(1)));
  uint16_t val11 = img.at<uint16_t>(uv_low(1), uv_low(0));
  uint16_t val12 = img.at<uint16_t>(uv_low(1), uv_high(0));
  uint16_t val21 = img.at<uint16_t>(uv_high(1), uv_low(0));
  uint16_t val22 = img.at<uint16_t>(uv_high(1), uv_high(0));

  float val = bilinear_interp<float>(uv(1), uv(0),
      uv_low(1), uv_high(1), uv_low(0), uv_high(0),
      val11, val12, val21, val22);

  return static_cast<uint16_t>(val);
}

bool PointCloudFusion::ProcFrame(
    const Eigen::Isometry3f &X_WC,
    const pcl::PointCloud<pcl::PointXYZRGB> &raw_cloud,
    const cv::Mat &raw_depth) {
  if (raw_cloud.size() < 3000)
    return false;

  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr current =
      boost::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();

  // Make a point cloud with normal.
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr points =
        boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    pcl::PointCloud<pcl::Normal>::Ptr normals =
        boost::make_shared<pcl::PointCloud<pcl::Normal>>();

    pcl::transformPointCloud(raw_cloud, *points, X_WC);
    points = perception::DownSample<pcl::PointXYZRGB>(points, resolution_);
    points = perception::RADOutlierRemoval<pcl::PointXYZRGB>(points, 20, 0.01);
    //points = perception::SOROutlierRemoval<pcl::PointXYZRGB>(points, 20);
    perception::EstimateNormal<pcl::PointXYZRGB>(points, normals,
                                                 X_WC.translation(), 0.02);
    pcl::concatenateFields<pcl::PointXYZRGB, pcl::Normal,
                           pcl::PointXYZRGBNormal>(*points, *normals, *current);

    //pcl::io::savePCDFileASCII<pcl::PointXYZRGBNormal>(
    //    "raw_nomal" + std::to_string(ctr++) + ".pcd", *current);
  }

  // First tick.
  std::unique_lock<std::mutex> lock1(lock_);
  if (fused_->size() == 0) {
    fused_ = current;
    world_update_.setIdentity();
    return true;
  }
  lock1.unlock();

  pcl::PointCloud<pcl::PointXYZRGBNormal> transformed_fused;
  pcl::IterativeClosestPoint<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>
      icp;

  // Only do icp on really down sampled cloud.
  icp.setInputSource(
      perception::DownSample<pcl::PointXYZRGBNormal>(fused_, 0.005));
  icp.setInputTarget(
      perception::DownSample<pcl::PointXYZRGBNormal>(current, 0.005));
  // icp.setInputSource(fused_);
  // icp.setInputTarget(current);

  icp.setMaxCorrespondenceDistance(0.02);
  icp.setMaximumIterations(50);
  icp.setTransformationEpsilon(1e-8);
  // Magic number need to set it to something to stop icp from going crazy and
  // "diverge".
  icp.setEuclideanFitnessEpsilon(1);
  icp.align(transformed_fused);
  Eigen::Isometry3f update;
  update.matrix() = icp.getFinalTransformation();

  /*
  std::cout << "icp has converged:" << icp.hasConverged()
            << " score: " << icp.getFitnessScore() << std::endl;
  */
  if (!icp.hasConverged())
    return false;

  // Fuse and filter phantom points on the dense clouds, using the
  // transforamtion from icp.
  pcl::transformPointCloudWithNormals(*fused_, transformed_fused, update);

  std::vector<int> valid_idx(transformed_fused.size());
  int valid_ctr = 0;

  Eigen::Isometry3f X_CW = X_WC.inverse();
  for (size_t n = 0; n < transformed_fused.size(); n++) {
    Eigen::Vector3f xyz(transformed_fused[n].x, transformed_fused[n].y,
                        transformed_fused[n].z);
    xyz = X_CW * xyz;

    Eigen::Vector2f uv = proj_func_(xyz);

    if (is_uv_in_image(uv, raw_depth)) {
      // xyz is in [m], raw depth is in [mm]
      uint16_t proj_depth = (uint16_t)(xyz[2] * 1000.);
      uint16_t real_depth = get_interp_val(uv, raw_depth);

      if ((proj_depth < real_depth - 50)) {
        // valid_idx[valid_ctr++] = n;
      } else {
        valid_idx[valid_ctr++] = n;
      }
    } else {
      valid_idx[valid_ctr++] = n;
    }
  }
  valid_idx.resize(valid_ctr);

  // Fuse.
  (*current) +=
      pcl::PointCloud<pcl::PointXYZRGBNormal>(transformed_fused, valid_idx);
  auto down_sampled =
      perception::DownSample<pcl::PointXYZRGBNormal>(current, resolution_);

  lock1.lock();
  fused_ = down_sampled;
  world_update_ = update;
  return true;
}

} // namespace perception

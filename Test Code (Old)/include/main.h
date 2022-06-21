#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/file_io.h>
#include <pcl/io/ply/ply_parser.h>
#include <pcl/io/ply/ply.h>

#include <pcl/point_types.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>

#include <pcl/common/transforms.h>
#include <pcl/common/geometry.h>
#include <pcl/common/common.h>
#include <pcl/common/common_headers.h>

#include <pcl/ModelCoefficients.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/gasd.h>
#include <pcl/features/normal_3d_omp.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing.h>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>

#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <filesystem>
#include <functional>
#include <regex>
#include <Eigen/Dense>
#include <regex>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <vtkPlaneSource.h>

#include "plyReader.h"

#define INAVLID_HEIGHT 55.555555
#define DEBUG 0

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudFiltered(new pcl::PointCloud<pcl::PointXYZRGB>());
pcl::PointCloud<pcl::PointXYZ>::Ptr cloudXYZ(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud <pcl::PointXYZRGB>::Ptr coloured_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOutliers(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::Normal>::Ptr cloudNormal(new pcl::PointCloud<pcl::Normal>());

typedef struct {
	float x;
	float y;
	float width;
	float height;
	int id;
	int r, g, b;
	float average_area;
	std::vector<cv::Rect> bounding_boxes;
} bounded_box_data;

inline void printUsage(const char* progName) {
	std::cout << "\nUsage: " << progName << " <input cloud> <leaf size>" << std::endl;
}

void downSample(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloudFiltered,
	float leafSize);

void removeOutliers(
	pcl::PointCloud<pcl::PointXYZ>::Ptr& inputCloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& outCloud
);

void calculateNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr& inputCloud,
	pcl::PointCloud<pcl::Normal>::Ptr& outputCloud);

int ransacSegmentation(
	pcl::PointCloud<pcl::Normal>::Ptr &cloudNormal,
	pcl::PointCloud<pcl::PointXYZ>::Ptr &cloudXYZ
);

void planeFitting(
	pcl::PointCloud<pcl::Normal>::Ptr &cloudNormal,
	pcl::PointCloud<pcl::PointXYZ>::Ptr &cloudXYZ
);

void regionSegmentation(
	pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
	pcl::PointCloud<pcl::Normal>::Ptr &cloudNormal,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloudOut,
	std::vector<pcl::PointIndices>& pointIndices
);

void pp_callback(const pcl::visualization::PointPickingEvent& event, void* viewer_void);

void computeInliersRANSAC(
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_xyz,
	pcl::PointCloud<pcl::Normal>::Ptr& cloud_normal,
	pcl::PointIndices& cluster
);

cv::Mat generateHeightMap(
	std::vector<int>& points,
	float adjusted_width_x, float adjusted_width_y, float step,
	float min_x, float min_y, float min_alt, float max_alt
);

bool load_ply(
	const std::string& input_filename,
	std::vector<std::string>& comments
);

void returnPointsFromImage(
	std::vector<int>& points,
	std::vector<int>& points_bb,
	cv::Mat& img,
	bounded_box_data& bounding_box,
	float min_x, float min_y, float step
);
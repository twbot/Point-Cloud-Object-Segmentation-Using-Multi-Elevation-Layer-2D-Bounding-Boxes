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

typedef struct {
	float x;
	float y;
	float width;
	float height;
	int id;
	int r, g, b;
	float average_area;
} bounded_box_data;

void printUsage(const char* progName) {
	std::cout << "\nUsage: " << progName << " <input cloud> <leaf size>" << std::endl;
}

void downSample(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloudFiltered,
	float leafSize)
{

	pcl::VoxelGrid<pcl::PointXYZRGB> sor;
	// pcl::toPCLPointCloud2(cloud, point_cloud2);
	sor.setInputCloud(cloud);
	sor.setLeafSize(leafSize, leafSize, leafSize); // was 0.85f
	sor.filter(*cloudFiltered);


	std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height
		<< " data points (" << pcl::getFieldsList(*cloud) << ")." << std::endl;
	std::cerr << "PointCloud after filtering: " << cloudFiltered->width * cloudFiltered->height
		<< " data points (" << pcl::getFieldsList(*cloudFiltered) << ")." << std::endl;
}

void removeOutliers(
	pcl::PointCloud<pcl::PointXYZ>::Ptr& inputCloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& outCloud	
)
{
	std::cout << "Using statistical outlier removal..." << std::endl;
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
	sor.setInputCloud(inputCloud);
	sor.setMeanK(100);
	sor.setStddevMulThresh(1.0);
	sor.filter(*outCloud);
	inputCloud = outCloud;
}

void calculateNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr& inputCloud,
	pcl::PointCloud<pcl::Normal>::Ptr& outputCloud)
{
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZ>);
	kdTree->setInputCloud(inputCloud);

	//Normal Estimation
	std::cout << "Using normal method estimation...";
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> estimator;
	estimator.setNumberOfThreads(3);
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	estimator.setInputCloud(inputCloud);
	estimator.setSearchMethod(kdTree);
	estimator.setKSearch(40); //It was 20
	estimator.compute(*normals);//Normals are estimated using standard method.

	// pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal> ());
	//pcl::concatenateFields(*inputCloud, *normals, *outputCloud);
	pcl::copyPointCloud(*normals, *outputCloud);

	std::cout << "Normal Estimation...[OK]" << std::endl;
}

int ransacSegmentation(
	pcl::PointCloud<pcl::Normal>::Ptr &cloudNormal,
	pcl::PointCloud<pcl::PointXYZ>::Ptr &cloudXYZ
) {
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	// Create the segmentation object
	pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
	seg.setInputNormals(cloudNormal);
	//pcl::SACSegmentation<pcl::PointXYZ> seg;
	// Optional
	seg.setOptimizeCoefficients(true);
	// Mandatory
	seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
	Eigen::Vector3f axis = Eigen::Vector3f(1.0, 0.0, 0.0);
	seg.setAxis(axis);
	seg.setEpsAngle(0.0f * (M_PI / 180.0f));
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(2000);
	seg.setDistanceThreshold(0.21);
	int id = 1;

	while (cloudXYZ->points.size() > 120000) {
		boost::shared_ptr<pcl::visualization::PCLVisualizer> view(new pcl::visualization::PCLVisualizer);
		view->addPointCloud(cloudXYZ);
		seg.setInputCloud(cloudXYZ);
		seg.segment(*inliers, *coefficients);

		if (inliers->indices.size() == 0)
		{
			PCL_ERROR("Could not estimate a planar model for the given dataset.");
			return (-1);
		}

		//Plane Model: ax+by+cz+d=0; saved in *coefficients

		float a, b, c, d;
		a = coefficients->values[0];
		b = coefficients->values[1];
		c = coefficients->values[2];
		d = coefficients->values[3];

		int i, j, k;
		int sizeIndices = inliers->indices.size();
		int sizeCoefficients = coefficients->values.size();
		//------------------------ Projection ------------------------
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_proj(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		pcl::ExtractIndices <pcl::Normal> extractNormals;
		pcl::PointXYZ point;
		for (i = 0; i < sizeIndices; i++)
		{
			int value = inliers->indices[i];
			float x, y, z;
			x = cloudXYZ->at(value).x;
			y = cloudXYZ->at(value).y;
			z = cloudXYZ->at(value).z;
			//std::cout << inliers->indices[i] << " ";
			point.x = x;
			point.y = y;
			point.z = z;
			/*point.x = ((b*b + c * c)*x - a * (b*y + c * z + d)) / (a*a + b * b + c * c);
			point.y = ((a*a + c * c)*y - b * (a*x + c * z + d)) / (a*a + b * b + c * c);
			point.z = ((b*b + a * a)*z - c * (a*x + b * y + d)) / (a*a + b * b + c * c);*/
			cloud_proj->push_back(point);
		}
		int r = rand() % 255;
		int g = rand() % 255;
		int bv = rand() % 255;
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(cloud_proj, r, g, bv);

		extract.setInputCloud(cloudXYZ);
		extract.setIndices(inliers);
		extract.setNegative(true);
		extract.filter(*cloudXYZ);

		extractNormals.setInputCloud(cloudNormal);
		extractNormals.setIndices(inliers);
		extractNormals.setNegative(true);
		extractNormals.filter(*cloudNormal);

		view->addPointCloud(cloud_proj, color_handler, std::to_string(id));
		view->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, std::to_string(id));
		id += 1;
		std::cout << "Cloud point size: " << cloudXYZ->points.size() << std::endl;
		while (!view->wasStopped())
		{
			view->spinOnce(100);
		}

		system("pause");
	}
}

void planeFitting(
	pcl::PointCloud<pcl::Normal>::Ptr &cloudNormal,
	pcl::PointCloud<pcl::PointXYZ>::Ptr &cloudXYZ
) {
	// Start timer
	const auto timeBegin = std::chrono::steady_clock::now();

	pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
	pcl::IndicesPtr indices(new std::vector <int>);
	pcl::PassThrough<pcl::PointXYZ> pass;
	pass.setFilterFieldName("z");
	pass.setFilterLimits(0.0, 1.0);
	pass.filter(*indices);

	reg.setMinClusterSize(20);
	reg.setMaxClusterSize(10000);
	reg.setSearchMethod(tree);
	reg.setNumberOfNeighbours(30);
	reg.setInputCloud(cloudXYZ);
	//reg.setIndices (indices);
	reg.setInputNormals(cloudNormal);
	reg.setSmoothnessThreshold(4.2 / 180.0 * M_PI);
	reg.setResidualThreshold(3.2);
	reg.setCurvatureThreshold(1.2);

	std::vector <pcl::PointIndices> clusters;
	reg.extract(clusters);


	pcl::PointCloud <pcl::PointXYZRGB>::Ptr coloured_cloud = reg.getColoredCloud();
	pcl::io::savePLYFileBinary("../coloured_cloud.ply", *coloured_cloud);
	boost::shared_ptr<pcl::visualization::PCLVisualizer> view(new pcl::visualization::PCLVisualizer);
	view->addPointCloud(coloured_cloud);

	// End timer
	const auto timeStop = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds> (timeStop - timeBegin);
	std::cout << "Time Duration: " << duration.count() << " seconds" << std::endl;

	while (!view->wasStopped())
	{
		view->spinOnce(100);
	}

	system("pause");
}

void regionSegmentation(
	pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
	pcl::PointCloud<pcl::Normal>::Ptr &cloudNormal,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloudOut,
	std::vector<pcl::PointIndices>& pointIndices
)
{
	pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
	pcl::IndicesPtr indices(new std::vector <int>);
	pcl::PassThrough<pcl::PointXYZ> pass;
	pass.setFilterFieldName("z");
	pass.setFilterLimits(0.0, 1.0);
	pass.filter(*indices);

	reg.setMinClusterSize(50);
	reg.setMaxClusterSize(100000000);
	reg.setSearchMethod(tree);
	reg.setNumberOfNeighbours(50);
	reg.setInputCloud(cloud);
	reg.setInputNormals(cloudNormal);
	// Max angle between normals (in radians converted to degrees)
	reg.setSmoothnessThreshold(4.1 / 180.0 * M_PI);
	reg.setResidualThreshold(1.2);
	reg.setCurvatureThreshold(0.5);

	reg.extract(pointIndices);
	cloudOut = reg.getColoredCloud();
}

void pp_callback(const pcl::visualization::PointPickingEvent& event, void* viewer_void)
{
	if (event.getPointIndex() == -1)
	{
		return;
	}
	std::cout << event.getPointIndex();
}

void computeInliersRANSAC(
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_xyz,
	pcl::PointCloud<pcl::Normal>::Ptr& cloud_normal, 
	pcl::PointIndices& cluster
) {
	int cluster_size = cluster.indices.size();
	int min_points = cluster_size * (2.0f / 3.0f);
	const auto partition = [cluster, cluster_size](
		int min_points
	) {
		pcl::PointIndices linVec;
		int step = ceil(min_points / 2);
		for (int i = 0; i <= cluster_size - 1; step++) {
			linVec.indices.push_back(cluster.indices[i]);
		}
		return linVec;
	};
	int iterations = 0;
	
	while (iterations < 30) {
		pcl::PointIndices idxs, test_idxs = partition(min_points);
		std::vector<pcl::PointNormal> data, test_data;
		for (int i = 0; i < idxs.indices.size(); i++) {
			pcl::PointXYZ point = cloud_xyz->points[idxs.indices[i]];
			pcl::Normal pointNormal = cloud_normal->points[idxs.indices[i]];
			pcl::PointNormal pt; pt.x = point.x; pt.y = point.y; pt.z = point.z;
			pt.normal_x = pointNormal.normal_x; pt.normal_y = pointNormal.normal_y; pt.normal_z = pointNormal.normal_z;
			data.push_back(pt); test_data.push_back(pt);
		}
	}
}

cv::Mat generateHeightMap(
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	float adjusted_width_x, float adjusted_width_y, float step,
	float min_x, float min_y, float min_alt, float max_alt
)
{
	//std::cout << "Generating height map..." << std::endl;
	// Convert point to grid coordinate system
	const auto convert = [](
		float measure, float min, float step
		) {
		float adjustedMeasure = (measure - min) / step;
		return adjustedMeasure;
	};

	// Normalize elevation value between 0 and 255
	const auto convertAlt = [](
		float alt,
		float min_alt,
		float max_alt
		)
	{
		float val = (alt - min_alt) / (max_alt - min_alt);
		return val * 255;
	};

	float currAlt;
	cv::Mat matPlaneMask(cv::Size(adjusted_width_y, adjusted_width_x), CV_8UC1, 0.0);

	for (int i = 0; i < cloud->points.size(); i++) {
		currAlt = cloud->points[i].z;

		float row = convert(cloud->points[i].y, min_y, step);
		float col = convert(cloud->points[i].x, min_x, step);
		int alt = convertAlt(currAlt, min_alt, max_alt);
#if DEBUG
		if (i < 100) {
			std::cout << "X: " << cloud->points[i].x << " Y: " << cloud->points[i].y << std::endl;
			std::cout << "Row: " << row << " col: " << col << std::endl;
			std::cout << "Curr Alt: " << currAlt << " Alt: " << alt << std::endl;
			std::cout << "====================" << std::endl;
		}
#endif
		if (row >= 0 && col >= 0 && row < adjusted_width_y && col < adjusted_width_x) {
			if (matPlaneMask.at<uchar>(row, col) < alt) {
				matPlaneMask.at<uchar>(row, col) = alt;
			}
		}
	}
	return matPlaneMask;
}

bool load_ply(
	const std::string& input_filename,
	std::vector<std::string>& comments
)
{
	try
	{
		pcl::console::print_highlight("Loading "); std::cout << input_filename << std::endl;

		std::ifstream ss_temp(input_filename, std::ios::binary);
		plyReader::PlyFile file_template(ss_temp);

		comments = file_template.comments;

		file_template.read(ss_temp);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: Could not load " << input_filename << ". " << e.what() << std::endl;
		return false;
	}
	return true;
}

void returnPointsFromImage(
	std::vector<int>& points,
	cv::Mat& img,
	bounded_box
)
{

}

int main(int argc, char** argv) {

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudFiltered(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudXYZ(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud <pcl::PointXYZRGB>::Ptr coloured_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOutliers(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::Normal>::Ptr cloudNormal(new pcl::PointCloud<pcl::Normal>());
	 
	// point clouds for 

	// File list and types
	std::vector<int> filenames;
	srand(time(NULL));

	if (argc < 3 || argc > 3) {
		printUsage(argv[0]);
		return -1;
	}

	pcl::console::TicToc tt;
	pcl::console::print_highlight("Loading ");
	filenames = pcl::console::parse_file_extension_argument(argc, argv, ".ply");
	if (filenames.size() == 1)
	{
		pcl::io::loadPLYFile(argv[filenames[0]], *cloud);
	}
	pcl::console::print_info("\nFound ply file.");
	pcl::console::print_info("[done, ");
	pcl::console::print_value("%g", tt.toc());
	pcl::console::print_info(" ms : ");
	pcl::console::print_value("%d", cloud->size());
	pcl::console::print_info(" points]\n");


	cloud->width = (int)cloud->points.size();
	cloud->height = 1;
	cloud->is_dense = true;

	if (cloud->height == 1) {
		pcl::console::print_info("Point cloud is unorganized\n");
	}
	else {
		pcl::console::print_info("Point cloud is organized\n");
	}
	std::vector<std::string> header;

	if (!load_ply(argv[filenames[0]], header))
	{
		std::cout << "Error: Could not load ply file." << std::endl;
		return 1;
	}
	else {
		std::cout << "\nLoaded PLY Files[done, " << tt.toc() << " ms]" << std::endl;
	}
	std::regex regexp("^Geometry, ([X-Z])=(-?[0-9]*\.[0-9]+):(-?[0-9]*\.[0-9]+):(-?[0-9]*\.[0-9]+)");
	std::smatch match;
	float min_x, min_y, max_x, max_y;
	float min_alt, max_alt, step;
	float adjusted_width_x, adjusted_width_y;

	for (int j = 0; j < header.size(); j++) {
		std::regex_search(header[j], match, regexp);
		std::string coord = match.str(1);
		if (coord == "X") {
			min_x = stof(match.str(2));
			max_x = stof(match.str(4));
			step = stof(match.str(3));
		}
		if (coord == "Y") {
			min_y = stof(match.str(2));
			max_y = stof(match.str(4));
		}
		if (coord == "Z") {
			min_alt = stof(match.str(2));
			max_alt = stof(match.str(4));
		}
	}
	adjusted_width_x = (float)((max_x - min_x) / step);
	adjusted_width_y = (float)((max_y - min_y) / step);

	std::string select_leaf_size = argv[2];
	float leaf_size = std::atof(select_leaf_size.c_str());
	// Downsample point cloud using given leaf size
	downSample(cloud, cloudFiltered, leaf_size);
	pcl::copyPointCloud(*cloudFiltered, *cloudXYZ);
	cloudFiltered->clear();
	//cloud->clear();

	// Filter outliers using point neighbourhood statistics
#if 0
	removeOutliers(cloudOutliers, cloudXYZ);
#endif
	// Calculate normals using OMP
	calculateNormals(cloudXYZ, cloudNormal);

	// Start timer
	const auto timeBegin = std::chrono::steady_clock::now();

	//new segmentation method (in testing)
	//how this segmentation method will work
	//1. Loop through the point cloud and group all 
	std::map<float, std::vector<int>> pointMap;
	int max_alt_found = std::numeric_limits<int>::min();
	int min_alt_found = std::numeric_limits<int>::max();
	for (int i = 0; i < cloudXYZ->points.size(); i++)
	{
		int key = ceil(cloudXYZ->points[i].z);
		if (pointMap.count(key) == 0) {
			std::vector<int> indices;
			indices.push_back(i);
			pointMap.insert({ key, indices });
			if (max_alt_found < key)
				max_alt_found = key;
			if (min_alt_found > key)
				min_alt_found = key;
		}
		else {
			(pointMap.at(key)).push_back(i);
		}
		
		/*if (key > max_alt) {
			max_alt = key;
		}
		if (key < min_alt) {
			min_alt = key;
		}*/
	}

	// detect when the data starts to become more clustered (i.e. for city, when we start to reach ground level)
	int largest_difference = 0;
	int change_alt = 0;
	std::cout << max_alt_found << " " << min_alt_found << std::endl;
	int bin_size = 8;
	for (int elev = min_alt_found; elev < max_alt_found; elev += bin_size) {
		int total_bin_size = (pointMap.at(elev)).size();
		int total_bin_size_next = (pointMap.at(elev + bin_size)).size();
		for (int j = 1; j < bin_size; j++)
		{
			if(elev + bin_size < max_alt_found)
				total_bin_size += (pointMap.at(elev + j)).size();
			if(elev + bin_size + j < max_alt_found)
				total_bin_size_next += (pointMap.at(elev + bin_size + j)).size();
		}
		int difference = total_bin_size - total_bin_size_next;
		std::cout << "Difference: " << difference << std::endl;
		std::cout << "Index: " << elev << std::endl;
		if (difference > largest_difference)
		{
			largest_difference = difference;
			change_alt = elev + 1;
			std::cout << "Heerrrreee" << std::endl;
		}
	}

	std::cout << "Largest difference is at alt: " << change_alt << std::endl;

	//Bounded boxes
	std::vector<bounded_box_data> bounded_boxes;

	//Objects
	std::map<int, std::vector<int>> objects;

	const auto margin = [](
		int val, int check
	)
	{
		int val_margin = 5;
		bool within = false;
		if (val > check - val_margin
			&& val < check + val_margin) {
			within = true;
		}
		return within;
	};

	std::map<float, std::vector<int>>::reverse_iterator itr;
	int index = 0;
	int dilationSize = 35;
	int exclusionCloseSize = 60;
	int exclusionOpenSize = 20;

	cv::Mat bb_image(cv::Size(adjusted_width_y, adjusted_width_x), CV_8UC3, 0.0);

	std::cout << "PointMap size: " << pointMap.size() << std::endl;
	for (itr = pointMap.rbegin(); itr != pointMap.rend(); ++itr) {

		// Add points to temp point cloud
		pcl::PointCloud<pcl::PointXYZ>::Ptr new_alt_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		std::vector<int> checkSize;
		if (itr->second.size() > checkSize.max_size())
		{
			std::cout << "Index: " << index << std::endl;
			std::cout << "Vector max size: " << checkSize.max_size() << std::endl;
			std::cout << "Vector too large! Size is: " << itr->second.size() << " " << std::endl;
		}
		std::vector<int> points = itr->second;
		for (int i = 0; i < points.size(); i++) {

			new_alt_cloud->points.push_back(cloudXYZ->points[points[i]]);

		}
		if (index % 20 == 0) {
			if (dilationSize > 25) {
				dilationSize -= 10;
			}
			if (exclusionOpenSize > 15) {
				exclusionOpenSize -= 1;
			}
		}

		// Generate heightmap from pointcloud
		cv::Mat heightMap = generateHeightMap(new_alt_cloud, adjusted_width_x, adjusted_width_y,
			step, min_x, min_y, min_alt, max_alt);
		new_alt_cloud->clear();
		// Get contours from heightmap
		cv::Mat img, img2, morphImg;
		cv::Mat se = getStructuringElement(cv::MORPH_RECT, cv::Size(dilationSize, dilationSize));
		cv::dilate(heightMap, img, se);
		cv::Mat seMorph = getStructuringElement(cv::MORPH_RECT, cv::Size(exclusionCloseSize, exclusionCloseSize));
		cv::morphologyEx(img, img2, cv::MORPH_CLOSE, exclusionCloseSize);
		//cv::imwrite("C:\\Users\\TskyDroneOps\\Desktop\\meshTest\\meshPCL\\src\\dilated_img.jpg", img);
		seMorph = getStructuringElement(cv::MORPH_RECT, cv::Size(exclusionOpenSize, exclusionOpenSize));
		cv::morphologyEx(img2, morphImg, cv::MORPH_OPEN, seMorph);
		//cv::cvtColor(morphImg, morphImg, cv::COLOR_GRAY2BGR);

		// Find all distinct objects in image
		cv::Mat labels, stats, centroids;
   		cv::connectedComponentsWithStats(morphImg, labels, stats, centroids);

		for (int i = 0; i < stats.rows; i++)
		{
			int x = stats.at<int>(cv::Point(0, i));
			int y = stats.at<int>(cv::Point(1, i));
			int w = stats.at<int>(cv::Point(2, i));
			int h = stats.at<int>(cv::Point(3, i));

			if (w > 10 && h > 10 &&
				w < heightMap.rows && h < heightMap.cols) {
				cv::Rect rect(x, y, w, h);
				bounded_box_data bb; bb.x = x; bb.y = y; bb.width = w; bb.height = h;
				bb.id = std::rand();
				if (bounded_boxes.size() > 0)
				{
					cv::Point2d centroid((x + w) / 2, (y + h) / 2);
					bool bounded_box_exists = true;
					for (int j = 0; j < bounded_boxes.size(); j++) {
						bounded_box_data curr_bb = bounded_boxes[j];
						cv::Point2d centroid_bounded_box(
							(curr_bb.x + curr_bb.width) / 2,
							(curr_bb.y + curr_bb.height) / 2
						);
						float bb_area = w * h;
						float curr_bb_area = curr_bb.width * curr_bb.height;
						// Calculate weighted average
						float w1 = curr_bb_area > bb_area ? 0.8 : 0.2; float w2 = 1 - w1;
						float average_area = (w1*curr_bb_area + w2 * bb_area);

						float centroid_distance = sqrt((pow(centroid_bounded_box.x - centroid.x, 2)
							+ pow(centroid_bounded_box.y - centroid.y, 2)));
						float area_difference = abs(bb_area - curr_bb_area);
						if ((centroid_distance < 80 && area_difference < 70000)
						)
						{
							objects[bounded_boxes[j].id].insert(objects[bounded_boxes[j].id].end(), points.begin(), points.end());
							if (bb_area > curr_bb_area) {
								bounded_boxes[j].x = x;
								bounded_boxes[j].y = y;
								bounded_boxes[j].width = w;
								bounded_boxes[j].height = h;
								bounded_boxes[j].average_area = average_area;
								cv::Scalar bb_color(bounded_boxes[j].b, bounded_boxes[j].g, bounded_boxes[j].r);
								cv::rectangle(bb_image, rect, bb_color, 4);
							}
							bounded_box_exists = true;
							break;
						}
						else if (
							(bb.x >= curr_bb.x && bb.y >= curr_bb.y && 
								bb.x+w <= curr_bb.x+curr_bb.width && bb.y + h <= curr_bb.y + curr_bb.height) &&
							(bb_area < average_area)
							) {
							bounded_box_exists = true;
							break;
						}
						else {
							bounded_box_exists = false;
						}
					}
					if(!bounded_box_exists){
						// Have color associated with each layer
						int r = rand() % 255;
						int g = rand() % 255;
						int b = rand() % 255;
						cv::Scalar color(b, g, r);
						bb.r = r; bb.g = g; bb.b = b;
						bounded_boxes.push_back(bb);
						objects[bb.id] = points;
						
						cv::rectangle(bb_image, rect, color, 4);
					}
				}
				else {
					// Have color associated with each layer
					int r = rand() % 255;
					int g = rand() % 255;
					int b = rand() % 255;
					cv::Scalar color(b, g, r);
					bb.r = r; bb.g = g; bb.b = b;
					objects[bb.id] = points;
					bounded_boxes.push_back(bb);
					cv::rectangle(bb_image, rect, color, 4);
				}
				cv::rectangle(morphImg, rect, cv::Scalar(255, 255, 255), 5);
			}
		}
		std::cout << "Writing to file image " << index << " with dilation size " << dilationSize << " and elevation " << itr->first << std::endl;
		std::string name = "C:\\Users\\TskyDroneOps\\Desktop\\PlaneFitting\\src\\results3\\" + std::to_string(index) + ".jpg";
		cv::imwrite(name , morphImg);

		//boost::shared_ptr<pcl::visualization::PCLVisualizer> view(new pcl::visualization::PCLVisualizer);
		//// add coordinate and point pick listener
		//view->registerPointPickingCallback(pp_callback, (void*)&view);
		//view->addCoordinateSystem(20.0);

		//// add point cloud data
		//view->addPointCloud(new_alt_cloud, "cloud2");
		////view->addPointCloud(coloured_cloud);
		//while (!view->wasStopped())
		//{
		//	view->spinOnce(100);
		//	//view->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.5, 0.5, "plane1", 0);
		//	//view->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 1.0, "normals");
		//}
		//system("pause");
		cv::Mat flipped_bb_image;
		cv::flip(bb_image, flipped_bb_image, 1);
		index++;

	}
	// End timer 
	const auto timeStop = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds> (timeStop - timeBegin);
	std::cout << "Time Duration: " << duration.count() << " seconds" << std::endl;


	//show point cloud
	std::map<int, std::vector<int>>::reverse_iterator itr2;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr new_alt_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	std::cout << "PointMap size: " << objects.size() << std::endl;
	for (itr2 = objects.rbegin(); itr2 != objects.rend(); ++itr2) {

		int r = rand() % 255;
		int g = rand() % 255;
		int b = rand() % 255;

		// Add points to temp point cloud
		
		std::vector<int> checkSize;
		if (itr2->second.size() > checkSize.max_size())
		{
			std::cout << "Index: " << index << std::endl;
			std::cout << "Vector max size: " << checkSize.max_size() << std::endl;
			std::cout << "Vector too large! Size is: " << itr2->second.size() << " " << std::endl;
		}
		std::vector<int> points = itr2->second;
		for (int i = 0; i < points.size(); i++) {
			pcl::PointXYZRGB point;
			point.x = cloudXYZ->points[points[i]].x;
			point.y = cloudXYZ->points[points[i]].y;
			point.z = cloudXYZ->points[points[i]].z;
			point.r = r; point.b = b; point.g = g;
			new_alt_cloud->points.push_back(point);
		}
	}

	boost::shared_ptr<pcl::visualization::PCLVisualizer> view(new pcl::visualization::PCLVisualizer);
	// add coordinate and point pick listener
	view->registerPointPickingCallback(pp_callback, (void*)&view);
	view->addCoordinateSystem(20.0);

	// add point cloud data
	view->addPointCloud(new_alt_cloud, "cloud2");
	//view->addPointCloud(coloured_cloud);
	while (!view->wasStopped())
	{
		view->spinOnce(100);
		//view->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.5, 0.5, "plane1", 0);
		//view->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 1.0, "normals");
	}
	system("pause");

	/*std::map<float, std::vector<int>>::iterator itr;
	for (itr = pointMap.begin(); itr != pointMap.end(); ++itr) {
		std::cout << "Key: " << itr->first << " value: " << itr->second.size() << std::endl;
	}*/
	
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//original segmentation method
	//std::vector<pcl::PointIndices> clusters;
	//regionSegmentation(cloudXYZ, cloudNormal, coloured_cloud, clusters);

	//pcl::io::savePLYFileBinary("../coloured_cloud.ply", *coloured_cloud);

	//// For each cluster determine seed point
	//// Find average angle between seed point normal and point normals in cluster
	///*for (std::size_t cluster = 0; cluster < clusters.size(); cluster++) {
	//	for(std::size_t point = 0; point < clusters[cluster].)
	//}*/

	//Eigen::Vector4f centroid;
	//pcl::compute3DCentroid(*cloudXYZ, centroid);

	//std::vector<pcl::PointXYZRGBNormal> clusterNormals;
	//std::vector<pcl::ModelCoefficients> planes;

	//pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pointNormalCloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	//pcl::PointCloud<pcl::Normal>::Ptr pointNormalOnlyCloud(new pcl::PointCloud<pcl::Normal>);
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloudCluster(new pcl::PointCloud<pcl::PointXYZRGB>);

	//for (int i = 0; i < clusters.size(); i++) {
	//	int size = clusters[i].indices.size();
	//	float divisor = 1.0 / size;
	//	pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	//	for (int j = 0; j < size; j++) {
	//		int index = clusters[i].indices[j];
	//		temp_cloud->points.push_back(cloudXYZ->points[clusters[i].indices[j]]);
	//		//computeInliersRANSAC(cloudXYZ, cloudNormal, clusters[i]);
	//	}
	//	//RANSAC plane model fitting
	//	std::vector<int> inliers;
	//	pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr
	//		model(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(temp_cloud));
	//	pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model);
	//	ransac.setDistanceThreshold(0.1);
	//	ransac.computeModel();
	//	ransac.getInliers(inliers);
	//	//model->~SampleConsensusModelPlane();

	//	int inlier_size = inliers.size();
	//	// compute normal from points
	//	pcl::PointXYZRGBNormal centroid;
	//	pcl::Normal clusterNormal;
	//	pcl::PointXYZRGB clusterCentroid;
	//	for (int j = 0; j < inlier_size; j++) {
	//		int index = clusters[i].indices[inliers[j]];
	//		pcl::PointXYZ point = cloudXYZ->points[index];
	//		pcl::Normal normal = cloudNormal->points[index];
	//		centroid.x += point.x; centroid.y += point.y; centroid.z += point.z;
	//		centroid.normal_x += normal.normal_x; centroid.normal_y += normal.normal_y; centroid.normal_z += normal.normal_z;
	//	}
	//	centroid.x /= inlier_size; centroid.y /= inlier_size; centroid.z /= inlier_size;
	//	centroid.normal_x /= inlier_size; centroid.normal_y /= inlier_size; centroid.normal_z /= inlier_size;
	//	clusterNormals.push_back(centroid);
	//	//for normal vis
	//	clusterNormal.normal_x = centroid.normal_x; clusterNormal.normal_y = centroid.normal_y; 
	//	clusterNormal.normal_z = centroid.normal_z;
	//	// for cluster centroid vis
	//	int r = rand() / 255;
	//	int g = rand() / 255;
	//	int b = rand() / 255;
	//	clusterCentroid.x = centroid.x; clusterCentroid.y = centroid.y; clusterCentroid.z = centroid.z;
	//	clusterCentroid.r = r; clusterCentroid.g = g; clusterCentroid.b = b;
	//	// add cluster normal to list of cluster normals
	//	if (!isnan(centroid.normal_x) && !isnan(centroid.normal_y) && !isnan(centroid.normal_z))
	//	{
	//		pointNormalCloud->points.push_back(centroid);
	//		pointNormalOnlyCloud->points.push_back(clusterNormal);
	//	}
	//	pointCloudCluster->points.push_back(clusterCentroid);
	//	// add centroid to list of cluster centroids
	//	/*averagePointCluster.x = averagePointCluster.x * divisor;
	//	averagePointCluster.y = averagePointCluster.y * divisor;
	//	averagePointCluster.z = averagePointCluster.z * divisor;

	//	normalPoint.x = averagePointCluster.x; normalPoint.normal_x = averageNormalCluster.normal_x;
	//	normalPoint.y = averagePointCluster.y; normalPoint.normal_y = averageNormalCluster.normal_y;
	//	normalPoint.z = averagePointCluster.z; normalPoint.normal_z = averageNormalCluster.normal_z;
	//	pointNormalCloud->points.push_back(normalPoint);*/

	///*	pcl::PointXYZRGBNormal point;
	//	point.normal_x = averageNormalCluster.normal_x; point.normal_y = averageNormalCluster.normal_y; point.normal_z = averageNormalCluster.normal_z;
	//	point.x = averagePointCluster.x; point.y = averagePointCluster.y; point.z = averagePointCluster.z;
	//	clusterNormals.push_back(point);*/

	//	// for viewing cluster centroids
	//	//cloudOutliers->points.push_back(averagePointCluster);

	//	float d = centroid.normal_x * cloudXYZ->points[clusters[i].indices[0]].x +
	//			centroid.normal_y * cloudXYZ->points[clusters[i].indices[0]].y +
	//			centroid.normal_z* cloudXYZ->points[clusters[i].indices[0]].z;
	//	d = -d;
	//	pcl::ModelCoefficients modelCoeff;
	//	modelCoeff.values.resize(4);
	//	modelCoeff.values[0] = centroid.normal_x; modelCoeff.values[1] = centroid.normal_y;
	//	modelCoeff.values[2] = centroid.normal_z; modelCoeff.values[3] = d;
	//	planes.push_back(modelCoeff);

	//	temp_cloud->clear();
	//	/*vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New();
	//	plane->SetCenter(averagePointCluster.x, averagePointCluster.y, averagePointCluster.z);
	//	plane->SetNormal(averageNormalCluster.normal_x, averageNormalCluster.normal_y, averageNormalCluster.normal_z);
	//	plane->Update();*/
	//}

	////struct remove_cluster : public std::unary_function<const float, bool>
	////{
	////	bool operator()(const float angle) const
	////	{
	////		float delta = 4.0 / 180.0 * M_PI;
	////		return angle < delta;
	////	}
	////};

	///*pcl::PointCloud<pcl::PointXYZRGB>::Ptr adjusted_coloured_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	//int a = 0;
	//std::vector<pcl::PointIndices> checkClusters;
	//checkClusters.push_back(clusters[558]);
	//checkClusters.push_back(clusters[598]);
	//for (int i = 0; i < checkClusters.size(); i++) {
	//	int r = rand() / 255;
	//	int g = rand() / 255;
	//	int b = rand() / 255;
	//	for (int j = 0; j < checkClusters[i].indices.size(); j++) {
	//		pcl::PointXYZ pt = cloudXYZ->points.at(checkClusters[i].indices[j]);
	//		pcl::PointXYZRGB ptRGB;
	//		ptRGB.x = pt.x; ptRGB.y = pt.y; ptRGB.z = pt.z;
	//		ptRGB.r = r; ptRGB.g = g; ptRGB.b = b;
	//		adjusted_coloured_cloud->points.push_back(ptRGB);
	//	}
	//	a++;
	//}*/

	//const auto magnitude = [](
	//	pcl::PointXYZRGBNormal p1
	//) {
	//	return sqrt(pow(p1.x, 2) + pow(p1.y, 2) + pow(p1.z, 2));
	//};

	//std::cout << "Cluster normal size: " << clusterNormals.size() << std::endl;
	//std::cout << "==================" << std::endl;
	//std::vector<int> indices;
	//std::vector<pcl::PointXYZRGBNormal>::iterator iter;
	//float delta = 4.5 * (M_PI/180.0); // 15.0
	//float dist_d = 35; float angle_max_proj = 5.0 * (M_PI/180.0);
	//for (int i = 0; i < clusterNormals.size(); i++) {
	//	for (int j = 0; j < clusterNormals.size(); j++) {
	//		if (i != j) {
	//			pcl::PointXYZRGBNormal pointI = clusterNormals[i];
	//			pcl::PointXYZRGBNormal pointJ = clusterNormals[j];

	//			// find angle between normals of clusters
	//			float nom = (pointI.normal_x * pointJ.normal_x) + (pointI.normal_y * pointJ.normal_y)
	//				+ (pointI.normal_z * pointJ.normal_z);
	//			float denom = sqrt(
	//				pow(pointI.normal_x, 2) + pow(pointI.normal_y, 2) + pow(pointI.normal_z, 2))
	//				* sqrt(
	//					pow(pointJ.normal_x, 2) + pow(pointJ.normal_y, 2) + pow(pointJ.normal_z, 2));
	//			float angle = acos(nom / denom);

	//			//calculate euclidean distance between centroids of clusters
	//			float centroid_distance = sqrt(
	//				pow((pointI.x - pointJ.x), 2)
	//				+ pow((pointI.y - pointJ.y), 2)
	//				+ pow((pointI.z - pointJ.z), 2)
	//			);

	//			// project plane onto tested plane and find angle
	//			float dist = abs(abs(pointJ.z) - abs(pointI.z));
	//			/*pcl::PointXYZRGBNormal hyp; pcl::PointXYZRGBNormal proj;
	//			hyp.x = pointJ.x - pointI.x; hyp.y = pointJ.y - pointI.y; hyp.z = pointJ.z - pointI.z;
	//			proj.x = pointJ.x - pointI.x; proj.y = pointJ.y - pointI.y; proj.z = pointI.z;
	//			float magHyp = magnitude(hyp); float magProj = magnitude(proj);
	//			float angle_proj = cos(magProj / magHyp);*/

	//			//indices.push_back(i);
	//			if (angle < delta && angle > (4.0 * (M_PI/180)) && angle > 0 && centroid_distance < dist_d && dist < 5) {
	//				bool exists = false;
	//				for (int k = 0; k < indices.size(); k++) {
	//					if (indices[k] == i)
	//						exists = true;
	//				}
	//				if (!exists) {
	//					indices.push_back(i);
	//					std::cout << "Angle between indices: " << (angle*180) / M_PI << std::endl;
	//					std::cout << " Indices: " << i << " && " << j << std::endl;
	//					std::cout << "I x: " << pointI.x << " I y: " << pointI.y << " I z: " << pointI.z << std::endl;
	//					std::cout << "I nx: " << pointI.normal_x << " I ny: " << pointI.normal_y << " I nz: " << pointI.normal_z << std::endl;
	//					std::cout << "Difference Z: " << dist << std::endl;
	//					std::cout << "J x: " << pointJ.x << " J y: " << pointJ.y << " J z: " << pointJ.z << std::endl;
	//					std::cout << "J nx: " << pointJ.normal_x << " J ny: " << pointJ.normal_y << " J nz: " << pointJ.normal_z << std::endl;
	//					std::cout << "==================" << std::endl;
	//				}
	//					//indices.push_back(i);
	//				/*std::cout << "Value at " << i << " and " << j << " is less than delta. Combine the two clusters" << std::endl;
	//				std::cout << "N1.x: " << clusterNormals[i].normal_x << " N2.x: " << clusterNormals[j].normal_x << std::endl;
	//				std::cout << "N1.y: " << clusterNormals[i].normal_y << " N2.y: " << clusterNormals[j].normal_y << std::endl;
	//				std::cout << "N1.z: " << clusterNormals[i].normal_z << " N2.z: " << clusterNormals[j].normal_z << std::endl;
	//				std::cout << "P1.x: " << clusterNormals[i].x << " P2.x: " << clusterNormals[j].x << std::endl;
	//				std::cout << "P1.y: " << clusterNormals[i].y << " P2.y: " << clusterNormals[j].y << std::endl;
	//				std::cout << "P1.z: " << clusterNormals[i].z << " P2.z: " << clusterNormals[j].z << std::endl;
	//				std::cout << "Angle: " << (angle*180)/M_PI << std::endl;
	//				std::cout << "Centroid distance: " << centroid_distance << std::endl;
	//				std::cout << "Cluster size before: " << clusters[i].indices.size() << std::endl;
	//				std::cout << "Clusers other size: " << clusters[j].indices.size() << std::endl;*/
	//				clusters[i].indices.insert(clusters[i].indices.end(), clusters[j].indices.begin(), clusters[j].indices.end());
	//				clusterNormals[i].normal_x = (clusterNormals[i].normal_x + clusterNormals[j].normal_x) * 0.5f;
	//				clusterNormals[i].normal_y = (clusterNormals[i].normal_y + clusterNormals[j].normal_y) * 0.5f;
	//				clusterNormals[i].normal_z = (clusterNormals[i].normal_z + clusterNormals[j].normal_z) * 0.5f;
	//				//std::cout << "Cluster size after: " << clusters[i].indices.size() << std::endl;
	//				//clusters.erase(clusters.begin() + j);
	//				//std::cout << "==========================================" << std::endl;
	//			}
	//		}
	//	}
	//}

	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr adjusted_coloured_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	//for (int i = 0; i < indices.size(); i++) {
	//	int r = rand() / 255;
	//	int g = rand() / 255;
	//	int b = rand() / 255;
	//	for (int j = 0; j < clusters[indices[i]].indices.size(); j++) {
	//		pcl::PointXYZ pt = cloudXYZ->points.at(clusters[indices[i]].indices[j]);
	//		pcl::PointXYZRGB ptRGB; 
	//		ptRGB.x = pt.x; ptRGB.y = pt.y; ptRGB.z = pt.z;
	//		ptRGB.r = r; ptRGB.g = g; ptRGB.b = b;
	//		adjusted_coloured_cloud->points.push_back(ptRGB);
	//	}
	//	/*std::cout << "Index: " << indices[i] << std::endl;
	//	std::cout << "Cluster size: " << clusters[indices[i]].indices.size() << std::endl;
	//	std::cout << "=========================" << std::endl;*/
	//}
	//std::cout << "Indices size: " << indices.size() << std::endl;
	//std::cout << "PointCloudSize: " << adjusted_coloured_cloud->points.size() << std::endl;
	//boost::shared_ptr<pcl::visualization::PCLVisualizer> view(new pcl::visualization::PCLVisualizer);

	//// add coordinate and point pick listener
	//view->registerPointPickingCallback(pp_callback, (void*)&view);
	//view->addCoordinateSystem(20.0);

	//// add point cloud data
	//view->addPointCloud(adjusted_coloured_cloud, "cloud2");
	////view->addPointCloud(coloured_cloud);

	//// End timer 
	//const auto timeStop = std::chrono::steady_clock::now();
	//auto duration = std::chrono::duration_cast<std::chrono::seconds> (timeStop - timeBegin);
	//std::cout << "Time Duration: " << duration.count() << " seconds" << std::endl;

	//while (!view->wasStopped())
	//{
	//	view->spinOnce(100);
	//	//view->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.5, 0.5, "plane1", 0);
	//	//view->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 1.0, "normals");
	//}
	//system("pause");
	
	return (0);
}

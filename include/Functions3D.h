#ifndef Functions3D_h
#define Functions3D_h

#include <numeric>
#include <limits>
#ifdef WIN32
#include <Windows.h>
#endif
#include <algorithm>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/features2d/features2d.hpp>

#include <DgmOctree.h>
#include <KdTree.h>
#include <CloudSamplingTools.h>
#include <DistanceComputationTools.h>
#include <Garbage.h>
#include <GenericProgressCallback.h>
#include <GenericIndexedMesh.h>
#include <GeometricalAnalysisTools.h>
#include <Jacobi.h>
#include <ManualSegmentationTools.h>
#include <NormalDistribution.h>
#include <ParallelSort.h>
#include <PointCloud.h>
#include <ReferenceCloud.h>
#include <ScalarFieldTools.h>
#include <RegistrationTools.h>

#include "Coordinates.h"
#include "MeshTypes.hpp"

#define DEBUG 0
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define LESS_THAN_EPSILON(x) x < ZERO_TOLERANCE_F

namespace Functions3D
{

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

	struct ModelCloud
	{
		ModelCloud() : cloud(nullptr), weights(nullptr) {}
		ModelCloud(const ModelCloud& m) = default;
		CCCoreLib::GenericIndexedCloudPersist* cloud;
		CCCoreLib::ScalarField* weights;
	};

	struct DataCloud
	{
		DataCloud() : cloud(nullptr), rotatedCloud(nullptr), weights(nullptr), CPSetRef(nullptr), CPSetPlain(nullptr) {}

		CCCoreLib::ReferenceCloud* cloud;
		CCCoreLib::PointCloud* rotatedCloud;
		CCCoreLib::ScalarField* weights;
		CCCoreLib::ReferenceCloud* CPSetRef;
		CCCoreLib::PointCloud* CPSetPlain;
	};

	RR_3DLib_API std::vector<Mesh_Types::POINT_CLOUD::PointType> nearestKSearch(
		std::unique_ptr<Mesh_Types::MESH>& mesh_,
		PointCoordinateType* point,
		double k
	);
	RR_3DLib_API std::vector<Mesh_Types::POINT_CLOUD::PointType> radiusSearch(
		std::unique_ptr<Mesh_Types::MESH>& mesh_,
		PointCoordinateType* point,
		double radius
	);
	RR_3DLib_API cv::Mat generateHeightMap(
		std::unique_ptr<Mesh_Types::MESH>& mesh_,
		float adjusted_width_x, float adjusted_width_y,
		std::vector<int>* points
	);
	RR_3DLib_API cv::Mat generateHeightMap(
		std::shared_ptr<Mesh_Types::POINT_CLOUD>& pointCloud,
		float adjusted_width_x, float adjusted_width_y,
		std::vector<int>* points
	);
	RR_3DLib_API std::shared_ptr<Mesh_Types::POINT_CLOUD> objectSegmentation(
		std::unique_ptr<Mesh_Types::MESH>& mesh_,
		int epsilon_centroid_distance = 80,
		int epsilon_area_difference = 70000
	);
	//transforms pointclouds relative to initial point cloud in vector
	//returns a vector of transformed key_point pointclouds
	RR_3DLib_API std::vector<std::shared_ptr<Mesh_Types::POINT_CLOUD>> keyPointRegistration(
		std::vector<std::shared_ptr<Mesh_Types::POINT_CLOUD>>& pointClouds,
		double leaf_size = 0.95
	);
	RR_3DLib_API Mesh_Types::POINT computeCloudMean(
		std::shared_ptr<Mesh_Types::POINT_CLOUD> cloud
	);
	RR_3DLib_API std::shared_ptr<Mesh_Types::POINT_CLOUD> pointCloudDifferencing(
		std::shared_ptr<Mesh_Types::POINT_CLOUD>& mesh_reference,
		std::shared_ptr<Mesh_Types::POINT_CLOUD>& mesh_compare
	);
	RR_3DLib_API std::shared_ptr<Mesh_Types::POINT_CLOUD> downsamplePointCloudMaxPoints(
		std::unique_ptr<Mesh_Types::MESH>& mesh_,
		int num_points
	);
	RR_3DLib_API std::shared_ptr<Mesh_Types::POINT_CLOUD> downsamplePointCloudSpatially(
		std::unique_ptr<Mesh_Types::MESH>& mesh_,
		Precision leaf_size
	);
	RR_3DLib_API void constructOctree(
		std::unique_ptr<Mesh_Types::MESH>& mesh_
	);
	RR_3DLib_API void constructKDTree(
		std::unique_ptr<Mesh_Types::MESH>& mesh_
	);

	namespace AncillaryFunctions {
		CCCoreLib::ICPRegistrationTools::RESULT_TYPE ICPRegister(
			CCCoreLib::GenericIndexedCloudPersist* inputModelCloud,
			CCCoreLib::GenericIndexedMesh* inputModelMesh,
			CCCoreLib::GenericIndexedCloudPersist* inputDataCloud,
			const CCCoreLib::ICPRegistrationTools::Parameters& params,
			CCCoreLib::ICPRegistrationTools::ScaledTransformation& transform,
			double& finalRMS,
			unsigned& finalPointCount,
			CCCoreLib::GenericProgressCallback* progressCb = nullptr
		);
		bool RegistrationProcedure(
			CCCoreLib::GenericCloud* P, //data
			CCCoreLib::GenericCloud* X, //model
			CCCoreLib::ICPRegistrationTools::ScaledTransformation& trans,
			bool adjustScale = false,
			CCCoreLib::ScalarField* coupleWeights = 0,
			PointCoordinateType aPrioriScale = 1.0f
		);
		static bool resampleCellAtLevel(
			const CCCoreLib::DgmOctree::octreeCell& cell,
			void** additionalParameters,
			CCCoreLib::NormalizedProgress* progress = nullptr
		);
		void returnPointsFromImage(
			std::unique_ptr<Mesh_Types::MESH>& mesh_,
			std::vector<int>& points,
			std::vector<int>& points_bb,
			cv::Mat& img,
			bounded_box_data& bounding_box
		);
		bool getPointsInCell(
			const Mesh_Types::POINT_CLOUD& pointCloud,
			CCCoreLib::DgmOctree& octree,
			CCCoreLib::DgmOctree::CellCode cellCode,
			unsigned char level,
			std::vector<CCVector3>& subset,
			bool isCodeTruncated = false
		);
		bool cmp(
			std::pair<int, std::vector<int>>& a,
			std::pair<int, std::vector<int>>& b
		);
		void sort(
			std::map<int, std::vector<int>>& M,
			std::vector<std::pair<int, std::vector<int>>>& B
		);
	}
}

#endif // Functions3D_h
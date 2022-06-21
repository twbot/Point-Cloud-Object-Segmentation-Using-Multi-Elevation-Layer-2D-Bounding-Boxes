#include "../include/Functions3D.h"

using namespace CCCoreLib;

namespace Functions3D {

	std::shared_ptr<Mesh_Types::POINT_CLOUD> downsamplePointCloudSpatially(
		std::unique_ptr<Mesh_Types::MESH>& mesh_,
		Precision leaf_size
	)
	{
		assert(mesh_);

		//Create octree if not already created
		if (!mesh_->octree_enabled) {
			constructOctree(mesh_);
		}

		//Create kdtree if not already created
		if (!mesh_->kdtree_enabled) {
			constructKDTree(mesh_);
		}

		std::shared_ptr<Mesh_Types::POINT_CLOUD> cloud = std::make_shared<Mesh_Types::POINT_CLOUD>();

		std::vector<char> markers; //DGM: upgraded from vector, as this can be quite huge!
		try
		{
			markers.resize(mesh_->pointcloud.size(), 1); //true by default
		}
		catch (const std::bad_alloc&)
		{
			return nullptr;
		}

		//best octree level (there may be several of them if we use parameter modulation)
		std::vector<unsigned char> bestOctreeLevel;
		try
		{
			unsigned char defaultLevel = mesh_->octree->findBestLevelForAGivenNeighbourhoodSizeExtraction(leaf_size);
			bestOctreeLevel.push_back(defaultLevel);
		}
		catch (const std::bad_alloc&)
		{
			return nullptr;
		}

		//for each point in the cloud that is still 'marked', we look
		//for its neighbors and remove their own marks
		bool error = false;
		//default octree level
		assert(!bestOctreeLevel.empty());
		unsigned char octreeLevel = bestOctreeLevel.front();
		//default distance between points
		PointCoordinateType minDistBetweenPoints = leaf_size;
		unsigned cloudSize = mesh_->pointcloud.size();
		for (unsigned i = 0; i < cloudSize; i++)
		{
			//no mark? we skip this point
			if (markers[i] != 0)
			{
				//init neighbor search structure
				const CCVector3* P = mesh_->pointcloud.getPoint(i);

				//look for neighbors and 'de-mark' them
				{
					DgmOctree::NeighboursSet neighbours;
					mesh_->octree->getPointsInSphericalNeighbourhood(*P, minDistBetweenPoints, neighbours, octreeLevel);
					for (DgmOctree::NeighboursSet::iterator it = neighbours.begin(); it != neighbours.end(); ++it)
						if (it->pointIndex != i)
							markers[it->pointIndex] = 0;
				}
			}
		}

#pragma omp parallel for
		for (unsigned i = 0; i < markers.size(); i++) {
			if (markers[i] != 0)
				cloud->add(mesh_->pointcloud.getPointFull(i));
		}

		//remove unnecessarily allocated memory
		if (error)
		{
			return nullptr;
		}
		else
		{
			return cloud;
		}
	}

	std::shared_ptr<Mesh_Types::POINT_CLOUD> downsamplePointCloudMaxPoints(
		std::unique_ptr<Mesh_Types::MESH>& mesh_,
		int num_points
	)
	{
		RR_ASSERT(mesh_);

		//Create octree if not already created
		if (!mesh_->octree_enabled) {
			constructOctree(mesh_);
		}

		//Create kdtree if not already created
		if (!mesh_->kdtree_enabled) {
			constructKDTree(mesh_);
		}

		int octreeLevel = mesh_->octree->findBestLevelForAGivenCellNumber(num_points);
		unsigned nCells = mesh_->octree->getCellNumber(octreeLevel);

		CloudSamplingTools::RESAMPLING_CELL_METHOD method = CloudSamplingTools::RESAMPLING_CELL_METHOD::CELL_CENTER;

		std::shared_ptr<Mesh_Types::POINT_CLOUD> new_cloud = std::make_shared<Mesh_Types::POINT_CLOUD>();
		void *additionalParameters[5] = {
			reinterpret_cast<void*>((mesh_->kdtree).get()),
			reinterpret_cast<void*>((mesh_->octree).get()),
			reinterpret_cast<void*>(&mesh_->pointcloud),
			reinterpret_cast<void*>(new_cloud.get()),
			reinterpret_cast<void*>(&method)
		};

		if (mesh_->octree->executeFunctionForAllCellsAtLevel(octreeLevel,
			&AncillaryFunctions::resampleCellAtLevel,
			additionalParameters,
			false, nullptr, "Cloud Resampling") == 0)
		{
			//something went wrong
			return nullptr;
		}

		return new_cloud;
	}

	std::shared_ptr<Mesh_Types::POINT_CLOUD> objectSegmentation(
		std::unique_ptr<Mesh_Types::MESH>& mesh_,
		int epsilon_centroid_distance,
		int epsilon_area_difference
	)
	{
		float adjusted_width_x, adjusted_width_y;
		int border_padding = 500;

		adjusted_width_x = (mesh_->pointcloud.max_vertex[0] - mesh_->pointcloud.min_vertex[0]) / mesh_->pointcloud.horizontal_step + border_padding;
		adjusted_width_y = (mesh_->pointcloud.max_vertex[1] - mesh_->pointcloud.min_vertex[1]) / mesh_->pointcloud.horizontal_step + border_padding;

		//Bounded boxes
		std::vector<bounded_box_data> bounded_boxes;

		//Objects
		std::map<int, std::vector<int>> objects;

		//new segmentation method (in testing)
		std::map<float, std::vector<int>> pointMap;
		int vertical_res = 100 / (mesh_->pointcloud.vertical_step * 100);
		for (int i = 0; i < mesh_->pointcloud.size(); i++)
		{
			float key = round((mesh_->pointcloud.getPointFull(i).coord.z * 1)) / 1;
			if (pointMap.count(key) == 0) {
				std::vector<int> indices;
				indices.push_back(i);
				pointMap.insert({ key, indices });
			}
			else {
				(pointMap.at(key)).push_back(i);
			}
		}

		// detect when the data starts to become more clustered (i.e. for city, when we start to reach ground level)
		int largest_difference = 0;
		// data for the base level data
		int change_alt = 0; int change_alt_id = std::rand();

		std::map<float, std::vector<int>>::iterator pointmap_itr;

		int bin_size = 1;
		for (pointmap_itr = pointMap.begin(); pointmap_itr != std::prev(pointMap.end()); ++pointmap_itr) {
			int total_bin_size = pointmap_itr->second.size();
			int total_bin_size_next = std::next(pointmap_itr)->second.size();
			int difference = total_bin_size - total_bin_size_next;
			if (difference > largest_difference && pointmap_itr->first < (mesh_->pointcloud.max_vertex[2] + mesh_->pointcloud.min_vertex[2]) / 2)
			{
				largest_difference = difference;
				change_alt = std::next(pointmap_itr)->first;
			}
		}

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
		int exclusionCloseSize = 20;
		int exclusionOpenSize = 20;

		cv::Mat bb_image(cv::Size(adjusted_width_y, adjusted_width_x), CV_8UC3, 0.0);

		for (itr = pointMap.rbegin(); itr != pointMap.rend(); ++itr) {
			std::vector<int> points = itr->second;
			if (itr->first > change_alt) {
				RR_ASSERT(points.size() < points.max_size());
				if (index % 20 == 0) {
					if (dilationSize > 25) {
						dilationSize -= 10;
					}
					if (exclusionOpenSize > 15) {
						exclusionOpenSize -= 1;
					}
				}

				// Generate heightmap from pointcloud
				cv::Mat heightMap = generateHeightMap(mesh_, adjusted_width_x, adjusted_width_y, &points);
				// Get contours from heightmap
				cv::Mat img, img2, morphImg;
				cv::Mat se = getStructuringElement(cv::MORPH_RECT, cv::Size(dilationSize, dilationSize));
				cv::dilate(heightMap, img, se);
				cv::Mat seMorph = getStructuringElement(cv::MORPH_RECT, cv::Size(exclusionCloseSize, exclusionCloseSize));
				cv::morphologyEx(img, img2, cv::MORPH_CLOSE, exclusionCloseSize);
				seMorph = getStructuringElement(cv::MORPH_RECT, cv::Size(exclusionOpenSize, exclusionOpenSize));
				cv::morphologyEx(img2, morphImg, cv::MORPH_OPEN, seMorph);

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
						bb.bounding_boxes.push_back(rect);

						// Get 3D points within bounding box
						std::vector<int> bounding_box_points;
						AncillaryFunctions::returnPointsFromImage(mesh_, points, bounding_box_points, heightMap, bb);

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
								if (centroid_distance < epsilon_centroid_distance
									&& area_difference < epsilon_area_difference
									)
								{
									objects[bounded_boxes[j].id].insert(objects[bounded_boxes[j].id].end(),
										bounding_box_points.begin(), bounding_box_points.end());
									if (bb_area > curr_bb_area) {
										bounded_boxes[j].x = x;
										bounded_boxes[j].y = y;
										bounded_boxes[j].width = w;
										bounded_boxes[j].height = h;
										bounded_boxes[j].average_area = average_area;
										bounded_boxes[j].bounding_boxes.push_back(rect);
										cv::Scalar bb_color(bounded_boxes[j].b, bounded_boxes[j].g, bounded_boxes[j].r);
										cv::rectangle(bb_image, rect, bb_color, 4);
									}
									bounded_box_exists = true;
									break;
								}
								else if (
									(bb.x >= curr_bb.x && bb.y >= curr_bb.y &&
										bb.x + w <= curr_bb.x + curr_bb.width && bb.y + h <= curr_bb.y + curr_bb.height) &&
										(bb_area < average_area)
									) {
									bounded_box_exists = true;
									break;
								}
								else {
									bounded_box_exists = false;
								}
							}
							if (!bounded_box_exists) {
								// Have color associated with each layer
								int r = rand() % 255;
								int g = rand() % 255;
								int b = rand() % 255;
								cv::Scalar color(b, g, r);
								bb.r = r; bb.g = g; bb.b = b;
								bounded_boxes.push_back(bb);
								objects[bb.id] = bounding_box_points;

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
							objects[bb.id] = bounding_box_points;
							bounded_boxes.push_back(bb);
							cv::rectangle(bb_image, rect, color, 4);
						}
						cv::rectangle(morphImg, rect, cv::Scalar(255, 255, 255), 5);
					}
				}
#if DEBUG
				std::cout << "Writing to file image " << index << " with dilation size " << dilationSize << ", exclusion size " << exclusionOpenSize <<
					" and elevation " << itr->first << std::endl;
				std::string name = "C:\\Users\\TskyDroneOps\\Desktop\\PlaneFitting\\src\\results\\" + std::to_string(index) + ".jpg";
				cv::imwrite(name, morphImg);

				cv::Mat flipped_bb_image;
				cv::flip(bb_image, flipped_bb_image, 1);

				std::string bb_name = "C:\\Users\\TskyDroneOps\\Desktop\\PlaneFitting\\src\\bb_results\\bb" + std::to_string(index) + ".jpg";
				cv::imwrite(bb_name, bb_image);
#endif
				index++;
			}
			else {
				if (objects[change_alt_id].size() < 0)
					objects[change_alt_id] = points;
				else
					objects[change_alt_id].insert(objects[change_alt_id].end(),
						points.begin(), points.end());
			}

		}

		bool post_processing = false;
		//Post-processing
		if (post_processing) {
			int average_bounding_box_size = 0;
			int size = bounded_boxes.size();
			for (int i = 0; i < size; i++)
			{
				average_bounding_box_size += bounded_boxes[i].bounding_boxes.size();
			}
			average_bounding_box_size = average_bounding_box_size / size;
#if DEBUG
			std::cout << "Avg bb size: " << average_bounding_box_size << std::endl;
#endif
			for (int i = 0; i < size; i++)
			{
				bounded_box_data bb = bounded_boxes[i];

				if (bb.bounding_boxes.size() < average_bounding_box_size) {

					//Generate heightmap from points within noise object
					cv::Mat heightMap = generateHeightMap(mesh_, adjusted_width_x, adjusted_width_y, &objects[bb.id]);

					//Centroid of noise object
					cv::Point2d centroid_bb(
						(bb.x + bb.width) / 2,
						(bb.y + bb.height) / 2
					);

					for (int j = 0; j < size; j++)
					{
						if (i != j) {

							bounded_box_data curr_bb = bounded_boxes[j];

							// Centroid of checked objects
							cv::Point2d centroid_bounded_box(
								(curr_bb.x + curr_bb.width) / 2,
								(curr_bb.y + curr_bb.height) / 2
							);

							// Calculate centroid distance
							float centroid_distance = sqrt((pow(centroid_bounded_box.x - centroid_bb.x, 2)
								+ pow(centroid_bounded_box.y - centroid_bb.y, 2)));

							// If noise object is within an bounding-box of another object, add noise object to said object
							/*if (bb.x >= curr_bb.x && bb.y >= curr_bb.y &&
								bb.x + bb.width <= curr_bb.x + curr_bb.width && bb.y + bb.height <= curr_bb.y + curr_bb.height &&
								centroid_distance < 80)
							{
								objects[curr_bb.id].insert(objects[curr_bb.id].end(),
									objects[bb.id].begin(), objects[bb.id].end());
								objects.erase(bb.id);
								bounded_boxes.erase(bounded_boxes.begin() + i);
								std::cout << "Found within ";
							}*/
							if (bb.x <= curr_bb.x && bb.y <= curr_bb.y &&
								bb.x + bb.width >= curr_bb.x + curr_bb.width && bb.y + bb.height >= curr_bb.y + curr_bb.height) {
								// Get 3D points within bounding box
								std::vector<int> bounding_box_points;
								AncillaryFunctions::returnPointsFromImage(mesh_, objects[bb.id], bounding_box_points, heightMap, curr_bb);
								objects[curr_bb.id].insert(objects[curr_bb.id].end(),
									bounding_box_points.begin(), bounding_box_points.end());
							}

							if (size != bounded_boxes.size()) {
								--i; size = bounded_boxes.size();
							}
						}
					}
				}
			}
		}

		// Add points to temp point cloud
		std::map<int, std::vector<int>>::reverse_iterator itr2;
		Mesh_Types::POINT_CLOUD pc;

#if DEBUG
		std::cout << "PointMap size: " << objects.size() << std::endl;
#endif

		std::vector<std::pair<int, std::vector<int>> > A;
		AncillaryFunctions::sort(objects, A);

		for (auto& object : A) {

			int r = rand() % 255;
			int g = rand() % 255;
			int b = rand() % 255;

			std::vector<int> points = object.second;
			RR_ASSERT(points.size() < points.max_size());
			for (int i = 0; i < points.size(); i++) {
				Mesh_Types::POINT point;
				point.coord.x = mesh_->pointcloud.getPointFull(points[i]).coord.x;
				point.coord.y = mesh_->pointcloud.getPointFull(points[i]).coord.y;
				point.coord.z = mesh_->pointcloud.getPointFull(points[i]).coord.z;
				point.color.r = r; point.color.b = b; point.color.g = g;
				pc.add(point);
			}
		}

		return std::make_shared<Mesh_Types::POINT_CLOUD>(pc);
	}

	Mesh_Types::POINT computeCloudMean(
		std::shared_ptr<Mesh_Types::POINT_CLOUD> cloud
	)
	{
		const size_t point_size = cloud->size();
		Eigen::Matrix<Precision, DYNAMIC, POLYGON_TYPE> reference_coords_centered;
		reference_coords_centered.resize(point_size);
		for (size_t i = 0; i < point_size; i++) {
			for (uchar j = 0; j < 3; j++) {
				reference_coords_centered(i, j) = cloud->getPointFull(i).getCoordinate(j);
			}
		}
		Mesh_Types::POINT pnt;
		pnt.coord.x = reference_coords_centered.col(0).mean();
		pnt.coord.y = reference_coords_centered.col(1).mean();
		pnt.coord.z = reference_coords_centered.col(2).mean();
		return pnt;
	}

	std::vector<std::shared_ptr<Mesh_Types::POINT_CLOUD>> keyPointRegistration(
		std::vector<std::shared_ptr<Mesh_Types::POINT_CLOUD>>& pointClouds,
		double leaf_size
	)
	{
		const size_t num_meshes = pointClouds.size();

		std::vector<std::shared_ptr<Mesh_Types::POINT_CLOUD>> keyPointClouds;

		std::vector<cv::Mat> heightmaps;
		std::vector<cv::Mat> keypointsMaps;

		// Seed timer
		srand(time(NULL));

		for (int i = 0; i < num_meshes; i++) {
			// Create temp point clouds for data storage
			std::shared_ptr<Mesh_Types::POINT_CLOUD> keyPointCloud = std::make_shared<Mesh_Types::POINT_CLOUD>();

			float adjusted_width_x = (float)((pointClouds[i]->max_vertex[0] - pointClouds[i]->min_vertex[0]) / pointClouds[i]->horizontal_step);
			float adjusted_width_y = (float)((pointClouds[i]->max_vertex[1] - pointClouds[i]->min_vertex[1]) / pointClouds[i]->vertical_step);

			// For debugging
#if DEBUG
			std::cout << "Min x: " << pointClouds[i]->min_vertex[0] << " Max x: " << pointClouds[i]->max_vertex[0] << std::endl;
			std::cout << "Min y: " << pointClouds[i]->min_vertex[1] << " Max y: " << pointClouds[i]->max_vertex[1] << std::endl;
			std::cout << "Min alt: " << pointClouds[i]->min_vertex[2] << " Max alt: " << pointClouds[i]->max_vertex[2] << std::endl;
			std::cout << "Step size (horizontal): " << pointClouds[i]->horizontal_step << "Step size (vertical): " << pointClouds[i]->vertical_step << std::endl;
			std::cout << "Adjusted_width_x: " << adjusted_width_x << " Adjusted_width_y: " << adjusted_width_y << std::endl;
#endif

			std::vector<int> emptyPnts;
			// Generate heightmap from pointcloud
			cv::Mat heightmap = generateHeightMap(pointClouds[i], adjusted_width_x, adjusted_width_y, &emptyPnts);
			heightmaps.push_back(heightmap);

			// AKAZE keypoint extractor
			std::cout << "Extracting keypoints from heightmap..." << std::endl;
			cv::Mat output;
			cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
			std::vector<cv::KeyPoint> keypoints;
			akaze->detectAndCompute(heightmap, cv::noArray(), keypoints, output);

			//generate random color value for keypoints
			uchar red = rand() % 256; uchar blue = rand() % 256; uchar green = rand() % 256;

			// plot keypoints on img
			cv::Mat outputImg;
			cv::drawKeypoints(heightmap, keypoints, outputImg, cv::Scalar(blue, green, red, 0.0));
			keypointsMaps.push_back(outputImg);

			// For debugging
#if DEBUG
			std::cout << "Keypoint size: " << keypoints.size() << std::endl;
			cv::imwrite(base_path + "heightmap" + std::to_string(i) + ".jpg", heightmap);
			cv::imwrite(base_path + "keyPoints" + std::to_string(i) + ".jpg", outputImg);
#endif

			// Convert altitude to original units
			const auto convertAlt = [](
				float alt,
				float min_alt,
				float max_alt
				)
			{
				float val = (alt / 255 * (max_alt - min_alt)) + min_alt;
				return val;
			};

			// Convert point to original coordinate system
			const auto convert = [](
				float measure, float min, float step
				) {
				float adjustedMeasure = (measure*step) + min;
				return adjustedMeasure;
			};

			for (std::size_t j = 0; j < keypoints.size(); j++) {
				Mesh_Types::POINT point;
				float x_grid = keypoints[j].pt.x;
				point.coord.x = convert(x_grid, pointClouds[i]->min_vertex[0], pointClouds[i]->horizontal_step);
				float y_grid = keypoints[j].pt.y;
				point.coord.y = convert(y_grid, pointClouds[i]->min_vertex[1], pointClouds[i]->vertical_step);
				uchar alt_norm = heightmap.at<uchar>(keypoints[j].pt.y, keypoints[j].pt.x);
				point.coord.z = convertAlt(alt_norm, pointClouds[i]->min_vertex[2], pointClouds[i]->max_vertex[2]);
				point.color.r = red;
				point.color.g = green;
				point.color.b = blue;
				keyPointCloud->add(point);
			}
			//set key point cloud center
			keyPointCloud->setCenter(pointClouds[i]->center);
			keyPointClouds.push_back(keyPointCloud);
		}

		//find rotation/translation matrix for pointclouds relative to initial cloud and transform
		for (int i = 1; i < num_meshes; i++) {
			// Point Cloud Registration
			std::cout << "Registering point clouds..." << std::endl;
			//procrustes(keyPointClouds[0], keyPointClouds[i]);
			RegistrationTools::ScaledTransformation transMatrix;
			double rmse; unsigned finalPointCount;
			ICPRegistrationTools::Parameters params;
			params.nbMaxIterations = 800; params.convType = ICPRegistrationTools::MAX_ERROR_CONVERGENCE;
			params.adjustScale = false; params.maxThreadCount = 0; params.finalOverlapRatio = 0.99;
			ICPRegistrationTools::RESULT_TYPE returnVal = AncillaryFunctions::ICPRegister(pointClouds[0].get(), nullptr, pointClouds[i].get(),
				params, transMatrix, rmse, finalPointCount);
		
			Vector3 translationVector; Matrix3x3 rotationMatrix;
			//Append values to eigen rotation matrix
			for (int32_t RowIter = 0; RowIter < 3; RowIter++) {
				for (int32_t ColIter = 0; ColIter < 3; ColIter++) {
					rotationMatrix(RowIter, ColIter) = transMatrix.R.getValue(RowIter, ColIter);
				}
			}
			translationVector(0) = transMatrix.T[0]; translationVector(1) = transMatrix.T[1]; translationVector(2) = transMatrix.T[2];
			pointClouds[i]->transform(rotationMatrix, translationVector);
			keyPointClouds[i]->transform(rotationMatrix, translationVector);
		}
		return keyPointClouds;
	}

	void constructOctree(std::unique_ptr<Mesh_Types::MESH>& mesh_)
	{
		mesh_->octree = std::make_shared<DgmOctree>(&mesh_->pointcloud);
		mesh_->octree->build();
		mesh_->octree_enabled = true;
	}

	void constructKDTree(std::unique_ptr<Mesh_Types::MESH>& mesh_)
	{
		mesh_->kdtree = std::make_shared<KDTree>();
		mesh_->kdtree->buildFromCloud(&mesh_->pointcloud);
		mesh_->kdtree_enabled = true;
	}

	std::shared_ptr<Mesh_Types::POINT_CLOUD> pointCloudDifferencing(
		std::shared_ptr<Mesh_Types::POINT_CLOUD>& pc_reference,
		std::shared_ptr<Mesh_Types::POINT_CLOUD>& pc_compare
	)
	{
		DgmOctree* octreeReference = new DgmOctree(pc_reference.get());
		DgmOctree* octreeCompared = new DgmOctree(pc_compare.get());
		octreeReference->build();
		CCVector3 bbMin0, bbMin1, bbMax0, bbMax1;
		octreeReference->getBoundingBox(bbMin0, bbMax0);
		octreeCompared->build();
		octreeCompared->getBoundingBox(bbMin1, bbMax1);

		uchar level = octreeReference->findBestLevelForComparisonWithOctree(octreeCompared);
		if (level < 6) //dont want to go below a certain level of precision
			level = 6;

		//Compute distances between two point clouds
		DistanceComputationTools::Cloud2CloudDistancesComputationParams c2cParams;
		c2cParams.octreeLevel = level; c2cParams.localModel = LOCAL_MODEL_TYPES::LS;
		c2cParams.kNNForLocalModel = 10;
		int dist_return_additions = DistanceComputationTools::computeCloud2CloudDistances(pc_compare.get(), pc_reference.get(), c2cParams, nullptr, octreeCompared, octreeReference);
		int dist_return_subtractions = DistanceComputationTools::computeCloud2CloudDistances(pc_reference.get(), pc_compare.get(), c2cParams, nullptr, octreeReference, octreeCompared);

		const auto convertDistanceToRGB = [](
			float distance,
			float min_distance,
			float max_distance
			)
		{
			float val = (distance - min_distance) / (max_distance - min_distance);
			return val ;
		};

		Mesh_Types::POINT_CLOUD pc1, pc2;
		//First compute the min/max distance and mean distance of scalar points in compared cloud
		float min_distance_add, min_distance_sub = std::numeric_limits<float>::max();
		float max_distance_add, max_distance_sub = std::numeric_limits<float>::min();
		float mean_add, mean_sub = 0.0; std::vector<float> median_vector; float median_add, median_sub = 0.0;
		if (dist_return_additions > 0) {
			for (int i = 0; i < pc_compare->size(); i++) {
				Mesh_Types::POINT pnt = pc_compare->getPointFull(i);
				if (pnt.scalar < min_distance_add)
					min_distance_add = pnt.scalar;
				if (pnt.scalar > max_distance_add)
					max_distance_add = pnt.scalar;
				mean_add += pnt.scalar;
				median_vector.push_back(pnt.scalar);
			}
			std::sort(median_vector.begin(), median_vector.end());
			size_t size_vector = median_vector.size();
			if (size_vector % 2 == 0)
				median_add = (median_vector[size_vector / 2 - 1] + median_vector[size_vector / 2]) / 2;
			else
				median_add = median_vector[size_vector / 2];
			mean_add /= pc_compare->size();
		}
		median_vector.clear();
		if (dist_return_subtractions > 0) {
			for (int i = 0; i < pc_reference->size(); i++) {
				Mesh_Types::POINT pnt = pc_reference->getPointFull(i);
				if (pnt.scalar < min_distance_sub)
					min_distance_sub = pnt.scalar;
				if (pnt.scalar > max_distance_sub)
					max_distance_sub = pnt.scalar;
				mean_sub += pnt.scalar;
				median_vector.push_back(pnt.scalar);
			}
			std::sort(median_vector.begin(), median_vector.end());
			size_t size_vector = median_vector.size();
			if (size_vector % 2 == 0)
				median_sub = (median_vector[size_vector / 2 - 1] + median_vector[size_vector / 2]) / 2;
			else
				median_sub = median_vector[size_vector / 2];
			mean_sub /= pc_reference->size();
		}
		if (dist_return_additions > 0 || dist_return_subtractions > 0) {

			Mesh_Types::POINT_CLOUD pc1;
			for (size_t j = 0; j < pc_compare->size(); j++) {
				Mesh_Types::POINT point = pc_compare->getPointFull(j);
				//uchar color = convertDistanceToRGB(point.scalar, min_distance_add, max_distance_add);
				uchar color = convertDistanceToRGB(point.scalar, 0, 1);
				float a = (1 - color) / 0.25;	//invert and group
				int X = std::floor(a);	//this is the integer part
				float Y = std::floor(255 * (a - X)); //fractional part from 0 to 255
				uchar r, g, b;
				/*switch (X)
				{
					case 0: r = 255; g = Y; b = 0; break;
					case 1: r = 255 - Y; g = 255; b = 0; break;
					case 2: r = 0; g = 255; b = Y; break;
					case 3: r = 0; g = 255 - Y; b = 255; break;
					case 4: r = 0; g = 0; b = 255; break;
				}*/
				switch (X)
				{
				case 1: r = 255 - (255 * 0.0); g = 0; b = 0; break;
				case 2: r = 255 - (255 * 0.25); g = 0; b = 0; break;
				case 3: r = 255 - (255 * 0.5); g = 0 ; b = 0; break;
				case 4: r = 255 - (255 * 0.75); g = 0; b = 0; break;
				}
				if (point.scalar > median_add) {
					point.color.r = r; point.color.g = g; point.color.b = b;
				}
				pc1.add(point);
			}
			/*bool createNew = true;
			std::unique_ptr<Mesh_Types::MESH> new_mesh_ = std::make_unique<Mesh_Types::MESH>();
			new_mesh_->set_name("differenced_cloud(additions)");
			new_mesh_->set_uuid(generate_uuid());
			new_mesh_->pointcloud.setPointCloud(pc1.get());
			if (createNew) {
				new_mesh_->pointcloud.setColorsAvailable(true);
				new_mesh_->pointcloud.init();
				FileHandler3D::convert_to_3DTiles(new_mesh_);
				std::string file_name = new_mesh_->get_name() + ".json";
				gui->CallJS("SetPointCloudSource", std::string_format("%s", file_name.c_str()),
					std::string_format("%s", new_mesh_->get_uuid().c_str()));
			}*/
			for (size_t j = 0; j < pc_reference->size(); j++) {
				Mesh_Types::POINT point = pc_reference->getPointFull(j);
				uchar color = convertDistanceToRGB(point.scalar, 0, 1);
				float a = (1 - color) / 0.25;	//invert and group
				int X = std::floor(a);	//this is the integer part
				float Y = std::floor(255 * (a - X)); //fractional part from 0 to 255
				uchar r, g, b;
				switch (X)
				{
				case 1: g = 255 - (255 * 0.0); r = 0; b = 0; break;
				case 2: g = 255 - (255 * 0.25); r = 0; b = 0; break;
				case 3: g = 255 - (255 * 0.5); r = 0; b = 0; break;
				case 4: g = 255 - (255 * 0.75); r = 0; b = 0; break;
				}
				/*switch (X)
				{
				case 0: r = 255; g = Y; b = 0; break;
				case 1: r = 255 - Y; g = 255; b = 0; break;
				case 2: r = 0; g = 255; b = Y; break;
				case 3: r = 0; g = 255 - Y; b = 255; break;
				case 4: r = 0; g = 0; b = 255; break;
				}*/
				if (point.scalar > median_add) {
					point.color.r = r; point.color.g = g; point.color.b = b;
					pc1.add(point);
				}
				/*uchar color = convertDistanceToRGB(point.scalar, min_distance_sub, max_distance_sub);
				if (point.scalar > median_sub) {
					point.color.r = color; point.color.g = 255; point.color.b = 255;
					pc1.add(point);
				}*/
			}
			/*std::unique_ptr<Mesh_Types::MESH> new_mesh_2 = std::make_unique<Mesh_Types::MESH>();
			new_mesh_2->set_name("differenced_cloud(subtractions)");
			new_mesh_2->set_uuid(generate_uuid());
			new_mesh_2->pointcloud.setPointCloud(pc2.get());
			if (createNew) {
				new_mesh_2->pointcloud.setColorsAvailable(true);
				new_mesh_2->pointcloud.init();
				FileHandler3D::convert_to_3DTiles(new_mesh_2);
				std::string file_name = new_mesh_2->get_name() + ".json";
				gui->CallJS("SetPointCloudSource", std::string_format("%s", file_name.c_str()),
					std::string_format("%s", new_mesh_2->get_uuid().c_str()));
			}*/
		}
		free(octreeCompared); free(octreeReference);

		return std::make_shared<Mesh_Types::POINT_CLOUD>(pc1);
	}

	std::vector<Mesh_Types::POINT_CLOUD::PointType> nearestKSearch(
		std::unique_ptr<Mesh_Types::MESH>& mesh_,
		PointCoordinateType* point,
		double k
	)
	{
		if (!mesh_->kdtree_enabled)
			constructKDTree(mesh_);
		std::vector<Mesh_Types::POINT_CLOUD::PointType> points;
		std::vector<unsigned> indices;
		size_t num_3d = mesh_->kdtree->findPointsLyingToDistance(point, k, 1e-5, indices);
		for (size_t j = 0; j < num_3d; j++) {
			Mesh_Types::POINT_CLOUD::PointType point(mesh_->pointcloud.getPointFull(indices[j]));
			points.push_back(point);
		}
		return points;
	}

	std::vector<Mesh_Types::POINT_CLOUD::PointType> radiusSearch(
		std::unique_ptr<Mesh_Types::MESH>& mesh_,
		PointCoordinateType* point,
		double radius
	)
	{
		if (!mesh_->kdtree_enabled)
			constructKDTree(mesh_);
		std::vector<Mesh_Types::POINT_CLOUD::PointType> points;
		std::vector<unsigned> indices;
		size_t num_3d = mesh_->kdtree->findPointsLyingToDistance(point, radius, 0.1, indices);
		for (size_t j = 0; j < num_3d; j++) {
			Mesh_Types::POINT_CLOUD::PointType point(mesh_->pointcloud.getPointFull(indices[j]));
			points.push_back(point);
		}
		return points;
	}

	//generate heightmap from mesh input
	cv::Mat generateHeightMap(
		std::unique_ptr<Mesh_Types::MESH>& mesh_,
		float adjusted_width_x, float adjusted_width_y,
		std::vector<int>* points
	)
	{
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

		Precision currAlt;
		cv::Mat matPlaneMask(adjusted_width_y, adjusted_width_x, CV_8UC1, 0.0);

		if (points->size() == 0) {
			size_t size = mesh_->pointcloud.size();
			points->resize(size);
			std::iota(std::begin(*points), std::end(*points), 0);
		}
		for (int i = 0; i < points->size(); i++) {

			currAlt = mesh_->pointcloud.getPointFull((*points)[i]).coord.z;
			float row = convert(mesh_->pointcloud.getPointFull((*points)[i]).coord.y, mesh_->pointcloud.min_vertex[1], mesh_->pointcloud.horizontal_step);
			float col = convert(mesh_->pointcloud.getPointFull((*points)[i]).coord.x, mesh_->pointcloud.min_vertex[0], mesh_->pointcloud.horizontal_step);
			int alt = convertAlt(currAlt, mesh_->pointcloud.min_vertex[2], mesh_->pointcloud.max_vertex[2]);

			if (row >= 0 && col >= 0 && row < adjusted_width_y - 1 && col < adjusted_width_x - 1) {
				if (matPlaneMask.at<uchar>(row, col) < alt) {
					matPlaneMask.at<uchar>(row, col) = alt;
				}
			}
		}

		return matPlaneMask;
	}

	// generate heightmap from point cloud input
	cv::Mat generateHeightMap(
		std::shared_ptr<Mesh_Types::POINT_CLOUD>& pointCloud,
		float adjusted_width_x, float adjusted_width_y,
		std::vector<int>* points
	)
	{
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

		Precision currAlt;
		cv::Mat matPlaneMask(adjusted_width_y, adjusted_width_x, CV_8UC1, 0.0);

		if (points->size() == 0) {
			size_t size = pointCloud->size();
			points->resize(size);
			std::iota(std::begin(*points), std::end(*points), 0);
		}
		for (int i = 0; i < points->size(); i++) {

			currAlt = pointCloud->getPointFull((*points)[i]).coord.z;
			float row = convert(pointCloud->getPointFull((*points)[i]).coord.y, pointCloud->min_vertex[1], pointCloud->horizontal_step);
			float col = convert(pointCloud->getPointFull((*points)[i]).coord.x, pointCloud->min_vertex[0], pointCloud->horizontal_step);
			int alt = convertAlt(currAlt, pointCloud->min_vertex[2], pointCloud->max_vertex[2]);

			if (row >= 0 && col >= 0 && row < adjusted_width_y - 1 && col < adjusted_width_x - 1) {
				if (matPlaneMask.at<uchar>(row, col) < alt) {
					matPlaneMask.at<uchar>(row, col) = alt;
				}
			}
		}

		return matPlaneMask;
	}

	namespace AncillaryFunctions {

		ICPRegistrationTools::RESULT_TYPE ICPRegister(
			GenericIndexedCloudPersist* inputModelCloud,
			GenericIndexedMesh* inputModelMesh,
			GenericIndexedCloudPersist* inputDataCloud,
			const ICPRegistrationTools::Parameters& params,
			ICPRegistrationTools::ScaledTransformation& transform,
			double& finalRMS,
			unsigned& finalPointCount,
			GenericProgressCallback* progressCb/*=nullptr*/
		)
		{
			if (!inputModelCloud || !inputDataCloud)
			{
				assert(false);
				return ICPRegistrationTools::RESULT_TYPE::ICP_NOTHING_TO_DO;
			}

			//hopefully the user will understand it's not possible ;)
			finalRMS = -1.0;

			Garbage<GenericIndexedCloudPersist> cloudGarbage;
			Garbage<ScalarField> sfGarbage;

			bool registerWithNormals = (params.normalsMatching != ICPRegistrationTools::NO_NORMAL);

			//DATA CLOUD (will move)
			DataCloud data;
			{
				//we also want to use the same number of points for registration as initially defined by the user!
				unsigned dataSamplingLimit = params.finalOverlapRatio != 1.0 ? static_cast<unsigned>(params.samplingLimit / params.finalOverlapRatio) : params.samplingLimit;

				if (inputDataCloud->size() == 0)
				{
					return ICPRegistrationTools::RESULT_TYPE::ICP_NOTHING_TO_DO;
				}
				else if (inputDataCloud->size() > dataSamplingLimit)
				{
					//we resample the cloud if it's too big (speed increase)
					data.cloud = CloudSamplingTools::subsampleCloudRandomly(inputDataCloud, dataSamplingLimit);
					if (!data.cloud)
					{
						return ICPRegistrationTools::RESULT_TYPE::ICP_ERROR_NOT_ENOUGH_MEMORY;
					}
					cloudGarbage.add(data.cloud);

					//if we need to resample the weights as well
					if (params.dataWeights)
					{
						data.weights = new ScalarField("ResampledDataWeights");
						sfGarbage.add(data.weights);

						unsigned destCount = data.cloud->size();
						if (data.weights->resizeSafe(destCount))
						{
							for (unsigned i = 0; i < destCount; ++i)
							{
								unsigned pointIndex = data.cloud->getPointGlobalIndex(i);
								data.weights->setValue(i, params.dataWeights->getValue(pointIndex));
							}
							data.weights->computeMinAndMax();
						}
						else
						{
							//not enough memory
							return ICPRegistrationTools::RESULT_TYPE::ICP_ERROR_NOT_ENOUGH_MEMORY;
						}
					}
				}
				else //no need to resample
				{
					//we still create a 'fake' reference cloud with all the points
					data.cloud = new ReferenceCloud(inputDataCloud);
					cloudGarbage.add(data.cloud);
					if (!data.cloud->addPointIndex(0, inputDataCloud->size()))
					{
						//not enough memory
						return ICPRegistrationTools::RESULT_TYPE::ICP_ERROR_NOT_ENOUGH_MEMORY;
					}
					if (params.dataWeights)
					{
						//we use the input weights
						data.weights = new ScalarField(*params.dataWeights);
						sfGarbage.add(data.weights);
					}
				}

				//eventually we'll need a scalar field on the data cloud
				if (!data.cloud->enableScalarField())
				{
					//not enough memory
					return ICPRegistrationTools::RESULT_TYPE::ICP_ERROR_NOT_ENOUGH_MEMORY;
				}

				//we need normals to register with normals ;)
				registerWithNormals &= inputDataCloud->normalsAvailable();
			}
			assert(data.cloud);

			//octree level for cloud/mesh distances computation
			unsigned char meshDistOctreeLevel = 8;

			//MODEL ENTITY (reference, won't move)
			ModelCloud model;
			if (inputModelMesh)
			{
				assert(!params.modelWeights);

				if (inputModelMesh->size() == 0)
				{
					return ICPRegistrationTools::RESULT_TYPE::ICP_ERROR_INVALID_INPUT;
				}

				//we'll use the mesh vertices to estimate the right octree level
				DgmOctree dataOctree(data.cloud);
				DgmOctree modelOctree(inputModelCloud);
				if (dataOctree.build() < static_cast<int>(data.cloud->size()) || modelOctree.build() < static_cast<int>(inputModelCloud->size()))
				{
					//an error occurred during the octree computation: probably there's not enough memory
					return ICPRegistrationTools::RESULT_TYPE::ICP_ERROR_NOT_ENOUGH_MEMORY;
				}

				meshDistOctreeLevel = dataOctree.findBestLevelForComparisonWithOctree(&modelOctree);

				//we need normals to register with normals ;)
				registerWithNormals &= inputModelMesh->normalsAvailable();
			}
			else /*if (inputModelCloud)*/
			{
				if (inputModelCloud->size() == 0)
				{
					return ICPRegistrationTools::RESULT_TYPE::ICP_ERROR_INVALID_INPUT;
				}
				else if (inputModelCloud->size() > params.samplingLimit)
				{
					//we resample the cloud if it's too big (speed increase)
					ReferenceCloud* subModelCloud = CloudSamplingTools::subsampleCloudRandomly(inputModelCloud, params.samplingLimit);
					if (!subModelCloud)
					{
						//not enough memory
						return ICPRegistrationTools::RESULT_TYPE::ICP_ERROR_NOT_ENOUGH_MEMORY;
					}
					cloudGarbage.add(subModelCloud);

					//if we need to resample the weights as well
					if (params.modelWeights)
					{
						model.weights = new ScalarField("ResampledModelWeights");
						sfGarbage.add(model.weights);

						unsigned destCount = subModelCloud->size();
						if (model.weights->resizeSafe(destCount))
						{
							for (unsigned i = 0; i < destCount; ++i)
							{
								unsigned pointIndex = subModelCloud->getPointGlobalIndex(i);
								model.weights->setValue(i, params.modelWeights->getValue(pointIndex));
							}
							model.weights->computeMinAndMax();
						}
						else
						{
							//not enough memory
							return ICPRegistrationTools::RESULT_TYPE::ICP_ERROR_NOT_ENOUGH_MEMORY;
						}
					}
					model.cloud = subModelCloud;
				}
				else
				{
					//we use the input cloud and weights
					model.cloud = inputModelCloud;
					model.weights = params.modelWeights;
				}
				assert(model.cloud);

				//we need normals to register with normals ;)
				registerWithNormals &= inputModelCloud->normalsAvailable();
			}

			//for partial overlap
			unsigned maxOverlapCount = 0;
			std::vector<ScalarType> overlapDistances;
			if (params.finalOverlapRatio < 1.0)
			{
				//we pre-allocate the memory to sort distance values later
				try
				{
					overlapDistances.resize(data.cloud->size());
				}
				catch (const std::bad_alloc&)
				{
					//not enough memory
					return ICPRegistrationTools::RESULT_TYPE::ICP_ERROR_NOT_ENOUGH_MEMORY;
				}
				maxOverlapCount = static_cast<unsigned>(params.finalOverlapRatio*data.cloud->size());
				assert(maxOverlapCount != 0);
			}

			//Closest Point Set (see ICP algorithm)
			if (inputModelMesh)
			{
				data.CPSetPlain = new PointCloud;
				cloudGarbage.add(data.CPSetPlain);
			}
			else
			{
				data.CPSetRef = new ReferenceCloud(model.cloud);
				cloudGarbage.add(data.CPSetRef);
			}

			//per-point couple weights
			ScalarField* coupleWeights = nullptr;
			if (model.weights || data.weights || registerWithNormals)
			{
				coupleWeights = new ScalarField("CoupleWeights");
				sfGarbage.add(coupleWeights);
			}

			//we compute the initial distance between the two clouds (and the CPSet by the way)
			//data.cloud->forEach(ScalarFieldTools::SetScalarValueToNaN); //DGM: done automatically in computeCloud2CloudDistances now
			if (inputModelMesh)
			{
				assert(data.CPSetPlain);
				DistanceComputationTools::Cloud2MeshDistancesComputationParams c2mDistParams;
				c2mDistParams.octreeLevel = meshDistOctreeLevel;
				c2mDistParams.signedDistances = params.useC2MSignedDistances;
				c2mDistParams.CPSet = data.CPSetPlain;
				c2mDistParams.maxThreadCount = params.maxThreadCount;
				if (DistanceComputationTools::computeCloud2MeshDistances(data.cloud, inputModelMesh, c2mDistParams, progressCb) < 0)
				{
					//an error occurred during distances computation...
					return ICPRegistrationTools::RESULT_TYPE::ICP_ERROR_DIST_COMPUTATION;
				}
			}
			else if (inputModelCloud)
			{
				assert(data.CPSetRef);
				DistanceComputationTools::Cloud2CloudDistancesComputationParams c2cDistParams;
				c2cDistParams.CPSet = data.CPSetRef;
				c2cDistParams.maxThreadCount = params.maxThreadCount;
				if (DistanceComputationTools::computeCloud2CloudDistances(data.cloud, model.cloud, c2cDistParams, progressCb) < 0)
				{
					//an error occurred during distances computation...
					return ICPRegistrationTools::RESULT_TYPE::ICP_ERROR_DIST_COMPUTATION;
				}
			}
			else
			{
				assert(false);
			}

			FILE* fTraceFile = nullptr;
	#ifdef CC_DEBUG
			fTraceFile = fopen("registration_trace_log.csv", "wt");
	#endif
			if (fTraceFile)
			{
				fprintf(fTraceFile, "Iteration; RMS; Point count;\n");
			}

			double lastStepRMS = -1.0;
			double initialDeltaRMS = -1.0;
			ICPRegistrationTools::ScaledTransformation currentTrans;
			ICPRegistrationTools::RESULT_TYPE result = ICPRegistrationTools::RESULT_TYPE::ICP_ERROR;

			for (unsigned iteration = 0;; ++iteration)
			{
				if (progressCb && progressCb->isCancelRequested())
				{
					result = ICPRegistrationTools::RESULT_TYPE::ICP_ERROR_CANCELED_BY_USER;
					break;
				}

				//shall we remove the farthest points?
				bool pointOrderHasBeenChanged = false;
				if (params.filterOutFarthestPoints)
				{
					NormalDistribution N;
					N.computeParameters(data.cloud);
					if (N.isValid())
					{
						ScalarType mu;
						ScalarType sigma2;
						N.getParameters(mu, sigma2);
						ScalarType maxDistance = static_cast<ScalarType>(mu + 2.5*sqrt(sigma2));

						DataCloud filteredData;
						filteredData.cloud = new ReferenceCloud(data.cloud->getAssociatedCloud());
						cloudGarbage.add(filteredData.cloud);

						if (data.CPSetRef)
						{
							filteredData.CPSetRef = new ReferenceCloud(data.CPSetRef->getAssociatedCloud()); //we must also update the CPSet!
							cloudGarbage.add(filteredData.CPSetRef);
						}
						else if (data.CPSetPlain)
						{
							filteredData.CPSetPlain = new PointCloud; //we must also update the CPSet!
							cloudGarbage.add(filteredData.CPSetPlain);
						}

						if (data.weights)
						{
							filteredData.weights = new ScalarField("ResampledDataWeights");
							sfGarbage.add(filteredData.weights);
						}

						unsigned pointCount = data.cloud->size();
						if (!filteredData.cloud->reserve(pointCount)
							|| (filteredData.CPSetRef && !filteredData.CPSetRef->reserve(pointCount))
							|| (filteredData.CPSetPlain && !filteredData.CPSetPlain->reserve(pointCount))
							|| (filteredData.weights && !filteredData.weights->reserveSafe(pointCount)))
						{
							//not enough memory
							result = ICPRegistrationTools::RESULT_TYPE::ICP_ERROR_NOT_ENOUGH_MEMORY;
							break;
						}

						//we keep only the points with "not too high" distances
						for (unsigned i = 0; i < pointCount; ++i)
						{
							if (data.cloud->getPointScalarValue(i) <= maxDistance)
							{
								filteredData.cloud->addPointIndex(data.cloud->getPointGlobalIndex(i));
								if (filteredData.CPSetRef)
									filteredData.CPSetRef->addPointIndex(data.CPSetRef->getPointGlobalIndex(i));
								else if (filteredData.CPSetPlain)
									filteredData.CPSetPlain->addPoint(*(data.CPSetPlain->getPoint(i)));
								if (filteredData.weights)
									filteredData.weights->addElement(data.weights->getValue(i));
							}
						}

						//resize should be ok as we have called reserve first
						filteredData.cloud->resize(filteredData.cloud->size()); //should always be ok as current size < pointCount
						if (filteredData.CPSetRef)
							filteredData.CPSetRef->resize(filteredData.CPSetRef->size());
						else if (filteredData.CPSetPlain)
							filteredData.CPSetPlain->resize(filteredData.CPSetPlain->size());
						if (filteredData.weights)
							filteredData.weights->resize(filteredData.weights->currentSize());

						//replace old structures by new ones
						cloudGarbage.destroy(data.cloud);
						if (data.CPSetRef)
							cloudGarbage.destroy(data.CPSetRef);
						else if (data.CPSetPlain)
							cloudGarbage.destroy(data.CPSetPlain);
						if (data.weights)
							sfGarbage.destroy(data.weights);
						data = filteredData;

						pointOrderHasBeenChanged = true;
					}
				}

				//shall we ignore/remove some points based on their distance?
				DataCloud trueData;
				unsigned pointCount = data.cloud->size();
				if (maxOverlapCount != 0 && pointCount > maxOverlapCount)
				{
					assert(overlapDistances.size() >= pointCount);
					for (unsigned i = 0; i < pointCount; ++i)
					{
						overlapDistances[i] = data.cloud->getPointScalarValue(i);
						assert(overlapDistances[i] == overlapDistances[i]);
					}

					ParallelSort(overlapDistances.begin(), overlapDistances.begin() + pointCount);

					assert(maxOverlapCount != 0);
					ScalarType maxOverlapDist = overlapDistances[maxOverlapCount - 1];

					DataCloud filteredData;
					filteredData.cloud = new ReferenceCloud(data.cloud->getAssociatedCloud());
					if (data.CPSetRef)
					{
						filteredData.CPSetRef = new ReferenceCloud(data.CPSetRef->getAssociatedCloud()); //we must also update the CPSet!
						cloudGarbage.add(filteredData.CPSetRef);
					}
					else if (data.CPSetPlain)
					{
						filteredData.CPSetPlain = new PointCloud; //we must also update the CPSet!
						cloudGarbage.add(filteredData.CPSetPlain);
					}
					cloudGarbage.add(filteredData.cloud);
					if (data.weights)
					{
						filteredData.weights = new ScalarField("ResampledDataWeights");
						sfGarbage.add(filteredData.weights);
					}

					if (!filteredData.cloud->reserve(pointCount) //should be maxOverlapCount in theory, but there may be several points with the same value as maxOverlapDist!
						|| (filteredData.CPSetRef && !filteredData.CPSetRef->reserve(pointCount))
						|| (filteredData.CPSetPlain && !filteredData.CPSetPlain->reserve(pointCount))
						|| (filteredData.CPSetPlain && !filteredData.CPSetPlain->enableScalarField()) //don't forget the scalar field with the nearest triangle index
						|| (filteredData.weights && !filteredData.weights->reserveSafe(pointCount)))
					{
						//not enough memory
						result = ICPRegistrationTools::RESULT_TYPE::ICP_ERROR_NOT_ENOUGH_MEMORY;
						break;
					}

					//we keep only the points with "not too high" distances
					for (unsigned i = 0; i < pointCount; ++i)
					{
						if (data.cloud->getPointScalarValue(i) <= maxOverlapDist)
						{
							filteredData.cloud->addPointIndex(data.cloud->getPointGlobalIndex(i));
							if (filteredData.CPSetRef)
							{
								filteredData.CPSetRef->addPointIndex(data.CPSetRef->getPointGlobalIndex(i));
							}
							else if (filteredData.CPSetPlain)
							{
								filteredData.CPSetPlain->addPoint(*(data.CPSetPlain->getPoint(i)));
								//don't forget the scalar field with the nearest triangle index!
								filteredData.CPSetPlain->addPointScalarValue(data.CPSetPlain->getPointScalarValue(i));
							}
							if (filteredData.weights)
							{
								filteredData.weights->addElement(data.weights->getValue(i));
							}
						}
					}
					assert(filteredData.cloud->size() >= maxOverlapCount);

					//resize should be ok as we have called reserve first
					filteredData.cloud->resize(filteredData.cloud->size()); //should always be ok as current size < pointCount
					if (filteredData.CPSetRef)
						filteredData.CPSetRef->resize(filteredData.CPSetRef->size());
					else if (filteredData.CPSetPlain)
						filteredData.CPSetPlain->resize(filteredData.CPSetPlain->size());
					if (filteredData.weights)
						filteredData.weights->resize(filteredData.weights->currentSize());

					//(temporarily) replace old structures by new ones
					trueData = data;
					data = filteredData;
				}

				//update couple weights (if any)
				if (coupleWeights)
				{
					unsigned count = data.cloud->size();
					assert(model.weights || data.weights || registerWithNormals);
					assert(!model.weights || (data.CPSetRef && data.CPSetRef->size() == count));

					if (coupleWeights->currentSize() != count && !coupleWeights->resizeSafe(count))
					{
						//not enough memory to store weights
						result = ICPRegistrationTools::RESULT_TYPE::ICP_ERROR_NOT_ENOUGH_MEMORY;
						break;
					}

					for (unsigned i = 0; i < count; ++i)
					{
						double w = 1.0;
						if (registerWithNormals)
						{
							//retrieve the data point normal
							const CCVector3* Nd = data.cloud->getNormal(i);

							//retrieve the nearest model point normal
							CCVector3 Nm;
							if (inputModelMesh)
							{
								unsigned triIndex = static_cast<unsigned>(data.CPSetPlain->getPointScalarValue(i));
								assert(triIndex >= 0 && triIndex < inputModelMesh->size());
								inputModelMesh->interpolateNormals(triIndex, *data.cloud->getPoint(i), Nm);
							}
							else
							{
								Nm = *inputModelCloud->getNormal(i);
							}

							//we assume the vectors are unitary!
							PointCoordinateType dp = Nd->dot(Nm);

							switch (params.normalsMatching)
							{
							case ICPRegistrationTools::NORMALS_MATCHING::OPPOSITE_NORMALS:
							{
								w = acos(dp) / M_PI; // 0 rad --> w = 0 / pi/2 rad --> w = 0.5 / pi rad --> w = 1
							}
							break;

							case ICPRegistrationTools::NORMALS_MATCHING::SAME_SIDE_NORMALS:
							{
								w = 1.0 - acos(dp) / M_PI; // 0 rad --> w = 1 / pi/2 rad --> w = 0.5 / pi rad --> w = 0
							}
							break;

							case ICPRegistrationTools::NORMALS_MATCHING::DOUBLE_SIDED_NORMALS:
							{
								dp = std::abs(dp);
								w = 1.0 - acos(dp) / M_PI_2; // 0 rad --> w = 1 / pi/2 rad --> w = 0
							}
							break;

							default:
								assert(false);
								break;
							}
						}
						if (data.weights)
						{
							w *= data.weights->getValue(i);
						}
						if (model.weights)
						{
							//model weights are only supported with a reference cloud!
							ScalarType wm = model.weights->getValue(data.CPSetRef->getPointGlobalIndex(i));
							w *= wm;
						}
						coupleWeights->setValue(i, static_cast<ScalarType>(w));
					}
					coupleWeights->computeMinAndMax();
				}

				//we can now compute the best registration transformation for this step
				//(now that we have selected the points that will be used for registration!)
				{
					//if we use weights, we have to compute weighted RMS!!!
					double meanSquareValue = 0.0;
					double wiSum = 0.0; //we normalize the weights by their sum

					for (unsigned i = 0; i < data.cloud->size(); ++i)
					{
						ScalarType V = data.cloud->getPointScalarValue(i);
						if (ScalarField::ValidValue(V))
						{
							double wi = 1.0;
							if (coupleWeights)
							{
								ScalarType w = coupleWeights->getValue(i);
								if (!ScalarField::ValidValue(w))
									continue;
								wi = std::abs(w);
							}
							double Vd = wi * V;
							wiSum += wi * wi;
							meanSquareValue += Vd * Vd;
						}
					}

					//12/11/2008 - A.BEY: ICP guarantees only the decrease of the squared distances sum (not the distances sum)
					double meanSquareError = (wiSum != 0 ? static_cast<ScalarType>(meanSquareValue / wiSum) : 0);

					double rms = sqrt(meanSquareError);

					if (fTraceFile)
					{
						fprintf(fTraceFile, "%u; %f; %u;\n", iteration, rms, data.cloud->size());
					}

					if (iteration == 0)
					{
						//progress notification
						if (progressCb)
						{
							//on the first iteration, we init/show the dialog
							if (progressCb->textCanBeEdited())
							{
								progressCb->setMethodTitle("Clouds registration");
								char buffer[256];
								sprintf(buffer, "Initial RMS = %f\n", rms);
								progressCb->setInfo(buffer);
							}
							progressCb->update(0);
							progressCb->start();
						}

						finalRMS = rms;
						finalPointCount = data.cloud->size();


						if (LESS_THAN_EPSILON(rms))
						{
							//nothing to do
							result = ICPRegistrationTools::RESULT_TYPE::ICP_NOTHING_TO_DO;
							break;
						}
					}
					else
					{
						assert(lastStepRMS >= 0.0);

						if (rms > lastStepRMS) //error increase!
						{
							result = iteration == 1 ? ICPRegistrationTools::RESULT_TYPE::ICP_NOTHING_TO_DO : ICPRegistrationTools::RESULT_TYPE::ICP_APPLY_TRANSFO;
							break;
						}

						//error update (RMS)
						double deltaRMS = lastStepRMS - rms;
						//should be better!
						assert(deltaRMS >= 0.0);

						//we update the global transformation matrix
						if (currentTrans.R.isValid())
						{
							if (transform.R.isValid())
								transform.R = currentTrans.R * transform.R;
							else
								transform.R = currentTrans.R;

							transform.T = currentTrans.R * transform.T;
						}

						if (params.adjustScale)
						{
							transform.s *= currentTrans.s;
							transform.T *= currentTrans.s;
						}

						transform.T += currentTrans.T;

						finalRMS = rms;
						finalPointCount = data.cloud->size();

						//stop criterion
						if ((params.convType == ICPRegistrationTools::CONVERGENCE_TYPE::MAX_ERROR_CONVERGENCE && deltaRMS < params.minRMSDecrease) //convergence reached
							|| (params.convType == ICPRegistrationTools::CONVERGENCE_TYPE::MAX_ITER_CONVERGENCE && iteration >= params.nbMaxIterations) //max iteration reached
							)
						{
							result = ICPRegistrationTools::RESULT_TYPE::ICP_APPLY_TRANSFO;
							break;
						}

						//progress notification
						if (progressCb)
						{
							if (progressCb->textCanBeEdited())
							{
								char buffer[256];
								if (coupleWeights)
									sprintf(buffer, "Weighted RMS = %f [-%f]\n", rms, deltaRMS);
								else
									sprintf(buffer, "RMS = %f [-%f]\n", rms, deltaRMS);
								progressCb->setInfo(buffer);
							}
							if (iteration == 1)
							{
								initialDeltaRMS = deltaRMS;
								progressCb->update(0);
							}
							else
							{
								assert(initialDeltaRMS >= 0.0);
								float progressPercent = static_cast<float>((initialDeltaRMS - deltaRMS) / (initialDeltaRMS - params.minRMSDecrease)*100.0);
								progressCb->update(progressPercent);
							}
						}
					}

					lastStepRMS = rms;
				}

				//single iteration of the registration procedure
				currentTrans = ICPRegistrationTools::ScaledTransformation();
				if (!RegistrationProcedure(data.cloud,
					data.CPSetRef ? static_cast<GenericCloud*>(data.CPSetRef) : static_cast<GenericCloud*>(data.CPSetPlain),
					currentTrans,
					params.adjustScale,
					coupleWeights))
				{
					result = ICPRegistrationTools::RESULT_TYPE::ICP_ERROR_REGISTRATION_STEP;
					break;
				}

				//restore original data sets (if any were stored)
				if (trueData.cloud)
				{
					cloudGarbage.destroy(data.cloud);
					if (data.CPSetRef)
						cloudGarbage.destroy(data.CPSetRef);
					else if (data.CPSetPlain)
						cloudGarbage.destroy(data.CPSetPlain);
					if (data.weights)
						sfGarbage.destroy(data.weights);
					data = trueData;
				}

				//shall we filter some components of the resulting transformation?
				if (params.transformationFilters != ICPRegistrationTools::SKIP_NONE)
				{
					//filter translation (in place)
					RegistrationTools::FilterTransformation(currentTrans, params.transformationFilters, currentTrans);
				}

				//get rotated data cloud
				if (!data.rotatedCloud || pointOrderHasBeenChanged)
				{
					//we create a new structure, with rotated points
					PointCloud* rotatedDataCloud = PointProjectionTools::applyTransformation(data.cloud, currentTrans);
					if (!rotatedDataCloud)
					{
						//not enough memory
						result = ICPRegistrationTools::RESULT_TYPE::ICP_ERROR_NOT_ENOUGH_MEMORY;
						break;
					}
					//replace data.rotatedCloud
					if (data.rotatedCloud)
						cloudGarbage.destroy(data.rotatedCloud);
					data.rotatedCloud = rotatedDataCloud;
					cloudGarbage.add(data.rotatedCloud);

					//update data.cloud
					data.cloud->clear();
					data.cloud->setAssociatedCloud(data.rotatedCloud);
					if (!data.cloud->addPointIndex(0, data.rotatedCloud->size()))
					{
						//not enough memory
						result = ICPRegistrationTools::RESULT_TYPE::ICP_ERROR_NOT_ENOUGH_MEMORY;
						break;
					}
				}
				else
				{
					//we simply have to rotate the existing temporary cloud
					currentTrans.apply(*data.rotatedCloud);
					data.rotatedCloud->invalidateBoundingBox(); //invalidate bb

					//DGM: warning, we must manually invalidate the ReferenceCloud bbox after rotation!
					data.cloud->invalidateBoundingBox();
				}

				//compute (new) distances to model
				if (inputModelMesh)
				{
					DistanceComputationTools::Cloud2MeshDistancesComputationParams c2mDistParams;
					c2mDistParams.octreeLevel = meshDistOctreeLevel;
					c2mDistParams.signedDistances = params.useC2MSignedDistances;
					c2mDistParams.CPSet = data.CPSetPlain;
					c2mDistParams.maxThreadCount = params.maxThreadCount;
					if (DistanceComputationTools::computeCloud2MeshDistances(data.cloud, inputModelMesh, c2mDistParams) < 0)
					{
						//an error occurred during distances computation...
						result = ICPRegistrationTools::RESULT_TYPE::ICP_ERROR_REGISTRATION_STEP;
						break;
					}
				}
				else if (inputDataCloud)
				{
					DistanceComputationTools::Cloud2CloudDistancesComputationParams c2cDistParams;
					c2cDistParams.CPSet = data.CPSetRef;
					c2cDistParams.maxThreadCount = params.maxThreadCount;
					if (DistanceComputationTools::computeCloud2CloudDistances(data.cloud, model.cloud, c2cDistParams) < 0)
					{
						//an error occurred during distances computation...
						result = ICPRegistrationTools::RESULT_TYPE::ICP_ERROR_REGISTRATION_STEP;
						break;
					}
				}
				else
				{
					assert(false);
				}
			}

			//end of tracefile
			if (fTraceFile)
			{
				fclose(fTraceFile);
				fTraceFile = nullptr;
			}

			//end of progress notification
			if (progressCb)
			{
				progressCb->stop();
			}

			return result;
		}

		bool RegistrationProcedure(
			GenericCloud* P, //data
			GenericCloud* X, //model
			ICPRegistrationTools::ScaledTransformation& trans,
			bool adjustScale/*=false*/,
			ScalarField* coupleWeights/*=0*/,
			PointCoordinateType aPrioriScale/*=1.0f*/
		)
		{
			//resulting transformation (R is invalid on initialization, T is (0,0,0) and s==1)
			trans.R.invalidate();
			trans.T = CCVector3d(0, 0, 0);
			trans.s = 1.0;

			if (P == nullptr || X == nullptr || P->size() != X->size() || P->size() < 3)
				return false;

			//centers of mass
			CCVector3 Gp = coupleWeights ? GeometricalAnalysisTools::ComputeWeightedGravityCenter(P, coupleWeights) : GeometricalAnalysisTools::ComputeGravityCenter(P);
			CCVector3 Gx = coupleWeights ? GeometricalAnalysisTools::ComputeWeightedGravityCenter(X, coupleWeights) : GeometricalAnalysisTools::ComputeGravityCenter(X);

			//specific case: 3 points only
			//See section 5.A in Horn's paper
			if (P->size() == 3)
			{
				//compute the first set normal
				P->placeIteratorAtBeginning();
				const CCVector3* Ap = P->getNextPoint();
				const CCVector3* Bp = P->getNextPoint();
				const CCVector3* Cp = P->getNextPoint();
				CCVector3 Np(0, 0, 1);
				{
					Np = (*Bp - *Ap).cross(*Cp - *Ap);
					double norm = Np.normd();
					if (LESS_THAN_EPSILON(norm))
					{
						return false;
					}
					Np /= static_cast<PointCoordinateType>(norm);
				}
				//compute the second set normal
				X->placeIteratorAtBeginning();
				const CCVector3* Ax = X->getNextPoint();
				const CCVector3* Bx = X->getNextPoint();
				const CCVector3* Cx = X->getNextPoint();
				CCVector3 Nx(0, 0, 1);
				{
					Nx = (*Bx - *Ax).cross(*Cx - *Ax);
					double norm = Nx.normd();
					if (LESS_THAN_EPSILON(norm))
					{
						return false;
					}
					Nx /= static_cast<PointCoordinateType>(norm);
				}
				//now the rotation is simply the rotation from Nx to Np, centered on Gx
				CCVector3 a = Np.cross(Nx);
				if (LESS_THAN_EPSILON(a.norm()))
				{
					trans.R = SquareMatrix(3);
					trans.R.toIdentity();
					if (Np.dot(Nx) < 0)
					{
						trans.R.scale(-PC_ONE);
					}
				}
				else
				{
					double cos_t = Np.dot(Nx);
					assert(cos_t > -1.0 && cos_t < 1.0); //see above
					double s = sqrt((1 + cos_t) * 2);
					double q[4] = { s / 2, a.x / s, a.y / s, a.z / s }; //don't forget to normalize the quaternion
					double qnorm = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
					assert(qnorm >= ZERO_TOLERANCE_D);
					qnorm = sqrt(qnorm);
					q[0] /= qnorm;
					q[1] /= qnorm;
					q[2] /= qnorm;
					q[3] /= qnorm;
					trans.R.initFromQuaternion(q);
				}

				if (adjustScale)
				{
					double sumNormP = (*Bp - *Ap).norm() + (*Cp - *Bp).norm() + (*Ap - *Cp).norm();
					sumNormP *= aPrioriScale;
					if (LESS_THAN_EPSILON(sumNormP))
					{
						return false;
					}
					double sumNormX = (*Bx - *Ax).norm() + (*Cx - *Bx).norm() + (*Ax - *Cx).norm();
					trans.s = static_cast<PointCoordinateType>(sumNormX / sumNormP); //sumNormX / (sumNormP * Sa) in fact
				}

				//we deduce the first translation
				trans.T = Gx.toDouble() - (trans.R*Gp) * (aPrioriScale*trans.s); //#26 in besl paper, modified with the scale as in jschmidt

				//we need to find the rotation in the (X) plane now
				{
					CCVector3 App = trans.apply(*Ap);
					CCVector3 Bpp = trans.apply(*Bp);
					CCVector3 Cpp = trans.apply(*Cp);

					double C = 0;
					double S = 0;
					CCVector3 Ssum(0, 0, 0);
					CCVector3 rx;
					CCVector3 rp;

					rx = *Ax - Gx;
					rp = App - Gx;
					C = rx.dot(rp);
					Ssum = rx.cross(rp);

					rx = *Bx - Gx;
					rp = Bpp - Gx;
					C += rx.dot(rp);
					Ssum += rx.cross(rp);

					rx = *Cx - Gx;
					rp = Cpp - Gx;
					C += rx.dot(rp);
					Ssum += rx.cross(rp);

					S = Ssum.dot(Nx);
					double Q = sqrt(S*S + C * C);
					if (LESS_THAN_EPSILON(Q))
					{
						return false;
					}

					PointCoordinateType sin_t = static_cast<PointCoordinateType>(S / Q);
					PointCoordinateType cos_t = static_cast<PointCoordinateType>(C / Q);
					PointCoordinateType inv_cos_t = 1 - cos_t;

					const PointCoordinateType& l1 = Nx.x;
					const PointCoordinateType& l2 = Nx.y;
					const PointCoordinateType& l3 = Nx.z;

					PointCoordinateType l1_inv_cos_t = l1 * inv_cos_t;
					PointCoordinateType l3_inv_cos_t = l3 * inv_cos_t;

					SquareMatrix R(3);
					//1st column
					R.m_values[0][0] = cos_t + l1 * l1_inv_cos_t;
					R.m_values[0][1] = l2 * l1_inv_cos_t + l3 * sin_t;
					R.m_values[0][2] = l3 * l1_inv_cos_t - l2 * sin_t;

					//2nd column
					R.m_values[1][0] = l2 * l1_inv_cos_t - l3 * sin_t;
					R.m_values[1][1] = cos_t + l2 * l2*inv_cos_t;
					R.m_values[1][2] = l2 * l3_inv_cos_t + l1 * sin_t;

					//3rd column
					R.m_values[2][0] = l3 * l1_inv_cos_t + l2 * sin_t;
					R.m_values[2][1] = l2 * l3_inv_cos_t - l1 * sin_t;
					R.m_values[2][2] = cos_t + l3 * l3_inv_cos_t;

					trans.R = R * trans.R;
					trans.T = Gx.toDouble() - (trans.R*Gp) * (aPrioriScale*trans.s); //update T as well
				}
			}
			else
			{
				CCVector3 bbMin;
				CCVector3 bbMax;
				X->getBoundingBox(bbMin, bbMax);

				//if the data cloud is equivalent to a single point (for instance
				//it's the case when the two clouds are very far away from
				//each other in the ICP process) we try to get the two clouds closer
				CCVector3 diag = bbMax - bbMin;
				if (LESS_THAN_EPSILON(std::abs(diag.x) + std::abs(diag.y) + std::abs(diag.z)))
				{
					trans.T = (Gx - Gp * aPrioriScale).toDouble();
					return true;
				}

				//Cross covariance matrix, eq #24 in Besl92 (but with weights, if any)
				SquareMatrixd Sigma_px = (coupleWeights ? GeometricalAnalysisTools::ComputeWeightedCrossCovarianceMatrix(P, X, Gp, Gx, coupleWeights)
					: GeometricalAnalysisTools::ComputeCrossCovarianceMatrix(P, X, Gp, Gx));
				if (!Sigma_px.isValid())
					return false;

#define USE_SVD
#ifdef USE_SVD

				SquareMatrixd U, S, V;
				if (!Sigma_px.svd(S, U, V))
					return false;
				SquareMatrixd UT = U.transposed();

				trans.R = V * UT;

#else
				//transpose sigma_px
				SquareMatrixd Sigma_px_t = Sigma_px.transposed();

				SquareMatrixd Aij = Sigma_px - Sigma_px_t;

				double trace = Sigma_px.trace(); //that is the sum of diagonal elements of sigma_px

				SquareMatrixd traceI3(3); //create the I matrix with eigvals equal to trace
				traceI3.m_values[0][0] = trace;
				traceI3.m_values[1][1] = trace;
				traceI3.m_values[2][2] = trace;

				SquareMatrixd bottomMat = Sigma_px + Sigma_px_t - traceI3;

				//we build up the registration matrix (see ICP algorithm)
				SquareMatrixd QSigma(4); //#25 in the paper (besl)

				QSigma.m_values[0][0] = trace;

				QSigma.m_values[0][1] = QSigma.m_values[1][0] = Aij.m_values[1][2];
				QSigma.m_values[0][2] = QSigma.m_values[2][0] = Aij.m_values[2][0];
				QSigma.m_values[0][3] = QSigma.m_values[3][0] = Aij.m_values[0][1];

				QSigma.m_values[1][1] = bottomMat.m_values[0][0];
				QSigma.m_values[1][2] = bottomMat.m_values[0][1];
				QSigma.m_values[1][3] = bottomMat.m_values[0][2];

				QSigma.m_values[2][1] = bottomMat.m_values[1][0];
				QSigma.m_values[2][2] = bottomMat.m_values[1][1];
				QSigma.m_values[2][3] = bottomMat.m_values[1][2];

				QSigma.m_values[3][1] = bottomMat.m_values[2][0];
				QSigma.m_values[3][2] = bottomMat.m_values[2][1];
				QSigma.m_values[3][3] = bottomMat.m_values[2][2];

				//we compute its eigenvalues and eigenvectors
				SquareMatrixd eigVectors;
				std::vector<double> eigValues;
				if (!Jacobi<double>::ComputeEigenValuesAndVectors(QSigma, eigVectors, eigValues, false))
				{
					//failure
					return false;
				}

				//as Besl says, the best rotation corresponds to the eigenvector associated to the biggest eigenvalue
				double qR[4];
				double maxEigValue = 0;
				Jacobi<double>::GetMaxEigenValueAndVector(eigVectors, eigValues, maxEigValue, qR);

				//these eigenvalue and eigenvector correspond to a quaternion --> we get the corresponding matrix
				trans.R.initFromQuaternion(qR);
#endif
				if (adjustScale)
				{
					//two accumulators
					double acc_num = 0.0;
					double acc_denom = 0.0;

					//now deduce the scale (refer to "Point Set Registration with Integrated Scale Estimation", Zinsser et. al, PRIP 2005)
					X->placeIteratorAtBeginning();
					P->placeIteratorAtBeginning();

					unsigned count = X->size();
					assert(P->size() == count);
					for (unsigned i = 0; i < count; ++i)
					{
						//'a' refers to the data 'A' (moving) = P
						//'b' refers to the model 'B' (not moving) = X
						CCVector3d a_tilde = trans.R * (*(P->getNextPoint()) - Gp);	// a_tilde_i = R * (a_i - a_mean)
						CCVector3d b_tilde = (*(X->getNextPoint()) - Gx);			// b_tilde_j =     (b_j - b_mean)

						acc_num += b_tilde.dot(a_tilde);
						acc_denom += a_tilde.dot(a_tilde);
					}

					//DGM: acc_2 can't be 0 because we already have checked that the bbox is not a single point!
					assert(acc_denom > 0.0);
					trans.s = static_cast<PointCoordinateType>(std::abs(acc_num / acc_denom));
				}

				//and we deduce the translation
				trans.T = Gx.toDouble() - (trans.R*Gp) * (aPrioriScale*trans.s); //#26 in besl paper, modified with the scale as in jschmidt
			}

			return true;
		}

		void returnPointsFromImage(
			std::unique_ptr<Mesh_Types::MESH>& mesh_,
			std::vector<int>& points,
			std::vector<int>& points_bb,
			cv::Mat& img,
			bounded_box_data& bounding_box
		)
		{
			// Convert point to grid coordinate system
			const auto convert = [](
				float measure, float min, float step
				) {
				float adjustedMeasure = (measure - min) / step;
				return adjustedMeasure;
			};

			for (int i = 0; i < points.size(); i++) {
				float x_pos = mesh_->pointcloud.getPointFull(points[i]).coord.x;
				float y_pos = mesh_->pointcloud.getPointFull(points[i]).coord.y;

				float row = convert(y_pos, mesh_->pointcloud.min_vertex[1], mesh_->pointcloud.horizontal_step);
				float col = convert(x_pos, mesh_->pointcloud.min_vertex[0], mesh_->pointcloud.horizontal_step);

				if (col >= bounding_box.x && col <= bounding_box.x + bounding_box.width)
					if (row >= bounding_box.y && row <= bounding_box.y + bounding_box.height)
						points_bb.push_back(points[i]);
			}
		}

		bool resampleCellAtLevel(
			const DgmOctree::octreeCell& cell,
			void** additionalParameters,
			NormalizedProgress* progress/*=nullptr*/
		)
		{
			KDTree* kdtree = static_cast<KDTree*>(additionalParameters[0]);
			DgmOctree* octree = static_cast<DgmOctree*>(additionalParameters[1]);
			Mesh_Types::POINT_CLOUD* pointcloud = static_cast<Mesh_Types::POINT_CLOUD*>(additionalParameters[2]);
			Mesh_Types::POINT_CLOUD* new_pointcloud = static_cast<Mesh_Types::POINT_CLOUD*>(additionalParameters[3]);
			CloudSamplingTools::RESAMPLING_CELL_METHOD resamplingMethod = *static_cast<CloudSamplingTools::RESAMPLING_CELL_METHOD*>(additionalParameters[4]);
			Mesh_Types::POINT pnt;
			unsigned index_nearest;
			PointCoordinateType* pnt_coord = new PointCoordinateType[3];

			if (resamplingMethod == CloudSamplingTools::RESAMPLING_CELL_METHOD::CELL_GRAVITY_CENTER)
			{
				const CCVector3* P = Neighbourhood(cell.points).getGravityCenter();
				if (!P)
					return false;
				pnt.coord = *P;
				pnt_coord[0] = pnt.coord.x; pnt_coord[1] = pnt.coord.y; pnt_coord[2] = pnt.coord.z;
			}
			else //if (resamplingMethod == CloudSamplingTools::RESAMPLING_CELL_METHOD::CELL_CENTER)
			{
				CCVector3 center;
				cell.parentOctree->computeCellCenter(cell.truncatedCode, cell.level, center, true);
				pnt.coord = center;
				pnt_coord[0] = pnt.coord.x; pnt_coord[1] = pnt.coord.y; pnt_coord[2] = pnt.coord.z;
			}
			kdtree->findNearestNeighbour(pnt_coord, index_nearest, 1000); //1000 only to ensure at least one point is captured
			pnt.color = pointcloud->getPointFull(index_nearest).color;
			new_pointcloud->add(pnt);
			free(pnt_coord);

			return true;
		}

		bool getPointsInCell(
			const Mesh_Types::POINT_CLOUD& pointCloud,
			DgmOctree& octree,
			DgmOctree::CellCode cellCode,
			unsigned char level,
			std::vector<CCVector3>& subset,
			bool isCodeTruncated/*=false*/
		)
		{
			unsigned char bitDec = DgmOctree::GET_BIT_SHIFT(level);
			if (!isCodeTruncated)
			{
				cellCode >>= bitDec;
			}

			unsigned cellIndex = octree.getCellIndex(cellCode, bitDec);
			//check that cell exists!
			if (cellIndex < octree.getNumberOfProjectedPoints())
			{

				//binary shift for cell code truncation
				unsigned char bitDec = DgmOctree::GET_BIT_SHIFT(level);

				//we look for the first index in 'm_thePointsAndTheirCellCodes' corresponding to this cell
				DgmOctree::cellsContainer pointsAndTheirCellCodes = octree.pointsAndTheirCellCodes();
				DgmOctree::cellsContainer::const_iterator p = pointsAndTheirCellCodes.begin() + cellIndex;
				DgmOctree::CellCode searchCode = (p->theCode >> bitDec);

				//while the (partial) cell code matches this cell
				while ((p != pointsAndTheirCellCodes.end()) && ((p->theCode >> bitDec) == searchCode))
				{
					subset.push_back(*pointCloud.getPoint(p->theIndex));
					++p;
				}
			}

			return true;
		}

		bool cmp(std::pair<int, std::vector<int>>& a,
			std::pair<int, std::vector<int>>& b)
		{
			return a.second.size() > b.second.size();
		}

		void sort(
			std::map<int, std::vector<int>>& M,
			std::vector<std::pair<int, std::vector<int>>>& B
		)
		{
			std::vector<std::pair<int, std::vector<int>> > A;

			for (auto& it : M) {
				A.push_back(it);
			}

			std::sort(A.begin(), A.end(), [](std::pair<int, std::vector<int>>& a,
				std::pair<int, std::vector<int>>& b) {return cmp(a, b); });
			B = A;
		}
	}
}
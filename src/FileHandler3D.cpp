
#include "../include/FileHandler3D.h"

namespace FileHandler3D {
	bool load_ply(
		const std::string& input_filename,
		std::unique_ptr<Mesh_Types::MESH>& mesh_
	)
	{
		try
		{
			std::ifstream ss_temp(input_filename, std::ios::binary);
			PlyReader::PlyFile file_template(ss_temp);
			bool vertex_type_float, normal_type_float;

			//Get LLA
			std::vector<std::string> comments = file_template.comments;
			for (int i = 0; i < comments.size(); i++) {
				int comp = strcmp(comments[i].substr(0, comments[i].find_first_of(' ')).c_str(), "LLA");
				if (comp == 0) {
					double lat, lon, alt;

					std::string lla_data = comments[i];
					//std::list<std::string> data = std::str_split(comments[i], ' ');
					std::string lat_index = lla_data.substr(lla_data.find_first_of(' ')+1, lla_data.size());
					std::string lon_index = lat_index.substr(lat_index.find_first_of(' ')+1, lat_index.size());
					std::string alt_index = lon_index.substr(lon_index.find_first_of(' ')+1, lon_index.size());
					lat_index = lat_index.substr(0, lat_index.find_first_of(' ')); lat = std::stod(lat_index);
					lon_index = lon_index.substr(0, lon_index.find_first_of(' ')); lon = std::stod(lon_index);
					alt_index = alt_index.substr(0, alt_index.size()); alt = std::stod(alt_index);

					mesh_->lla = RedRiver::LLA(lat, lon, alt);
				}
			}

			//Get vertex type
			std::vector<PlyReader::PlyElement> elements = file_template.get_elements();
			for (unsigned char i = 0; i < elements.size(); i++) {
				if (elements[i].name == "vertex") {
					std::vector<PlyReader::PlyProperty> properties = elements[i].properties;
					for (unsigned char j = 0; j < properties.size(); j++) {
						if (properties[j].name == "x") {
							if (properties[j].propertyType == PlyReader::PlyProperty::Type::FLOAT32) {
								vertex_type_float = true;
							}
							else {
								vertex_type_float = false;
							}
						}
						if (properties[j].name == "nx") {
							if (properties[j].propertyType == PlyReader::PlyProperty::Type::FLOAT32) {
								normal_type_float = true;
							}
							else {
								normal_type_float = false;
							}
						}
					}
				}
			}
			mesh_->pointcloud.header = file_template.comments;

			std::vector<double> verticesDouble, vertexNormalsDouble;
			std::vector<float> verticesFloat, vertexNormalsFloat;
			std::vector<unsigned char> vertexColors;
			std::vector<int> vertexIndices;

			uint64_t vertexCount, normalCount;
			size_t numVertices, numVertexColors, numVertexNormals;
			bool vertexNormalZeros = false;

			if (!vertex_type_float) {
				vertexCount = file_template.request_properties_from_element("vertex", { "x", "y", "z" }, verticesDouble);
				numVertices = verticesDouble.size();
			}
			else {
				vertexCount = file_template.request_properties_from_element("vertex", { "x", "y", "z" }, verticesFloat);
				numVertices = verticesFloat.size();
			}
			if (!normal_type_float) {
				normalCount = file_template.request_properties_from_element("vertex", { "nx", "ny", "nz" }, vertexNormalsDouble);
				numVertexNormals = vertexNormalsDouble.size();
			}
			else {
				normalCount = file_template.request_properties_from_element("vertex", { "nx", "ny", "nz" }, vertexNormalsFloat);
				numVertexNormals = vertexNormalsFloat.size();
			}

			uint64_t colorCount = file_template.request_properties_from_element("vertex", { "red", "green", "blue" }, vertexColors);
			numVertexColors = vertexColors.size();
			//Meshes not currently supported for PLY file type
			//uint64_t faceCount = file_template.request_properties_from_element("face", {"vertex_index"}, vertexIndices);

			if (vertexCount != (numVertices / POLYGON_TYPE)
				&& colorCount != (numVertexColors / POLYGON_TYPE)
				&& normalCount != (numVertexNormals / POLYGON_TYPE)
				)
			{
				std::cout << "Error: Only triangle mesh is supported. Abort" << std::endl;
				if (colorCount != (numVertexColors / POLYGON_TYPE)) {
					std::cout << "Error: Vertex colour count != vertex coordinate count. Abort" << std::endl;
				}
				if (normalCount != (numVertexNormals / POLYGON_TYPE)) {
					std::cout << "Error: Issue with vertex normals. Aborting" << std::endl;
				}
				return false;
			}

			file_template.read(ss_temp);


			bool vertexColorZeros = std::all_of(vertexColors.begin(), vertexColors.end(), [](int i) { return i == 0; });
			bool coloredCloud = numVertexColors > 0 && !vertexColorZeros ? true : false;
			bool has_normals = (numVertexNormals > 0) && !vertexNormalZeros ? true : false;

			Precision prec_lim_max = (std::numeric_limits<Precision>::max)();
			Precision prec_lim_min = (std::numeric_limits<Precision>::min)();
			Precision largest_x = prec_lim_min, largest_y = prec_lim_min, largest_z = prec_lim_min;
			Precision smallest_x = prec_lim_max, smallest_y = prec_lim_max, smallest_z = prec_lim_max;
			//convert to local point type
			Mesh_Types::POINT pnt;
			for (size_t i = 0; i < numVertices; i++) {
				int index = i % POLYGON_TYPE;
				Precision vertex_value = vertex_type_float ? static_cast<Precision>(verticesFloat[i]) : static_cast<Precision>(verticesDouble[i]);
				Precision normal_value;
				if (has_normals)
					if (normal_type_float)
						normal_value = static_cast<Precision>(vertexNormalsFloat[i]);
					else
						normal_value = static_cast<Precision>(vertexNormalsDouble[i]);
				else
					normal_value = 0.0;
				unsigned char color_value = coloredCloud ? static_cast<unsigned char>(vertexColors[i]) : 255; //set to all white vertices if uncolored
				if (index == 0) {
					pnt.coord.x = vertex_value;
					if (vertex_value > largest_x) { largest_x = vertex_value; }
					if (vertex_value < smallest_x) { smallest_x = vertex_value; }
					mesh_->pointcloud.center[0] += pnt.coord.x;
					pnt.color.r = color_value;
					pnt.normal.x = normal_value;
				}
				else if (index == 1) {
					pnt.coord.y = vertex_value;
					if (vertex_value > largest_y) { largest_y = vertex_value; }
					if (vertex_value < smallest_y) { smallest_y = vertex_value; }
					mesh_->pointcloud.center[1] += pnt.coord.y;
					pnt.color.g = color_value;
					pnt.normal.y = normal_value;
				}
				else {
					pnt.coord.z = vertex_value;
					if (vertex_value > largest_z) { largest_z = vertex_value; }
					if (vertex_value < smallest_z) { smallest_z = vertex_value; }
					mesh_->pointcloud.center[2] += pnt.coord.z;
					pnt.color.b = color_value;
					pnt.normal.z = normal_value;
					mesh_->pointcloud.add(pnt);
				}
			}
			numVertices = numVertices / POLYGON_TYPE;
			mesh_->pointcloud.center[0] /= numVertices; mesh_->pointcloud.center[1] /= numVertices; mesh_->pointcloud.center[2] /= numVertices;
			//Normalize coordinates if center not within a certain range
			Precision range_x = largest_x - smallest_x;
			Precision range_y = largest_y - smallest_y;
			Precision range_z = largest_z - smallest_z;
	#pragma omp parallel for
			for (size_t i = 0; i < mesh_->pointcloud.size(); i++) {
				Mesh_Types::POINT pnt = mesh_->pointcloud.getPointFull(i);
				if (abs(mesh_->pointcloud.center[0]) > range_x) {
					pnt.coord.x = ((pnt.coord.x - smallest_x) / (largest_x - smallest_x)) * (range_x);
				}
				if (abs(mesh_->pointcloud.center[1]) > range_y) {
					pnt.coord.y = ((pnt.coord.y - smallest_y) / (largest_y - smallest_y)) * (range_y);
				}
				if (abs(mesh_->pointcloud.center[2]) > range_z) {
					pnt.coord.z = ((pnt.coord.z - smallest_z) / (largest_z - smallest_z)) * (range_z);
				}
				mesh_->pointcloud.setPoint(i, pnt);
			}
			//TODO: Ask user if they want to change to aligned
			mesh_->pointcloud.center[0] = ((mesh_->pointcloud.center[0] - smallest_x) / (largest_x - smallest_x)) * (range_x);
			mesh_->pointcloud.center[1] = ((mesh_->pointcloud.center[1] - smallest_y) / (largest_y - smallest_y)) * (range_y);
			mesh_->pointcloud.center[2] = ((mesh_->pointcloud.center[2] - smallest_z) / (largest_z - smallest_z)) * (range_z);
			mesh_->pointcloud.setColorsAvailable(coloredCloud);
			mesh_->pointcloud.setNormalsAvailable(has_normals);
		}
		catch (const std::exception& e)
		{
			std::cerr << "Error: Could not load " << input_filename << ". " << e.what() << std::endl;
			return false;
		}
		return true;
	}

	bool load_mtl(
		std::string input_filename,
		std::unique_ptr<Mesh_Types::MESH>& mesh_
	)
	{
		FILE * file = fopen(input_filename.c_str(), "r");
		std::string parentPath = input_filename.substr(0, input_filename.find_last_of("/\\"));

		float floatBuff[3];
		char lineHeader[128];
		char stupidBuffer[1000];
		std::string fullPath;
		std::string relPath;
		while (1) {
			int res = fscanf(file, "%s", lineHeader);
			if (res == EOF)
				break;

			if (strcmp(lineHeader, "newmtl") == 0) {
				Mesh_Types::RR_MATERIAL tempMat;
				fscanf(file, "%s\n", &tempMat.materialId);

				//eat 8 lines containing material info, TODO
				for (int i = 0; i < 8; i++) {
					fgets(stupidBuffer, 1000, file);
				}

				fscanf(file, "map_Kd %s\n", stupidBuffer);
				std_filesystem::path matPath = stupidBuffer; 
 
 
				if (matPath.is_absolute()) {
					tempMat.imagePath = stupidBuffer;;
				}
				else {
					std_filesystem::path filename = std_filesystem::path(stupidBuffer).filename();
					tempMat.imagePath = parentPath + "\\"+filename.u8string();;
				}

				mesh_->materials.push_back(tempMat);

			}
		}
		return true;
	}

	bool load_obj(
		const std::string& input_filename,
		std::unique_ptr<Mesh_Types::MESH>& mesh_
	)
	{
		unsigned int vPosition = 0, vtPosition = 0, vnPosition = 0;

		FILE * file = fopen(input_filename.c_str(), "r");
		Mesh_Types::RR_MATERIAL *currentMat = nullptr;

		if (file == NULL) {
			return false;
		}

		while (1) {

			char lineHeader[128];
			int res = fscanf(file, "%s", lineHeader);
			if (res == EOF)
				break;

			if (strcmp(lineHeader, "v") == 0) {
				Mesh_Types::POINT pnt;
				fscanf(file, "%f %f %f\n", &pnt.coord.x, &pnt.coord.y, &pnt.coord.z);
				pnt.color.r = 255; pnt.color.g = 255; pnt.color.b = 255;
				mesh_->pointcloud.add(pnt);
				vPosition++;
			}
			else if (strcmp(lineHeader, "vt") == 0) {
				Mesh_Types::RR_TEXTURE<Precision> texture;
				fscanf(file, "%f %f\n", &(texture.u), &(texture.v));

				//obj and opengl have y coordinate in opposite direciton, reverse it
				texture.v = 1-texture.v;
				mesh_->textures.push_back(texture);

				vtPosition++;
			}
			else if (strcmp(lineHeader, "vn") == 0) {
				Mesh_Types::POINT* pnt = mesh_->pointcloud.getPointPointer(vnPosition);
				fscanf(file, "%f %f %f\n", &(pnt->normal.x), &(pnt->normal.y), &(pnt->normal.z));
				vnPosition++;
			}
			else if (strcmp(lineHeader, "f") == 0) {
				Mesh_Types::FACE face_;
				face_.material = currentMat;
				int matches;
				int tempfaces[9];
				if (vnPosition == 0 && vtPosition == 0) {
					matches = fscanf(file, "%d %d %d\n", tempfaces, tempfaces + 1, tempfaces + 2);
					for (int i = 0; i < 3; i++) {
						face_.vp[i] = tempfaces[i] - 1;
					}
				}
				else if (vnPosition > 0 && vtPosition > 0) {
					matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n",
						tempfaces, tempfaces + 3, tempfaces + 6,
						tempfaces + 1, tempfaces + 4, tempfaces + 7,
						tempfaces + 2, tempfaces + 5, tempfaces + 8);

					//obj face index starts at 1...
					for (int i = 0; i < 3; i++) {
						face_.vp[i] = tempfaces[i] - 1;
					}

					for (int i = 0; i < 3; i++) {
						face_.tp[i] = tempfaces[i + 3] - 1;
					}

					for (int i = 0; i < 3; i++) {
						face_.np[i] = tempfaces[i + 6] - 1;
					}

				}
				else if (vnPosition > 0 && vtPosition == 0) {
					matches = fscanf(file, "%d//%d %d//%d %d//%d\n", tempfaces, tempfaces + 3,
						tempfaces + 1, tempfaces + 4, tempfaces + 2, tempfaces + 5);
					//obj face index starts at 1...
					for (int i = 0; i < 3; i++) {
						face_.vp[i] = tempfaces[i] - 1;
					}

					for (int i = 0; i < 3; i++) {
						face_.np[i] = tempfaces[i + 3] - 1;
					}
				}
				else if (vnPosition == 0 && vtPosition > 0) {
					matches = fscanf(file, "%d/%d %d/%d %d/%d\n", tempfaces, tempfaces + 3,
						tempfaces + 1, tempfaces + 4, tempfaces + 2, tempfaces + 5);
					//obj face index starts at 1...
					for (int i = 0; i < 3; i++) {
						face_.vp[i] = tempfaces[i] - 1;
					}

					for (int i = 0; i < 3; i++) {
						face_.tp[i] = tempfaces[i + 3] - 1;
					}


				}

				if (matches % 3 != 0) {
					printf("File can't be read by our simple parser :-( Try exporting with other options\n");
					fclose(file);
					return false;
				}
				mesh_->faces.push_back(face_);
			}
			else if (strcmp(lineHeader, "#LLA") == 0) {
				float lat, lon, alt;
				fscanf(file, "%f %f %f\n", &lat, &lon, &alt);
				mesh_->lla = LLA(lat, lon, alt);
			}
			else if (strcmp(lineHeader, "#YPR") == 0) {
				float yaw, pitch, roll;
				fscanf(file, "%f %f %f\n", &yaw, &pitch, &roll);
				mesh_->setYPROffset(yaw, pitch, roll);
			}
			else if (strcmp(lineHeader, "#Z_Offset") == 0) {
				float z_offset;
				fscanf(file, "%f\n", &z_offset);
				mesh_->z_offset = z_offset;
			}
			//load in a material file
			else if (strcmp(lineHeader, "mtllib") == 0) {
				char pathString[200];
				fscanf(file, "%s\n", pathString);
				std_filesystem::path path = pathString;

				
				if (path.is_absolute()) {
					load_mtl(pathString, mesh_);
				}
				//NOTE: this results in undefined behavior if .mtl is relative path but not in same directory as our .obj mesh
				else {
					std::string absPath = std_filesystem::path(input_filename).parent_path().u8string() + "\\"+path.filename().u8string();
					load_mtl(absPath, mesh_);
				}

			}
			else if (strcmp(lineHeader, "usemtl") == 0) {
				std::string id;
				fscanf(file, "%s\n", &id);

				//find mat in list of mat and select it
				for (Mesh_Types::RR_MATERIAL& mat : mesh_->materials) {
					if (strcmp(mat.materialId.c_str(), id.c_str()) == 0) {
						currentMat = &mat;
						break;
					}
				}

			}
			else {
				// Comment, eat up the rest of the line
				char stupidBuffer[1000];
				fgets(stupidBuffer, 1000, file);
			}

		}
		if (mesh_->faces.size() > 0)
			mesh_->is_polygon_mesh = true;
		if (vtPosition != 0)
			mesh_->is_textured = true;
		if (vnPosition != 0)
			mesh_->pointcloud.setNormalsAvailable(true);

		fclose(file);
		return true;
	}
}
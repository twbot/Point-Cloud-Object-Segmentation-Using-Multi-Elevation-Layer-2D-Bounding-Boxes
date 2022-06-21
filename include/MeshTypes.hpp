#ifndef MeshTypes_h
#define MeshTypes_h

#include <memory>
#include <vector>
#include <random>

#include <CCGeom.h>
#include <GenericIndexedCloudPersist.h>
#include <DgmOctree.h>
#include <KdTree.h>

#include "Coordinates.h"
#include "IAABB.h"
#include "AABB.h"

using namespace Coordinates;

enum class PolygonTypes : int {
	Triangles = 3,
	Quadrilaterals = 4 //Not yet supported
};

enum class MeshPrimitives : int
{
	Points = 0,
	Lines = 1,
	Line_Loop = 2,
	Line_Strip = 3,
	Triangles = 4,
	Triangle_Strip = 5,
	Triangle_Fan = 6
};

const int POLYGON_TYPE = static_cast<int>(PolygonTypes::Triangles);
using Precision = float; //3D Tiles only supports float currently, but user may operate within the space of doubles

namespace Mesh_Types {

	using uchar = unsigned char;

	struct RR_MATERIAL {
		std::string imagePath;
		std::string materialId;
	};

	/*struct facet : public IAABB {
		int id;
		Eigen::Vector3f corners[3];
		Eigen::Vector3f trans_corners[3];
		Eigen::Vector3f centroid;
		Eigen::Vector3f normal;
		int positions[3];
		float angle = (std::numeric_limits<float>::min)();
		int text_position[3];
		int normal_position;
		AABB aabb;
		AABB getAABB() const {
			return aabb;
		};
		Eigen::Vector3f getCentroid() const {
			return centroid;
		};
	};*/

	struct RR_FACE : public IAABB {
		//vertex position
		unsigned int vp[POLYGON_TYPE] = { 0 };
		//texture position
		unsigned int tp[POLYGON_TYPE] = { 0 };
		//normal position
		unsigned int np[POLYGON_TYPE] = { 0 };

		//this is the material that this face pertains to 
		RR_MATERIAL *material = nullptr;

		AABB getAABB() const {
			return aabb;
		}

		Eigen::Vector3f getCentroid() const {
			return centroid;
		}

		//doesn't do normals yet
		std::string to_String(uint8_t faceIndex) {
			std::string output = "";
			output += std::to_string(vp[faceIndex]);
			output += "/" + std::to_string(tp[faceIndex]);
			return output;
		}
		
		private:
			//centroid of face
			Eigen::Vector3f centroid;
			//axis aligned bounding box for this polygonal face
			AABB aabb;
	};

	struct RR_COLOR {
		uchar r, g, b;
	};

	template<typename T>
	struct RR_TEXTURE {
		T u, v;
	};

	//texture specialized to float for reflection
	template<>
	struct RR_TEXTURE<float> {
		float u, v;
	};

	template<typename Type>
	struct RR_POINT {
		//typedef T Type;
		CCVector3 coord;
		RR_COLOR color;
		RR_TEXTURE<Type> texture;
		CCVector3 normal;
		ScalarType scalar;
		Type getCoordinate(int index) {
			if (index == 0)
				return coord.x;
		else if (index == 1)
				return coord.y;
			else if (index == 2)
				return coord.z;
			else return 0.0;
		}
		uchar getColor(int index) {
			if (index == 0)
				return color.r;
			else if (index == 1)
				return color.g;
			else if (index == 2)
				return color.b;
			else return 0;
		}
		RR_POINT() {
		}
		RR_POINT(Type x_, Type y_, Type z_)
		{
			coord.x = x_; coord.y = y_; coord.z = z_;
		}
		RR_POINT(Type pnt[3])
		{
			coord = coord.fromArray(pnt);
		}
		RR_POINT<Type> operator- (RR_POINT<Type> point) {
			RR_POINT<Type> pnt;
			pnt.coord = coord - point.coord;
			return pnt;
		}
		RR_POINT<Type> operator- (Type point[3]) {
			RR_POINT<Type> pnt; CCVector3 type;
			pnt.coord = coord - type.fromArray(point);
			return pnt;
		}
		RR_POINT<Type> operator+ (RR_POINT<Type> point) {
			RR_POINT<Type> pnt;
			pnt.coord = coord + point.coord;
			return pnt;
		}
		RR_POINT<Type> operator+ (Type point[3]) {
			RR_POINT<Type> pnt; CCVector3 type;
			pnt.coord = coord + type.fromArray(point);
			return pnt;
		}
	};

	template<typename Type>
	struct RR_POINTCLOUD : public CCCoreLib::GenericIndexedCloudPersist {
	public:
		//typedef T Type;
		typedef RR_POINT<Type> PointType;

		RR_POINTCLOUD() {
		}

		RR_POINTCLOUD(size_t size) {
			points.reserve(size);
		}
		//Copy constructor
		RR_POINTCLOUD(const RR_POINTCLOUD<Type>& pointcloud) {
			for (uchar i = 0; i < 3; i++) {
				center[i] = pointcloud.center[i];
				max_vertex[i] = pointcloud.max_vertex[i];
				min_vertex[i] = pointcloud.min_vertex[i];
				max_color[i] = pointcloud.max_color[i];
				min_color[i] = pointcloud.min_color[i];
			}
			header = pointcloud.header;
			has_normals = pointcloud.normalsAvailable();
			is_colored = pointcloud.colorsAvailable();
			points = pointcloud.get();
		}
		~RR_POINTCLOUD() {
		}

		union {
			Type center[3] = { 0.0 };
			struct center {
				Type x, y, z;
			};
		};
		union {

			//initialize max at min possible values for later comparison
			Type max_vertex[3] = {
			static_cast<Type>(std::numeric_limits<Precision>::min()),
			static_cast<Type>(std::numeric_limits<Precision>::min()),
			static_cast<Type>(std::numeric_limits<Precision>::min()) };

			struct max_vertex {
				Type x, y, z;
			};
		};
		union {
			//initialize min at max possible values for later comparison
			Type min_vertex[3] = {
			static_cast<Type>(std::numeric_limits<Precision>::max()),
			static_cast<Type>(std::numeric_limits<Precision>::max()),
			static_cast<Type>(std::numeric_limits<Precision>::max()) };
			struct min_vertex {
				Type x, y, z;
			};
		};
		union {

			Type min_color[3] = {
			static_cast<Type>(std::numeric_limits<uchar>::max()),
			static_cast<Type>(std::numeric_limits<uchar>::max()),
			static_cast<Type>(std::numeric_limits<uchar>::max()) };

			struct min_color {
				uchar r, g, b;
			};
		};
		union {

			Type max_color[3] = {
			static_cast<Type>(std::numeric_limits<uchar>::min()),
			static_cast<Type>(std::numeric_limits<uchar>::min()),
			static_cast<Type>(std::numeric_limits<uchar>::min()) };

			struct max_color {
				uchar r, g, b;
			};
		};

		Type horizontal_step = 0.25, vertical_step = 0.25;
		std::vector<std::string> header;
		void add(Type x, Type y, Type z) {
			PointType pnt;
			pnt.coord.x = x; pnt.coord.y = y; pnt.coord.z = z;
			points.push_back(pnt);
		}
		void add(PointType pnt) {
			points.push_back(pnt);
		}
		void clear() {
			points.clear();
			for (uchar i = 0; i < sizeof(max_vertex) / sizeof(Type); i++) {
				max_vertex[i] = std::numeric_limits<Type>::min();
				min_vertex[i] = std::numeric_limits<Type>::max();
				center[i] = 0.0;
			}
			for (uchar i = 0; i < sizeof(max_color) / sizeof(uchar); i++) {
				max_color[i] = std::numeric_limits<uchar>::min();
				min_color[i] = std::numeric_limits<Type>::max();
			}
		}
		void reset() {
			clear();
		}
		void setPointCloud(const std::vector<PointType>& pnts) {
			points = pnts;
		}
		void setCenter(Type c_[3]) {
			center[0] = c_[0]; center[1] = c_[1]; center[2] = c_[2];
		}
		void getBoundingBox(CCVector3& bbMin, CCVector3& bbMax) {
			bbMin = bbMin.fromArray(min_vertex);
			bbMax = bbMax.fromArray(max_vertex);
		}
		void forEach(std::function<void(const CCVector3&, ScalarType &)>) {}
		void placeIteratorAtBeginning() { iterator = 0; }
		std::vector<PointType> get() const { return points; }
		PointType getPointFull(unsigned index) { return points[index]; }
		PointType *getPointPointer(unsigned index) { return points.data() + index; }
		const CCVector3* getPoint(unsigned index) const { return &points[index].coord; }
		void getPoint(unsigned index, CCVector3& P) const { P = points[index].coord; }
		const CCVector3* getPointPersistentPtr(unsigned index) const { return &points[index].coord; }
		const CCVector3* getNormal(unsigned index) const { return &points[index].normal; }
		void setPoint(int index, PointType pnt) { points[index] = pnt; }
		unsigned size() const { return points.size(); }
		const CCVector3* getNextPoint() {
			CCVector3* pnt = &points[iterator].coord;
			iterator++;
			return pnt;
		}
		bool normalsAvailable() const { return has_normals; }
		bool setNormalsAvailable(bool value) {
			has_normals = value;
			return value;

		}
		bool colorsAvailable() const { return is_colored; }
		void setColorsAvailable(bool value) { is_colored = value; }
		bool enableScalarField() {
			if (!scalar_field_enabled)
				scalar_field_enabled = true;
			return scalar_field_enabled;
		}
		bool isScalarFieldEnabled() const { return scalar_field_enabled; }
		void setPointScalarValue(unsigned pointIndex, ScalarType value) {
			points[pointIndex].scalar = value;
		}
		ScalarType getPointScalarValue(unsigned pointIndex) const {
			CCVector3 pnt = points[pointIndex].coord;
			return pnt.norm2();
		}

		size_t kdtree_get_point_count() const { return size(); }
		std::vector<Type> getVerticesSerial() {
			std::vector<Type> vertices;
			for (size_t i = 0; i < points.size(); i++)
			{
				vertices.push_back(static_cast<Type>(points[i].coord.x));
				vertices.push_back(static_cast<Type>(points[i].coord.y));
				vertices.push_back(static_cast<Type>(points[i].coord.z));
			}
			return vertices;
		}
		std::vector<Type> getVertexTexturesSerial() {
			std::vector<Type> textures;
			for (size_t i = 0; i < points.size(); i++)
			{
				textures.push_back(static_cast<Type>(points[i].texture.u));
				textures.push_back(static_cast<Type>(points[i].texture.v));
			}
			return textures;
		}
		std::vector<uchar> getVertexColorsSerial() {
			std::vector<uchar> vertexColors;
			for (size_t i = 0; i < points.size(); i++)
			{
				vertexColors.push_back(points[i].color.r);
				vertexColors.push_back(points[i].color.g);
				vertexColors.push_back(points[i].color.b);
			}
			return vertexColors;
		}
		void transform(Matrix3x3 rotateMatrix, Vector3 translateVector) {
#pragma omp parallel for
			for (unsigned i = 0; i < points.size(); i++) {
				Vector3 temp_data(points[i].coord.x, points[i].coord.y, points[i].coord.z);
				temp_data = rotateMatrix * temp_data.transpose() + translateVector.transpose();
				points[i].coord = CCVector3(temp_data[0], temp_data[1], temp_data[2]);
			}
		}
		std::vector<Type> getNormalizedVertexColorsSerial(bool include_alpha) {
			std::vector<Type> vertexColors;
			for (size_t i = 0; i < points.size(); i++)
			{
				vertexColors.push_back((Type)(static_cast<Type>(points[i].color.r) / 255));
				vertexColors.push_back((Type)(static_cast<Type>(points[i].color.g) / 255));
				vertexColors.push_back((Type)(static_cast<Type>(points[i].color.b) / 255));
				if (include_alpha)
					vertexColors.push_back(1.0);
			}
			return vertexColors;
		}
		std::vector<Type> getVertexNormalsSerial() {
			std::vector<Type> vertexNormals;
			for (size_t i = 0; i < points.size(); i++)
			{
				vertexNormals.push_back(static_cast<Type>(points[i].normal.x));
				vertexNormals.push_back(static_cast<Type>(points[i].normal.y));
				vertexNormals.push_back(static_cast<Type>(points[i].normal.z));
			}
			return vertexNormals;
		}
		template <class BBOX>
		bool kdtree_get_bbox(BBOX&) const {
			return false;
		}
		Type kdtree_get_pt(const size_t index, int dim) const {
			return getPointFull(index).getCoordinate(dim);
		}
		void initMinMaxCoordinates() {
			std::vector<Type> vertices = getVerticesSerial();
			std::vector<uchar> vertexColors = getVertexColorsSerial();
			for (int i = 0; i < vertices.size(); i++) {
				int index = i % 3;
				float vertex_val = vertices[i];
				// Min/Max vertex
				if (vertex_val > max_vertex[index])
					max_vertex[index] = vertex_val;
				if (vertex_val < min_vertex[index])
					min_vertex[index] = vertex_val;

				if (vertexColors.size() > 0) {
					float normalized_color = (float)(vertexColors[i]) / 255;
					// Min/Max color
					if (normalized_color > max_color[index])
						max_color[index] = normalized_color;
					if (normalized_color < min_color[index])
						min_color[index] = normalized_color;
				}
			}
		}
		void init() {
			std::vector<Type> vertices = getVerticesSerial();
			std::vector<uchar> vertexColors = getVertexColorsSerial();
			center[0] = 0.0; center[1] = 0.0; center[2] = 0.0;
			for (int i = 0; i < vertices.size(); i++) {
				int index = i % 3;
				float vertex_val = vertices[i];
				//Find center
				if (index == 0) {
					center[0] += vertex_val;
				}
				else if (index == 1) {
					center[1] += vertex_val;
				}
				else if (index == 2) {
					center[2] += vertex_val;
				}


				// Min/Max vertex
				if (vertex_val > max_vertex[index])
					max_vertex[index] = vertex_val;
				if (vertex_val < min_vertex[index])
					min_vertex[index] = vertex_val;

				if (vertexColors.size() > 0) {
					float normalized_color = (float)(vertexColors[i]) / 255;
					// Min/Max color
					if (normalized_color > max_color[index])
						max_color[index] = normalized_color;
					if (normalized_color < min_color[index])
						min_color[index] = normalized_color;
				}
			}
			unsigned size_ = vertices.size() / POLYGON_TYPE;
			center[0] /= size_;
			center[1] /= size_;
			center[2] /= size_;
		}

		std::string name;
		std::string get_uuid() {
			return uuid;
		}

	private:
		bool has_normals = false;
		bool is_colored = false;
		bool scalar_field_enabled = true;
		unsigned iterator = 0;
	public:
		std::vector<PointType> points;

	};

	/*
	This is made for writing materials to gltf. Gltf requires that vertices with different materials are seperated
	into different chunks. To generate mesh chunks, call getMeshChunks on an RR_Mesh.
	*/
	struct RR_Mesh_Chunk {

		//serialized face data, referencing the linear position in the indexToVertex map
		std::vector<unsigned int> facesSerial;

		//Material that this Mesh Chunk is associated with
		RR_MATERIAL material;

		//this is a map with face indexes from pool associated with point
		std::map<std::string, RR_POINT<Precision>> indexToVertex;
		
		Precision max_vertex[3];

		Precision min_vertex[3];

		std::vector<Precision> getVerticesSerial() {
			std::vector<Precision> vertices;

			for (auto const&[key, val] : indexToVertex)
			{
				vertices.push_back(static_cast<float>(val.coord.x));
				vertices.push_back(static_cast<float>(val.coord.y));
				vertices.push_back(static_cast<float>(val.coord.z));
			}
			
			return vertices;
		}

		std::vector<Precision> getVertexTexturesSerial() {
			std::vector<Precision> textures;

			for (auto const&[key, val] : indexToVertex)
			{
				textures.push_back(static_cast<Precision>(val.texture.u));
				textures.push_back(static_cast<Precision>(val.texture.v));
			}

			return textures;
		}

		void initMinMaxCoordinates(std::vector<Precision> vertices) {
			for (int i = 0; i < 3; i++) {
				max_vertex[i] = vertices[i];
				min_vertex[i] = vertices[i];
			}

			for (int i = 0; i < vertices.size(); i++) {
				int index = i % 3;
				float vertex_val = vertices[i];
				// Min/Max vertex
				bool test = (vertex_val > max_vertex[index]);
				if (vertex_val > max_vertex[index])
					max_vertex[index] = vertex_val;
				if (vertex_val < min_vertex[index])
					min_vertex[index] = vertex_val;
			}
		}

	};

	template<typename Type>
	struct RR_MESH {
		//typedef T Type;
		bool is_polygon_mesh = false;
		bool is_textured = false;
		bool has_normals = false;
		bool octree_enabled = false;
		bool kdtree_enabled = false;
		RR_POINTCLOUD<Type> pointcloud;

		//materials must be written with absolute paths
		std::vector<RR_MATERIAL> materials;

		std::shared_ptr<CCCoreLib::KDTree> kdtree;
		std::shared_ptr<CCCoreLib::DgmOctree> octree;
		LLA lla;
		
		double z_offset = 0;

		/*
		Some filetypes store textures as a pool, while others associate a texture with each point,
		which is why we also have texture in point
		*/
		std::vector<RR_TEXTURE<Precision>> textures;

		std::vector<RR_FACE> faces;

		std::array<std::vector<int>, 3> getFaces() {
			std::array<std::vector<int>, 3> face_array;
			for (int i = 0; i < faces.size(); i++) {
				RR_FACE tmp_face = faces[i];
				for (uchar j = 0; j < POLYGON_TYPE; j++) {
					face_array[0].push_back(tmp_face.vp[j]);
					if (is_textured)
						face_array[1].push_back(tmp_face.tp[j]);
					if (has_normals)
						face_array[2].push_back(tmp_face.np[j]);
				}
			}
			return face_array;
		}

		std::vector<Eigen::Vector3i> getFacesDense3D() {

			std::vector<Eigen::Vector3i> faces_eigen;

			for (int i = 0; i < faces.size(); i++) {
				Eigen::Vector3i face(faces[i].vp[0], faces[i].vp[1], faces[i].vp[2]);
				faces_eigen.push_back(face);
			}
			return faces_eigen;
		}

		std::vector<Eigen::Vector3f> getPointsDense3D() {
			
			std::vector<Eigen::Vector3f> points_eigen;

			for (int i = 0; i < faces.size(); i++) {
				Eigen::Vector3i face(faces[i].vp[0], faces[i].vp[1], faces[i].vp[2]);
				Eigen::Vector3f point;
				for (int j = 0; j < POLYGON_TYPE; j++) {
					point[j] = pointcloud.getPointFull(face[j]).getCoordinate(j);
				}
				points_eigen.push_back(point);
			}
			return points_eigen;
		}

		void set_uuid(std::string new_uuid) {
			uuid = new_uuid;
		}

		std::string get_uuid() {
			return uuid;
		}

		void set_name(std::string new_name) {
			name = new_name;
		}

		std::string get_name() {
			return name;
		}

		void setYPROffset(double ypr[3]) {
			ypr_offset[0] = ypr[0];
			ypr_offset[1] = ypr[1];
			ypr_offset[2] = ypr[2];
		}

		void setYPROffset(double yaw, double pitch, double roll) {
			ypr_offset[0] = yaw;
			ypr_offset[1] = pitch;
			ypr_offset[2] = roll;
		}

		void getYPROffset(double& yaw, double& pitch, double& roll) {
			yaw = ypr_offset[0];
			pitch = ypr_offset[1];
			roll = ypr_offset[2];
		}

		//TODO: for this fuction to work, faces must be ordered by material
		//This function generates Mesh chunks from an RR_Mesh, used for converting to gltf
		std::vector<RR_Mesh_Chunk> getMeshChunks() {
			std::vector<RR_Mesh_Chunk> chunks;
			int i = 0;

			std::map<std::string, RR_POINT<Precision>> indexToVertex;

			for (RR_MATERIAL &currentMat : materials) {
				RR_Mesh_Chunk tempChunk;
				std::vector<RR_FACE> tempFaces;

				//Go through all faces matching with current material, and add faces and points to our map
				for (; i < faces.size();i++) {

					if (faces[i].material != &currentMat) {
						break;
					}

					tempFaces.push_back(faces[i]);

					for (int j = 0; j < POLYGON_TYPE; j++) {

						int vertPos = faces[i].vp[j];

						int texturePos = faces[i].tp[j];
						
						std::string currentIndex = faces[i].to_String(j);

						if (indexToVertex.count(currentIndex) == 0) {
							RR_POINT<Precision> tempPoint;
							tempPoint.coord = pointcloud.points[vertPos].coord;
							tempPoint.texture = textures[texturePos];
							tempChunk.indexToVertex[currentIndex] = tempPoint;
						}
					}
				}

				//now we have to find the linear position of faces in our map as if it was array
				//to do this, generate string array of faces, and search through this to find face positions
				std::vector<std::string> stringList;
				for (auto const&[key, val] : tempChunk.indexToVertex)
				{
					stringList.push_back(key);
				}

				for (RR_FACE &face : tempFaces)
				{
					for (int j = 0; j < POLYGON_TYPE; j++) {
						std::string currentIndex = face.to_String(j);
						auto iter = std::lower_bound(stringList.begin(), stringList.end(), currentIndex);
						unsigned int pos = iter-stringList.begin();
						tempChunk.facesSerial.push_back(pos);
					}
				}
				tempChunk.material = currentMat;
				chunks.push_back(tempChunk);
			}

			return chunks;

		}

	private:
		double ypr_offset[3];
		std::string name;
		std::string uuid;
	};

	using MESH = Mesh_Types::RR_MESH<Precision>;
	using POINT = Mesh_Types::RR_POINT<Precision>;
	using POINT_CLOUD = Mesh_Types::RR_POINTCLOUD<Precision>;
	using FACE = Mesh_Types::RR_FACE;
}

#endif
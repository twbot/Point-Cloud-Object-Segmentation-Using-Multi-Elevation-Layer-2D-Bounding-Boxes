#ifndef FileHandler_h
#define FileHandler_h
#include <map>
#include <string>
#include <typeinfo>

#include "Coordinates.h"
#include "MeshTypes.hpp"
#include "PlyReader.h"

#ifndef M_PI
#define M_PI 3.1415926535
#endif // !M_PI

namespace FileHandler3D {

	enum class FileTypes
	{
		PLY,
		GLTF,
		TILE,
		OBJ,
		JSON,
		PNTS,
		GLB
	};

	bool load_ply(
		const std::string& input_filename,
		std::unique_ptr<Mesh_Types::MESH>& mesh_
	);
	bool load_mtl(
		std::string input_filename,
		std::unique_ptr<Mesh_Types::MESH>& mesh_
	);
	bool load_obj(
		const std::string& input_filename,
		std::unique_ptr<Mesh_Types::MESH>& mesh_
	);
}

#endif // FileHandler_h
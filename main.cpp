#include "include/MeshTypes.hpp"
#include "include/Functions3D.h"
#include "include/FileHandler3D.h"

#ifdef _WIN32
//wxWidgets
#include <wx/wxprec.h>
#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif
#include <wx/dialog.h>
#endif

#define DEBUG 0

std::map<std::string, std::unique_ptr<Mesh_Types::MESH>> meshes;
std::map<FileHandler3D::FileTypes, std::string> fileTypeMap = { {FileHandler3D::FileTypes::PLY, "ply"}, {FileHandler3D::FileTypes::OBJ, "obj"} };

void addMesh(
	std::unique_ptr<Mesh_Types::MESH>& mesh_
)
{
	meshes[mesh_->get_uuid()] = std::move(mesh_);
}

void SelectPointCloud()
{
	//Create dialog
	wxFileDialog* filedlg = new wxFileDialog(nullptr, _("Import Mesh"), "", "",
		"All Known Formats (*.json; *.ply; *.obj; *.gltf; *.glb)|*.json;*.ply;*.obj;*.gltf;*.glb | 3D Tile Files (*.json)|*.json| PLY Files (*.ply)|*.ply| OBJ Files (*.obj)|*.obj| GLTF Files (*.gltf)|*.gltf| GLB Files (*.glb)|*.glb",
		wxFD_OPEN | wxFD_MULTIPLE);
	std::vector<std::string> full_path_file, file_name, file_type;
	std::vector < std::unique_ptr<Mesh_Types::MESH>> meshes;
	wxArrayString files;

	if (filedlg->ShowModal() == wxID_OK)
	{
		filedlg->GetPaths(files);
		for (uchar i = 0; i < files.Count(); i++) {
			std::string full_path_file_local;
			std::unique_ptr<Mesh_Types::MESH> mesh_ = std::make_unique<Mesh_Types::MESH>();
			full_path_file_local = files.Item(i).mb_str();
			size_t fileNameIndex = full_path_file_local.find_last_of("\\");
			size_t typeIndex = full_path_file_local.find_last_of(".");
			file_type.push_back(full_path_file_local.substr(typeIndex + 1, full_path_file_local.size()));
			file_name.push_back(full_path_file_local.substr(fileNameIndex + 1, full_path_file_local.size()));

			typeIndex = file_name[i].find_last_of(".");
			file_name[i] = file_name[i].substr(0, typeIndex);

			mesh_->set_uuid(generate_uuid());
			mesh_->set_name(file_name[i]);
			meshes.push_back(std::move(mesh_));
			full_path_file.push_back(full_path_file_local);
		}
	}

	filedlg->Destroy();

	for (uchar i = 0; i < files.Count(); i++) {
		std::unique_ptr<Mesh_Types::MESH> mesh_ = std::move(meshes[i]);
		//PLY loading currently only supports point clouds, no polygon mesh (yet)
		if (file_type[i] == fileTypeMap[FileHandler3D::FileTypes::PLY]) {
			FileHandler3D::load_ply(full_path_file[i], mesh_);
			mesh_->pointcloud.init();
		}
		//OBJ supports point clouds & polygon mesh
		else if (file_type[i] == fileTypeMap[FileHandler3D::FileTypes::OBJ]) {
			FileHandler3D::load_obj(full_path_file[i], mesh_);
			mesh_->pointcloud.init();
		}
		addMesh(mesh_);
	}
}

int main(int argc, char** argv)
{

	//Have user select point cloud
	SelectPointCloud();

	//Choose initial mesh (for now)
	std::unique_ptr<Mesh_Types::MESH> mesh_ = std::move(meshes.begin()->first);

	bool downsample = false;
	bool calculateNormals = false;
	bool createNewPointCloudOuput = true;

	int epsilon_centroid_distance = 80; //change depending on scene
	int epsilon_area_difference = 70000; //change depending on scene

	//Downsample point cloud before applying operation
	if (downsample) {
		//downSample(mesh_, InArguments);
	}

	// Calculate normals using OMP
	if (calculateNormals) {
		//calculateNormals(mesh_, InArguments);
	}

	// Start timer
	const auto timeBegin = std::chrono::steady_clock::now();

	std::shared_ptr<Mesh_Types::POINT_CLOUD> pc;
	pc = Functions3D::objectSegmentation(mesh_, epsilon_centroid_distance, epsilon_area_difference);

	if (!createNewPointCloudOuput) {
		mesh_->pointcloud.setPointCloud(pc->points);
		mesh_->pointcloud.init();
		mesh_->pointcloud.setColorsAvailable(true);
	}
	else if (createNewPointCloudOuput) {
		std::unique_ptr<Mesh_Types::MESH> new_mesh_ = std::make_unique<Mesh_Types::MESH>();
		new_mesh_->set_name("segmented_" + mesh_->get_name());
		new_mesh_->set_uuid(generate_uuid());
		new_mesh_->pointcloud.setPointCloud(pc->points);
		new_mesh_->pointcloud.init();
		new_mesh_->pointcloud.setColorsAvailable(true);
	}

	// End timer 
	const auto timeStop = std::chrono::steady_clock::now();
	auto duration_s = std::chrono::duration_cast<std::chrono::seconds> (timeStop - timeBegin);
	auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds> (timeStop - timeBegin);
#if DEBUG
	std::cout << "Time Duration: " << duration_s.count() << " seconds" << std::endl;
#endif

	return (0);
}

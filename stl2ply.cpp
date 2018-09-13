#include <iostream>
#include <fstream>
#include <time.h>
#include <algorithm>
#include <string>
#include <stack>
#include <queue>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <map>
#include <dirent.h>
#include <sys/types.h>

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/surface/mls.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/common/transforms.h>
#include <vtkVersion.h>
#include <vtkPLYReader.h>
#include <vtkOBJReader.h>
#include <vtkTriangle.h>
#include <vtkTriangleFilter.h>
#include <vtkPolyDataMapper.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>

using namespace std;

const std::string BASE_DATASET_PATH = "../82_dataset/";

struct Coordinate {
    float x, y, z;
    bool operator<(const Coordinate& rhs){
        return x<rhs.x || (x == rhs.x&&y<rhs.y) || (x == rhs.x&&y == rhs.y&&z<rhs.z);
    }
    bool operator==(const Coordinate& rhs){
        return x == rhs.x&&y == rhs.y && z == rhs.z;
    }
};
vector<Coordinate> vecSorted, vecOrigin;
vector<Coordinate>::iterator iter, iterBegin;

int numberOfFacets;
int numberOfPoints;
int index;


char c1[] = "ply\nformat binary_little_endian 1.0\ncomment By ET \nelement vertex ";
char c2[] = "\nproperty float x\nproperty float y\nproperty float z\nelement face ";
char c3[] = "\nproperty list uchar int vertex_indices\nend_header\n";

inline double
uniform_deviate(int seed)
{
	double ran = seed * (1.0 / (RAND_MAX + 1.0));
	return ran;
}

inline void
randomPointTriangle(float a1, float a2, float a3, float b1, float b2, float b3, float c1, float c2, float c3,
Eigen::Vector4f& p)
{
	float r1 = static_cast<float> (uniform_deviate(rand()));
	float r2 = static_cast<float> (uniform_deviate(rand()));
	float r1sqr = std::sqrt(r1);
	float OneMinR1Sqr = (1 - r1sqr);
	float OneMinR2 = (1 - r2);
	a1 *= OneMinR1Sqr;
	a2 *= OneMinR1Sqr;
	a3 *= OneMinR1Sqr;
	b1 *= OneMinR2;
	b2 *= OneMinR2;
	b3 *= OneMinR2;
	c1 = r1sqr * (r2 * c1 + b1) + a1;
	c2 = r1sqr * (r2 * c2 + b2) + a2;
	c3 = r1sqr * (r2 * c3 + b3) + a3;
	p[0] = c1;
	p[1] = c2;
	p[2] = c3;
	p[3] = 0;
}

inline void
randPSurface(vtkPolyData * polydata, std::vector<double> * cumulativeAreas, double totalArea, Eigen::Vector4f& p, bool calcNormal, Eigen::Vector3f& n)
{
	float r = static_cast<float> (uniform_deviate(rand()) * totalArea);

	std::vector<double>::iterator low = std::lower_bound(cumulativeAreas->begin(), cumulativeAreas->end(), r);
	vtkIdType el = vtkIdType(low - cumulativeAreas->begin());

	double A[3], B[3], C[3];
	vtkIdType npts = 0;
	vtkIdType *ptIds = NULL;
	polydata->GetCellPoints(el, npts, ptIds);
	polydata->GetPoint(ptIds[0], A);
	polydata->GetPoint(ptIds[1], B);
	polydata->GetPoint(ptIds[2], C);
	if (calcNormal)
	{
		// OBJ: Vertices are stored in a counter-clockwise order by default
		Eigen::Vector3f v1 = Eigen::Vector3f(A[0], A[1], A[2]) - Eigen::Vector3f(C[0], C[1], C[2]);
		Eigen::Vector3f v2 = Eigen::Vector3f(B[0], B[1], B[2]) - Eigen::Vector3f(C[0], C[1], C[2]);
		n = v1.cross(v2);
		n.normalize();
	}
	randomPointTriangle(float(A[0]), float(A[1]), float(A[2]),
		float(B[0]), float(B[1]), float(B[2]),
		float(C[0]), float(C[1]), float(C[2]), p);
}

void
uniform_sampling(vtkSmartPointer<vtkPolyData> polydata, size_t n_samples, bool calc_normal, pcl::PointCloud<pcl::PointNormal> & cloud_out)
{
	polydata->BuildCells();
	vtkSmartPointer<vtkCellArray> cells = polydata->GetPolys();

	double p1[3], p2[3], p3[3], totalArea = 0;
	std::vector<double> cumulativeAreas(cells->GetNumberOfCells(), 0);
	size_t i = 0;
	vtkIdType npts = 0, *ptIds = NULL;
	for (cells->InitTraversal(); cells->GetNextCell(npts, ptIds); i++)
	{
		polydata->GetPoint(ptIds[0], p1);
		polydata->GetPoint(ptIds[1], p2);
		polydata->GetPoint(ptIds[2], p3);
		totalArea += vtkTriangle::TriangleArea(p1, p2, p3);
		cumulativeAreas[i] = totalArea;
	}

	cloud_out.points.resize(n_samples);
	cloud_out.width = static_cast<pcl::uint32_t> (n_samples);
	cloud_out.height = 1;

	for (i = 0; i < n_samples; i++)
	{
		Eigen::Vector4f p;
		Eigen::Vector3f n;
		randPSurface(polydata, &cumulativeAreas, totalArea, p, calc_normal, n);
		cloud_out.points[i].x = p[0];
		cloud_out.points[i].y = p[1];
		cloud_out.points[i].z = p[2];
		if (calc_normal)
		{
			cloud_out.points[i].normal_x = n[0];
			cloud_out.points[i].normal_y = n[1];
			cloud_out.points[i].normal_z = n[2];
		}
	}
}

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

const int default_number_samples = 100000;
const float default_leaf_size = 0.01f;

void
printHelp(int, char **argv)
{
	print_error("Syntax is: %s input.{ply,obj} output.pcd <options>\n", argv[0]);
	print_info("  where options are:\n");
	print_info("                     -n_samples X      = number of samples (default: ");
	print_value("%d", default_number_samples);
	print_info(")\n");
	print_info(
		"                     -leaf_size X  = the XYZ leaf size for the VoxelGrid -- for data reduction (default: ");
	print_value("%f", default_leaf_size);
	print_info(" m)\n");
	print_info("                     -write_normals = flag to write normals to the output pcd\n");
	print_info(
		"                     -no_vis_result = flag to stop visualizing the generated pcd\n");
}

int string2chararray(string s, const char *p);
int list_stl_files();
int list_ply_files();
int pc_upsampling(std::string sPlyFileName = "test.ply", std::string sOutPcdName = "test.pcd");
int stl2ply(std::string sFileName);

int string2chararray(string s, const char *p)
{
	p = s.c_str();
	cout << p << endl;
	return 1;
}

int list_stl_files()
{
	DIR *dp;
	struct dirent *dirp;
	vector<string>filename;

	if ((dp = opendir(BASE_DATASET_PATH.c_str())) == NULL)
	perror("open dir error.");
	while ((dirp = readdir(dp)) != NULL)
	{
		string s = dirp->d_name;
// 		if ((s.size() > 4 && s.substr(s.size() - 4, 4) == ".cpp") || (s.size() > 2 && s.substr(s.size() - 2, 2) == ".h"))
		if (((s.size() > 4) && (s.substr(s.size() - 4, 4) == ".stl")) || ((s.size() > 4) && (s.substr(s.size() - 4, 4) == ".STL")))
		{
			filename.push_back(dirp->d_name);
		}
	}

	cout << "当前文件夹下的所有STL文件如下：\n";
	char tmp_str[1000];
	int line_count = 0;
	while (!filename.empty())
	{
		string tmp_fn = filename.back();
		cout << tmp_fn << endl;

		// To process with the stl file here
		// stl2ply(BASE_DATASET_PATH + tmp_fn);
		// To process with the stl file here

		filename.pop_back();
	}

	closedir(dp);
	system("pause");

	return 0;
}

int list_ply_files()
{
	DIR *dp;
	struct dirent *dirp;
	vector<string>filename;

	if ((dp = opendir(BASE_DATASET_PATH.c_str())) == NULL)
		perror("open dir error.");
	while ((dirp = readdir(dp)) != NULL)
	{
		string s = dirp->d_name;
		// 		if ((s.size() > 4 && s.substr(s.size() - 4, 4) == ".cpp") || (s.size() > 2 && s.substr(s.size() - 2, 2) == ".h"))
		if (((s.size() > 4) && (s.substr(s.size() - 4, 4) == ".ply")) || ((s.size() > 4) && (s.substr(s.size() - 4, 4) == ".PLY")))
		{
			filename.push_back(dirp->d_name);
		}
	}

	cout << "当前文件夹下的所有ply文件如下：\n";
	char tmp_str[1000];
	int line_count = 0;
	while (!filename.empty())
	{
		string tmp_fn = filename.back();
		cout << tmp_fn << endl;

		// To process with the ply file here
		std::string sPlyName = BASE_DATASET_PATH + tmp_fn;
		int nPos = sPlyName.find(".ply");
		std::string sOutPcdName = sPlyName.substr(0, nPos) + ".pcd";
		pc_upsampling(sPlyName, sOutPcdName);
		// To process with the stl file here

		filename.pop_back();
	}

	closedir(dp);
	system("pause");

	return 0;
}

int pc_upsampling(std::string sPlyFileName, std::string sOutPcdName)
{
	int SAMPLE_POINTS_ = default_number_samples;
	float leaf_size = 1.0; // default_leaf_size;
	bool vis_result = false;
	const bool write_normals = false;
	printf("SAMPLE_POINTS_ and leaf_size: %d, %.3lf\n\n", SAMPLE_POINTS_, leaf_size);

	vtkSmartPointer<vtkPolyData> polydata1 = vtkSmartPointer<vtkPolyData>::New();
	bool bIfPly = true;
	if (bIfPly)
	{
		pcl::PolygonMesh mesh;
		pcl::io::loadPolygonFilePLY(sPlyFileName.c_str(), mesh);
		pcl::io::mesh2vtk(mesh, polydata1);
	}
	else // obj
	{
		vtkSmartPointer<vtkOBJReader> readerQuery = vtkSmartPointer<vtkOBJReader>::New();
		readerQuery->SetFileName(sPlyFileName.c_str());
		readerQuery->Update();
		polydata1 = readerQuery->GetOutput();
	}

	//make sure that the polygons are triangles!
	vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
#if VTK_MAJOR_VERSION < 6
	triangleFilter->SetInput(polydata1);
#else
	triangleFilter->SetInputData(polydata1);
#endif
	triangleFilter->Update();

	vtkSmartPointer<vtkPolyDataMapper> triangleMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	triangleMapper->SetInputConnection(triangleFilter->GetOutputPort());
	triangleMapper->Update();
	polydata1 = triangleMapper->GetInput();

	bool INTER_VIS = false;

	if (INTER_VIS)
	{
		visualization::PCLVisualizer vis;
		vis.addModelFromPolyData(polydata1, "mesh1", 0);
		vis.setRepresentationToSurfaceForAllActors();
		vis.spin();
	}

	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_1(new pcl::PointCloud<pcl::PointNormal>);
	uniform_sampling(polydata1, SAMPLE_POINTS_, write_normals, *cloud_1);

	if (INTER_VIS)
	{
		visualization::PCLVisualizer vis_sampled;
		vis_sampled.addPointCloud<pcl::PointNormal>(cloud_1);
		if (write_normals)
			vis_sampled.addPointCloudNormals<pcl::PointNormal>(cloud_1, 1, 0.02f, "cloud_normals");
		vis_sampled.spin();
	}

	// Voxelgrid
	VoxelGrid<PointNormal> grid_;
	grid_.setInputCloud(cloud_1);
	grid_.setLeafSize(leaf_size, leaf_size, leaf_size);

	pcl::PointCloud<pcl::PointNormal>::Ptr voxel_cloud(new pcl::PointCloud<pcl::PointNormal>);
	grid_.filter(*voxel_cloud);

	if (vis_result)
	{
		visualization::PCLVisualizer vis3("VOXELIZED SAMPLES CLOUD");
		vis3.addPointCloud<pcl::PointNormal>(voxel_cloud);
		if (write_normals)
			vis3.addPointCloudNormals<pcl::PointNormal>(voxel_cloud, 1, 0.02f, "cloud_normals");
		vis3.spin();
	}

	if (!write_normals)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
		// Strip uninitialized normals from cloud:
		pcl::copyPointCloud(*voxel_cloud, *cloud_xyz);
		savePCDFileASCII(sOutPcdName, *cloud_xyz);
	}
	else
	{
		savePCDFileASCII(sOutPcdName, *voxel_cloud);
	}

	return 0;
}

int stl2ply(std::string sFileName)
{
	clock_t start, finish;
	double totaltime;
	start = clock();

	int length;
	int position = 80;
	fstream fileIn(sFileName, ios::in | ios::binary);
	fileIn.seekg(0, ios::end);
	length = (int)fileIn.tellg();
	fileIn.seekg(0, ios::beg);
	char* buffer = new char[length];
	fileIn.read(buffer, length);
	fileIn.close();




	numberOfFacets = *(int*)&(buffer[position]);
	position += 4;
	cout << "Number of Facets: " << numberOfFacets << endl;
	for (int i = 0; i < numberOfFacets; i++)
	{
		Coordinate tmpC;
		position += 12;
		for (int j = 0; j < 3; j++)
		{
			tmpC.x = *(float*)&(buffer[position]);
			position += 4;
			tmpC.y = *(float*)&(buffer[position]);
			position += 4;
			tmpC.z = *(float*)&(buffer[position]);
			position += 4;
			vecOrigin.push_back(tmpC);
		}
		position += 2;
	}

	free(buffer);



	vecSorted = vecOrigin;
	sort(vecSorted.begin(), vecSorted.end());
	iter = unique(vecSorted.begin(), vecSorted.end());
	vecSorted.erase(iter, vecSorted.end());
	numberOfPoints = vecSorted.size();



//	ofstream fileOut("test.ply", ios::binary | ios::out | ios::trunc);

	int nPos = sFileName.find(".STL");
	std::string sSaveName = sFileName.substr(0, nPos);
	std::cout << "sFileName, nPos, and sSaveName: " << sFileName << " " << nPos << " " << sSaveName << std::endl;
	ofstream fileOut(sSaveName + ".ply", ios::binary | ios::out | ios::trunc);
	
	fileOut.write(c1, sizeof(c1) - 1);
	fileOut << numberOfPoints;
	fileOut.write(c2, sizeof(c2) - 1);
	fileOut << numberOfFacets;
	fileOut.write(c3, sizeof(c3) - 1);


	buffer = new char[numberOfPoints * 3 * 4];
	position = 0;
	for (int i = 0; i < numberOfPoints; i++)
	{
		buffer[position++] = *(char*)(&vecSorted[i].x);
		buffer[position++] = *((char*)(&vecSorted[i].x) + 1);
		buffer[position++] = *((char*)(&vecSorted[i].x) + 2);
		buffer[position++] = *((char*)(&vecSorted[i].x) + 3);
		buffer[position++] = *(char*)(&vecSorted[i].y);
		buffer[position++] = *((char*)(&vecSorted[i].y) + 1);
		buffer[position++] = *((char*)(&vecSorted[i].y) + 2);
		buffer[position++] = *((char*)(&vecSorted[i].y) + 3);
		buffer[position++] = *(char*)(&vecSorted[i].z);
		buffer[position++] = *((char*)(&vecSorted[i].z) + 1);
		buffer[position++] = *((char*)(&vecSorted[i].z) + 2);
		buffer[position++] = *((char*)(&vecSorted[i].z) + 3);
	}


	fileOut.write(buffer, numberOfPoints * 3 * 4);




	free(buffer);

	buffer = new char[numberOfFacets * 13];

	for (int i = 0; i < numberOfFacets; i++)
	{
		buffer[13 * i] = (unsigned char)3;
	}

	iterBegin = vecSorted.begin();
	position = 0;
	for (int i = 0; i < numberOfFacets; i++)
	{
		position++;
		for (int j = 0; j < 3; j++)
		{
			iter = lower_bound(vecSorted.begin(), vecSorted.end(), vecOrigin[3 * i + j]);
			index = iter - iterBegin;
			buffer[position++] = *(char*)(&index);
			buffer[position++] = *((char*)(&index) + 1);
			buffer[position++] = *((char*)(&index) + 2);
			buffer[position++] = *((char*)(&index) + 3);

		}
	}


	fileOut.write(buffer, numberOfFacets * 13);



	free(buffer);
	fileOut.close();


	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC * 1000;
	cout << "All Time: " << totaltime << "ms\n";

	return 0;
}

// main.exe input.stl output.pcd
int main(int argc, char** argv)
{
	list_ply_files();
	// pc_upsampling();

	return 0;
}

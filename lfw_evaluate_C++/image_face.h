#pragma once

#include <iostream>
#include "opencv2/core/core.hpp"

#include "MTCNN.h"
#include "CenterFace.h"
#include "SimilarityTransform.h"
#include "readPath.h"


using namespace std;
using namespace cv;

//for union_set
struct Pair_Dist
{
	int i, j;
	double dist;
	Pair_Dist(int ii, int jj, double dist_)
	{
		i = ii;
		j = jj;
		dist = dist_;
	}
};

struct FaceTemp
{
	string name;
	vector<float> feature;
	Mat face;
};

struct FaceResult
{
	string name;
	double similarity;
};

class ImageFace
{
	//Ptr<MTCNN> mtcnn;
	
	bool align_crop;     //align or not
	double margin_ratio; //for face boundingbox, extend it to a bigger one
	const double dist_threshold;

	vector<vector<float>> feature_faces_total;
	vector<FaceTemp> faces_temps;
	double th_person;
public:

	ImageFace();
	Ptr<MTCNN> mtcnn;
	Ptr<CenterFace> center_face;
	Ptr<AlignFace> aligner;

	void calculateDistanceFace(const vector<Mat> &faces, vector<Pair_Dist> &dists);
	void face_align(const vector<Mat> &img_in, const vector<vector<Point2f>> &alignments, vector<Mat> &img_out);
	void compareImages(const string &folder_in);
	void compareImages(const string &folder_1, const string &folder_2);
	vector<vector<int>> findFaceClusters(const vector<Mat> &faces);
	void makeFaceTemplates(const string &face_folder, const string &file_out);
	
	void readFaceTemplates(const string &temp_file, vector<FaceTemp> &face_temps);
	void addPersonsToTemplate(const string &temp_file, const string &person_folder);
	void replacePersonsToTemplate(const string &temp_file, const string &person_folder);
	vector<FaceResult> faceRecognition(const string &img_path, const vector<FaceTemp> &face_temps_in, int k);
	vector<FaceResult> faceRecognition(const string &img_path, int k);
	vector<vector<FaceResult>> faceRecognition_folder(const string &img_path, int k);


	double distTwoImages(const string img_path1, const string img_path_2);
	vector<FaceResult> faceClassification(const string &img_path, int k);
	vector<vector<FaceResult>> faceClassification_folder(const string &folder, int k);
	vector<string> actors_names;
private:
	void rect_extend(Rect &rec, double r, int width, int height);
	bool is_frontalFace(const Rect &box, const vector<Point2f> &landermarks);
	

	//for union_set
	vector<int> uset;
	vector<int> set_size;

	inline int find(int x) {
		if (x != uset[x]) uset[x] = find(uset[x]);
		return uset[x];
	}
	inline void unionSet(int x, int y) {
		if ((x = find(x)) == (y = find(y))) return;
		if (set_size[x] >= set_size[y])
		{
			uset[y] = x;
			set_size[x] += set_size[y];
		}
		else {
			uset[x] = y;
			set_size[y] += set_size[x];
		}
	}
};

void makeFaceTemplates(const string &face_folder, const string &img_folder_out, const string &file_out, const int num_thread);
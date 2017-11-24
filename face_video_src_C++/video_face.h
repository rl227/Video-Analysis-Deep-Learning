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
	Pair_Dist(int ii, int jj,double dist_)
	{
		i = ii;
		j = jj;
		dist = dist_;
	}
};


class VideoFace
{
	Ptr<MTCNN> mtcnn;
	Ptr<CenterFace> center_face;
	Ptr<AlignFace> aligner;
	bool align_crop;     //align or not
	double margin_ratio; //for face boundingbox, extend it to a bigger one

	const int _frame_num_limite; //4500
	const int _face_num_limite; //50
	const int _step;   //10

	const double dist_threshold;
	const int num_face_reserve;  //leading face

	vector<vector<float>> feature_faces_total;
	vector<vector<float>> leading_faces_features;

public:

	VideoFace();
	void video_face_for_show(const string &video_path);
	void video_face_for_show_no_pass(const string &video_path);
	void video_face_detection(const string &video_path, vector<Mat> &face_imgs);
	void video_face_detection(const string &video_path, vector<Mat> &face_imgs, vector<vector<Point2f>> &alignments);
	void video_face_batch(const string &video_folder, const string &foler_out);

	void calculateDistanceFace(const vector<Mat> &faces, vector<Pair_Dist> &dists);
	vector<int> findLeadingActors(const vector<Mat> &faces);
	void face_align(const vector<Mat> &img_in, const vector<vector<Point2f>> &alignments, vector<Mat> &img_out);

	void videos_vertification(const vector<string> &video_paths);

	void video_2_faces(const string &video_path, const string &folder_out);
    void video_2_faces(const string &video_path, const string &video_folder, const string &class_name);
	void video_face_samples(const string &video_path, vector<Mat> &face_imgs);
	vector<vector<int>> findFaceClusters(const vector<Mat> &faces);

private:
	void rect_extend(Rect &rec, double r, int width, int height);
	bool video_match(const vector<vector<float>> &feature_video_1, const vector<vector<float>> &feature_video_2);
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

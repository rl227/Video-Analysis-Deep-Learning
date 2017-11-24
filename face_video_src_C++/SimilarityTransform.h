#pragma once

#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

class AlignFace
{
public:

	AlignFace();
	AlignFace(const vector<Point2f> &template_, const Size &template_size_);
	Mat getSimilarityTransform_mat(const vector<Point2f> &src, const vector<Point2f> &dst);
	void similarity_transform(const Mat &src, const vector<Point2f> &landmarks, Mat &out);

private:
	vector<Point2f> face_template;
	int landermark_nums;
	Size template_size;
};
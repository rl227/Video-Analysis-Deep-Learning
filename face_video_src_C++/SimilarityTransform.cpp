#include "SimilarityTransform.h"

AlignFace::AlignFace()
{
	landermark_nums = 5;
	face_template.resize(landermark_nums);
	face_template[0] = Point2f(30.2946, 51.6963);
	face_template[1] = Point2f(65.5318, 51.5014);
	face_template[2] = Point2f(48.0252, 71.7366);
	face_template[3] = Point2f(33.5493, 92.3655);
	face_template[4] = Point2f(62.7299, 92.2041);
	template_size = Size(96, 112);
}

AlignFace::AlignFace(const vector<Point2f> &template_,const Size &template_size_):landermark_nums(template_.size()),face_template(template_),template_size(template_size_)
{
	//landermark_nums = template_.size();
	//face_template = template_;
	//template_size = template_size_;
}

Mat AlignFace::getSimilarityTransform_mat(const vector<Point2f> &src, const vector<Point2f> &dst)
{
	Mat warp_mat(2, 3, CV_32FC1, Scalar::all(0));

	if (src.size() != dst.size()) {
		printf("Error: vectors not the same size\n");
		return warp_mat;
	}

	int num_pairs = src.size();
	//printf("finding similarity based on [%d] pairs of points\n", num_pairs);

	Mat srcMat(4, 2 * num_pairs, CV_32F);
	Mat dstMat(1, 2 * num_pairs, CV_32F);
	for (int i = 0; i < num_pairs; i++) {
		srcMat.at<float>(0, i) = src[i].x;
		srcMat.at<float>(1, i) = -src[i].y;
		srcMat.at<float>(2, i) = 1;
		srcMat.at<float>(3, i) = 0;

		srcMat.at<float>(0, i + num_pairs) = src[i].y;
		srcMat.at<float>(1, i + num_pairs) = src[i].x;
		srcMat.at<float>(2, i + num_pairs) = 0;
		srcMat.at<float>(3, i + num_pairs) = 1;

		dstMat.at<float>(0, i) = dst[i].x;
		dstMat.at<float>(0, i + num_pairs) = dst[i].y;
		//dstMat.at<float>(2, i) = 1;
	}

	// transform * src = dst
	// transform = dst * src' * inv(src * src')

	//A = dst * src'
	Mat A = dstMat*srcMat.t();

	//B = src * src'
	Mat B = srcMat*srcMat.t();

	//invert B...
	Mat B_inv = B.inv();

	//multiply A and B^(-1)
	Mat T = A*B_inv;

	// set warp_mat
	warp_mat.at<float>(0, 0) = T.at<float>(0, 0);
	warp_mat.at<float>(0, 1) = -T.at<float>(0, 1);
	warp_mat.at<float>(0, 2) = T.at<float>(0, 2);
	warp_mat.at<float>(1, 0) = T.at<float>(0, 1);
	warp_mat.at<float>(1, 1) = T.at<float>(0, 0);
	warp_mat.at<float>(1, 2) = T.at<float>(0, 3);

	//cout << "warp mat" << warp_mat << endl;
	return warp_mat;
}

void AlignFace::similarity_transform(const Mat &src, const vector<Point2f> &landmarks,Mat &out)
{
	Mat warp_mat = getSimilarityTransform_mat(landmarks, face_template);
	warpAffine(src, out, warp_mat, template_size);
}
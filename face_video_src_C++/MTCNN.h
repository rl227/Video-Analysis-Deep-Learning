//#define CPU_ONLY

#ifndef MTCNN_MTCNN_H
#define MTCNN_MTCNN_H

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "SimilarityTransform.h"

using namespace caffe;
using namespace cv;

struct Bounding_Box
{
	float x1;
	float y1;
	float x2;
	float y2;
	Bounding_Box(float x1_, float y1_, float x2_, float y2_)
	{
		x1 = x1_;
		y1 = y1_;
		x2 = x2_;
		y2 = y2_;
	}
	Bounding_Box()
	{

	}
};

struct Regression_Box
{
	float dx1;
	float dy1;
	float dx2;
	float dy2;
	Regression_Box(){}
	Regression_Box(float dx1_, float dy1_, float dx2_, float dy2_)
	{
		dx1 = dx1_;
		dy1 = dy1_;
		dx2 = dx2_;
		dy2 = dy2_;
	}
};

struct Face_Infor
{
	double confidence;
	Bounding_Box bounding_box;
	Regression_Box regression_box;
	vector<cv::Point2f> landmarks;
};

class MTCNN {

public:

    MTCNN();
    MTCNN(const std::vector<std::string> &model_file, const std::vector<std::string> &trained_file);
    ~MTCNN();

	void detection(const cv::Mat& img);
    void detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles);
	void detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles,std::vector<std::vector<cv::Point2f>>& alignment);

	cv::Mat Preprocess(const cv::Mat &img);
	void resize_img(cv::Mat &img,vector<cv::Mat> &img_resized,vector<double> &scales);

	void P_Net(cv::Mat &img, const double scale);
    void R_Net(const Mat &img);
    void O_Net(const Mat &img_data);
	void NMS(vector<Face_Infor> &candidata_faces,double thres,const char u_m);

    void Predict(const cv::Mat& img, int i);
    void Predict(const std::vector<cv::Mat> &imgs, int i);
    void WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat> *input_channels, int i);
    void WrapInputLayer(const vector<cv::Mat> &imgs, std::vector<cv::Mat> *input_channels, int i);

    float IoU(const Bounding_Box &rect1, const Bounding_Box &rect2);
    float IoM(const Bounding_Box &rect1, const Bounding_Box &rect2);
   
	void GenerateBoxs(int i, const int image_w, const int img_h, const double scale);
	
    void BoxRegress(vector<Face_Infor> &faces, int stage);
	void Bbox2Square(std::vector<Face_Infor>& bboxes);
	void Padding(std::vector<Face_Infor>& faces, int img_w, int img_h);
    cv::Mat crop(const cv::Mat &img, Bounding_Box &rect);


	Mat image_face_detection_align(const Mat &img);
    void img_show(cv::Mat &img);
	vector<Face_Infor> face_total;

private:
    //param for P, R, O, L net
    std::vector<std::shared_ptr<Net<float>>> nets_;
    std::vector<cv::Size> input_geometry_;
    int num_channels_;

    //paramter for the threshold
    int minSize_ =80;
    float factor_ = 0.709;
    double threshold_[3] = {0.6, 0.8, 0.95};
};

#endif //MTCNN_MTCNN_H

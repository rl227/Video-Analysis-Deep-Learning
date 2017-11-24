

#define CPU_ONLY

#ifndef _CENTERFACE_H
#define _CENTERFACE_H

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace caffe;
using namespace cv;

class CenterFace {

public:

	CenterFace();
	CenterFace(const std::string &model_file, const std::string &trained_file);
    ~CenterFace();

	void extractFeature(const cv::Mat &img, vector<float> &feature_vector);
	void extractFeature(const vector<cv::Mat> &imgs, vector<vector<float>> &feature_vector);
    Mat Preprocess(const cv::Mat &img);
   
	vector<float> Predict(const cv::Mat& img);
	vector<vector<float>> Predict(const std::vector<cv::Mat> imgs);
    void WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat> *input_channels);
    void WrapInputLayer(const vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels);
    void Padding(std::vector<cv::Rect>& bounding_box, int img_w,int img_h);
    cv::Mat crop(cv::Mat img, cv::Rect& rect);
    //param for net
    std::shared_ptr<Net<float>> net_;
    cv::Size input_geometry_;
    int num_channels_;
	int vector_size;

    //variable for the output of the neural network
//    std::vector<cv::Rect> regression_box_;
	//std::vector<float> feature_vector;
};


#endif //_CENTERFACE_H

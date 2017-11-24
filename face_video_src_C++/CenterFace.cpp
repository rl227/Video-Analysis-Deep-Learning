//
// Created by Young on 2016/11/27.
//

/*
 * TO DO : change the P-net and update the generat box
 */

#include "CenterFace.h"

CenterFace::CenterFace() {}

CenterFace::CenterFace(const std::string &model_file, const std::string &trained_file):vector_size(512)
{
//#ifdef CPU_ONLY
//	Caffe::set_mode(Caffe::CPU);
//#else
	Caffe::set_mode(Caffe::GPU);
//#endif

	//std::shared_ptr<Net<float>> net;
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
	//net_ = net;
}

CenterFace::~CenterFace() {}

void CenterFace::extractFeature(const cv::Mat &img,vector<float> &feature_vector)
{
	CV_Assert(img.size() == input_geometry_);
	CV_Assert(img.channels() == num_channels_);
	Mat img_data=Preprocess(img);
	feature_vector=Predict(img_data);
}

void CenterFace::extractFeature(const vector<cv::Mat> &imgs, vector<vector<float>> &feature_vector)
{
	//CV_Assert(img[0].size() == input_geometry_);
	//CV_Assert(img[0].channels() == num_channels_);
	vector<Mat> img_datas(imgs.size());
	for (int i = 0; i < imgs.size(); ++i)
	{
		img_datas[i] = Preprocess(imgs[i]);
	}
	
	feature_vector = Predict(img_datas);
}

Mat CenterFace::Preprocess(const cv::Mat &img)
{
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample.convertTo(sample_float, CV_32FC3);
	else
		sample.convertTo(sample_float, CV_32FC1);


	//cv::cvtColor(sample_float, sample_float, cv::COLOR_BGR2RGB);

	//sample_float = sample_float.t();
	sample_float = (sample_float - 127.5) / 128.;
	return sample_float;
}

/*
 * Predict function input is a image without crop
 * the reshape of input layer is image's height and width
 */
vector<float> CenterFace::Predict(const cv::Mat& img)
{
	std::shared_ptr<Net<float>> net = net_;

	Blob<float>* input_layer = net->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		img.rows, img.cols);
	/* Forward dimension change to all layers. */
	net->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(img, &input_channels);
	net->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	return std::vector<float>(begin, end);
}

/*
 * Predict(const std::vector<cv::Mat> imgs, int i) function
 * used to input is a group of image with crop from original image
 * the reshape of input layer of net is the number, channels, height and width of images.
 */
vector<vector<float>> CenterFace::Predict(const std::vector<cv::Mat> imgs)
{
	std::shared_ptr<Net<float>> net = net_;

	Blob<float>* input_layer = net->input_blobs()[0];
	input_layer->Reshape(imgs.size(), num_channels_,
		input_geometry_.height, input_geometry_.width);
	int num = input_layer->num();
	/* Forward dimension change to all layers. */
	net->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(imgs, &input_channels);

	net->Forward();

	vector<vector<float>> features(imgs.size());
	Blob<float>* output_layer = net->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();

	/* Copy the output layer to a std::vector */
	//You can also try to use the blob_by_name()
	for (int i = 0; i < imgs.size(); ++i)
	{
		int s = i*vector_size;
		features[i] = vector<float>(begin + s, begin + s + vector_size);
	}
	return features;
}

void CenterFace::WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat> *input_channels)
{
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int j = 0; j < input_layer->channels(); ++j)
	{
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}

	//cv::Mat sample_normalized;
	//cv::subtract(img, mean_[i], img);
	/* This operation will write the separate BGR planes directly to the
	 * input layer of the network because it is wrapped by the cv::Mat
	 * objects in input_channels. */
	cv::split(img, *input_channels);

}

/*
 * WrapInputLayer(const vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels, int i) function
 * used to write the separate BGR planes directly to the input layer of the network
 */
void CenterFace::WrapInputLayer(const vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels)
{
	Blob<float> *input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	int num = input_layer->num();
	float *input_data = input_layer->mutable_cpu_data();

	for (int j = 0; j < num; j++) {
		//std::vector<cv::Mat> *input_channels;
		for (int k = 0; k < input_layer->channels(); ++k) {
			cv::Mat channel(height, width, CV_32FC1, input_data);
			input_channels->push_back(channel);
			input_data += width * height;
		}
		cv::Mat img = imgs[j];
		cv::split(img, *input_channels);
		input_channels->clear();
	}
}


void CenterFace::Padding(std::vector<cv::Rect>& bounding_box, int img_w, int img_h)
{
	for (int i = 0; i < bounding_box.size(); i++)
	{
		bounding_box[i].x = (bounding_box[i].x > 0) ? bounding_box[i].x : 0;
		bounding_box[i].y = (bounding_box[i].y > 0) ? bounding_box[i].y : 0;
		bounding_box[i].width = (bounding_box[i].x + bounding_box[i].width < img_w) ? bounding_box[i].width : img_w;
		bounding_box[i].height = (bounding_box[i].y + bounding_box[i].height < img_h) ? bounding_box[i].height : img_h;
	}
}

cv::Mat CenterFace::crop(cv::Mat img, cv::Rect& rect)
{
	cv::Rect rect_old = rect;

	//    if(rect.width > rect.height)
	//    {
	//        int change_to_square = rect.width - rect.height;
	//        rect.height += change_to_square;
	//        rect.y -= change_to_square * 0.5;
	//    }
	//    else
	//    {
	//        int change_to_square = rect.height - rect.width;
	//        rect.width += change_to_square;
	//        rect.x -= change_to_square * 0.5;
	//    }

	cv::Rect padding;

	if (rect.x < 0)
	{
		padding.x = -rect.x;
		rect.x = 0;
	}
	if (rect.y < 0)
	{
		padding.y = -rect.y;
		rect.y = 0;
	}
	if (img.cols < (rect.x + rect.width))
	{
		padding.width = rect.x + rect.width - img.cols;
		rect.width = img.cols - rect.x;
	}
	if (img.rows < (rect.y + rect.height))
	{
		padding.height = rect.y + rect.height - img.rows;
		rect.height = img.rows - rect.y;
	}
	if (rect.width < 0 || rect.height < 0)
	{
		rect = cv::Rect(0, 0, 0, 0);
		padding = cv::Rect(0, 0, 0, 0);
	}
	cv::Mat img_cropped = img(rect);
	if (padding.x || padding.y || padding.width || padding.height)
	{
		cv::copyMakeBorder(img_cropped, img_cropped, padding.y, padding.height, padding.x, padding.width, cv::BORDER_CONSTANT, cv::Scalar(0));
		//here, the rect should be changed
		rect.x -= padding.x;
		rect.y -= padding.y;
		rect.width += padding.width + padding.x;
		rect.width += padding.height + padding.y;
	}

	//    cv::imshow("crop", img_cropped);
	//    cv::waitKey(0);

	return img_cropped;
}

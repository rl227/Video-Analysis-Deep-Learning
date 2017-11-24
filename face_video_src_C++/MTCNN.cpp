
#include "MTCNN.h"

MTCNN::MTCNN() {}

MTCNN::MTCNN(const std::vector<std::string> &model_file, const std::vector<std::string> &trained_file)
{
//#ifdef CPU_ONLY
//	Caffe::set_mode(Caffe::CPU);
//#else
	Caffe::set_mode(Caffe::GPU);
//#endif

	for (int i = 0; i < model_file.size(); i++)
	{
		std::shared_ptr<Net<float>> net;

		cv::Size input_geometry;
		int num_channel;

		net.reset(new Net<float>(model_file[i], TEST));
		net->CopyTrainedLayersFrom(trained_file[i]);

		Blob<float>* input_layer = net->input_blobs()[0];
		num_channel = input_layer->channels();
		input_geometry = cv::Size(input_layer->width(), input_layer->height());

		nets_.push_back(net);
		input_geometry_.push_back(input_geometry);
		if (i == 0)
			num_channels_ = num_channel;
		else if (num_channels_ != num_channel)
			std::cout << "Error: The number channels of the nets are different!" << std::endl;
	}
}

MTCNN::~MTCNN() {}

void MTCNN::detection(const cv::Mat& img)
{
	if (!face_total.empty()) face_total.clear();
	Mat img_data = Preprocess(img);
	vector<Mat> img_resized;
	vector<double> scales;
	resize_img(img_data, img_resized, scales);
	for (int i = 0; i < img_resized.size(); ++i)
	{
		double scale = scales[i];
		Mat img_tmp = img_resized[i];
		P_Net(img_tmp, scale);
	}
	if (face_total.empty()) return;
	NMS(face_total, 0.7, 'u');
	BoxRegress(face_total, 1);
	Bbox2Square(face_total);

	R_Net(img_data);
	if (face_total.empty()) return;
	NMS(face_total, 0.7, 'u');
	BoxRegress(face_total, 2);
	Bbox2Square(face_total);

	O_Net(img_data);
	if (face_total.empty()) return;
	BoxRegress(face_total, 3);
	NMS(face_total, 0.7, 'm');

}

void MTCNN::detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles)
{
	detection(img);
	for (auto &face : face_total)
	{
		//if (face.bounding_box.x1<0 || face.bounding_box.y1<0 || face.bounding_box.x2>img.cols - 1 || face.bounding_box.y2>img.rows - 1)
			//continue;
		cv::Rect rect = Rect(round(face.bounding_box.y1), round(face.bounding_box.x1),
			round(face.bounding_box.y2) - round(face.bounding_box.y1)+1, round(face.bounding_box.x2) - round(face.bounding_box.x1)+1);

		rectangles.push_back(rect);
	}
}

void MTCNN::detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<std::vector<cv::Point2f>>& alignment)
{
	//cout << "tt" << endl;
	detection(img, rectangles);

	alignment.resize(face_total.size());
	for (int i = 0; i < face_total.size(); ++i)
	{
		alignment[i] = face_total[i].landmarks;
	}
}

Mat MTCNN::Preprocess(const cv::Mat &img)
{
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	/*
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else*/
	sample = img;

	cv::Mat sample_float;
	//if (num_channels_ == 3)
	sample.convertTo(sample_float, CV_32FC3);
	//else
		//sample.convertTo(sample_float, CV_32FC1);


	cv::cvtColor(sample_float, sample_float, cv::COLOR_BGR2RGB);
	sample_float = sample_float.t();

	return sample_float;
}
void MTCNN::resize_img(cv::Mat &img, vector<cv::Mat> &img_resized, vector<double> &scales)
{
	img_resized.reserve(20);
	scales.reserve(20);

	int height = img.rows;
	int width = img.cols;

	int minSize = minSize_;
	float factor = factor_;
	double scale = 12. / minSize;
	double minWH = std::min(height, width) * scale;

	while (minWH >= 12)
	{
		scales.push_back(scale);
		int resized_h = std::ceil(height*scale);
		int resized_w = std::ceil(width*scale);

		cv::Mat resized;
		cv::resize(img, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);
		resized.convertTo(resized, CV_32FC3, 0.0078125, -127.5*0.0078125);
		img_resized.push_back(resized);

		minWH *= factor;
		scale *= factor;
	}
}

void MTCNN::P_Net(cv::Mat &img, const double scale)
{
	Predict(img, 0);
	GenerateBoxs(0, img.cols, img.rows, scale);
}

void MTCNN::R_Net(const Mat &img_data)
{
	float thresh = threshold_[1];
	vector<Mat> cur_imgs;
	for (int j = 0; j < face_total.size(); j++) {
		cv::Mat img = crop(img_data, face_total[j].bounding_box);
		if (img.size() == cv::Size(0, 0))
			continue;
		if (img.rows == 0 || img.cols == 0)
			continue;
		if (img.size() != input_geometry_[1])
			cv::resize(img, img, input_geometry_[1]);
		Mat img_data;
		img.convertTo(img_data, CV_32FC3, 0.0078125, -127.5*0.0078125);
		cur_imgs.push_back(img_data);
	}

	Predict(cur_imgs, 1);
	std::shared_ptr<Net<float>> net = nets_[1];
	Blob<float>* confidence = net->output_blobs()[1];
	const float* confidence_begin = confidence->cpu_data();
	int count = cur_imgs.size();

	Blob<float>* rect = net->output_blobs()[0];
	const float* rect_begin = rect->cpu_data();

	vector<Face_Infor> faces_tmp;
	faces_tmp.reserve(face_total.size());
	for (int j = 0; j < count; j++)
	{
		float conf = confidence_begin[2 * j + 1];
		if (conf > thresh) {
			//bounding box
			Face_Infor face_tmp = face_total[j];
			//float height = face_tmp.bounding_box.y2 - face_tmp.bounding_box.y1;
			//float width = face_tmp.bounding_box.x2 - face_tmp.bounding_box.x1;
			//regression box : y x height width
			/*face_tmp.bounding_box.y1 += rect_begin[4 * j] * height;
			face_tmp.bounding_box.x1 +=  rect_begin[4 * j + 1] * width;
			face_tmp.bounding_box.y2 +=  rect_begin[4 * j + 2] * height;
			face_tmp.bounding_box.x2 +=  rect_begin[4 * j + 3] * width;*/
			face_tmp.regression_box.dx1 = rect_begin[4 * j + 1];
			face_tmp.regression_box.dy1 = rect_begin[4 * j + 0];
			face_tmp.regression_box.dx2 = rect_begin[4 * j + 3];
			face_tmp.regression_box.dy2 = rect_begin[4 * j + 2];

			face_tmp.confidence = conf;
			faces_tmp.push_back(face_tmp);
		}
	}
	face_total = faces_tmp;
}

void MTCNN::O_Net(const Mat &img_data)
{
	float thresh = threshold_[2];
	vector<Mat> cur_imgs;
	for (int j = 0; j < face_total.size(); j++) {
		cv::Mat img = crop(img_data, face_total[j].bounding_box);
		if (img.size() == cv::Size(0, 0))
			continue;
		if (img.rows == 0 || img.cols == 0)
			continue;
		if (img.size() != input_geometry_[2])
			cv::resize(img, img, input_geometry_[2]);
		Mat img_data;
		img.convertTo(img_data, CV_32FC3, 0.0078125, -127.5*0.0078125);
		cur_imgs.push_back(img_data);
	}

	Predict(cur_imgs, 2);
	std::shared_ptr<Net<float>> net = nets_[2];
	Blob<float>* confidence = net->output_blobs()[2];
	const float* confidence_begin = confidence->cpu_data();
	int count = cur_imgs.size();

	Blob<float>* rect = net->output_blobs()[0];
	const float* rect_begin = rect->cpu_data();

	Blob<float>* points = net->output_blobs()[1];
	const float* points_begin = points->cpu_data();

	vector<Face_Infor> faces_tmp;
	faces_tmp.reserve(face_total.size());
	for (int j = 0; j < count; j++)
	{
		float conf = confidence_begin[2 * j + 1];
		if (conf > thresh) {
			//bounding box
			Face_Infor face_tmp = face_total[j];
			float height = (face_tmp.bounding_box.y2) - (face_tmp.bounding_box.y1) + 1;
			float width = (face_tmp.bounding_box.x2) - (face_tmp.bounding_box.x1) + 1;
			//regression box : y x height width
			//face_tmp.bounding_box.y1 += rect_begin[4 * j] * height;
			//face_tmp.bounding_box.x1 += rect_begin[4 * j + 1] * width;
			//face_tmp.bounding_box.y2 += rect_begin[4 * j + 2] * height;
			//face_tmp.bounding_box.x2 += rect_begin[4 * j + 3] * width;

			face_tmp.regression_box.dx1 = rect_begin[4 * j + 1];
			face_tmp.regression_box.dy1 = rect_begin[4 * j + 0];
			face_tmp.regression_box.dx2 = rect_begin[4 * j + 3];
			face_tmp.regression_box.dy2 = rect_begin[4 * j + 2];

			//face alignment
			std::vector<cv::Point2f> align(5);
			for (int k = 0; k < 5; k++)
			{
				align[k].y = face_tmp.bounding_box.x1 + width * points_begin[10 * j + k + 5] - 1;
				align[k].x = face_tmp.bounding_box.y1 + height * points_begin[10 * j + k] - 1;
			}
			face_tmp.landmarks = align;

			face_tmp.confidence = conf;
			faces_tmp.push_back(face_tmp);
		}
	}
	face_total = faces_tmp;
}

void MTCNN::NMS(vector<Face_Infor> &candidata_faces, double threshold, const char u_m)
{
	vector<int> index(candidata_faces.size());
	for (int i = 0; i < index.size(); ++i) index[i] = i;
	sort(index.begin(), index.end(), [&](int i, int j) {return candidata_faces[i].confidence > candidata_faces[j].confidence; });

	vector<bool> pass(index.size(), false);
	for (int i = 0; i < index.size(); i++)
	{
		int confidence_i = index[i];
		if (pass[confidence_i] == true) continue;
		for (int j = i + 1; j < index.size(); ++j)
		{
			int confidence_j = index[j];
			if (pass[confidence_j] == true) continue;
			double r = u_m == 'u' ? IoU(candidata_faces[confidence_i].bounding_box, candidata_faces[confidence_j].bounding_box) :
				IoM(candidata_faces[confidence_i].bounding_box, candidata_faces[confidence_j].bounding_box);

			if (r > threshold)
			{
				pass[confidence_j] = true;
			}
		}
	}
	vector<Face_Infor> faces_res;
	faces_res.reserve(candidata_faces.size());
	for (int i = 0; i < pass.size(); ++i)
	{
		if (pass[i] == false)
		{
			faces_res.push_back(candidata_faces[i]);
		}
	}
	candidata_faces = faces_res;
}

/*
 * Predict function input is a image without crop
 * the reshape of input layer is image's height and width
 */
void MTCNN::Predict(const cv::Mat& img, int i)
{
	std::shared_ptr<Net<float>> net = nets_[i];

	Blob<float>* input_layer = net->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		img.rows, img.cols);
	/* Forward dimension change to all layers. */
	net->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(img, &input_channels, i);
	net->Forward();
}

/*
 * Predict(const std::vector<cv::Mat> imgs, int i) function
 * used to input is a group of image with crop from original image
 * the reshape of input layer of net is the number, channels, height and width of images.
 */
void MTCNN::Predict(const std::vector<cv::Mat> &imgs, int i)
{
	std::shared_ptr<Net<float>> net = nets_[i];

	Blob<float>* input_layer = net->input_blobs()[0];
	input_layer->Reshape(imgs.size(), num_channels_,
		input_geometry_[i].height, input_geometry_[i].width);
	int num = input_layer->num();
	/* Forward dimension change to all layers. */
	net->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(imgs, &input_channels, i);

	net->Forward();
}

void MTCNN::WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat> *input_channels, int i)
{
	Blob<float>* input_layer = nets_[i]->input_blobs()[0];

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
void MTCNN::WrapInputLayer(const vector<cv::Mat> &imgs, std::vector<cv::Mat> *input_channels, int i)
{
	Blob<float> *input_layer = nets_[i]->input_blobs()[0];

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

float MTCNN::IoU(const Bounding_Box &rect1, const Bounding_Box &rect2)
{
	float x_overlap, y_overlap, intersection, unions;
	x_overlap = std::max(float(0.), std::min(rect1.x2, rect2.x2) - std::max(rect1.x1, rect2.x1));
	y_overlap = std::max(float(0.), std::min(rect1.y2, rect2.y2) - std::max(rect1.y1, rect2.y1));
	intersection = x_overlap * y_overlap;
	unions = (rect1.x2 - rect1.x1)*(rect1.y2 - rect1.y1) + (rect2.x2 - rect2.x1)*(rect2.y2 - rect2.y1) - intersection;
	return float(intersection) / unions;
}

float MTCNN::IoM(const Bounding_Box &rect1, const Bounding_Box &rect2)
{
	float x_overlap, y_overlap, intersection, unions;
	x_overlap = std::max(float(0.), std::min(rect1.x2, rect2.x2) - std::max(rect1.x1, rect2.x1));
	y_overlap = std::max(float(0.), std::min(rect1.y2, rect2.y2) - std::max(rect1.y1, rect2.y1));
	intersection = x_overlap * y_overlap;
	float min_area = std::min((rect1.x2 - rect1.x1)*(rect1.y2 - rect1.y1), (rect2.x2 - rect2.x1)*(rect2.y2 - rect2.y1));
	return float(intersection) / min_area;
}


void MTCNN::GenerateBoxs(int i, const int image_w, const int img_h, const double scale)
{

	int stride = 2;
	int cellSize = 12;

	/* Copy the output layer to a std::vector */
	std::shared_ptr<Net<float>> net = nets_[i];
	Blob<float>* rect = net->output_blobs()[0];
	Blob<float>* confidence = net->output_blobs()[1];
	int count = confidence->count() / 2;

	const float* rect_begin = rect->cpu_data();
	const float* confidence_begin = confidence->cpu_data() + count;

	int feature_map_w = confidence->shape(3);
	int feature_map_h = confidence->shape(2);

	float thresh = threshold_[0];

	vector<Face_Infor> faces_tmp;
	faces_tmp.reserve(count);

	for (int i = 0; i < count; i++)
	{
		if (confidence_begin[i] < thresh)
			continue;

		Face_Infor face;
		face.confidence = (confidence_begin[i]);

		int y = i / feature_map_w;
		int x = i - feature_map_w * y;

		//the regression box from the neural network
		//regression box : y x height width
		face.regression_box = Regression_Box(rect_begin[i + count], rect_begin[i],
			rect_begin[i + count * 3], rect_begin[i + count * 2]);

		//the bounding box combined with regression box
		float x1 = (x*stride + 1) / scale;
		float y1 = (y*stride + 1) / scale;
		float x2 = (x*stride + cellSize - 1 + 1) / scale;
		float y2 = (y*stride + cellSize - 1 + 1) / scale;

		face.bounding_box = Bounding_Box(x1, y1, x2, y2);
		faces_tmp.push_back(face);
	}

	NMS(faces_tmp, 0.5, 'u');
	face_total.insert(face_total.end(), faces_tmp.begin(), faces_tmp.end());
}

void MTCNN::BoxRegress(vector<Face_Infor> &faces, int stage)
{
	for (int i = 0; i < faces.size(); i++)
	{
		float x1 = faces[i].bounding_box.x1;
		float y1 = faces[i].bounding_box.y1;
		float x2 = faces[i].bounding_box.x2;
		float y2 = faces[i].bounding_box.y2;

		float width = stage == 1 ? x2 - x1 : x2 - x1 + 1;
		float height = stage == 1 ? y2 - y1 : y2 - y1 + 1;


		x1 += faces[i].regression_box.dx1 * width;
		y1 += faces[i].regression_box.dy1* height;
		x2 += faces[i].regression_box.dx2* width;
		y2 += faces[i].regression_box.dy2* height;

		faces[i].bounding_box.x1 = x1;
		faces[i].bounding_box.y1 = y1;
		faces[i].bounding_box.x2 = x2;
		faces[i].bounding_box.y2 = y2;
	}
}

void MTCNN::Bbox2Square(std::vector<Face_Infor>& faces) {
	for (int i = 0; i < faces.size(); i++) {
		float h = faces[i].bounding_box.y2 - faces[i].bounding_box.y1;
		float w = faces[i].bounding_box.x2 - faces[i].bounding_box.x1;
		float side = h > w ? h : w;

		faces[i].bounding_box.x1 += (w - side)*0.5;
		faces[i].bounding_box.y1 += (h - side)*0.5;

		faces[i].bounding_box.x2 = faces[i].bounding_box.x1 + side;
		faces[i].bounding_box.y2 = faces[i].bounding_box.y1 + side;

		//round to integer
		faces[i].bounding_box.x1 = round(faces[i].bounding_box.x1);
		faces[i].bounding_box.y1 = round(faces[i].bounding_box.y1);
		faces[i].bounding_box.x2 = round(faces[i].bounding_box.x2);
		faces[i].bounding_box.y2 = round(faces[i].bounding_box.y2);

	}
}
void MTCNN::Padding(std::vector<Face_Infor>& faces, int img_w, int img_h)
{
	/*for (int i = 0; i < faces.size(); i++)
	{
		faces[i].bounding_box.x = (faces[i].bounding_box.x > 0) ? faces[i].bounding_box.x : 0;
		faces[i].bounding_box.y = (faces[i].bounding_box.y > 0) ? faces[i].bounding_box.y : 0;
		faces[i].bounding_box.width = (faces[i].bounding_box.x + faces[i].bounding_box.width < img_w) ? faces[i].bounding_box.width : img_w;
		faces[i].bounding_box.height = (faces[i].bounding_box.y + faces[i].bounding_box.height < img_h) ? faces[i].bounding_box.height : img_h;
	}*/
}

cv::Mat MTCNN::crop(const cv::Mat &img, Bounding_Box &bbox)
{
	cv::Rect rect = Rect((bbox.x1), (bbox.y1), (bbox.x2) - (bbox.x1) + 1, (bbox.y2) - (bbox.y1) + 1);
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
	if (img.rows  < (rect.y + rect.height))
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
	/*	rect.x -= padding.x;
		rect.y -= padding.y;
		rect.width += padding.width + padding.x;
		rect.height += padding.height + padding.y;

		bbox.x1 = rect.x;
		bbox.y1 = rect.y;
		bbox.x2 = rect.x + rect.width - 1;
		bbox.y2 = rect.y + rect.height - 1;*/
	}
	return img_cropped;
}

void MTCNN::img_show(cv::Mat &img)
{
	for (int i = 0; i < face_total.size(); i++)
	{
		Face_Infor face = face_total[i];
		cv::Rect rect = Rect(round(face.bounding_box.y1), round(face.bounding_box.x1),
			round(face.bounding_box.y2) - round(face.bounding_box.y1) + 1, round(face.bounding_box.x2) - round(face.bounding_box.x1) + 1);

		rectangle(img, rect, cv::Scalar(0, 0, 255), 1);
		cv::putText(img, std::to_string(face_total[i].confidence), cvPoint((int)face_total[i].bounding_box.y1 + 3, (int)face_total[i].bounding_box.x1 + 13),
			cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);

		vector<Point2f> alignment = face_total[i].landmarks;
		for (int j = 0; j < alignment.size(); j++)
		{
			cv::circle(img, cv::Point(round(alignment[j].x), round(alignment[j].y)), 1, cv::Scalar(255, 255, 0), 2);
		}
	}
}

Mat MTCNN::image_face_detection_align(const Mat &img)
{
	vector<Rect> faces;
	vector<vector<Point2f>> alignments;
	detection(img, faces, alignments);

	Mat res;
	if (faces.size() > 1) return res;

	Mat img_aligned;
	AlignFace aligner;
	aligner.similarity_transform(img, alignments[0], img_aligned);
	return img_aligned;
}

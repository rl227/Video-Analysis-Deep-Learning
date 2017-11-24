#include "video_face.h"


VideoFace::VideoFace():_frame_num_limite(9000), _face_num_limite(100), _step(10), dist_threshold(0.8), num_face_reserve(5)
{
	vector<string> mtcnn_model_file = {
		"model/det1.prototxt",
		"model/det2.prototxt",
		"model/det3.prototxt"
	};

	vector<string> mtcnn_trained_file = {
		"model/det1.caffemodel",
		"model/det2.caffemodel",
		"model/det3.caffemodel"
	};

	string centerFace_model_file = "model/face_deploy.prototxt";
	//string centerFace_trained_file = "model/face_model.caffemodel";
	string centerFace_trained_file = "model/face_iter_28000.caffemodel";
	//string centerFace_trained_file = "center_loss_ms/center_loss_ms.caffemodel";

	//string centerFace_model_file = "SphereFace/sphereface_deploy.prototxt";
	//string centerFace_trained_file = "SphereFace/sphereface_model.caffemodel";

	mtcnn = new MTCNN(mtcnn_model_file, mtcnn_trained_file);
	center_face = new CenterFace(centerFace_model_file, centerFace_trained_file);

	aligner = new AlignFace();

	/*_frame_num_limite = 4500;
	_face_num_limite = 50;
	_step = 10;

	dist_threshold = 0.9;
	num_face_reserve = 5;*/
}

void VideoFace::rect_extend(Rect &rec, double r, int width, int height)
{

	Rect pr = rec;
	int dx = cvRound(pr.width*(r - 1) / 2.0);
	int dy = cvRound(pr.height*(r - 1) / 2.0);

	int w = cvRound(pr.width*r);
	int h = cvRound(pr.height*r);

	int x = pr.x - dx;
	int y = pr.y - dy;

	if (x < 0) x = 0;
	if (y < 0) y = 0;
	if (x + w>width - 1) w = width - 1 - x;
	if (y + h>height - 1) h = height - 1 - y;

	pr.x = x;
	pr.y = y;
	pr.width = w;
	pr.height = h;

	rec = pr;
}

void VideoFace::video_face_for_show(const string &video_path)
{
	Mat image;
	VideoCapture cap(video_path);
	if (!cap.isOpened())
		cout << "fail to open!" << endl;
	cap >> image;
	if (!image.data) {
		cout << "¶ÁÈ¡ÊÓÆµÊ§°Ü" << endl;
		return;
	}
	clock_t start;
	int stop = 120000;
	int step = 10;
	while (stop--) {
		start = clock();
		cap >> image;
		if (image.cols == 0 || image.rows == 0) break;
		mtcnn->detection(image);
		mtcnn->img_show(image);
		bool face = mtcnn->face_total.size() > 0;
		if (!face)
		{
			step = 100;
		}
		else
		{
			imshow("result", image);
			step = 25;
			waitKey(1);
		}

		while (step--)
		{
			cap >> image;
			if (image.cols == 0 || image.rows == 0) break;
		}
		start = clock() - start;
		cout << "time is  " << start << endl;
	}
}

void VideoFace::video_face_for_show_no_pass(const string &video_path)
{
	Mat image;
	VideoCapture cap(video_path);
	if (!cap.isOpened())
		cout << "fail to open!" << endl;
	cap >> image;
	if (!image.data) {
		cout << "¶ÁÈ¡ÊÓÆµÊ§°Ü" << endl;
		return;
	}
	clock_t start;
	int stop = 120000;
	int step = 10;
	while (stop--) {
		start = clock();
		cap >> image;
		if (image.cols == 0 || image.rows == 0) break;
		mtcnn->detection(image);
		mtcnn->img_show(image);
		bool face = mtcnn->face_total.size() > 0;
	
		imshow("result", image);
		step = 0;
		waitKey(1);

		while (step--) cap >> image;
	
		start = clock() - start;
		cout << "time is  " << start << endl;
	}
}


bool VideoFace::is_frontalFace(const Rect &box, const vector<Point2f> &landermarks)
{
	Point2f left_eye = landermarks[0];
	Point2f right_eye = landermarks[1];
	Point2f nose = landermarks[2];
	Point2f mouth_left = landermarks[3];
	Point2f mouth_right = landermarks[4];

	/*double face_width = box.width;
	double r = abs(right_eye.x - left_eye.x)/face_width;
	if(r < 0.3) return false;*/
	double dist_l = nose.x - left_eye.x;
	double dist_r = right_eye.x-nose.x;
	double r_d = dist_l / dist_r;
	if (dist_l<0  || dist_r<0 || r_d>3 || r_d<1/3) return false;

	dist_l = nose.x - mouth_left.x;
	dist_r = mouth_right.x - nose.x;
	r_d = dist_l / dist_r;
	if (dist_l<0 || dist_r<0 || r_d>3 || r_d<1 / 3) return false;

	return true;
}

void VideoFace::video_face_detection(const string &video_path, vector<Mat> &face_imgs)
{
	Mat image;
	VideoCapture cap(video_path);
	if (!cap.isOpened()) {
		cout << "fail to open!" << endl; return;
	}

	cap >> image;
	if (!image.data) {
		cout << "read image failed" << endl;
		return;
	}

	clock_t start;

	int count_face = 0;
	int frame_num_limite = _frame_num_limite;
	int face_num_limite = _face_num_limite;
	int step = _step;

	while (frame_num_limite--) {
		cap >> image;
		if (image.cols == 0 || image.rows == 0) break;

		vector<Rect> faces;
		mtcnn->detection(image, faces);
		bool have_face = faces.size() > 0;
		if (!have_face) step = 125;
		else
		{
			step = 25; count_face += faces.size();
			for (int k = 0; k<faces.size(); ++k)
			{
				Rect r = faces[k];
				rect_extend(r, 2, image.cols, image.rows);
				Mat face(image, r);

				face_imgs.push_back(face.clone());
			}
			if (count_face > face_num_limite) break;
		}
		while (step--)
		{
			cap >> image;
			if (image.cols == 0 || image.rows == 0) break;
		}
	}
}

void VideoFace::video_face_detection(const string &video_path, vector<Mat> &face_imgs, vector<vector<Point2f>> &alignments)
{
	Mat image;
	VideoCapture cap(video_path);
	if (!cap.isOpened()) {
		cout << "fail to open!" << endl; return;
	}

	cap >> image;
	if (!image.data) {
		cout << "read image failed" << endl;
		return;
	}

	clock_t start;
	
	int count_face = 0;
	int frame_num_limite = _frame_num_limite;
	int face_num_limite = _face_num_limite;
	int step = _step;
	int have_face_count = 0;
	int frame_count = 0;
	while (frame_count<=frame_num_limite) {
		cap >> image;
		frame_count++;
		if (image.cols == 0 || image.rows == 0) break;

		vector<Rect> faces;
		vector<vector<Point2f>> alignments_tmp;
		mtcnn->detection(image,faces,alignments_tmp);
		//mtcnn->img_show(image);
		bool have_face = faces.size() > 0;
		if (!have_face || have_face_count>=5) {
			step = 125; have_face_count = 0;
		}
		else
		{
			have_face_count++;
			step = 15; count_face += faces.size();
			for (int k=0;k<faces.size();++k)
			{
				Rect r = faces[k];
				//double w_h_r = (double)r.width / r.height;
				//if (w_h_r > 2 || w_h_r < 0.5) continue;
				vector<Point2f> landmarks = alignments_tmp[k];
				if (is_frontalFace(r, landmarks) == false) continue;

				rect_extend(r, 2, image.cols, image.rows);
				for (auto &l : landmarks)
				{
					l.x -= r.x;
					l.y -= r.y;
				}
				Mat face(image, r);

				face_imgs.push_back(face.clone());
				alignments.push_back(landmarks);
			}
			if (count_face > face_num_limite) break;
		}
		while (step--)
		{
			cap >> image;
			frame_count++;
			if (image.cols == 0 || image.rows == 0) break;
		}
	}
}

void VideoFace::video_face_batch(const string &video_folder, const string &foler_out)
{
    vector<string> filePaths=getFiles(video_folder, true);
	for (int i = 0; i < filePaths.size(); ++i)
	{
		cout << "\n\nface detection......" << endl;
		double t = (double)getTickCount();

		vector<Mat> raw_faces;
		vector<vector<Point2f>> alignments;
		string video_path = filePaths[i];
		string video_name = getFileName(video_path);
		video_face_detection(video_path, raw_faces, alignments);

		t = ((double)getTickCount() - t) / getTickFrequency();
		cout << "Times passed in ms: " << t * 1000 << endl;


		cout << "\n face aligning ......" << endl;
		t = (double)getTickCount();
		vector<Mat> faces_aligned;

		/*for (int i = 0; i < raw_faces.size(); ++i)
		{
			Mat raw = raw_faces[i];
			Mat resized;
			resize(raw, resized, Size(96, 112));
			faces_aligned.push_back(resized);
		}*/
		face_align(raw_faces, alignments, faces_aligned);

		t = ((double)getTickCount() - t) / getTickFrequency();
		cout << "Times passed in ms: " << t * 1000 << endl;


		cout << "\n extracting feature ....." << endl;
		t = (double)getTickCount();
		vector<int> indexs = findLeadingActors(faces_aligned);
		t = ((double)getTickCount() - t) / getTickFrequency();
		cout << "Times passed in ms: " << t * 1000 << endl;
		///////////////////////////*result*/////////////////////////////////////////////////////////////////
		string video_folder = foler_out +"/" + video_name;
		{
			int flag = mkdir(video_folder.c_str(),0777);
			if (flag == 0) cout << "make successfully" << endl;
			else cout << "make errorly" << endl;
			CV_Assert(flag == 0);
		}

		string faces_out = video_folder + "/faces";
		{
			int flag = mkdir(faces_out.c_str(),0777);
			if (flag == 0) cout << "make successfully" << endl;
			else cout << "make errorly" << endl;
			CV_Assert(flag == 0);
		}
		
		for (int i = 0; i < raw_faces.size(); ++i)
		{
			string folder_out = faces_out + "/";
			string img_name = folder_out + to_string(i) + ".png";
			imwrite(img_name, faces_aligned[i]);
		}

		string faces_index_out = faces_out+ "/faces_index";
		{
			int flag = mkdir(faces_index_out.c_str(),0777);
			if (flag == 0) cout << "make successfully" << endl;
			else cout << "make errorly" << endl;
			CV_Assert(flag == 0);
		}
		for (int i = 0; i < indexs.size(); ++i)
		{
			int k = indexs[i];
			string folder_out = faces_index_out + "/";
			string img_name = folder_out + to_string(k) + ".png";
			imwrite(img_name, faces_aligned[k]);
		}
	}

}

void VideoFace::calculateDistanceFace(const vector<Mat> &faces, vector<Pair_Dist> &dists)
{
	int n_faces = faces.size();
	vector<vector<float>> feature_faces(n_faces);
	center_face->extractFeature(faces, feature_faces);
	/*for (int i = 0; i < faces.size(); ++i)
	{
		vector<float> res;
		center_face->extractFeature(faces[i], res);
		feature_faces[i] = res;
	}*/
	feature_faces_total = feature_faces;   //

	for (int i = 0; i != n_faces; ++i)
	{
		for (int j = i+1; j != n_faces; ++j)
		{
			double norm_i = norm(feature_faces[i]);
			double norm_j = norm(feature_faces[j]);
			double cos_dist = Mat(feature_faces[i]).dot(Mat(feature_faces[j]))/(norm_i*norm_j);
		/*	Mat vi(feature_faces[i]);
			Mat vj(feature_faces[j]);
			Mat v_dist = vi - vj;
			double dist = norm(v_dist);*/
			if (cos_dist < dist_threshold) continue;
			dists.push_back(Pair_Dist(i, j, cos_dist));
		}
	}

	sort(dists.begin(), dists.end(), [](Pair_Dist a, Pair_Dist b) {return a.dist > b.dist; });
}

vector<int> VideoFace::findLeadingActors(const vector<Mat> &faces)
{
	vector<Pair_Dist> dists;
	calculateDistanceFace(faces, dists);
	int num = faces.size();

	uset.resize(num);
	set_size.resize(num);
	for (int i = 0; i < num; ++i) { uset[i] = i; set_size[i] = 1; }
	for (int k = 0; k < dists.size(); ++k)
	{
		int i = dists[k].i;
		int j = dists[k].j;
		unionSet(i, j);
	}

	vector<int> counter(num, 0);
	vector<int> index_sorted(num, 0);
	for (int i = 0; i != num; ++i)
	{
		counter[uset[i]]++;
		index_sorted[i] = i;
	}
	
	size_t num_tmp = num_face_reserve;
	num_tmp = min(index_sorted.size(), num_tmp);
	sort(index_sorted.begin(), index_sorted.end(), [&counter](int i, int j) {return counter[i] > counter[j]; });

	vector<int> res;
	for (int i = 0; i < index_sorted.size(); ++i)
	{
		if (uset[index_sorted[i]] == index_sorted[i])
		{
			res.push_back(index_sorted[i]);
		}
		if (res.size() == num_tmp) break;
	}
	return res;
}

void VideoFace::face_align(const vector<Mat> &img_in, const vector<vector<Point2f>> &alignments,vector<Mat> &img_out)
{
	for (int i = 0; i < img_in.size(); ++i)
	{
		Mat img = img_in[i];
		Mat img_aligned;
		//cout << "i: " << i << endl;
		aligner->similarity_transform(img, alignments[i], img_aligned);
		img_out.push_back(img_aligned);
	}
}


bool VideoFace::video_match(const vector<vector<float>> &feature_video_1, const vector<vector<float>> &feature_video_2)
{
	int simi_count = 0;
	for (int i = 0; i != feature_video_1.size(); ++i)
	{
		for (int j = 0; j != feature_video_2.size(); ++j)
		{
			double norm_i = norm(feature_video_1[i]);
			double norm_j = norm(feature_video_2[j]);
			double cos_dist = Mat(feature_video_1[i]).dot(Mat(feature_video_2[j])) / (norm_i*norm_j);
			if (cos_dist > dist_threshold)
			{
				simi_count++;
				if (simi_count > 1) return true;
			}
		}
	}
	return false;
}


void VideoFace::videos_vertification(const vector<string> &video_paths)
{
	int video_count = video_paths.size();
	vector<vector<vector<float>>> feature_videos(video_count);

	for (int i = 0; i < video_paths.size(); ++i)
	{
		cout << "processing " << video_paths[i] << endl;
		string temp_path = video_paths[i];
		vector<Mat> raw_faces;
		vector<vector<Point2f>> alignments;
		video_face_detection(temp_path, raw_faces, alignments);

		vector<Mat> faces_aligned;
	/*	for (int i = 0; i < raw_faces.size(); ++i)
		{
			Mat raw = raw_faces[i];
			Mat resized;
			resize(raw, resized, Size(96, 112));
			faces_aligned.push_back(resized);
		}*/
		face_align(raw_faces, alignments, faces_aligned);

		vector<int> indexs=findLeadingActors(faces_aligned);
		for (int j = 0; j < indexs.size(); ++j)
		{
			feature_videos[i].push_back(feature_faces_total[j]);
		}
	}
	
	for (int i = 0; i < video_paths.size(); ++i)
	{
		string video_name = getFileName(video_paths[i]);
		cout << video_name << "   " << i << endl;
	}
	for (int i = 0; i != video_count; ++i)
	{
		//cout << i << "	" << endl;
		for (int j = i+1; j != video_count; ++j)
		{
			bool isIn = video_match(feature_videos[i], feature_videos[j]);
			
			if(isIn) cout << i << " and " << j << " : " << isIn << endl;
		}
	}
}

void VideoFace::video_face_samples(const string &video_path, vector<Mat> &face_imgs)
{
	Mat image;
	VideoCapture cap(video_path);
	if (!cap.isOpened()) {
		cout << "fail to open!" << endl; return;
	}

	cap >> image;
	if (!image.data) {
		cout << "read image failed" << endl;
		return;
	}

	clock_t start;

	int count_face = 0;
	int frame_num_limite = 50000;
	int face_num_limite = 2000;
	int step = 0;
	int have_face_count = 0;
	int frame_count = 0;
	while (frame_count <= frame_num_limite) {
		cap >> image;
		frame_count++;
		if (image.cols == 0 || image.rows == 0) break;

		vector<Rect> faces;
		vector<vector<Point2f>> alignments_tmp;
		mtcnn->detection(image, faces, alignments_tmp);
		//mtcnn->img_show(image);
		bool have_face = faces.size() > 0;
		if (!have_face) {
			step = 125; have_face_count = 0;
		}
		else
		{
			have_face_count++;
			step = 10; count_face += faces.size();
			for (int k = 0; k<faces.size(); ++k)
			{
				vector<Point2f> landmarks = alignments_tmp[k];
				Mat img_aligned;
				aligner->similarity_transform(image, landmarks, img_aligned);
				face_imgs.push_back(img_aligned);
			}
			if (count_face > face_num_limite) break;
		}
		while (step--)
		{
			cap >> image;
			frame_count++;
			if (image.cols == 0 || image.rows == 0) break;
		}
	}
}


vector<vector<int>> VideoFace::findFaceClusters(const vector<Mat> &faces)
{
	vector<Pair_Dist> dists;
	calculateDistanceFace(faces, dists);
	int num = faces.size();

	uset.resize(num);
	set_size.resize(num);
	for (int i = 0; i < num; ++i) { uset[i] = i; set_size[i] = 1; }
	for (int k = 0; k < dists.size(); ++k)
	{
		int i = dists[k].i;
		int j = dists[k].j;
		unionSet(i, j);
	}

	vector<int> counter(num, 0);
	vector<int> index_sorted(num, 0);
	for (int i = 0; i != num; ++i)
	{
		counter[uset[i]]++;
		index_sorted[i] = i;
	}

	vector<vector<int>> tmp(num);
	for (int i = 0; i < uset.size(); ++i)
	{
		tmp[uset[i]].push_back(i);
	}
	vector<vector<int>> res;

	for (int i = 0; i < num; ++i)
	{
		if (!tmp[i].empty())
			res.push_back(tmp[i]);
	}
	return res;
}


void VideoFace::video_2_faces(const string &video_path,const string &video_folder)
{
	vector<Mat> faces;
	video_face_samples(video_path, faces);
	vector<vector<int>> indexs = findFaceClusters(faces);

	for (int i = 0; i < indexs.size(); ++i)
	{
		string faces_out = video_folder + "/"+to_string(i);
		if (access(faces_out.c_str(), 0) == -1)
		{
			cout << faces_out << " is not existing" << endl;
			int flag = mkdir(faces_out.c_str(),0777);
			if (flag == 0) cout << "make successfully" << endl;
			else cout << "make errorly" << endl;
			CV_Assert(flag == 0);
		}
		for (int j = 0; j < indexs[i].size(); ++j)
		{
			string img_out = faces_out + "/" + to_string(i)+"_"+to_string(j) + ".png";
			imwrite(img_out, faces[indexs[i][j]]);
		}
	}
}

void VideoFace::video_2_faces(const string &video_path, const string &video_folder,const string &class_name)
{
    vector<Mat> faces;
    video_face_samples(video_path, faces);
                
    for (int i = 0; i < faces.size(); ++i)
    {
        string faces_out = video_folder + "/" + class_name+"_"+to_string(i)+".png";
        imwrite(faces_out, faces[i]);
    }
}

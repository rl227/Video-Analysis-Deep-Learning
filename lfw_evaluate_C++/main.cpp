#include <iostream>
#include "CenterFace.h"
#include "MTCNN.h"
#include "SimilarityTransform.h"
#include "image_face.h"

using namespace std;
using namespace cv;

#define LFW_eval

#ifdef LFW_eval

//注意：当字符串为空时，也会返回一个空字符串  
void split(std::string& s, std::string& delim, std::vector< std::string >& ret)
{
	size_t last = 0;
	size_t index = s.find_first_of(delim, last);
	while (index != std::string::npos)
	{
		ret.push_back(s.substr(last, index - last));
		last = index + 1;
		index = s.find_first_of(delim, last);
	}
	if (index - last>0)
	{
		ret.push_back(s.substr(last, index - last));
	}
}

int main()
{
	ImageFace image_facer;
	ifstream pair("pairs.txt");
	vector<string> pair_pos;
	vector<string> pair_neg;
	vector<double> dist_pos;
	vector<double> dist_neg;
	string lfw_path = "lfw";
	string temp;
	int count = 0;
	
	vector<string> lines;
	while (getline(pair, temp))
	{
		lines.push_back(temp);
	}
    //Caffe::SetDevice(0);
    //Caffe::set_mode(Caffe::GPU);
	for(int i=0;i<lines.size();++i)
	{
		temp = lines[i];
		vector<string> t;
		string delim = "\t";
		split(temp,delim,t);
		count++;
		//if (count == 1000) break;
		cout << "i:  " << count << endl;
		if (t.size() == 3)
		{
			string folder = t[0];
			string index_1 = t[1];
			string index_2 = t[2];
			while (index_1.size() != 4)
				index_1 = "0" + index_1;
			while (index_2.size() != 4)
				index_2 = "0" + index_2;
			string img_path_1 = lfw_path + "/" + folder + "/" + folder + "_" + index_1 + ".jpg";
			string img_path_2 = lfw_path + "/" + folder + "/" + folder + "_" + index_2 + ".jpg";
			double dist = image_facer.distTwoImages(img_path_1, img_path_2);
			if (dist < 0.5) cout << "i:  " << i << "dist: "<<dist<<endl;
			dist_pos.push_back(dist);
		}
		else if (t.size() == 4)
		{
			string folder_1 = t[0];
			string index_1 = t[1];
			string folder_2 = t[2];
			string index_2 = t[3];

			while (index_1.size() != 4)
				index_1 = "0" + index_1;
			while (index_2.size() != 4)
				index_2 = "0" + index_2;
			string img_path_1 = lfw_path + "/" + folder_1 + "/" + folder_1 + "_" + index_1 + ".jpg";
			string img_path_2 = lfw_path + "/" + folder_2 + "/" + folder_2 + "_" + index_2 + ".jpg";
			double dist = image_facer.distTwoImages(img_path_1, img_path_2);
			dist_neg.push_back(dist);
		}
	}

	sort(dist_pos.begin(), dist_pos.end(), greater<double>());
	sort(dist_neg.begin(), dist_neg.end(), greater<double>());
	ofstream dist_pos_out("dist_pos_out.txt");
	ofstream dist_neg_out("dist_neg_out.txt");
	for (auto d : dist_pos)
		dist_pos_out << d << "\n";
	for (auto d : dist_neg)
		dist_neg_out << d << "\n";

	return 0;
}

#else

int main()
{
	ImageFace image_facer;
	
	int choose = 6;
	if (choose == 0)
	{
		cout << "make templates" << endl;
		string tem_file = "face_template_latest.xml";
		if (_access(tem_file.c_str(), 0) == 0)
		{
			cout << "the template file already exits" << endl;
			return 0;
		}
		string img_folder = "celeb/images";
		image_facer.makeFaceTemplates(img_folder, tem_file);
	}
	else if (choose == 1)
	{
		cout << "add people to templates" << endl;
		string person_folder = "celeb/images/林青霞.jpg";
		string tem_file = "face_templates_10000.xml";
		//image_facer.addPersonsToTemplate(tem_file, person_folder);
		image_facer.replacePersonsToTemplate(tem_file, person_folder);
	}
	else if (choose == 2)
	{
		cout << "face recongnition" << endl;
		vector<FaceTemp> faces_temps;
		string tem_file = "face_template_latest.xml";
		image_facer.readFaceTemplates(tem_file, faces_temps);
		string img_path = "林青霞_1.jpg";

		vector<FaceResult> res = image_facer.faceRecognition(img_path, faces_temps, 10);
		for (auto &r : res)
		{
			cout << r.name << "  :  " << r.similarity << endl;
		}
	}
	else if (choose == 3)
	{
		cout << "face recognition in folder" << endl;
		string folder = "test";
		vector<vector<FaceResult>> res = image_facer.faceRecognition_folder(folder, 1);
		for (auto r : res)
		{
			if (r.empty())
				cout << "none" << endl;
			else
				cout << r[0].name << "  :  " << r[0].similarity << endl;
		}
	}
	else if (choose == 4)
	{
		cout << "make templates from muti-images of one person" << endl;
		string tem_file = "face_templates_celeb_test.xml";
		string folder_in = "G:/数据集/douban_actor/actors";
		makeFaceTemplates("tt_in", "tt_out", tem_file, 2);
		//image_facer.makeFaceTemplates("celeb/images", tem_file);
	}
	else if (choose == 5)
	{
		cout << "face image classification" << endl;
		string img_path = "tt.png";
		vector<FaceResult> res = image_facer.faceClassification(img_path, 10);

		for (auto r : res)
			cout << r.name<<"  " << r.similarity <<endl;
	}
	else if (choose == 6)
	{
		cout << "face classification in folder" << endl;
		string folder = "test";
		vector<vector<FaceResult>> res = image_facer.faceClassification_folder(folder, 1);
		for (auto r : res)
		{
			if (r.empty())
				cout << "none" << endl;
			else
				cout << r[0].name << "  :  " << r[0].similarity << endl;
		}
	}
	
	/*string img_path = "3.JPG";
	string faces_out = "faces";
	Mat img = imread(img_path);
	ImageFace image_facer;*/
	//vector<Mat> faces_aligned;
	//image_facer.mtcnn->image_face_detection_align(img, faces_aligned);

	//
	//if(_access(faces_out.c_str(),0)== -1)
	//{
	//	int flag = _mkdir(faces_out.c_str());
	//	if (flag == 0) cout << "make successfully" << endl;
	//	else cout << "make errorly" << endl;
	//	CV_Assert(flag == 0);
	//}

	//for (int i = 0; i < faces_aligned.size(); ++i)
	//{
	//	string folder_out = faces_out + "/";
	//	string img_name = folder_out + "3_"+to_string(i) + ".png";
	//	imwrite(img_name, faces_aligned[i]);
	//}

	//image_facer.compareImages(faces_out);
	//image_facer.compareImages("faces_1","faces_3");
	
	waitKey(0);
	return 0;
}

#endif

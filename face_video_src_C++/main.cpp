#include <iostream>
#include "CenterFace.h"
#include "MTCNN.h"
#include "SimilarityTransform.h"
//#include "video_face.h"
#include "videoProcess.h"
//#include "image_face.h"

using namespace std;
using namespace cv;


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

//string centerFace_model_file = "model/face_deploy.prototxt";
//string centerFace_trained_file = "model/face_iter_28000.caffemodel";
//string centerFace_model_file = "center_loss_ms/center_loss_ms.prototxt";
//string centerFace_trained_file = "center_loss_ms/center_loss_ms.caffemodel";

//string centerFace_model_file = "SphereFace/sphereface_deploy.prototxt";
//string centerFace_trained_file = "SphereFace/sphereface_model.caffemodel";


//int main()
//{
//	string video_folder = "path.txt";
//	vector<string> video_paths;
//	readImageFile(video_folder, video_paths);
//	VideoFace video_facer;
//	video_facer.videos_vertification(video_paths);
//}

int main(int argc, char** argv)
{
	string video_folder = argv[1];
	string faces_folder = argv[2];
    int num_thread=atoi(argv[3]);
	processVideos2FacesThreads(video_folder, faces_folder, num_thread);

    return 0;
}


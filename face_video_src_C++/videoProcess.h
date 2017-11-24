#ifndef _VIDEOPROCESS_H
#define _VIDEOPROCESS_H

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  

#include <iostream>  
#include <string>
#include <fstream>
#include <thread>

#include "readPath.h"
#include "video_face.h"

using namespace std;
using namespace cv;

void processVideos2FacesThreads(const string &folder_in, const string &folder_out, const int num_thread);

#endif

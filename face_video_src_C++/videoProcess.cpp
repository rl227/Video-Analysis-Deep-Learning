#include "videoProcess.h"


void processVideo_thread(const vector<string> &video_paths, const int s, const int t, const string &dst_classFolder, VideoFace video_face)
{
	for (int i = s; i < s+t; ++i)
	{
		string video_path = video_paths[i];
		cout << "i: " << i << "   " << video_path << endl;

		int a = video_path.find_last_of('.');
		int b = video_path.find_last_of('/', a - 1);

		string className = video_path.substr(b+1, a-b-1);
		string dst_name = dst_classFolder + "/" + className;

		video_face.video_2_faces(video_path, dst_name,className);
	}
}

void processVideos2FacesThreads(const string &folder_in, const string &folder_out, const int num_thread)
{
	vector<string> filePaths = getFiles(folder_in, true);
	vector<string> classNames = getFiles(folder_in, false);
	for (auto &s : classNames)
	{
		int n = s.find_last_of('.');
		s = s.substr(0, n);
	}

	vector<string> video_paths(filePaths);
	for (int i=0;i<filePaths.size();++i)
	{
		string dst_classFolder = folder_out+"/" + classNames[i];
		if (access(dst_classFolder.c_str(), 0) == -1)
		{
			cout << dst_classFolder << " is not existing" << endl;
			int flag = mkdir(dst_classFolder.c_str(),0777);
			if (flag == 0) cout << "make successfully" << endl;
			else cout << "make errorly" << endl;
			CV_Assert(flag == 0);
		}
	}
	int total_videos = video_paths.size();
	int videos_per_thread = (int)ceil((double)total_videos / num_thread);
	vector<thread> threads;

	int s = 0;
	for (int i = 0; i < num_thread; ++i)
	{
		VideoFace video_face;
		int t = min(videos_per_thread, total_videos - s);
		threads.push_back(thread(processVideo_thread, video_paths, s, t, folder_out,video_face));
		s += t;
	}

	for (int i = 0; i<threads.size(); i++)
		threads[i].join();

}

#include "readPath.h"

/**
* @function: ªÒ»°cate_dir
* @param: cate_dir - string
* @result£∫vector<string>
*/

vector<string> getFiles(const string &cate_dir,bool append=false)
{
	vector<string> files;  

	DIR *dir;
	struct dirent *ptr;
	char base[1000];

	if ((dir = opendir(cate_dir.c_str())) == NULL)
	{
		perror("Open dir error...");
		exit(1);
	}
    
	while ((ptr = readdir(dir)) != NULL)
	{
		if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)    ///current dir OR parrent dir  
			continue;
		else if (ptr->d_type == 8)    ///file  
        {
            if(append) files.push_back(cate_dir+"/"+ptr->d_name);
            else files.push_back(ptr->d_name);
        }
		else if (ptr->d_type == 10)    ///link file  
			continue;
		else if (ptr->d_type == 4)    ///dir  
		{
            if(append) files.push_back(cate_dir+"/"+ptr->d_name);
            else files.push_back(ptr->d_name);
			/*
			memset(base,'\0',sizeof(base));
			strcpy(base,basePath);
			strcat(base,"/");
			strcat(base,ptr->d_nSame);
			readFileList(base);
			*/
		}
	}
	closedir(dir);

	sort(files.begin(), files.end());
	return files;
}



void split_train_test(const string &folder, const string &train_out, const string &test_out, double split_rate)
{
	ofstream train(train_out);
	ofstream test(test_out);

	vector<string> filePaths = getFiles(folder, true);
    cout<<"get files: "<<filePaths.size()<<endl;

	for (int i = 0; i < filePaths.size(); ++i)
	{
		string subFolder = filePaths[i];
		vector<string> subFilePaths = getFiles(subFolder, true);
		for (auto &s : subFilePaths)
		{
			s += " " + to_string(i);
		}

		int max = subFilePaths.size();
		vector<int> index(max, 0);
		for (int i = 0; i < max; ++i) index[i] = i;
		random_shuffle(index.begin(), index.end());
		int num_test = max*split_rate;
		for (int i = 0; i < num_test;++i)
		{
			test << subFilePaths[index[i]] << endl;
		}
		for (int i = num_test; i < max; ++i)
		{
			train << subFilePaths[index[i]] << endl;
		}
	}
	train.close();
	test.close();
}


void shuffle_train_test(const string &fileList_in, const string &out)
{
	ifstream in(fileList_in);
	vector<string> path_in;
	int count = 0;
	while (in)
	{
		string buf;
		if (getline(in, buf))
		{
			path_in.push_back(buf);
			count++;
		}
	}
	in.close();

	vector<int> index(count, 0);
	vector<int> seed_tmp(count, 0);
	vector<string> path_out(count);
	for (int i = 0; i != count; ++i) index[i] = i;
	for(int i = 0; i != count; ++i)
	{
		int seed = rand() % (count - i);
		//cout << seed << endl;
		seed_tmp[i] = index[seed];
		path_out[i] = path_in[index[seed]];
		index[seed] = index[count - i - 1];
	}
	sort(seed_tmp.begin(), seed_tmp.end());
	ofstream out_shuffled(out);
	for(auto p:path_out)
	{
		out_shuffled << p << endl;
	}
	out_shuffled.close();
}

string getFileName(const string &str)
{
	int m = str.find_last_of('/');
	int n = str.find_last_of('.');

	return str.substr(m + 1, n - m - 1);
}

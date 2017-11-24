#ifndef _READPATH_H
#define _READPATH_H

#include <iostream>  
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
  
#include <unistd.h>  
#include <dirent.h>  
#include <sys/types.h>  
#include <sys/stat.h> 


#include <stdlib.h>  
#include <stdio.h>  
#include <string.h> 
  

using namespace std;

vector<string> getFiles(const string &cate_dir, bool append);
void split_train_test(const string &folder, const string &train_out, const string &test_out, double split_rate = 0.3);
void shuffle_train_test(const string &fileList_in, const string &out);

string getFileName(const string &str);

#endif

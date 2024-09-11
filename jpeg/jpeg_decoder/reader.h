#ifndef READER_H
#define READER_H
#include<bits/stdc++.h>
#include"fake_string.h"
class Reader
{
	FILE *fp;
	char *buffer;
	char *bufferBegin;
	unsigned int fileSize;
	char *bufferEnd;
	char *filePath;
public:
	fakeString data;
	Reader(){}
	Reader(char *Path):filePath(Path){}
	~Reader()
	{
		free(bufferBegin);
	}
	void read();
	int get();
};
#endif

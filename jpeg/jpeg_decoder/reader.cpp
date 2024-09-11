#include"reader.h"
void Reader::read()
{
	fp=fopen(filePath,"rb");
	fseek(fp,0,SEEK_END);
	fileSize=ftell(fp);
	fseek(fp,0,SEEK_SET);
	bufferBegin=(char*)malloc(fileSize+10);
	fread(bufferBegin,fileSize,1,fp);
	bufferEnd=bufferBegin+fileSize;
	buffer=bufferBegin;
	data=fakeString(buffer,fileSize);
}
int Reader::get()
{
	return data.get();
}

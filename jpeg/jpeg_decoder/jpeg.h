#ifndef JPEG_H
#define JPEG_H
#include<bits/stdc++.h>
#include "fake_string.h"
struct SOI
{
	void read(fakeString &data);
};
struct APP0
{
	int length;
	long long exchangeType;
	int mainVersion;
	int minorVersion;
	int densityUnit;
	int xDensity;
	int yDensity;
	int xPixels;
	int yPixels;
	int *rgbThumbnail; //
	~APP0()
	{
		free(rgbThumbnail);
	}
	void read(fakeString &data);
};
struct APP1
{
	int length;
	long long exchangeType;
	int byteAlign;
	int tagMark;
	int offset;
	void read(fakeString &data);
};
struct QT
{
	int message;
	int number;
	int bits;
	int qt[66];
	bool operator < (const QT& rhs)const 
	{
		return number < rhs.number;
	}
};
struct HT
{
	int message;
	int type;//0 DC, 1 AC
	int number;
	int hlt[22];
	int sum;
	int *hvt;
	bool operator < (const HT& rhs)const 
	{
		return number<rhs.number;
	}
	HT(){}
	HT(const HT &rhs)
	{
		message=rhs.message;
		type=rhs.type;
		number=rhs.number;
		for(int i=1;i<=16;i++)hlt[i]=rhs.hlt[i];
		sum=rhs.sum;
		hvt=(int*)malloc(sizeof(int)*(sum+10));
		for(int i=0;i<sum;i++)hvt[i]=rhs.hvt[i];
	}
	HT& operator = (const HT& rhs)
	{
		message=rhs.message;
		type=rhs.type;
		number=rhs.number;
		for(int i=1;i<=16;i++)hlt[i]=rhs.hlt[i];
		sum=rhs.sum;
		hvt=(int*)malloc(sizeof(int)*(sum+10));
		for(int i=0;i<sum;i++)hvt[i]=rhs.hvt[i];
		return *this;
	}
	~HT()
	{
		free(hvt);
	}
};
struct DQT
{
	int length;
	void read(fakeString &data, std::vector<QT> &qts);
};
struct SOF0
{
	int length;
	int deepth;
	int height;
	int width;
	int channelSum;
	// Y U V | id, ratio, qt_number
	// id: 1Y 2Cb 3Cr 4I 5Q
	int channels[4][3];
	void read(fakeString &data);
};
struct DHT
{
	int length;
	void read(fakeString &data,std::map<int,HT> &hts);
};
struct DRI
{
	int length;
	int interval;
	void read(fakeString &data);
};
struct SOS
{
	int length;
	void read(fakeString &data);
};
class JPEG
{
public:
	SOI soi;
	APP0 app0;
	APP1 app1;
	std::vector<DQT> dqts;
	std::vector<QT> qts;
	std::vector<DHT> dhts;
	std::map<int,HT> hts;
	SOF0 sof0;
	DRI dri;
	SOS sos;
	
	void read(fakeString &data);
	
};
#endif

#ifndef FAKE_STRING_H
#define FAKE_STRING_H
#include<bits/stdc++.h>
class fakeString
{
	char *data;
	int dataLength;
public:
	fakeString(){}
	fakeString(char *data,int dataLength):data(data),dataLength(dataLength){}
	fakeString substr(int start,int len);
	int length();
	long long get(int x);
	int get();
	bool empty();
};
#endif

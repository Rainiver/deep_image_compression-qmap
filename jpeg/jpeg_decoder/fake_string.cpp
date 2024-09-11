#include"fake_string.h"
fakeString fakeString::substr(int start,int len)
{
	if(len+start>dataLength)len=dataLength-start;
	return fakeString(data+start,len);
}
int fakeString::length()
{
	return dataLength;
}
long long fakeString::get(int x)
{
	assert(x<=dataLength);
	long long ret=0;
	for(int i=0;i<x;i++)
	{
		int ch=get();
		dataLength--;
		ret=ret*256+ch;
	}
	return ret;
}
int fakeString::get()
{
	assert(!empty());
	unsigned char ch=*(data++);
	dataLength--;
	if((int)ch==0xff)
		if((unsigned char)(*data)==0x00)data++,dataLength--;
	return (int)ch;
}
bool fakeString::empty()
{
	return dataLength==0;
}

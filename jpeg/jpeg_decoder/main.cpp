#include <bits/stdc++.h>
#include "reader.h"
#include "jpeg.h"

char pp[11]="test.jpeg";

void decode(Reader &reader)
{
	JPEG image;
	image.read(reader.data);
    QT vqt=image.qts[0];
    QT uvqt=image.qts[1];
    freopen("quant_table.txt","a+",stdout);
    for(int i=0;i<64;i++)
	{
		if(i%8==0&&i>0)puts("");
		printf("%d ",vqt.qt[i]);
	}
    puts("");
    for(int i=0;i<64;i++)
	{
		if(i%8==0&&i>0)puts("");
		printf("%d ",uvqt.qt[i]);
	}
    puts("");
}
extern "C"
{
void get_QT(char* path,int *YQT, int *UVQT)
{
    Reader reader(path);
    reader.read();
	JPEG image;
	image.read(reader.data);
    assert(image.qts.size()==2);
    QT vqt=image.qts[0];
    QT uvqt=image.qts[1];
    for(int i=0;i<64;i++)YQT[i]=vqt.qt[i];
    for(int i=0;i<64;i++)UVQT[i]=uvqt.qt[i];
}
}

int main(int argc, char** argv) {
	Reader reader(argv[1]);
	reader.read();
	decode(reader);
	return 0;
}

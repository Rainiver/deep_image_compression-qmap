#include "jpeg.h"
void JPEG::read(fakeString &data)
{
	int ident,type;
	while(!data.empty())
	{
		ident=data.get();
		assert(ident==0xff);
		type=data.get();
		while(type==0xff)type=data.get();
		
		if(type==0xD8)soi.read(data);
		else if(type==0xe0)app0.read(data);
		else if(type==0xe1)app1.read(data);
		else if(type==0xdb)
		{
			DQT dqt;
			dqt.read(data,qts);
			dqts.push_back(dqt);
		}
		else if(type==0xc0)sof0.read(data);
		else if(type==0xc4)
		{
			DHT dht;
			dht.read(data,hts);
			dhts.push_back(dht);
		}
		else if(type==0xdd)dri.read(data);
		else if(type==0xda)
		{
			return ;
			sos.read(data);
		}
		else 
		{
			printf("%x\n",type);
			return;
		}
	}
}
void SOI::read(fakeString &data)
{
	return;
}
void APP0::read(fakeString &data)
{
	puts("--------------APP0---------------");
	length=data.get(2);
	printf("app0 length: %d\n",length);
	
	exchangeType=data.get(5);
	assert(exchangeType==0x4A46494600);
	printf("exchange type: JFIF\n");
	
	mainVersion=data.get();
	printf("main version: %d\n",mainVersion);
	
	minorVersion=data.get();
	printf("minor version %d\n",minorVersion);
	
	densityUnit=data.get();
	printf("density unit: %d\n",densityUnit);
	
	xDensity=data.get(2);
	printf("x density: %d\n",xDensity);
	
	yDensity=data.get(2);
	printf("y density: %d\n",yDensity);
	
	xPixels=data.get();
	printf("thumbnail x pixels: %d\n",xPixels);
	
	yPixels=data.get();
	printf("thumbnail x pixels: %d\n",yPixels);
	assert(xPixels*yPixels*3+16==length);
	
	rgbThumbnail=(int*)malloc(sizeof(int)*(xPixels*yPixels+10));
	if(xPixels*yPixels==0)return;
	for(int i=0;i<xPixels;i++)
		for(int j=0;j<yPixels;j++)
			rgbThumbnail[i*yPixels+j]=data.get();
}
void APP1::read(fakeString &data)
{
	puts("--------------APP1---------------");
	length=data.get(2);
	printf("APP1 length: %d\n",length);
	
	exchangeType=data.get(6);
	assert(exchangeType==0x457869660000);
	
	printf("exchangeType: exif\n");
	
	data.get(length-8);
}
void DQT::read(fakeString &data,std::vector<QT> &qts)
{
	puts("---------------DQT---------------");
	length=data.get(2);
	printf("DQT length: %d\n",length);
	
	int tot=2;
	
	while(tot<length)
	{
		QT table;
		table.message=data.get();
		tot++;
		table.number=(table.message&16);
		table.bits=(table.message>>4);
		printf("QT number: %d\nQT bits: %d\n",table.number,table.bits?16:8);
		for(int j=0;j<64;j++)
			table.qt[j]=data.get(1+table.bits);
			
		tot+=(1+table.bits)*64;
		qts.push_back(table);
	}
}
void SOF0::read(fakeString &data)
{
	puts("--------------SOF0---------------");
	length=data.get(2);
	printf("SOF0 length: %d\n",length);
	
	deepth=data.get();
	printf("deepth: %d\n",deepth);
	
	height=data.get(2);
	printf("height: %d\n",height);
	
	width=data.get(2);
	printf("width: %d\n",width);
	
	channelSum=data.get(1);
	assert(length=8+3*channelSum);
	printf("channel sum: %d\n",channelSum);
	
	for(int i=0;i<channelSum;i++)
		for(int j=0;j<3;j++)
			channels[i][j]=data.get();
}
void DHT::read(fakeString &data,std::map<int,HT> &hts)
{
	puts("---------------DHT---------------");
	length=data.get(2);
	printf("DHT length: %d\n",length);
	
	int tot=2;
	
	while(tot<length)
	{
		HT table;
		table.message=data.get();
		table.number=(table.message&16);
		table.type=(table.message>>4);
		printf("HT number: %d\nHT type: %s\n",table.number,table.type?"DC":"AC");
		tot+=1;
		table.sum=0;
		for(int i=1;i<=16;i++)
			table.hlt[i]=data.get(1),table.sum+=table.hlt[i];
		tot+=16;
		
		table.hvt=(int*)malloc(sizeof(int)*(table.sum+10));
		for(int i=1;i<=table.sum;i++)table.hvt[i]=data.get();
		tot+=table.sum;
		hts[table.number]=table;
	}
}
void DRI::read(fakeString &data)
{
	puts("---------------DRI---------------");
	length=data.get();
	assert(length==0x0004);
	printf("DRI length: %d\n",length);
	
	interval=data.get(2);
	printf("inteval: %d\n",interval);
}
void SOS::read(fakeString &data)
{
	puts("---------------SOS---------------");
	length=data.get();
	printf("SOS length: %d\n",length);
	return ;
	data.get(length-2);
}

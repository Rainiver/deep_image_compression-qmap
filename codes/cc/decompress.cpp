#include<bits/stdc++.h>
#define rre(i,r,l) for(int i=(r);i>=(l);i--)
#define re(i,l,r) for(int i=(l);i<=(r);i++)
#define Clear(a,b) memset(a,b,sizeof(a))
#define inout(x) printf("%d",(x))
#define douin(x) scanf("%lf",&x)
#define strin(x) scanf("%s",(x))
#define op operator
typedef unsigned long long ULL;
typedef uint32_t uint;
typedef long long LL;
using namespace std;
class InputStream
{
public:
	istream &input;
	int currentByte,numBitsRemaining;
	InputStream(istream &in):input(in)
	{
		currentByte=0;
		numBitsRemaining=0;
	}
	inline int read()
	{
		if(currentByte==-1)
			return -1;
		if(numBitsRemaining==0)
		{
			currentByte=input.get();  // Note: istream.get() returns int, not char
			if (currentByte==EOF)
				return -1;
			if (currentByte<0||currentByte>255)
				throw std::logic_error("Assertion error");
			numBitsRemaining=8;
		}
		if(numBitsRemaining<=0)
			throw logic_error("Assertion error");
		numBitsRemaining--;
		return(currentByte>>numBitsRemaining)&1;
	}
};
class Table_
{
public:
	vector<uint> frequencies;
	mutable vector<uint> cumulative;
	uint total;
	inline uint checkedAdd(uint x,uint y)const 
	{
		if (x>UINT32_MAX-y)
			throw overflow_error("Arithmetic overflow");
		return x+y;
	}
	inline void initCumulative(bool checkTotal=true)const
	{
		if(!cumulative.empty())return;
		uint sum=0;
		cumulative.push_back(sum);
		for(uint freq:frequencies)
			sum=checkedAdd(freq,sum),cumulative.push_back(sum);
		if(checkTotal&&sum!=total)
			throw logic_error("Assertion error");
	}
	inline uint getHigh(uint symbol)const 
	{
		initCumulative();
		return cumulative.at(symbol+1);
	}
	inline uint getLow(uint symbol)const 
	{
		initCumulative();
		return cumulative.at(symbol);
	}
	inline uint getTotal()const
	{
		return total;
	}
	inline void increment(uint symbol)
	{
		if (frequencies.at(symbol)==UINT32_MAX)
			throw overflow_error("Arithmetic overflow");
		total=checkedAdd(total,1);
		frequencies.at(symbol)++;
		cumulative.clear();
	}
	inline uint getSymbolLimit()const
	{
		return static_cast<uint>(frequencies.size());
	}
	Table_(const vector<uint> &freqs)
	{
		if(freqs.size()>UINT32_MAX-1)
			throw length_error("Too many symbols");
		uint size = static_cast<uint>(freqs.size());
		if(size<1)
			throw std::invalid_argument("At least 1 symbol needed");
		
		frequencies=freqs;
		cumulative.reserve(size+1);
		initCumulative(false);
		total=getHigh(size-1);
	}
};
class Decoder
{
public:
	int numStateBits;
	ULL fullRange,halfRange,quarterRange,minimumRange,maximumTotal,stateMask;
	ULL low,high;
	InputStream &input;
	ULL code;
	inline int readCodeBit()
	{
		int temp=input.read();
		if (temp==-1)
			temp=0;
		return temp;	
	}
	inline void shift()
	{
		code=((code<<1)&stateMask)|readCodeBit();
	}
	inline void underflow()
	{
		code=(code&halfRange)|((code<<1)&(stateMask>>1))|readCodeBit();
	}
	inline void update(const Table_ &freqs, uint symbol)
	{
		if(low>=high||(low&stateMask)!=low||(high&stateMask)!=high)
			throw logic_error("Assertion error: Low or high out of range");
		ULL range=high-low+1;
		if(range<minimumRange||range>fullRange)
			throw logic_error("Assertion error: Range out of range");
		
		// Frequency table values check
		uint total=freqs.getTotal();
		uint symLow=freqs.getLow(symbol);
		uint symHigh=freqs.getHigh(symbol);
		if(symLow==symHigh)
			throw invalid_argument("Symbol has zero frequency");
		if(total>maximumTotal)
			throw invalid_argument("Cannot code symbol because total is too large");
		
		// Update range
		ULL newLow=low+symLow*range/total;
		ULL newHigh=low+symHigh*range/total-1;
		low=newLow;
		high=newHigh;
		
		// While low and high have the same top bit value, shift them out
		while(((low^high)&halfRange)==0)
		{
			shift();
			low=((low<<1)&stateMask);
			high=((high<<1)&stateMask)|1;
		}
		// Now low's top bit must be 0 and high's top bit must be 1
		
		// While low's top two bits are 01 and high's are 10, delete the second highest bit of both
		while((low&~high&quarterRange)!=0)
		{
			underflow();
			low=(low<<1)^halfRange;
			high=((high^halfRange)<<1)|halfRange|1;
		}
	}
	inline uint read(const Table_ &freqs)
	{
		// Translate from coding range scale to frequency table scale
		uint total=freqs.getTotal();
		if (total>maximumTotal)
			throw invalid_argument("Cannot decode symbol because total is too large");
		uint64_t range=high-low+1;
		uint64_t offset=code-low;
		uint64_t value=((offset+1)*total-1)/range;
		if (value*range/total>offset)
			throw std::logic_error("Assertion error");
		if (value>=total)
			throw std::logic_error("Assertion error");
		
		// A kind of binary search. Find highest symbol such that freqs.getLow(symbol) <= value.
		uint start=0;
		uint end=freqs.getSymbolLimit();
		while(end-start>1) {
			uint middle=(start+end)>>1;
			if(freqs.getLow(middle)>value)
				end=middle;
			else
				start=middle;
		}
		if (start+1!=end)
			throw logic_error("Assertion error");
		
		uint symbol=start;
		if(offset<freqs.getLow(symbol)*range/total||freqs.getHigh(symbol)*range/total<=offset)
			throw std::logic_error("Assertion error");
		update(freqs, symbol);
		if (code<low||code>high)
			throw logic_error("Assertion error: Code out of range");
		return symbol;
	}
	inline uint golomb()
	{
		int temp=readCodeBit(),top=0;
		while(!temp)temp=readCodeBit(),top++;
		uint ret=temp;
		re(i,1,top)ret<<=1,ret|=readCodeBit();
		return ret-1;
	}
	Decoder(int numBits,InputStream &in,bool initialed=1):input(in)
	{
		if (numBits<1||numBits>63)
			throw std::domain_error("State size out of range");
		numStateBits=numBits;
		fullRange=static_cast<decltype(fullRange)>(1)<<numStateBits;
		halfRange=fullRange>>1;  // Non-zero
		quarterRange=halfRange>>1;  // Can be zero
		minimumRange=quarterRange+2;  // At least 2
		maximumTotal=min(numeric_limits<decltype(fullRange)>::max()/fullRange,minimumRange);
		stateMask=fullRange-1;
		low=0;
		high=stateMask;
		code=0;
		if(initialed)
			re(i,0,numStateBits-1)
				code=code<<1|readCodeBit();
	}
};
extern "C"
{
/*
file_dir: char*, file_dir of input file
offset: int, sum of **bytes** needed to be ignore(shape len, shape, min, table_len)
p_table: int**, tables of data
table_sum,tabel_len = p_table.shape
a: int*, spaces used to save data decompressed
index: int*,  indexes of p_table of each number in a
len: int, length of a and index
adaptive: bool, if use adaptive decompress
*/
int decompress(const char *file_dir,int offset,int *p_tabel,int table_sum,int table_len,int *a,int *index,int len,bool adaptive)
{
	ifstream in(file_dir, ios::binary);
	ifstream gin(string("golomb")+file_dir,ios::binary);
	char _;while(offset--)in.read(&_,1);
	InputStream bin(in);
	InputStream gbin(gin);
	int tot=0;
	try {
		std::vector<Table_> tabels;
		for(int i=0;i<table_sum;i++)
		{
			vector<uint> freq_vec;
            for(int  j = 0; j < table_len; ++j) {
                freq_vec.push_back((uint)p_tabel[i*table_len + j]);
            }
			Table_ freqs(freq_vec);
			tabels.push_back(freqs);
		}
		Decoder dec(32,bin);
		Decoder gdec(32,gbin,0);
		while(tot<=len)
		{
			uint symbol;
			if(tot<len&&index[tot]==-1)
			{
				symbol=gdec.golomb();
				a[tot++]=static_cast<int>(symbol);
				continue;
			}
			Table_ &freqs=tabels[tot==len?0:index[tot]];
			symbol = dec.read(freqs);
			if (symbol == table_len-1)break;
			else if(tot==len)throw std::logic_error("Assertion error: too long!");
			int b = static_cast<int>(symbol);
			/*if (std::numeric_limits<char>::is_signed)b -= (b >> 7) << 8;
			out.put(static_cast<char>(b));*/
			a[tot++]=b;
			if(adaptive)
				freqs.increment(symbol);
		}
		if(tot!=len)throw std::logic_error("Assertion error: too short!");
		return EXIT_SUCCESS;	
	}
	catch (const char *msg) 
	{
		std::cerr << msg << std::endl;
		return EXIT_FAILURE;
	}
}
	
}
/*
int a[3000000],in[3000000];
int main()
{
	int p[1000];
	re(i,0,999)p[i]=i+1;
	re(i,0,2999999)a[i]=i%99;
	re(i,0,2999999)in[i]=i%10-1;
	decompress("test.out",0,p,10,100,a,in,3000000,0);
	return 0;
}*/

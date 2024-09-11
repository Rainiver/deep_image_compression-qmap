#include<bits/stdc++.h>

#define rre(i, r, l) for(int i=(r);i>=(l);i--)
#define re(i, l, r) for(int i=(l);i<=(r);i++)
#define Clear(a, b) memset(a,b,sizeof(a))
typedef unsigned long long ULL;
typedef uint32_t uint;
typedef long long LL;
using namespace std;

class OutputStream {
public:
    ostream &output;
    int currentByte;
    int numBitsFilled;

    OutputStream(ostream &out) : output(out) {
        currentByte = 0;
        numBitsFilled = 0;
    }

    inline void write(int b) {
        if (b != 0 && b != 1)
            throw std::domain_error("Argument must be 0 or 1");
        currentByte = (currentByte << 1) | b;
        numBitsFilled++;
        if (numBitsFilled == 8) {
            // Note: ostream.put() takes char, which may be signed/unsigned
            if (std::numeric_limits<char>::is_signed)
                currentByte -= (currentByte >> 7) << 8;
            output.put(static_cast<char>(currentByte));
            currentByte = 0;
            numBitsFilled = 0;
        }
    }

    inline void finish() {
        while (numBitsFilled != 0)
            write(0);
    }
};

class Table {
public:
    vector <uint> frequencies;
    mutable vector <uint> cumulative;
    uint total;

    inline uint checkedAdd(uint x, uint y) const {
        if (x > UINT32_MAX - y)
            throw overflow_error("Arithmetic overflow");
        return x + y;
    }

    inline void initCumulative(bool checkTotal = 1) const {
        if (!cumulative.empty())
            return;
        uint sum = 0;
        cumulative.push_back(sum);
        for (uint freq:frequencies) {
            // This arithmetic should not throw an exception, because invariants are being maintained
            // elsewhere in the data structure. This implementation is just a defensive measure.
            sum = checkedAdd(freq, sum);
            cumulative.push_back(sum);
        }
        if (checkTotal && sum != total)
            throw logic_error("Assertion error");
    }

    inline uint getHigh(uint symbol) const {
        initCumulative();
        return cumulative.at(symbol + 1);
    }

    inline uint getLow(uint symbol) const {
        initCumulative();
        return cumulative.at(symbol);
    }

    inline uint getTotal() const {
        return total;
    }

    inline void increment(uint symbol) {
        if (frequencies.at(symbol) == UINT32_MAX)
            throw overflow_error("Arithmetic overflow");
        total = checkedAdd(total, 1);
        frequencies.at(symbol)++;
        cumulative.clear();
    }

    Table(const vector <uint> &freqs) {
        if (freqs.size() > UINT32_MAX - 1)
            throw length_error("Too many symbols");
        uint size = static_cast<uint>(freqs.size());
        if (size < 1)
            throw std::invalid_argument("At least 1 symbol needed");

        frequencies = freqs;
        cumulative.reserve(size + 1);
        initCumulative(false);
        total = getHigh(size - 1);
    }
};

class Encoder {
public:
    int numStateBits;
    ULL fullRange, halfRange, quarterRange, minimumRange, maximumTotal, stateMask;
    ULL low, high;
    int sta[66], top;

    OutputStream &output;
    uint numUnderflow;

    Encoder(int numBits, OutputStream &out) : output(out) {
        if (numBits < 1 || numBits > 63)
            throw std::domain_error("State size out of range");
        numStateBits = numBits;
        fullRange = static_cast<decltype(fullRange)>(1) << numStateBits;
        halfRange = fullRange >> 1;  // Non-zero
        quarterRange = halfRange >> 1;  // Can be zero
        minimumRange = quarterRange + 2;  // At least 2
        maximumTotal = min(numeric_limits<decltype(fullRange)>::max() / fullRange, minimumRange);
        stateMask = fullRange - 1;
        low = 0;
        high = stateMask;

        numUnderflow = 0;
    }

    inline void shift() {
        int bit = static_cast<int>(low >> (numStateBits - 1));
        output.write(bit);

        // Write out the saved underflow bits
        for (; numUnderflow > 0; numUnderflow--)
            output.write(bit ^ 1);
    }

    inline void underflow() {
        if (numUnderflow == numeric_limits<decltype(numUnderflow)>::max())
            throw overflow_error("Maximum underflow reached");
        numUnderflow++;
    }

    inline void write(const Table &freqs, uint symbol) {
        if (low >= high || (low & stateMask) != low || (high & stateMask) != high)
            throw logic_error("Assertion error: Low or high out of range");
        ULL range = high - low + 1;
        if (range < minimumRange || range > fullRange)
            throw logic_error("Assertion error: Range out of range");

        // Frequency table values check
        uint total = freqs.getTotal();
        uint symLow = freqs.getLow(symbol);
        uint symHigh = freqs.getHigh(symbol);
        if (symLow == symHigh)
            throw invalid_argument("Symbol has zero frequency");
        if (total > maximumTotal)
            throw invalid_argument("Cannot code symbol because total is too large");

        // Update range
        ULL newLow = low + symLow * range / total;
        ULL newHigh = low + symHigh * range / total - 1;
        low = newLow;
        high = newHigh;

        // While low and high have the same top bit value, shift them out
        while (((low ^ high) & halfRange) == 0) {
            shift();
            low = ((low << 1) & stateMask);
            high = ((high << 1) & stateMask) | 1;
        }
        // Now low's top bit must be 0 and high's top bit must be 1

        // While low's top two bits are 01 and high's are 10, delete the second highest bit of both
        while ((low & ~high & quarterRange) != 0) {
            underflow();
            low = (low << 1) ^ halfRange;
            high = ((high ^ halfRange) << 1) | halfRange | 1;
        }
    }

    inline void golomb(uint symbol) {
        symbol += 1, top = 0;
        while (symbol)sta[++top] = (symbol & 1), symbol >>= 1;
        re(i, 2, top)output.write(0);
        rre(i, top, 1)output.write(sta[i]);
    }

    inline void finish() {
        output.write(1);
    }
};
/*
file_dir: char*, file_dir of output file
p_table: int**, tables of data
table_sum,tabel_len = p_table.shape
a: int*, data to be compressed
index: int*,  indexes of p_table of each number in a
len: int, length of a and index
adaptive: bool, if use adaptive compress
*/
extern "C"
{

int compress(const char *file_dir,
             int *p_tabel, int table_sum, int table_len,
             int *a, int *index, int len,
             bool adaptive,
             bool append) {
    // printf("cc compressing! adaptive=%d\n", (int) adaptive);
    ofstream file;
    file.open(file_dir, (append ? std::ios::app : std::ios::binary) | std::ios::binary);
    ofstream gfile;
    gfile.open(string("golomb") + file_dir, (append ? std::ios::app : std::ios::binary) | std::ios::binary);
    OutputStream bout(file);
    OutputStream gbout(gfile);
    try {
        vector <Table> tabels;
        for (int i = 0; i < table_sum; i++) {
            vector <uint> freq_vec;
            for (int j = 0; j < table_len; ++j) {
                freq_vec.push_back((uint) p_tabel[i * table_len + j]);
            }
            Table freqs(freq_vec);
            tabels.push_back(freqs);
        }
        Encoder enc(32, bout);
        Encoder genc(32, gbout);
        for (int i = 0; i < len; i++) {
            int symbol = a[i];
            if (index[i] == -1) {
                genc.golomb(static_cast<uint>(symbol));
                continue;
            }
            Table &freqs = tabels[index[i]];
            if (symbol < 0 || symbol > table_len - 2) {
                cout << symbol << "\n";
                throw std::logic_error("Assertion error");
            }
            enc.write(freqs, static_cast<uint>(symbol));
            if (adaptive)
                freqs.increment(static_cast<uint>(symbol));
        }
        enc.write(tabels[0], table_len - 1);
        enc.finish();
        bout.finish();
        return EXIT_SUCCESS;
    }
    catch (const char *msg) {
        cerr << msg << "\n";
        return EXIT_FAILURE;
    }
}

vector<uint> make_t(
        vector <vector<uint>> &a,
        vector<uint> &index,
        vector<uint> &omega_t
) {
    vector<uint> c(a[0].size());
    for (auto idx : index) {
        uint omega = idx % omega_t.size();
        uint mu_sigma = idx / omega_t.size();
        for (int i = 0; i < a[0].size(); i++) {
            c[i] += a[mu_sigma][i] * omega_t[omega];
        }
    }
    return c;
}

int compress_gmm(
        const char *file_dir,
        int *p_tabel, int table_sum, int table_len,
        int *omega_t, int omega_l,
        int *a, int len,
        int *index, int k,
        bool adaptive, bool append) {
//    printf("cc gmm compressing! adaptive=%d\n", (int) adaptive);
    ofstream file;
    file.open(file_dir, (append ? std::ios::app : std::ios::binary) | std::ios::binary);
    ofstream gfile;
    gfile.open(string("golomb") + file_dir, (append ? std::ios::app : std::ios::binary) | std::ios::binary);
    OutputStream bout(file);
    OutputStream gbout(gfile);
    try {
        vector <uint> omega_vec;
        for (int i = 0; i < omega_l; i++) {
            omega_vec.push_back((uint) omega_t[i]);
        }

        vector <vector<uint>> tabels;
        for (int i = 0; i < table_sum; i++) {
            vector <uint> freq_vec;
            for (int j = 0; j < table_len; ++j) {
                freq_vec.push_back((uint) p_tabel[i * table_len + j]);
            }
            tabels.push_back(freq_vec);
        }

        Encoder enc(32, bout);
        Encoder genc(32, gbout);
        for (int i = 0; i < len; i++) {
            int symbol = a[i];

            bool f = false;
            vector <uint> index_i;
            for (int j = 0; j < k; j++) {
                int tmp = (uint) index[len * j + i];
                if (tmp == -1) {
                    f = true;
                    break;
                }
                index_i.push_back(tmp);
            }

            if (f) {
                genc.golomb(static_cast<uint>(symbol));
                continue;
            }

            Table freqs(make_t(tabels, index_i, omega_vec));
            if (symbol < 0 || symbol > table_len - 2) {
                cout << symbol << "\n";
                throw std::logic_error("Assertion error");
            }
            enc.write(freqs, static_cast<uint>(symbol));
            if (adaptive)
                freqs.increment(static_cast<uint>(symbol));
        }
        enc.write(tabels[0], table_len - 1);
        enc.finish();
        bout.finish();
        return EXIT_SUCCESS;
    }
    catch (const char *msg) {
        cerr << msg << "\n";
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
	compress("test.out",p,10,100,a,in,3000000,0,0);
	return 0;
}*/

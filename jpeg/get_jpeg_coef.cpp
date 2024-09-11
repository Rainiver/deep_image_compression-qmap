#include <algorithm>
#include <array>
#include <ctime>
#include <iostream>
#include <numeric>
#include <optional>
#include <vector>
#include <sstream>

#include <jpeglib.h>

typedef std::vector<std::array<JCOEF, 2>> dim_t;
typedef std::array<JCOEF, DCTSIZE2> dct_t;
typedef std::vector<dct_t> quant_t;
typedef std::vector<std::vector<dct_t>> coeff_t;

struct jpeg_data
{
    dim_t dimensions;
    quant_t quantization;
    coeff_t Y_coefficients;
    std::optional<std::array<coeff_t, 2>> CrCb_coefficients;
};

class libjpeg_exception : public std::exception
{
private:
    char *error;

public:
    libjpeg_exception(j_common_ptr cinfo)
    {
        error = new char[JMSG_LENGTH_MAX];
        (cinfo->err->format_message)(cinfo, error);
    }

    virtual const char *what() const throw()
    {
        return error;
    }
};

void raise_libjpeg(j_common_ptr cinfo)
{
    throw libjpeg_exception(cinfo);
}

long jdiv_round_up(long a, long b)
/* Compute a/b rounded up to next integer, ie, ceil(a/b) */
/* Assumes a >= 0, b > 0 */
{
    return (a + b - 1L) / b;
}

void extract_channel(const jpeg_decompress_struct &srcinfo,
                     jvirt_barray_ptr *src_coef_arrays,
                     int compNum,
                     coeff_t &coefficients,
                     quant_t &quantization)
{
    for (JDIMENSION rowNum = 0; rowNum < srcinfo.comp_info[compNum].height_in_blocks; rowNum++)
    {
        JBLOCKARRAY rowPtrs = srcinfo.mem->access_virt_barray((j_common_ptr)&srcinfo, src_coef_arrays[compNum],
                                                              rowNum, 1, FALSE);

        auto width = srcinfo.comp_info[compNum].width_in_blocks;
        std::vector<dct_t> row(width);
        for (JDIMENSION blockNum = 0; blockNum < width; blockNum++)
        {
            std::copy_n(rowPtrs[0][blockNum], DCTSIZE2, row[blockNum].begin());
        }
        coefficients.emplace_back(std::move(row));
    }

    quantization.emplace_back(dct_t());
    std::copy_n(srcinfo.comp_info[compNum].quant_table->quantval, DCTSIZE2,
                quantization.back().begin());
}

jpeg_data read_coefficients_using(jpeg_decompress_struct &srcinfo)
{
    jpeg_read_header(&srcinfo, TRUE);

    // channels x 2
    dim_t dimensions(srcinfo.num_components);
    for (auto i = 0; i < srcinfo.num_components; i++)
    {
        dimensions[i][0] = srcinfo.comp_info[i].downsampled_height;
        dimensions[i][1] = srcinfo.comp_info[i].downsampled_width;
    }

    // read coefficients
    jvirt_barray_ptr *src_coef_arrays = jpeg_read_coefficients(&srcinfo);

    coeff_t Y_coefficients;
    quant_t quantization;

    // extract Y channel
    extract_channel(srcinfo, src_coef_arrays, 0, Y_coefficients, quantization);

    // extract CrCb channels
    std::optional<std::array<coeff_t, 2>> CrCb_coefficients;

    if (srcinfo.num_components > 1)
    {
        CrCb_coefficients.emplace(std::array<coeff_t, 2>());
        extract_channel(srcinfo, src_coef_arrays, 1, (*CrCb_coefficients)[0], quantization);
        extract_channel(srcinfo, src_coef_arrays, 2, (*CrCb_coefficients)[1], quantization);
    }

    // cleanup
    jpeg_finish_decompress(&srcinfo);

    return {
        std::move(dimensions),
        std::move(quantization),
        std::move(Y_coefficients),
        std::move(CrCb_coefficients)};
}

jpeg_data read_coefficients(const std::string &path)
{
    // open the file
    FILE *infile;
    if ((infile = fopen(path.c_str(), "rb")) == nullptr)
    {
        std::ostringstream ss;
        ss << "Unable to open file for reading: " << path;
        throw std::runtime_error(ss.str());
    }

    // start decompression
    jpeg_decompress_struct cinfo{};

    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jerr.error_exit = raise_libjpeg;

    jpeg_create_decompress(&cinfo);

    jpeg_stdio_src(&cinfo, infile);

    auto ret = read_coefficients_using(cinfo);

    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    return ret;
}

void set_quantization(j_compress_ptr cinfo, const quant_t &quantization)
{
    int num_components = quantization.size();
    std::copy_n(quantization[0].begin(), DCTSIZE2, cinfo->quant_tbl_ptrs[0]->quantval);

    if (num_components > 1)
    {
        std::copy_n(quantization[1].begin(), DCTSIZE2, cinfo->quant_tbl_ptrs[1]->quantval);
    }
}

jvirt_barray_ptr *request_block_storage(j_compress_ptr cinfo)
{
    auto block_arrays = (jvirt_barray_ptr *)(*cinfo->mem->alloc_small)((j_common_ptr)cinfo,
                                                                       JPOOL_IMAGE,
                                                                       sizeof(jvirt_barray_ptr *) *
                                                                           cinfo->num_components);

    std::transform(cinfo->comp_info, cinfo->comp_info + cinfo->num_components, block_arrays,
                   [&](jpeg_component_info &compptr)
                   {
                       int MCU_width = jdiv_round_up((long)cinfo->jpeg_width, (long)compptr.MCU_width);
                       int MCU_height = jdiv_round_up((long)cinfo->jpeg_height, (long)compptr.MCU_height);

                       return (cinfo->mem->request_virt_barray)((j_common_ptr)cinfo,
                                                                JPOOL_IMAGE,
                                                                TRUE,
                                                                MCU_width,
                                                                MCU_height,
                                                                compptr.v_samp_factor);
                   });

    return block_arrays;
}

void fill_extended_defaults(j_compress_ptr cinfo, int color_samp_factor = 2)
{

    cinfo->jpeg_width = cinfo->image_width;
    cinfo->jpeg_height = cinfo->image_height;

    jpeg_set_defaults(cinfo);

    cinfo->comp_info[0].component_id = 0;
    cinfo->comp_info[0].h_samp_factor = 1;
    cinfo->comp_info[0].v_samp_factor = 1;
    cinfo->comp_info[0].quant_tbl_no = 0;
    cinfo->comp_info[0].width_in_blocks = jdiv_round_up(cinfo->jpeg_width, DCTSIZE);
    cinfo->comp_info[0].height_in_blocks = jdiv_round_up(cinfo->jpeg_height, DCTSIZE);
    cinfo->comp_info[0].MCU_width = 1;
    cinfo->comp_info[0].MCU_height = 1;

    if (cinfo->num_components > 1)
    {
        cinfo->comp_info[0].h_samp_factor = color_samp_factor;
        cinfo->comp_info[0].v_samp_factor = color_samp_factor;
        cinfo->comp_info[0].MCU_width = color_samp_factor;
        cinfo->comp_info[0].MCU_height = color_samp_factor;

        for (int c = 1; c < cinfo->num_components; c++)
        {
            cinfo->comp_info[c].component_id = c;
            cinfo->comp_info[c].h_samp_factor = 1;
            cinfo->comp_info[c].v_samp_factor = 1;
            cinfo->comp_info[c].quant_tbl_no = 1;
            cinfo->comp_info[c].width_in_blocks = jdiv_round_up(cinfo->jpeg_width, DCTSIZE * color_samp_factor);
            cinfo->comp_info[c].height_in_blocks = jdiv_round_up(cinfo->jpeg_width, DCTSIZE * color_samp_factor);
            cinfo->comp_info[c].MCU_width = 1;
            cinfo->comp_info[c].MCU_height = 1;
        }
    }

    cinfo->min_DCT_h_scaled_size = DCTSIZE;
    cinfo->min_DCT_v_scaled_size = DCTSIZE;
}

void set_channel(const jpeg_compress_struct &cinfo,
                 const coeff_t &coefficients,
                 jvirt_barray_ptr *dest_coef_arrays,
                 int compNum)
{
    for (JDIMENSION rowNum = 0; rowNum < cinfo.comp_info[compNum].height_in_blocks; rowNum++)
    {
        JBLOCKARRAY rowPtrs = cinfo.mem->access_virt_barray((j_common_ptr)&cinfo, dest_coef_arrays[compNum],
                                                            rowNum, 1, TRUE);

        for (JDIMENSION blockNum = 0; blockNum < cinfo.comp_info[compNum].width_in_blocks; blockNum++)
        {
            std::copy_n(coefficients[rowNum][blockNum].begin(), DCTSIZE2, rowPtrs[0][blockNum]);
        }
    }
}

void write_coefficients(const std::string &path, const jpeg_data &data)
{
    FILE *outfile;
    if ((outfile = fopen(path.c_str(), "wb")) == nullptr)
    {
        std::ostringstream ss;
        ss << "Unable to open file for reading: " << path;
        throw std::runtime_error(ss.str());
    }

    jpeg_compress_struct cinfo{};

    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jerr.error_exit = raise_libjpeg;

    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_height = data.dimensions[0][0];
    cinfo.image_width = data.dimensions[0][1];
    cinfo.input_components = data.CrCb_coefficients ? 3 : 1;
    cinfo.in_color_space = data.CrCb_coefficients ? JCS_RGB : JCS_GRAYSCALE;

    fill_extended_defaults(&cinfo);

    set_quantization(&cinfo, data.quantization);

    jvirt_barray_ptr *coef_dest = request_block_storage(&cinfo);
    jpeg_write_coefficients(&cinfo, coef_dest);

    set_channel(cinfo, data.Y_coefficients, coef_dest, 0);

    if (data.CrCb_coefficients)
    {
        set_channel(cinfo, data.CrCb_coefficients.value()[0], coef_dest, 1);
        set_channel(cinfo, data.CrCb_coefficients.value()[1], coef_dest, 2);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    fclose(outfile);
}

int main(int argc, char *argv[])
{
    if (argc == 3)
    {
        clock_t start = clock();
        auto data = read_coefficients(argv[1]);
        clock_t read = clock();
        write_coefficients(argv[2], data);
        clock_t write = clock();
        std::cout << "Read  time:" << double(read - start) / CLOCKS_PER_SEC << "s" << std::endl;
        std::cout << "Write time:" << double(write - read) / CLOCKS_PER_SEC << "s" << std::endl;
        std::cout << "Total time:" << double(write - start) / CLOCKS_PER_SEC << "s" << std::endl;
        return 0;
    }
    std::cout << "Please input \"" << argv[0] << " input.jpg output.jpg\"" << std::endl;
    return 1;
}

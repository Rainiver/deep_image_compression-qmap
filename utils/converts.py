import codecs
import os
import re
import shutil
import tempfile
import time

import cv2
from PIL import Image

from utils.execute import execute
from utils.timer import Timer


def bpp_capture(output):
    _result = re.search(r"(\d+(\.?\d*|)) bpp", output, re.MULTILINE)
    return float(_result.group(1))


def bpp_calculate(src, dst):
    pixel_num = cv2.imread(src).size / 3.0
    return os.path.getsize(dst) * 8.0 / pixel_num


class _VerboseModule:
    def __init__(self, verbose=False):
        self.verbose = verbose


class ImageTransform:
    def __call__(self, source, target):
        raise NotImplementedError

    def __add__(self, other):
        _self_func = self.__dump_to_func(self)
        _other_func = self.__dump_to_func(other)

        def _new_func(source, target):
            with tempfile.NamedTemporaryFile() as temp:
                _self_ret = _self_func(source, temp.name + ".jpg")
                _other_ret = _other_func(temp.name + ".jpg", target)

            return _self_ret, _other_ret

        return self.__load_from_func(_new_func)

    def __radd__(self, other):
        if isinstance(other, ImageTransform):
            return other + self
        elif hasattr(other, "__call__"):
            return self.__load_from_func(other) + self
        else:
            raise TypeError("Not a ImageTransform at left value.")

    @classmethod
    def __dump_to_func(cls, func):
        if isinstance(func, cls):
            def _func(source, target):
                return func(source, target)

            return _func
        elif hasattr(func, "__call__"):
            return func
        else:
            return "func not callable, cannot dump to func."

    @classmethod
    def __load_from_func(cls, func):
        if isinstance(func, cls):
            return func
        elif hasattr(func, "__call__"):
            class _Transform(ImageTransform):
                def __call__(self, source, target):
                    return func(source, target)

            return _Transform()
        else:
            raise TypeError("func not callable, cannot be loaded.")


class ImageEncoder(ImageTransform, _VerboseModule):
    def __init__(self, verbose=False, **kwargs):
        _VerboseModule.__init__(self, verbose)
        self._timer = Timer()

    def _before_encode(self, source, target):
        pass

    def _encode(self, source, target):
        raise NotImplementedError

    def _after_encode(self, source, target, ret):
        return ret

    def encode(self, source, target=None):
        source = os.path.abspath(source)
        target = os.path.abspath(target)
        if self.verbose:
            print("Encoding from {src} to {dst} ...".format(src=repr(source), dst=repr(target)))

        key = "{cls}_{id}_encode_{time}".format(
            cls=self.__class__.__name__, id=id(self),
            time=int(time.time() * 1000.0),
        )

        self._before_encode(source, target)
        with self._timer.with_time(key):
            _return = self._encode(source, target)
        _return = self._after_encode(source, target, _return)

        _duration = self._timer[key]
        if self.verbose:
            print("Encode complete! {duration} second(s) cost.".format(duration=_duration))
        return target, _return, _duration

    def __call__(self, source, target=None):
        return self.encode(source, target)


class ImageDecoder(ImageTransform, _VerboseModule):
    def __init__(self, verbose=False, **kwargs):
        _VerboseModule.__init__(self, verbose)
        self._timer = Timer()

    def _before_decode(self, source, target):
        pass

    def _decode(self, source, target):
        raise NotImplementedError

    def _after_decode(self, source, target, ret):
        return ret

    def decode(self, source, target=None):
        source = os.path.abspath(source)
        target = os.path.abspath(target)
        if self.verbose:
            print("Decoding from {src} to {dst} ...".format(src=repr(source), dst=repr(target)))

        key = "{cls}_{id}_decode_{time}".format(
            cls=self.__class__.__name__, id=id(self),
            time=int(time.time() * 1000.0),
        )

        self._before_decode(source, target)
        with self._timer.with_time(key):
            _return = self._decode(source, target)
        _return = self._after_decode(source, target, _return)

        _duration = self._timer[key]
        if self.verbose:
            print("Decode complete! {duration} second(s) cost.".format(duration=_duration))
        return target, _return, _duration

    def __call__(self, source, target=None):
        return self.decode(source, target)


class ImageProcessor(ImageTransform, _VerboseModule):
    def __init__(self, verbose=False, **kwargs):
        _VerboseModule.__init__(self, verbose)
        self._timer = Timer()

    def _before_process(self, source, target):
        pass

    def _process(self, source, target):
        raise NotImplementedError

    def _after_process(self, source, target, ret):
        return ret

    def process(self, source, target=None):
        source = os.path.abspath(source)
        target = os.path.abspath(target)
        if self.verbose:
            print("Processing from {src} to {dst} ...".format(src=repr(source), dst=repr(target)))

        key = "{cls}_{id}_process_{time}".format(
            cls=self.__class__.__name__, id=id(self),
            time=int(time.time() * 1000.0),
        )

        self._before_process(source, target)
        with self._timer.with_time(key):
            _return = self._process(source, target)
        _return = self._after_process(source, target, _return)

        _duration = self._timer[key]
        if self.verbose:
            print("Process complete! {duration} second(s) cost.".format(duration=_duration))
        return target, _return, _duration

    def __call__(self, source, target=None):
        return self.process(source, target)


class ImageCommandLineEncoder(ImageEncoder):
    def _get_command_line(self, source, target):
        raise NotImplementedError

    def _encode(self, source, target):
        _command = self._get_command_line(source, target)
        if self.verbose:
            print("Command line :", _command)
        if isinstance(_command, list):
            return [execute(*_cmd) for _cmd in _command]
        return execute(*_command)

    def _after_encode(self, source, target, ret):
        _exitcode, _, _stderr = list(zip(*ret)) if isinstance(ret, list) else ret
        if self.verbose:
            print("Encode exitcode : {}, with stderr ({} chars) : {}{}".format(
                _exitcode, len(_stderr), os.linesep, _stderr
            ))
        assert sum(_exitcode) == 0, _exitcode
        _bpp = bpp_calculate(source, target)
        print('_bpp', _bpp)
        return _stderr, _bpp


class ImageCommandLineDecoder(ImageDecoder):
    def _get_command_line(self, source, target):
        raise NotImplementedError

    def _decode(self, source, target):
        _command = self._get_command_line(source, target)
        if self.verbose:
            print("Command line :", _command)
        if isinstance(_command, list):
            return [execute(*_cmd) for _cmd in _command]
        return execute(*_command)

    def _after_decode(self, source, target, ret):
        _exitcode, _, _stderr = list(zip(*ret)) if isinstance(ret, list) else ret
        if self.verbose:
            print("Decode exitcode : {}, with stderr ({} chars) : {}{}".format(
                _exitcode, len(_stderr), os.linesep, _stderr
            ))
        assert sum(_exitcode) == 0, _exitcode
        return _stderr


class ImageCommandLineProcessor(ImageProcessor):
    def _get_command_line(self, source, target):
        raise NotImplementedError

    def _process(self, source, target):
        _command = self._get_command_line(source, target)
        if self.verbose:
            print("Command line :", _command)
        return execute(*_command)

    def _after_process(self, source, target, ret):
        _exitcode, _, _stderr = ret
        if self.verbose:
            print("Process exitcode : {}, with stderr ({} chars) : {}{}".format(
                _exitcode, len(_stderr), os.linesep, _stderr
            ))
        assert _exitcode == 0, _exitcode
        return _stderr


class WebpEncoder(ImageCommandLineEncoder):
    def __init__(self, quality, verbose=False, **kwargs):
        ImageCommandLineEncoder.__init__(self, verbose, **kwargs)
        self.quality = quality

    def _get_command_line(self, source, target):
        return (
            "/mnt/lustre/share/fe/wangyan1/codec/libwebp-1.0.3-linux-x86-64/bin/cwebp",
            "-q", self.quality, source, "-o", target,
        )


class WebpDecoder(ImageCommandLineDecoder):
    def __init__(self, verbose=False, **kwargs):
        ImageCommandLineDecoder.__init__(self, verbose, **kwargs)

    def _get_command_line(self, source, target):
        return (
            "/mnt/lustre/share/fe/wangyan1/codec/libwebp-1.0.3-linux-x86-64/bin/dwebp",
            source, "-o", target,
        )


class BpgEncoder(ImageCommandLineEncoder):
    __DEFAULT_CFMT = 444

    def __init__(self, quality, cfmt=__DEFAULT_CFMT, verbose=False, **kwargs):
        ImageCommandLineEncoder.__init__(self, verbose, **kwargs)
        self.quality = quality
        self.cfmt = cfmt

    def _get_command_line(self, source, target):
        return (
            "/mnt/lustre/share/fe/wangyan1/codec/libbpg-0.9.8/bpgenc",
            "-f", self.cfmt, "-q", self.quality, source, "-o", target, "-v",
        )


class BpgDecoder(ImageCommandLineDecoder):
    def __init__(self, verbose=False, **kwargs):
        ImageCommandLineDecoder.__init__(self, verbose, **kwargs)

    def _get_command_line(self, source, target):
        return (
            "/mnt/lustre/share/fe/wangyan1/codec/libbpg-0.9.8/bpgdec",
            source, "-o", target,
        )


class Jp2Encoder(ImageCommandLineEncoder):
    def __init__(self, verbose=False, **kwargs):
        ImageCommandLineEncoder.__init__(self, verbose, **kwargs)

    def _get_command_line(self, source, target):
        return (
            "/mnt/lustre/share/fe/wangyan1/codec/openjpeg/build/bin/opj_compress",
            "-i", source, "-o", target,
        )


class Jp2Decoder(ImageCommandLineDecoder):
    def __init__(self, verbose=False, **kwargs):
        ImageCommandLineDecoder.__init__(self, verbose, **kwargs)

    def _get_command_line(self, source, target):
        return (
            "/mnt/lustre/share/fe/wangyan1/codec/openjpeg/build/bin/opj_decompress",
            "-i", source, "-o", target,
        )


class HeifEncoder(ImageCommandLineEncoder):
    def __init__(self, quality, verbose=False, **kwargs):
        ImageCommandLineEncoder.__init__(self, verbose, **kwargs)
        self.quality = quality

    def _get_command_line(self, source, target):
        return (
            "/mnt/lustre/share/fe/wangliangzhou/codec/libheif/heif-enc",
            "-q", self.quality, source, "-o", target,
        )


class HeifDecoder(ImageCommandLineDecoder):
    def __init__(self, verbose=False, **kwargs):
        ImageCommandLineDecoder.__init__(self, verbose, **kwargs)

    def _get_command_line(self, source, target):
        return (
            "/mnt/lustre/share/fe/wangliangzhou/codec/libheif/heif-convert",
            source, target,
        )


class JpegEncoder(ImageEncoder):
    def __init__(self, quality, verbose=False, **kwargs):
        ImageEncoder.__init__(self, verbose, **kwargs)
        self.quality = quality

    def _encode(self, source, target):
        if self.verbose:
            print("PIL transform from {} to {}...".format(repr(source), repr(target)))
        with codecs.open(source, 'rb') as file:
            image = Image.open(file)
            image.save(target, quality=self.quality, format='jpeg')
        return None

    def _after_encode(self, source, target, ret):
        _bpp = bpp_calculate(source, target)
        return ret, _bpp


class JpegCopyDecoder(ImageDecoder):
    def __init__(self, verbose=False, **kwargs):
        ImageDecoder.__init__(self, verbose, **kwargs)

    def _decode(self, source, target):
        try:
            if self.verbose:
                print("Copying from {} to {}...".format(repr(source), repr(target)))
            shutil.copy2(source, target)
        except shutil.SameFileError:
            pass
        finally:
            return None


class LeptonProcessor(ImageCommandLineProcessor):
    def __init__(self, verbose=False, **kwargs):
        ImageCommandLineProcessor.__init__(self, verbose, **kwargs)

    def _get_command_line(self, source, target):
        return (
            "/mnt/lustre/share/fe/face_data/codec/compression_encode/lepton",
            source, target,
        )


class JpegLeptonEncoder(ImageEncoder):
    def __init__(self, quality, verbose=False, **kwargs):
        ImageEncoder.__init__(self, verbose, **kwargs)
        self.__encoder = JpegEncoder(quality, verbose, **kwargs) + LeptonProcessor(verbose, **kwargs)

    def _encode(self, source, target):
        self.__encoder(source, target)

    def _after_encode(self, source, target, ret):
        _bpp = bpp_calculate(source, target)
        return ret, _bpp


class JpegLeptonDecoder(ImageDecoder):
    def __init__(self, verbose=False, **kwargs):
        ImageDecoder.__init__(self, verbose, **kwargs)
        self.__decoder = LeptonProcessor(verbose, **kwargs) + JpegCopyDecoder(verbose, **kwargs)

    def _decode(self, source, target):
        self.__decoder(source, target)

class VtmEncoder(ImageCommandLineEncoder):
    _ffmpeg = "/mnt/lustre/share/wangyuan/ffmpeg_4.3.1/ffmpeg"
    _vtm_enc = "/mnt/lustre/share/yuhongjiu/tools/VVCSoftware_VTM/bin/EncoderAppStatic"
    _vtm_cfg = "/mnt/lustre/share/yuhongjiu/tools/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg"

    def __init__(self, quality, verbose=False, **kwargs):
        ImageCommandLineEncoder.__init__(self, verbose, **kwargs)
        self.quality = quality

    def _before_encode(self, source, target):
        obj = re.match(r'.*_(\d*)x(\d*)', target)
        self.h, self.w = obj.group(1), obj.group(2)

    def _get_command_line(self, source, target):
        return [(self._ffmpeg, "-y", "-i", source, "-pix_fmt", "yuv444p", target + '.yuv'),
                (self._vtm_enc, "-c", self._vtm_cfg, "-fr", 1, "-wdt", self.w, "-hgt", self.h, "-f", 1,
                "--InputChromaFormat=444", "--ConformanceWindowMode=1", "-i", target + '.yuv', "-q", self.quality, "-b", target),
                ("rm", "-f", target + '.yuv')]


class VtmDecoder(ImageCommandLineDecoder):
    _ffmpeg = "/mnt/lustre/share/wangyuan/ffmpeg_4.3.1/ffmpeg"
    _vtm_dec = "/mnt/lustre/share/yuhongjiu/tools/VVCSoftware_VTM/bin/DecoderAppStatic"

    def __init__(self, verbose=False, **kwargs):
        ImageCommandLineDecoder.__init__(self, verbose, **kwargs)

    def _before_decode(self, source, target):
        obj = re.match(r'.*_(\d*)x(\d*)', target)
        self.h, self.w = obj.group(1), obj.group(2)

    def _get_command_line(self, source, target):
        return [(self._vtm_dec, "-b", source, "-o", target + '.yuv'),
                (self._ffmpeg, "-pix_fmt", "yuv444p", "-s", f"{self.w}x{self.h}", "-i", target + '.yuv', target),
                ("rm", "-f", target + '.yuv')]


class JpegTurboEncoder(ImageCommandLineEncoder):
    def __init__(self, quality, verbose=False, **kwargs):
        ImageCommandLineEncoder.__init__(self, verbose, **kwargs)
        self.quality = quality

    def _get_command_line(self, source, target):
        return [(
            "/mnt/lustre/share/marui/mozjpeg-bin/cjpeg-static",
            "-outfile", target,
            "-q", self.quality,
            source,
        )]


class JpegTurboDecoder(ImageCommandLineDecoder):
    def __init__(self, verbose=False, **kwargs):
        ImageCommandLineDecoder.__init__(self, verbose, **kwargs)

    def _get_command_line(self, source, target):
        return [(
            "/mnt/lustre/share/marui/mozjpeg-bin/djpeg-static",
            "-outfile", target,
            "-bmp",
            source,
        )]

class JpegXlEncodeProcessor(ImageCommandLineProcessor):
    def __init__(self, quality, verbose=False, **kwargs):
        ImageCommandLineProcessor.__init__(self, verbose, **kwargs)

    def _get_command_line(self, source, target):
        return (
            "/mnt/lustre/share/marui/jpeg-xl-bin/cjxl",
            source, target,
            "-d", 0,
        )

class JpegXlEncoder(ImageCommandLineEncoder):
    def __init__(self, quality, verbose=False, **kwargs):
        ImageEncoder.__init__(self, verbose, **kwargs)
        self.__encoder = JpegEncoder(quality, verbose, **kwargs) + JpegXlEncodeProcessor(verbose, **kwargs)

    def _encode(self, source, target):
        self.__encoder(source, target)

    def _after_encode(self, source, target, ret):
        _bpp = bpp_calculate(source, target)
        return ret, _bpp

class JpegXlDecodeProcessor(ImageCommandLineProcessor):
    def __init__(self, verbose=False, **kwargs):
        ImageCommandLineProcessor.__init__(self, verbose, **kwargs)
    def _get_command_line(self, source, target):
        return (
            "/mnt/lustre/share/marui/jpeg-xl-bin/djxl",
            source, target,
        )

class JpegXlDecoder(ImageCommandLineDecoder):
    def __init__(self, verbose=False, **kwargs):
        ImageDecoder.__init__(self, verbose, **kwargs)
        self.__decoder = JpegXlDecodeProcessor(verbose, **kwargs) + JpegCopyDecoder(verbose, **kwargs)

    def _decode(self, source, target):
        self.__decoder(source, target)
        return ([0,0],"","")

class CmixEncodeProcessor(ImageCommandLineProcessor):
    def __init__(self, quality, verbose=False, **kwargs):
        ImageCommandLineProcessor.__init__(self, verbose, **kwargs)

    def _get_command_line(self, source, target):
        return (
            "/mnt/lustre/share/marui/cmix",
            "-c",
            source, target,
        )

class CmixEncoder(ImageCommandLineEncoder):
    def __init__(self, quality, verbose=False, **kwargs):
        ImageEncoder.__init__(self, verbose, **kwargs)
        self.__encoder = JpegEncoder(quality, verbose, **kwargs) + CmixEncodeProcessor(verbose, **kwargs)

    def _encode(self, source, target):
        self.__encoder(source, target)

    def _after_encode(self, source, target, ret):
        _bpp = bpp_calculate(source, target)
        return ret, _bpp

class CmixDecodeProcessor(ImageCommandLineProcessor):
    def __init__(self, verbose=False, **kwargs):
        ImageCommandLineProcessor.__init__(self, verbose, **kwargs)
    def _get_command_line(self, source, target):
        return (
            "/mnt/lustre/share/marui/cmix",
            "-d",
            source, target,
        )

class CmixDecoder(ImageCommandLineDecoder):
    def __init__(self, verbose=False, **kwargs):
        ImageDecoder.__init__(self, verbose, **kwargs)
        self.__decoder = CmixDecodeProcessor(verbose, **kwargs) + JpegCopyDecoder(verbose, **kwargs)

    def _decode(self, source, target):
        self.__decoder(source, target)
        return ([0,0],"","")
import os
import json
import numpy as np
import pandas as pd
from glob import glob


def parse_json_file(json_fn):
    with open(json_fn, "r") as f:
        profile_res = json.load(f)

    time_type = profile_res.get("displayTimeUnit", None)
    if time_type == "ns":
        factor = 1e9
    else:
        raise ValueError(f"Not supported time type: {time_type}")

    factor = 1e6

    trace_events = profile_res.get("traceEvents", None)
    if trace_events is None:
        raise ValueError("Empty trace events.")

    one_pass_key_list = [
        "difcoder.open",
        "difcoder.CreateNeuralNetwork",
        "difcoder.OpenProbTable",
        "difcoder.close"
    ]

    overall_codec_key = ["difcoder.process"]

    encode_key_list = [
        "difcoder.CompressShape",
        "difcoder.PrepareAndReshape",
        "difcoder.YEncoderForward",
        "difcoder.AbsAndCopy",
        "difcoder.ZEncoderForward",
        "difcoder.RoundAndCopy",
        "difcoder.ZDecoderForward",
        "difcoder.Compress",
        "difcoder.PrepareSigma",
        "difcoder.CompressY",
        "difcoder.CompressZ"
    ]

    decode_key_list = [
        "difcoder.DecompressShape",
        "difcoder.DecompressZ",
        "difcoder.PrepareZ",
        "difcoder.ForwardZDecoder",
        "difcoder.DecompressY",
        "difcoder.PrepareY",
        "difcoder.ForwardY",
        "difcoder.MergeBlock",
        "difcoder.SaveFrame"
    ]

    # one pass time statistics
    print("One pass time statistics ", "= " * 30)
    stats = []
    for _key in one_pass_key_list:
        _process_list = [p for p in trace_events if p["name"] == _key]
        _process_list = sorted(_process_list, key=lambda x: x["ts"])
        _begin_list = _process_list[::2]
        _end_list = _process_list[1::2]
        _avg_process_time = get_average_process_time(_begin_list, _end_list)
        stats.append(("one_pass", _key, _avg_process_time / factor))
        print(f"{_key} - {_avg_process_time / factor:.7f}    test {len(_begin_list)} times")

    # overall encode / decode statistics
    print("Overall codec time statistics ", "= " * 30)
    for _key in overall_codec_key:
        _process_list = [p for p in trace_events if p["name"] == _key]
        _process_list = sorted(_process_list, key=lambda x: x["ts"])
        _begin_list = _process_list[::2]
        _end_list = _process_list[1::2]
        # encode time
        _encode_begin_list = _begin_list[::2]
        _encode_end_list = _end_list[::2]
        _avg_process_time = get_average_process_time(_encode_begin_list, _encode_end_list)
        stats.append(("overall", "encode", _avg_process_time / factor))
        print(f"encode - {_avg_process_time / factor:.7f}    test {len(_encode_begin_list)} times")

        # decode time
        _decode_begin_list = _begin_list[1::2]
        _decode_end_list = _end_list[1::2]
        _avg_process_time = get_average_process_time(_decode_begin_list, _decode_end_list)
        stats.append(("overall", "decode", _avg_process_time / factor))
        print(f"decode - {_avg_process_time / factor:.7f}    test {len(_decode_begin_list)} times")

    # encode time statistics
    print("Encode time statistics ", "= " * 30)
    for _key in encode_key_list:
        _process_list = [p for p in trace_events if p["name"] == _key]
        _process_list = sorted(_process_list, key=lambda x: x["ts"])
        _begin_list = _process_list[::2]
        _end_list = _process_list[1::2]
        _avg_process_time = get_average_process_time(_begin_list, _end_list)
        stats.append(("encode", _key, _avg_process_time / factor))
        print(f"{_key} - {_avg_process_time / factor}    test {len(_begin_list)} times")

    # decode time statistics
    print("Decode time statistics ", "= " * 30)
    for _key in decode_key_list:
        _process_list = [p for p in trace_events if p["name"] == _key]
        _process_list = sorted(_process_list, key=lambda x: x["ts"])
        _begin_list = _process_list[::2]
        _end_list = _process_list[1::2]
        _avg_process_time = get_average_process_time(_begin_list, _end_list)
        stats.append(("decode", _key, _avg_process_time / factor))
        print(f"{_key} - {_avg_process_time / factor:.7f}    test {len(_begin_list)} times")
    return stats


def get_average_process_time(begin_list, end_list):
    res = []
    for _b, _e in zip(begin_list, end_list):
        _b_ts = int(_b["ts"])
        _e_ts = int(_e["ts"])
        # print(_e_ts, _b_ts)
        _wall_duration = _e_ts - _b_ts
        res.append(_wall_duration)
    return np.mean(res)


if __name__ == "__main__":
    # json_fn = "../jupyter/kestrel_visualize_0.json"
    # stats = parse_json_file(json_fn)
    # df = pd.DataFrame([stats])
    # print(df)

    kestrel_profiler_analyse_tool = \
        "/home/SENSETIME/wangyuanyuan/workspace/difcoder/deps/tools/kestrel_profiler_analyse_tool"
    tar_list = glob("../jupyter/adela_results/*/*/*.tar")
    print(tar_list)
    results = pd.DataFrame()
    for tar_fn in tar_list:
        print(tar_fn)
        _dir = os.path.dirname(tar_fn)
        _fn = os.path.basename(tar_fn)

        precision = tar_fn.split("/")[3]
        platform = tar_fn.split("/")[4]
        resolution = _fn[:-4]

        _cmd = f"cd {_dir} && " \
               f"mkdir -p {_fn[:-4]} && " \
               f"tar -xvf {_fn} -C {_fn[:-4]} && " \
               f"cd {resolution}/result/1_100 && " \
               f"{kestrel_profiler_analyse_tool} " \
               f"--visualize kestrel_tracing.dat " \
               f"--output kestrel_visualize"
        os.system(_cmd)

        json_fn = f"{_dir}/{resolution}/result/1_100/kestrel_visualize_0.json"
        if os.path.exists(json_fn):
            _stats = parse_json_file(json_fn)
            if results.columns.size == 0:
                results["process"] = [_s[:2] for _s in _stats]
            column_name = f"{resolution}_{platform}_{precision}"
            print(precision, platform, resolution, _stats)
            results[column_name] = [_s[2] for _s in _stats]
        else:
            raise ValueError(f"File does not exist: {json_fn}")
    print(results)
    results.to_csv("adela_results.csv", index=False)

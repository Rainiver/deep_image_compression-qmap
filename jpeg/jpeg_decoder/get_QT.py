from PIL import Image
import os
import subprocess
import numpy as np
from tqdm import tqdm

x = np.zeros([128, 128, 3])
x = Image.fromarray(np.uint8(x))
if os.path.exists("quant_table.txt"):
    os.remove("quant_table.txt")
if os.path.exists("decoder.exe"):
    exe = "decoder.exe"
elif os.path.exists("decoder"):
    exe = "./decoder"
else:
    print("please make first!")
    assert 0

# x = Image.open("origin.jpeg")
for i in range(1, 101):
    x.save("test" + str(i) + ".jpeg", quality=i)
for i in tqdm([j for j in range(1, 101)]):
    p = subprocess.Popen(exe + " test" + str(i) + ".jpeg", shell=True, stdout=subprocess.PIPE)
    p.wait()
for i in range(1, 101):
    os.remove("test" + str(i) + ".jpeg")

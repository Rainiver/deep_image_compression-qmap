import os

with open('/mnt/lustre/share/zhengyaoyan/dataset_choosen.txt') as f:
    line=f.readline()
    i=0
    while line:
        i+=1
        print(i,'/',10000)
        path='/mnt/lustre/share/images/test/'+line
        pp='cp '+path[0:-1]+' ./train/'
        print(pp)
        os.system(pp)
        line=f.readline()


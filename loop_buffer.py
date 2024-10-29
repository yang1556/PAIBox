import numpy as np

## fc: 4*4 -> 10
data_len=16
m=4
n=4
buffer = np.zeros((m, n))
data = [i+1 for i in range(data_len)]
tr = [0, 4, 8, 12]
twe = [16, 12, 8, 4]
end = 0
for i in range(len(data)):
    if i>0 and i%4==0:
        end=end+1
    for j in range(len(tr)-end):
        buffer[j][(tr[j]+i)%n]=data[i]
    print(buffer)
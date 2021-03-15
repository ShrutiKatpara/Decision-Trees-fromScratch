import matplotlib.pyplot as plt
import csv

fig = plt.figure(figsize=(8,24))

key = []
value = []

flag = 0 # 0: learning 1: predict

row_count = 0

with open("./dataset/riro.txt", "r") as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        row_count+=1
        if (row[0]=="learning"):
            continue
        elif (row[0]=="predict"):
            plt.subplot(4,2,1)
            plt.plot(key, value, label="learning (time in secs)")
            plt.xlabel("samples")
            plt.ylabel("RIRO: learning (time in secs)")
            key = []
            value = []
        else:
            key.append(float(row[0]))
            value.append(float(row[1]))

plt.subplot(4,2,2)
plt.plot(key,value, label="RIRO: predict (time in millisecs)")
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
plt.xlabel("samples")
plt.ylabel("RIDO: predict (time in millisecs)")
# plt.suptitle("Case: RIRO")


key = []
value = []

flag = 0 # 0: learning 1: predict

row_count = 0

with open("./dataset/rido.txt", "r") as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        row_count+=1
        if (row[0]=="learning"):
            continue
        elif (row[0]=="predict"):
            plt.subplot(4,2,3)
            plt.plot(key, value, label="learning (time in secs)")
            plt.xlabel("samples")
            plt.ylabel("RIDO: learning (time in secs)")
            key = []
            value = []
        else:
            key.append(float(row[0]))
            value.append(float(row[1]))

plt.subplot(4,2,4)
plt.plot(key,value, label="predict (time in millisecs)")
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
plt.xlabel("samples")
plt.ylabel("RIDO: predict (time in millisecs)")
# plt.suptitle("Case: RIDO")


key = []
value = []

flag = 0 # 0: learning 1: predict

row_count = 0

with open("./dataset/diro.txt", "r") as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        row_count+=1
        if (row[0]=="learning"):
            continue
        elif (row[0]=="predict"):
            plt.subplot(4,2,5)
            plt.plot(key, value, label="learning (time in secs)")
            plt.xlabel("samples")
            plt.ylabel("DIRO: learning (time in secs)")
            key = []
            value = []
        else:
            key.append(float(row[0]))
            value.append(float(row[1]))

plt.subplot(4,2,6)
plt.plot(key,value, label="predict (time in millisecs)")
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
plt.xlabel("samples")
plt.ylabel("DIRO: predict (time in millisecs)")
# plt.suptitle("Case: DIRO")


key = []
value = []

flag = 0 # 0: learning 1: predict

row_count = 0

with open("./dataset/dido.txt", "r") as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        row_count+=1
        if (row[0]=="learning"):
            continue
        elif (row[0]=="predict"):
            plt.subplot(4,2,7)
            plt.plot(key, value, label="learning (time in secs)")
            plt.xlabel("samples")
            plt.ylabel("DIDO: learning (time in secs)")
            key = []
            value = []
        else:
            key.append(float(row[0]))
            value.append(float(row[1]))

plt.subplot(4,2,8)
plt.plot(key,value, label="predict (time in millisecs)")
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
plt.xlabel("samples")
plt.ylabel("DIDO: predict (time in millisecs)")
# plt.suptitle("Case: DIDO")

plt.savefig("./figures/q4_new.png", dpi=400)
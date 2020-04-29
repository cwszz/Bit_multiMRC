from tqdm import tqdm 
loss = []
frequence = []
matrix =[]
cnt = []
epoch = 20
with open('true_train_w2_15.txt','r',encoding='utf-8',newline='\n') as t:
    temp_loss = 0
    for i in range(epoch):
        matrix.append(0)
        cnt.append(0)
        ttt = []
        for j in range(7):
            ttt.append(0)
        frequence.append(ttt)
    for line in tqdm(t.readlines()):
        # gggg = line[-3:1]
        num = line.split('--')[-1].strip()
        matrix[int(num)] += float(line.split('[')[1].split(']')[0])
        cnt [int(num)] += 1
        frequence[int(num)][int(float(line.split('[')[1].split(']')[0]))] += 1
    for i in range(epoch):
        if(cnt[i]!=0):
            loss.append({str(i):matrix[i]/cnt[i]})
    print(loss)
    for q in frequence:
        print(q)
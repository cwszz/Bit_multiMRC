from tqdm import tqdm 
loss = []
frequence = []
matrix =[]
cnt = []
epoch = 30
check = [4,5,6,7,8,9]
with open('Final_train_5.txt','r',encoding='utf-8',newline='\n') as t:
    temp_loss = 0
    for i in range(len(check)):
        matrix.append(0)
        cnt.append(0)
        ttt = []
        for j in range(7):
            ttt.append(0)
        frequence.append(ttt)
    for line in tqdm(t.readlines()):
        num = line.split('--')[-1].strip()
        if(int(num) in check):
            matrix[int(num)-check[0]] += float(line.split('[')[1].split(']')[0])
            cnt [int(num)-check[0]] += 1
            frequence[int(num)-check[0]][int(float(line.split('[')[1].split(']')[0]))] += 1
    for i in range(len(check)):
        if(cnt[i]!=0):
            loss.append({str(i):matrix[i]/cnt[i],'Numbers':cnt[i]})
    print(loss)
    for i,q in enumerate(frequence):
        # if(i in check):
        print(q)
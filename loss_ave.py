from tqdm import tqdm 
loss = []
matrix =[0,0,0,0]
cnt = [0,0,0,0]
with open('true_train.txt','r',encoding='utf-8',newline='\n') as t:
    temp_loss = 0
    for line in tqdm(t.readlines()):
        matrix[int(line[-2])] += float(line.split('[')[1].split(']')[0])
        cnt [int(line[-2])] += 1
    loss.append({'0':matrix[0]/cnt[0],'1':matrix[1]/cnt[1],'2':matrix[2]/cnt[2]})
    print(loss)
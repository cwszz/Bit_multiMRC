import json
from tqdm import tqdm
cnt =0
with open('data/preprocessed/devset/zhidao.dev.json','r',encoding='utf-8',newline='\n') as t:
    with open('data/preprocessed/test/dev_test2.json','a+',encoding='utf-8',newline='\n') as w:
        for line in tqdm(t.readlines(), desc="processing..."):
            sample = json.loads(line)
            # if 'answer_docs' not in sample or len(sample['answer_docs']) < 1:
            #     continue
            # if('match_scores' not in sample or len(sample['match_scores'])<1): 
            #     continue
            # elif(sample['match_scores'][0]<0.70):
            #     continue
            # else:
            cnt += 1
            if cnt > 200:
                break
            pre_sample = {}
            pre_sample = sample
            w.write(json.dumps(pre_sample,ensure_ascii=False)+'\n')
 
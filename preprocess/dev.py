import json
from tqdm import tqdm

with open('data/preprocessed/my_dev/dev.json','r',encoding='utf-8',newline='\n') as t:
    with open('data/preprocessed/my_dev/dev_ref.json','w',encoding='utf-8',newline='\n') as w:
        for line in tqdm(t.readlines(), desc="processing..."):
            sample = json.loads(line)
            # if 'answer_docs' not in sample or len(sample['answer_docs']) < 1:
            #     continue
            # if('match_scores' not in sample or len(sample['match_scores'])<1): 
            #     continue
            # elif(sample['match_scores'][0]<0.70):
            #     continue
            # else:
            pre_sample = {}
            pre_sample['question'] = sample['question']
            pre_sample['question_type'] = sample['question_type']
            pre_sample['question_id'] = sample['question_id']
            pre_sample['answers'] = sample['answer']
            pre_sample['source'] = 'search'
            w.write(json.dumps(pre_sample,ensure_ascii=False)+'\n')
 
#!/usr/bin/env python
# encoding: utf-8
"""
File Description: 抓取百度搜索的结果
Author: nghuyong
Mail: nghuyong@163.com
Created Time: 2019-09-26 16:56
"""
import json
import time
from multiprocessing import Pool
from pprint import pprint
from urllib.parse import quote

import jieba
import requests
from lxml import etree

headers = {
    "Cookie":"BIDUPSID=E8B8CF8D28F96EA14D978A3A0D26354E; PSTM=1573401411; BAIDUID=E8B8CF8D28F96EA18D0A6E2F63F659F4:FG=1; BD_UPN=123253; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; BDUSS=tzT2IyY2VwYkJPTDBpVzlBakRIdmtBdTdDZXBDRVVYNnJxaGNrZHhKbXV-fnhkRVFBQUFBJCQAAAAAAAAAAAEAAADCuxZOs-7O0srH1sfVzwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAK5y1V2uctVdME; pgv_pvi=9685763072; delPer=0; BD_CK_SAM=1; PSINO=1; H_PS_PSSID=1443_21107_30043_29567_29700_29220_26350; BD_HOME=1; COOKIE_SESSION=14512_0_8_4_21_18_0_2_8_4_1_0_0_0_8_0_1574328882_0_1574346897%7C9%2386586_3_1574249153%7C2; sugstore=1; H_PS_645EC=fc04lk6mNINJARd2o0r3bFl38HAuESMhcjuFeGCFg%2FER7bQscBq%2Bhyq2c4U",
    "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36"
    #"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36"
}

def get_raw_html(url, code='UTF-8'):
    head = {
        'Cookie':'BAIDUID=C65ABBB8CD05F990788B54482EE33F74:FG=1; BIDUPSID=C65ABBB8CD05F990788B54482EE33F74; PSTM=1570170444; BD_UPN=12314753; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; sugstore=1; H_PS_PSSID=1457_21099_30210_30071_26350; delPer=0; BD_CK_SAM=1; ZD_ENTRY=baidu; BD_HOME=0; rsv_jmp_slow=1575637504422; PSINO=1; H_PS_645EC=3a41aPX8pcqOJSbqZV7kM5KdBN6PdK%2F%2FzlMuhkb4Zd%2B5hEFhj2BHtTYr85Q; BDSVRTM=11',
        'User-Agent': "ozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36"
    }
    try:
        r = requests.get(url, headers=head)
        r.encoding = code
        html = r.text
    except BaseException:
        print("open error", url)
        return ""
    return html
def crawl_baidu_search(keyword, num=5):
    url = "https://www.baidu.com/s?wd=" + keyword
    #print("url:" + url)
    response = requests.get(url, headers=headers)
    content_tree = etree.HTML(response.text)
    # 取前三个搜索结果
    search_data = []
    search_results = content_tree.xpath('//div[@class="result c-container "]')[:10]
    for index, search_result in enumerate(search_results, 1):
        try:
            abstract = search_result.xpath('.//div[@class="c-abstract"]')[0]
            abstract = abstract.xpath('string(.)')
            source = search_result.xpath('.//a[@data-click]')[0]
            source_link = source.xpath('./@href')[0]
            title = source.xpath('string(.)')
            baidu_cache_link = search_result.xpath('.//a[text()="百度快照"]/@href')[0] + "&fast=y"
            search_data.append(
                {
                    'question_id': index,
                    'question': keyword,
                    'title': title,
                    'abstract': abstract,
                    'source_link': source_link,
                    'baidu_cache_link': baidu_cache_link
                }
            )
            if len(search_data) == num:
                break
        except Exception as e:
            print(e)
    return search_data

def get_div_text(response):
    txt = response.xpath('string(.)')
    scripts = response.xpath('//script')
    for it in scripts:
        ws = it.xpath('string(.)')
        if ws in txt:
          txt = txt.replace(ws, '')
    return txt

def baidu_find_answer(key):
    s = quote(key)
    url = 'https://www.baidu.com/s?wd=' + s
    html = get_raw_html(url)
    print('isok', key, url)
    if html == '': # open error
        return None
    # with open('D://test.html', 'w', encoding='utf') as f:
    #     f.write(html)
    response = etree.HTML(html)
    ans = response.xpath('//div[@class="c-border"]')
    if ans:
        ans = ans[0]
        url = ans.xpath('//span[@class="op_exactqa_s_abstract_more"]/a/@href')
        url = url[0] if url else None
        print(url)
        abstract = ans.xpath('string(.)')
        txt = None
        html = get_raw_html(url)
        if html != '':
            res = etree.HTML(html)
            txt = get_div_text(res)
        valid_lines = []
        for line in abstract.split('\n'):
            line = line.strip()
            if line:
                valid_lines.append(line)
        abstract = "。".join(valid_lines)
        valid_lines2 = []
        for line in txt.split('\n'):
            line = line.strip()
            if line:
                valid_lines2.append(line)
        txt = "。".join(valid_lines2)
        return (abstract, txt)
        # print(abstract)
        # print('#'*25)
        # print(txt)
    else:
        ans = response.xpath('//div[@class="c-result-content"]')
        # print(ans)
        if ans:
            ans = ans[0]
            abstract = ans.xpath('string(.)')
            valid_lines = []
            for line in abstract.split('\n'):
                line = line.strip()
                if line:
                    valid_lines.append(line)
            abstract = "。".join(valid_lines)
            return (abstract, None)
    return None

def crawl_baidu_cache_page(url):
    response = requests.get(url, headers=headers)
    response.encoding = 'gbk'
    content_tree = etree.HTML(response.text)
    try:
        raw_content = content_tree.xpath('//div[@style="position:relative"]')[0]
    except:
        return ""
    raw_content = raw_content.xpath('string(.)')
    valid_lines = []
    for line in raw_content.split('\n'):
        line = line.strip()
        if line:
            valid_lines.append(line)
    valid_content = "。".join(valid_lines)
    return valid_content



def crawl(keyword):
    # words = list(jieba.cut(keyword))
    # right_ans = baidu_find_answer(keyword)
    # print(right_ans[0])
    search_data = crawl_baidu_search(keyword)
    results = []
    pool = Pool(processes=3)
    for each_search_data in search_data:
        results.append(pool.apply_async(crawl_baidu_cache_page, args=(each_search_data['baidu_cache_link'],)))
    pool.close()
    pool.join()
    for each_search_data, result in zip(search_data, results):
        each_search_data['content'] = result.get()
        each_search_data['doc_tokens'] = list(jieba.cut(each_search_data['abstract']))
        each_search_data['temp_tokens'] = list(jieba.cut(each_search_data['content']))
        # head = len(each_search_data['temp_tokens'])
        # end = 0
        # for word in words:
        #     for token in each_search_data['temp_tokens']:
        #         if(word  in token):
        #             end = max(each_search_data['temp_tokens'].index(token)+30,end)
        #             end = min(len(each_search_data['temp_tokens']),end)
        #             head = min(head,each_search_data['temp_tokens'].index(token)- 30)
        #             head = max(0,head)
        #     print("Head : " + str(head) + ",End : " + str(end))
        # each_search_data['temp_tokens'] = each_search_data['temp_tokens'][head:end]
    # right_ans = {}
    # content = baidu_find_answer(keyword)
    # if(content !=None):
    #     right_ans['content'] = content[0]
    #     right_ans['doc_tokens']=list(jieba.cut(right_ans['content']))
    #     right_ans['temp_tokens'] = right_ans['doc_tokens']
    #     right_ans['question_id'] = 6
    #     right_ans['question'] = keyword
    #     search_data.insert(0,right_ans)
    #     if(len(search_data)>5):
    #         search_data = search_data[0:4]
    for each_search_data in search_data:
        print('抓取的数据为', each_search_data['content'])
    return search_data


# if __name__ == '__main__':
#     start_time = time.time()
#     data = crawl("我和我的祖国七位导演都是谁？")
#     print(data)


# coding:utf-8
import itchat
import re
from snownlp import SnowNLP
from snownlp import sentiment
from pyecharts import Map
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
import os 
import numpy as np
import PIL.Image as Image
import jieba
itchat.login()
friends = itchat.get_friends(update=True)[0:]
emotions = []
tList = []
for i in friends:
    signature = i["Signature"].replace(" ", "").replace("span", "").replace("class", "").replace("emoji", "")
    rep = re.compile("1f\d.+")
    signature = rep.sub("", signature)
    tList.append(signature)
    # 拼接字符串
    text = "".join(tList)
    if(len(signature)>0):
                nlp=SnowNLP(signature)
                emotions.append(nlp.sentiments)
# jieba分词
wordlist_jieba = jieba.cut(text, cut_all=True)
wl_space_split = " ".join(wordlist_jieba)
# wordcloud词云
d= os.path.dirname(os.path.abspath( __file__ ))
alice_coloring = np.array(Image.open(os.path.join(d, "timg.jpg")))
my_wordcloud = WordCloud(background_color="white", max_words=2000,mask=alice_coloring,max_font_size=400, random_state=420,font_path='/Users/Pluto/Desktop/Python/SourceHanSans-Heavy.otf').generate(wl_space_split)
image_colors = ImageColorGenerator(alice_coloring)
plt.imshow(my_wordcloud.recolor(color_func=image_colors))
plt.imshow(my_wordcloud)
plt.axis("off")
plt.show()

count_good = len(list(filter(lambda x:x>0.66,emotions)))
count_normal = len(list(filter(lambda x:x>=0.33 and x<=0.66,emotions)))
count_bad = len(list(filter(lambda x:x<0.33,emotions)))
labels = [u'负面消极',u'中性',u'正面积极']
values = (count_bad,count_normal,count_good)
plt.rcParams['font.sans-serif'] = ['simHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel(u'情感判断')
plt.ylabel(u'频数')
plt.xticks(range(3),labels)
plt.legend(loc='upper right',)
plt.bar(range(3), values, color = 'rgb')
plt.title(u'%s的微信好友签名信息情感分析' % friends[0]['NickName'])
plt.show()
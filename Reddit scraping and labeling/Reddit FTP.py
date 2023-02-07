
from os import listdir
import os
import json
import glob
import pandas as pd
import subprocess

link_df = pd.read_csv('link_df.csv')

for i in range(1, 250, 1):
  data_link = link_df.loc[i]['Link']
  print(data_link)
  os.makedirs('/content/reddit_data/' + str(i) + "/")
  bashCommand = "wget -P /content/reddit_data/" + str(i) + " " + data_link
  process = subprocess.run(bashCommand.split(), stdout=subprocess.PIPE)
  zip_dir = "/content/reddit_data/" + str(i)
  zip_file = "/content/reddit_data/" + str(i) + "/" + listdir(zip_dir)[0]
  bashCommand = "unzip -q " + zip_file + " -d /content/reddit_data/" + str(i) 
  process = subprocess.run(bashCommand.split(), stdout=subprocess.PIPE)
  file = glob.glob(zip_dir + '/*.jsonl')[0]
  with open(file, 'r', encoding='utf-8') as f:
    parents_to_delete = {}
    parents = {}
    id = []
    text = []
    parent = []
    time = []
    scores = []
    for line in f:
      jsonobj = json.loads(line)
      if jsonobj['id'] == jsonobj['root']:
        current_score = jsonobj['meta']['score']
        current_time = jsonobj['timestamp']
        current_id = jsonobj['id']
        scores.append(current_score)
        parents_to_delete[current_id] = [current_score, current_time]
        id.append(current_id)
        text.append(jsonobj['text'])
        time.append(current_time)
    corpus = pd.DataFrame(list(zip(id, text, time, scores)), columns=['id', 'text', 'time', 'score'])
    corpus.to_csv('/content/drive/MyDrive/Boost Project/data/reddit/titles_' + str(i) + '.csv', index=False)
    del corpus
    number_to_divide_by = 10
    if i > 100:
      number_to_divide_by = 8
    if i > 150:
      number_to_divide_by = 6
    decile = round(len(scores) / 8)
    scores.sort(reverse=True)
    threshold = scores[decile]
    for k, v in parents_to_delete.items():
      if v[0] >= threshold:
        parents[k] = v
    del scores
    del parents_to_delete
    f.close()
  with open(file, 'r', encoding='utf-8') as f:
    id = []
    text = []
    root = []
    reply = []
    parent = []
    time = []
    score = []
    for line in f:
      jsonobj = json.loads(line)
      rt = jsonobj['root']
      cid = jsonobj['id']
      if (rt in parents) and (cid not in parents):
        time_since = jsonobj['timestamp'] - parents[rt][1]
        if (time_since < 40000) and (time_since > 5):
          txt = jsonobj['text']
          if (len(txt.split()) >= 5) and (jsonobj['meta']['stickied'] != 'false'):
              id.append(cid)
              text.append(str(txt))
              root.append(rt)
              time.append(time_since)
              score.append(int(jsonobj['meta']['score']))
              reply.append(str(jsonobj['reply_to']))
      if len(text) > 30000000:
        break
    corpus = pd.DataFrame(list(zip(id, text, root, time, score, reply)), columns=['id', 'text', 'root', 'time_since', 'score', 'reply'])
    f.close()    
  corpus.to_csv('/content/drive/MyDrive/Boost Project/data/reddit/comments_level_1_' + str(i) + '.csv', index=False)
  parents2 = dict(zip(list(corpus.id), [0] * len(corpus.id)))
  del corpus
  with open(file, 'r', encoding='utf-8') as f:
    id = []
    text = []
    root = []
    reply = []
    parent = []
    time = []
    score = []
    for line in f:
      jsonobj = json.loads(line)
      level_1 = jsonobj['reply_to']
      cid = jsonobj['id']
      if (level_1 in parents2) and (cid not in parents2):
          txt = jsonobj['text']
          if (len(txt.split()) >= 5) and (jsonobj['meta']['stickied'] != 'false'):
              id.append(cid)
              text.append(str(txt))
              root.append(jsonobj['root'])
              time.append(jsonobj['timestamp'])
              score.append(int(jsonobj['meta']['score']))
              reply.append(str(jsonobj['reply_to']))
      if len(text) > 30000000:
        break
    corpus = pd.DataFrame(list(zip(id, text, root, time, score, reply)), columns=['id', 'text', 'root', 'time_since', 'score', 'reply'])
    f.close()    
  corpus.to_csv('/content/drive/MyDrive/Boost Project/data/reddit/comments_level_2_' + str(i) + '.csv', index=False)
  parents3 = dict(zip(list(corpus.id), [0] * len(corpus.id)))
  del corpus
  with open(file, 'r', encoding='utf-8') as f:
    id = []
    text = []
    root = []
    reply = []
    parent = []
    time = []
    score = []
    for line in f:
      jsonobj = json.loads(line)
      level_1 = jsonobj['reply_to']
      cid = jsonobj['id']
      if (level_1 in parents3) and (cid not in parents3):
          txt = jsonobj['text']
          if (len(txt.split()) >= 5) and (jsonobj['meta']['stickied'] != 'false'):
              id.append(cid)
              text.append(str(txt))
              root.append(jsonobj['root'])
              time.append(jsonobj['timestamp'])
              score.append(int(jsonobj['meta']['score']))
              reply.append(str(jsonobj['reply_to']))
      if len(text) > 30000000:
        break
    corpus = pd.DataFrame(list(zip(id, text, root, time, score, reply)), columns=['id', 'text', 'root', 'time_since', 'score', 'reply'])
    f.close()    
  corpus.to_csv('/content/drive/MyDrive/Boost Project/data/reddit/comments_level_3_' + str(i) + '.csv', index=False)
  del corpus  
  bashCommand = "rm -r /content/reddit_data/" + str(i) 
  process = subprocess.run(bashCommand.split(), stdout=subprocess.PIPE)

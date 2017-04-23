import json
import gzip
import pickle 

head_list = []
desc_list = []
keywords_list = []
count = 0
file = gzip.open('signalmedia-1m.jsonl.gz')
for each_line in file:
	record = json.loads(each_line)
	head_list.append(record['title'])
	desc_list.append(record['content'])
	keywords_list.append(None)
	count += 1
	if count >= 100:
		break

# (heads,desc,keywords) = (head_list,desc_list,keywords_list)
file_open = open('tokens.pkl','wb')
pickle.dump((head_list,desc_list,keywords_list),file_open,-1)
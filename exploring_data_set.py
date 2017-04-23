import gzip
import json







file_open = open('train_v1.1.json','r')


# file_open = open('test','w')
# file = gzip.open('train_v1.1.json.gz')

for line in file_open:
	print line
	raw_input()



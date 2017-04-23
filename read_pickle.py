import cPickle as pickle
FN0 = 'tokens_sample' # this is the name of the data file which I assume you already have



fp = open('%s.pkl'%FN0, 'rb')
heads, desc, keywords = pickle.load(fp) # keywords are not used in this project


print len(heads)
print len(desc)
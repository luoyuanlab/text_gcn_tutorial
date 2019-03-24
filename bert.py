from bert_serving.client import BertClient
import sys

if len(sys.argv) != 2:
	sys.exit("Use: python bert.py <dataset>")

dataset = sys.argv[1]
bc = BertClient()

f = open('data/corpus/' + dataset + '.txt', 'r')
lines = f.readlines()
f.close()
doc_list = [x.strip() for x in lines]

doc_vecs = bc.encode(doc_list)

doc_vec_str_list = []
for doc_vec in doc_vecs:
    doc_vec = [str(x) for x in doc_vec]
    string = 'doc_vec' + ' '.join(doc_vec)
    doc_vec_str_list.append(string)

doc_vec_str_list_str = '\n'.join(doc_vec_str_list)

f = open('data/' + dataset + '_bert.txt', 'w')
f.write(doc_vec_str_list_str)
f.close()

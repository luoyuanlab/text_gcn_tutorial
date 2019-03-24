from collections import Counter

f = open('data/ohsumed.txt', 'r')
lines = f.readlines()
f.close()

cnt = Counter()
for line in lines:
    temp = line.strip().split()
    cnt[temp[-1]] += 1

print(cnt)
top_5 = dict(cnt.most_common(5))
print(top_5)

f = open('data/corpus/ohsumed.txt', 'r')
doc_lines = f.readlines()
f.close()

top_5_meta_list = []
top_5_doc_list = []

for i in range(len(lines)):
    temp = lines[i].strip().split()
    if temp[-1] in top_5:
        top_5_meta_list.append(lines[i].strip())
        top_5_doc_list.append(doc_lines[i].strip())
        #print(lines[i].strip())

print(len(top_5_meta_list))

meta_str = '\n'.join(top_5_meta_list)
f = open('data/ohsumed_5.txt', 'w')
f.write(meta_str)
f.close()

docs_str = '\n'.join(top_5_doc_list)
f = open('data/corpus/ohsumed_5.txt', 'w')
f.write(docs_str)
f.close()
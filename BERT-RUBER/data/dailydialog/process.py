import sys
path = sys.argv[1]
print(f'path: {path}')
with open(path) as f:
    c = []
    for line in f.readlines():
        line = line.replace('<user0>', '').replace('<user1>', '').strip()
        c.append(line)
with open(path, 'w') as f:
    for i in c:
        f.write(f'{i}\n')

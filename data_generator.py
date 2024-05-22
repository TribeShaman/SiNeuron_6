import random


generated_data = []


for _ in range(1000):
    x1 = random.randint(0, 25)
    x2 = random.randint(0, 25)
    a = random.randint(-5,5)
    b = random.randint(0,5)
    if x2 >= (a*x1)+b:
        label = 0
    else:
        label = 1
    generated_data.append([x1, x2, label])
for data in generated_data:
    print(','.join(map(str, data)))

import random

generated_data = []

for _ in range(100):
    x = random.randint(0, 25)
    y = random.randint(0, 25)

    if y >= (-(1/3)*x)+15:
        label = 0
    else:
        label = 1

    generated_data.append([x, y, label])
for data in generated_data:
    print(','.join(map(str, data)))




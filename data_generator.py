import random

# Lista przechowująca wygenerowane dane
generated_data = []

# Generowanie 100 linii danych
for _ in range(10):
    x1 = random.randint(0, 25)
    x2 = random.randint(0, 25)
    # Ustalenie etykiety klasy na podstawie pozycji punktu względem prostej y = x
    if x2 >= x1:
        label = 1
    else:
        label = 0
    generated_data.append([x1, x2, label])

# Wyświetlenie wygenerowanych danych
for data in generated_data:
    print(','.join(map(str, data)))

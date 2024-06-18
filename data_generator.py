import random
import os

def generate_data():
    # Upewnij się, że katalog "Dane" istnieje
    os.makedirs('Dane', exist_ok=True)

    generated_data = []

    # Generowanie danych
    for _ in range(250):
        x = random.randint(0, 25)
        y = random.randint(0, 25)

        if y >= -(2 * x)+10:
            label = 0
        else:
            label = 1

        generated_data.append([x, y, label])

    # Zapisywanie danych do pliku
    with open('Dane/in.tab', 'w') as file:
        for data in generated_data:
            file.write(','.join(map(str, data)))
            file.write('\n')





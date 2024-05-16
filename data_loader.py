import numpy as np

# Otwarcie pliku do odczytu
with open('Dane/in.tab') as file:
    # Iteracja po każdym wierszu pliku
    for line in file:
        # Wczytanie pojedynczego wiersza jako listy wartości oddzielonych tabulatorami
        data = line.strip().split(',')
        # Tutaj możesz wykonywać operacje na wczytanym wierszu
        print("Dane:", data)

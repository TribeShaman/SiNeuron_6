import matplotlib
import tkinter as tk

# Funkcja obsługująca kliknięcie przycisku
def on_button_click():
    print("Przycisk został kliknięty!")

# Tworzenie głównego okna
root = tk.Tk()
root.title("Proste okno z przyciskiem")

# Tworzenie przycisku
button = tk.Button(root, text="Kliknij mnie!", command=on_button_click)
button.pack(pady=10)

# Uruchomienie pętli głównej
root.mainloop()

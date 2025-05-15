import os
import numpy as np
from PIL import Image, ImageTk
from tkinter import Tk, Label, Button, filedialog
from sklearn.neighbors import KNeighborsClassifier
import joblib

# ---------- SETTINGS ----------
image_size = (50, 50)
data_dir = "C:/Users/hp/Desktop/Fruit-Images-Dataset-master/Training"

# ---------- TRAINING ----------
def load_and_train():
    X, y, label_names = [], [], []
    for idx, folder in enumerate(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            label_names.append(folder)
            for file in os.listdir(folder_path)[:100]:  # limit 100 images per class
                img_path = os.path.join(folder_path, file)
                try:
                    img = Image.open(img_path).convert('RGB').resize(image_size)
                    X.append(np.array(img).flatten())
                    y.append(idx)
                except:
                    continue
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    joblib.dump((model, label_names), "fruit_model.pkl")

# ---------- PREDICTION ----------
def predict_image(img_path):
    model, label_names = joblib.load("fruit_model.pkl")
    img = Image.open(img_path).convert('RGB').resize(image_size)
    img_array = np.array(img).flatten().reshape(1, -1)
    pred = model.predict(img_array)[0]
    return label_names[pred]

# ---------- GUI ----------
class FruitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fruit Classifier")

        self.label = Label(root, text="Upload an image of a fruit", font=('Arial', 14))
        self.label.pack()

        self.img_label = Label(root)
        self.img_label.pack()

        self.result_label = Label(root, text="", font=('Arial', 16), fg='blue')
        self.result_label.pack()

        self.upload_button = Button(root, text="Browse Image", command=self.browse)
        self.upload_button.pack()

    def browse(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if path:
            img = Image.open(path).resize((150, 150))
            tk_img = ImageTk.PhotoImage(img)
            self.img_label.config(image=tk_img)
            self.img_label.image = tk_img

            result = predict_image(path)
            self.result_label.config(text=f"Prediction: {result}")

# ---------- MAIN ----------
if __name__ == "__main__":
    if not os.path.exists("fruit_model.pkl"):
        print("Training model, please wait...")
        load_and_train()
        print("Model saved!")

    root = Tk()
    app = FruitApp(root)
    root.mainloop()

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image
from predictions import predict
import threading
from ttkthemes import ThemedTk


class App:
    def __init__(self, root):
        self.result_text = None
        self.choose_button = None
        self.progress_bar = None

        self.root = root
        self.root.title('Handwritten text recognition')
        self.create_widgets()

    def create_widgets(self):
        self.choose_button = ttk.Button(self.root, text='Choose image', command=self.choose_image)
        self.choose_button.pack(pady=10)

        self.result_text = tk.Text(self.root, height=5, width=50, wrap="word")
        self.result_text.pack(pady=10)

        self.progress_bar = ttk.Progressbar(self.root, orient='horizontal', length=300, mode='indeterminate')
        self.progress_bar.pack(pady=10)

    def choose_image(self):
        file_path = filedialog.askopenfilename(filetypes=[('Image files', '*.png;*.jpg;*.jpeg')])
        if file_path:
            self.progress_bar.start()
            threading.Thread(target=self.run_predict, args=(file_path,)).start()
        else:
            result_message = 'Invalid data'
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result_message)

    def run_predict(self, file_path):
        # try:
        #
        # except Exception as e:
        #     result_message = 'Something went wrong'
        #     print(e)
        result_message = predict(file_name=file_path)

        self.progress_bar.stop()

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result_message)


if __name__ == '__main__':
    app_root = ThemedTk(theme='scidpink')
    app = App(app_root)
    app_root.mainloop()

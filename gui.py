import tkinter as tk
import numpy as np
from PIL import Image, ImageTk, ImageGrab
from image_analyzer import ImageAnalyzer
import pandas as pd
import json

MESSAGE_PATH = 'data/message_clean.csv'
OPT_NAME_PATH = 'data/options.csv'
ONE_OPT_ADV_PATH = 'data/one_option_adv.json'
TWO_OPT_ADV_PATH = 'data/two_option_adv.json'


class ClipboardImageApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Clipboard Image")

        self.canvas = tk.Canvas(self.window, width=300, height=200)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.button = tk.Button(self.window, text="Get Image from Clipboard", command=self.on_button_click)
        self.button.pack()

        self.top_left_button = tk.Button(self.window, text="Top Left", command=self.set_top_left_mode)
        self.top_left_button.pack()

        self.bottom_right_button = tk.Button(self.window, text="Bottom Right", command=self.set_bottom_right_mode)
        self.bottom_right_button.pack()


        self.image_np = None
        self.current_mode = 'top_left'
        self.tl_h = None
        self.tl_w = None
        self.br_h = None
        self.br_w = None

        self.image_analyzer = ImageAnalyzer()

        self.messages = pd.read_csv(MESSAGE_PATH)
        self.options = pd.read_csv(OPT_NAME_PATH)
        with open(ONE_OPT_ADV_PATH,'r') as f:
            self.one_option_adv = json.load(f)
        with open(TWO_OPT_ADV_PATH,'r') as f:
            self.two_option_adv = json.load(f)

    def analyze_image(self):
        if self.image_np is not None:
            options, is_avail, opt_gauge, adv_pred, opt_one_pred, opt_two_pred_1, opt_two_pred_2, enchant_n_pred= self.image_analyzer.analyze(self.image_np)
            print(f"Options: {options}")
            print(f"Is available: {is_avail}")
            print(f"Option gauge: {opt_gauge}")
            for ap, op, tp1, tp2 in zip(adv_pred, opt_one_pred, opt_two_pred_1, opt_two_pred_2):
                adv_str = self.messages.iloc[ap].Desc1
                if ap in self.one_option_adv:
                    adv_str = adv_str.replace('{0}', self.options.iloc[op].option_name)
                elif ap in self.two_option_adv:
                    adv_str = adv_str.replace('{0}', self.options.iloc[tp1].option_name)
                    adv_str = adv_str.replace('{1}', self.options.iloc[tp2].option_name)
                print(f"Advice prediction: {adv_str}")
            print(f"Enchantment prediction: {enchant_n_pred}")

        else:
            print("No image to analyze")

    def resize_image_for_display(self, img, max_width=1280, max_height=720):
        width, height = img.size
        aspect_ratio = float(width) / float(height)

        if width > max_width:
            width = max_width
            height = int(width / aspect_ratio)

        if height > max_height:
            height = max_height
            width = int(height * aspect_ratio)

        return img.resize((width, height), Image.Resampling.BILINEAR)
    

    def get_image_from_clipboard(self):
        try:
            img = ImageGrab.grabclipboard()
            img_rgb = img.convert('RGB')
            img_rgb = img_rgb.resize((1920,1080), Image.Resampling.BILINEAR)
            self.image_np = np.array(img_rgb)
            self.analyze_image()
            return img_rgb
        except Exception as e:
            print(f"Error getting image from clipboard: {e}")
            return None

    def display_image(self, img_rgb):
        if img_rgb:
            # Resize the image for display
            img_resized = self.resize_image_for_display(img_rgb)

            width, height = img_resized.size
            self.canvas.config(width=width, height=height)

            tk_img = ImageTk.PhotoImage(img_resized)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
            self.canvas.image = tk_img

        else:
            print("No image found in clipboard")
    def update_image(self):
        if self.image_np is not None:
            img = Image.fromarray(self.image_np)
            width, height = img.size
            self.canvas.config(width=width, height=height)

            tk_img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
            self.canvas.image = tk_img
        else:
            print("No image_np to update")

    def on_button_click(self):
        img = self.get_image_from_clipboard()
        self.display_image(img)

    def on_canvas_click(self, event):
        self.click_h = event.y
        self.click_w = event.x

        if self.image_np is not None and 0 <= self.click_h < self.image_np.shape[0] and 0 <= self.click_w < self.image_np.shape[1]:
            print(f"Clicked on image at height: {self.click_h}, width: {self.click_w}")
            print(f"Pixel value: {self.image_np[self.click_h][self.click_w]}")

            if self.current_mode == "top_left":
                self.tl_h = self.click_h
                self.tl_w = self.click_w
                print(f"Top-left corner set to height: {self.tl_h}, width: {self.tl_w}")

            elif self.current_mode == "bottom_right":
                self.br_h = self.click_h
                self.br_w = self.click_w
                print(f"Bottom-right corner set to height: {self.br_h}, width: {self.br_w}")

        else:
            print("Clicked outside of the image")
            
    def set_top_left_mode(self):
        self.current_mode = "top_left"

    def set_bottom_right_mode(self):
        self.current_mode = "bottom_right"


    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = ClipboardImageApp()
    app.run()

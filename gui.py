import tkinter as tk
import numpy as np
from PIL import Image, ImageTk, ImageGrab
from image_analyzer import ImageAnalyzer
from value_analyzer import ValueAnalyzer
from index_converter import AdvIdxConverter
import pandas as pd
import json

MESSAGE_PATH = 'data/message_clean.csv'
OPT_NAME_PATH = 'data/options.csv'
ONE_OPT_ADV_PATH = 'data/one_option_adv.json'
TWO_OPT_ADV_PATH = 'data/two_option_adv.json'
ADV_IDX_CONV = 'data/adv_idx_convert.csv'
ADV_IDX_CONV_TWO = 'data/adv_idx_convert_two_opt.json'


class ClipboardImageApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Clipboard Image")

        self.canvas = tk.Canvas(self.window, width=300, height=200)
        self.canvas.grid(row=0, column=0, columnspan=3, sticky="nsew")

        self.button = tk.Button(self.window, text="Get Image from Clipboard", command=self.on_button_click)
        self.button.grid(row=1, column=1, sticky="nsew")

        self.option_state_vars = [tk.StringVar() for _ in range(6)]
        self.option_name_vars = [tk.StringVar() for _ in range(6)]
        for i in range(6):
            label_state = tk.Label(self.window, textvariable=self.option_state_vars[i])
            label_state.grid(row=i+2, column=1)
            label_name = tk.Label(self.window, textvariable=self.option_name_vars[i])
            label_name.grid(row=i+2, column=2)
        # Caution text color red
        
        self.caution_0 = tk.Label(self.window, text="주의: 연산 효율을 위해 충분히 구린 옵션은 0으로 뜹니다.",
                                  fg='red')
        self.caution_0.grid(row=2, column=0)
        self.caution_1 = tk.Label(self.window, text="따라서 반드시 확률이 0임을 의미하지 않습니다.",
                                  fg='red')
        self.caution_1.grid(row=3, column=0)
        self.caution_2 = tk.Label(self.window, text="주의: 옵션의 가치는 고려되지 않습니다.",
                                  fg='red')
        self.caution_2.grid(row=5, column=0)
        self.caution_3 = tk.Label(self.window, text="특정 옵션 저격은 스스로 판단해주세요.",
                                  fg='red')
        self.caution_3.grid(row=6, column=0)

        self.rect_frames = []
        self.rect_texts = []
        for i in range(3):
            rect_frame = tk.Frame(self.window, bg="white", height=50)
            rect_frame.grid(row=8, column=i, sticky="nsew")
            self.window.columnconfigure(i, weight=1)
            self.rect_frames.append(rect_frame)

            rect_text = tk.Text(rect_frame, bg="white", height=11,width=40, wrap=tk.WORD)
            rect_text.tag_configure('center', justify='center')  # Center-align the text
            rect_text.pack(fill=tk.BOTH, expand=True)
            self.rect_texts.append(rect_text)

        self.options_idx = None

        self.image_np = None

        self.image_analyzer = ImageAnalyzer()
        self.value_analyzer = ValueAnalyzer()
        self.adv_idx_converter = AdvIdxConverter()

        self.messages = pd.read_csv(MESSAGE_PATH)
        self.options = pd.read_csv(OPT_NAME_PATH)
        with open(ONE_OPT_ADV_PATH,'r') as f:
            self.one_option_adv = json.load(f)
        with open(TWO_OPT_ADV_PATH,'r') as f:
            self.two_option_adv = json.load(f)


    def analyze_image(self):
        if self.image_np is not None:
            (options, 
             is_avail, 
             adv_gauge, 
             adv_pred, 
             opt_one_pred,
             opt_two_pred_1, 
             opt_two_pred_2, 
             enchant_n_pred, 
             options_idx)= self.image_analyzer.analyze(self.image_np)
            self.options_idx = options_idx
            options_str = ["▣"*i + "□"*(10-i) for i in options]
            is_avail_str = ['가능' if i else '봉인' for i in is_avail]
            options_name_str = [self.options.iloc[i].option_name for i in options_idx]
            converted_adv_pred = self.adv_idx_converter.convert(
                self.options_idx,
                adv_pred,
                opt_one_pred,
                opt_two_pred_1,
                opt_two_pred_2
            )
            adv_vals, curve_vals, final_vals, recommand_idx = self.value_analyzer.get_value(
                options,
                is_avail,
                adv_gauge,
                converted_adv_pred,
                enchant_n_pred,
            )
            self.white_rect_frame()
            self.update_rect_frame_color(recommand_idx, '#87CEEB')
            for i in range(5):
                self.option_state_vars[i].set(f"{is_avail_str[i]} {options_str[i]}")
                self.option_name_vars[i].set(f"{options_name_str[i]}")
            for i, (ag, ap, op, tp1, tp2) in enumerate(zip(adv_gauge, adv_pred, opt_one_pred, opt_two_pred_1, opt_two_pred_2)):
                adv_str = self.messages.iloc[ap].Desc1
                if ap in self.one_option_adv:
                    adv_str = adv_str.replace('{0}', self.options.iloc[op].option_name)
                elif ap in self.two_option_adv:
                    adv_str = adv_str.replace('{0}', self.options.iloc[tp1].option_name)
                    adv_str = adv_str.replace('{1}', self.options.iloc[tp2].option_name)
                if ag == 0:
                    gauge_str = ''
                elif ag > 0:
                    gauge_str = '◆' * ag + '◇' * (3 - ag)
                elif ag < 0:
                    gauge_str = '●' * (-ag) + '○' * (6 + ag)
                
                if isinstance(adv_vals[i], list):
                    rect_text = (
                        gauge_str + '\n' + adv_str + '\n' +
                        "합8 확률  |스택 가중치 | 최종 점수\n"
                    )
                    max_idx = np.argmax(final_vals[i])
                    for j in range(5):
                        if j == max_idx:
                            rect_text += f"*추천*{adv_vals[i][j]*100:05.2F}    |     {curve_vals[i]*1000:05.2F}  |    {final_vals[i][j]*100000:05.2F}*추천*\n"
                        else:
                            rect_text += f"{adv_vals[i][j]*100:05.2F}    |     {curve_vals[i]*1000:05.2F}  |    {final_vals[i][j]*100000:05.2F}\n"
                else:
                    rect_text = (
                        gauge_str + '\n' + adv_str + '\n' +
                        "합8 확률  |스택 가중치 | 최종 점수\n" +
                        f"{adv_vals[i]*100:05.2F}    |     {curve_vals[i]*1000:05.2F}  |    {final_vals[i]*100000:05.2F}"
                    )
                self.update_rect_text(i, rect_text)
                # self.update_rect_text(i, gauge_str + '\n' + adv_str)
            self.option_state_vars[5].set(f"남은 연성 횟수: {enchant_n_pred+1}")
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

        except Exception as e:
            print(f"Error getting image from clipboard: {e}")
            return None
        self.analyze_image()

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
            self.display_image(img)
        else:
            print("No image_np to update")

    def on_button_click(self):
        self.get_image_from_clipboard()
        self.update_image()

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

    def update_rect_text(self, index, new_text):
        if 0 <= index < len(self.rect_texts):
            self.rect_texts[index].delete(1.0, tk.END)
            self.rect_texts[index].insert(tk.END, new_text, 'center')
        else:
            print(f"Invalid rectangle index: {index}")

    def update_rect_frame_color(self, index, new_color):
        if 0 <= index < len(self.rect_texts):
            self.rect_texts[index].config(bg=new_color)
        else:
            print(f"Invalid rectangle index: {index}")

    def white_rect_frame(self):
        for rt in self.rect_texts:
            rt.config(bg='#FFFFFF')

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = ClipboardImageApp()
    app.run()

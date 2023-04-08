import torch
import numpy as np
import json
import models
import pandas as pd

RATIO_DATA = 'data/ratio_data.json'
ADVICE_MODEL = 'advice_resnet18_aug_1'
ADVICE_ONE_MODEL = 'advice_one_resnet18_1'
ADVICE_TWO_MODEL = 'advice_two_resnet18_1'
ENCHANT_N_MODEL = 'enchant_n_resnet18_aug_1'
MESSAGE_PATH = 'data/message_clean.csv'
OPT_NAME_PATH = 'data/options.csv'
ONE_OPT_ADV_PATH = 'data/one_option_adv.json'
TWO_OPT_ADV_PATH = 'data/two_option_adv.json'

class ImageAnalyzer():
    def __init__(
        self,
        ratio_data_path=RATIO_DATA,
        advice_model_name=ADVICE_MODEL,
        advice_one_model_name=ADVICE_ONE_MODEL,
        advice_two_model_name=ADVICE_TWO_MODEL,
        enchant_n_model_name=ENCHANT_N_MODEL,
        message_path=MESSAGE_PATH,
        opt_name_path=OPT_NAME_PATH,
        one_opt_adv_path=ONE_OPT_ADV_PATH,
        two_opt_adv_path=TWO_OPT_ADV_PATH
    ) -> None:
        with open(ratio_data_path, 'r') as f:
            self.ratio_data = json.load(f)
        self.per_slot_height_ratio = self.ratio_data["per_slot_height_ratio"]
        self.per_slot_width_ratio = self.ratio_data["per_slot_width_ratio"]
        self.prob_top_center_to_left_top_ratio_w = self.ratio_data["prob_top_center_to_left_top_ratio_w"]
        self.prob_width_to_slot_width_ratio = self.ratio_data["prob_width_to_slot_width_ratio"]
        self.prob_height_to_slot_height_ratio = self.ratio_data["prob_height_to_slot_height_ratio"]
        self.adv_dist_to_slot_width_ratio = self.ratio_data["adv_dist_to_slot_width_ratio"]
        self.chaos_spacing_to_slot_width_ratio = self.ratio_data["chaos_spacing_to_slot_width_ratio"]
        self.order_spacing_to_slot_width_ratio = self.ratio_data["order_spacing_to_slot_width_ratio"]
        self.radius_to_slot_width_ratio = self.ratio_data["radius_to_slot_width_ratio"]
        self.adv_1_chaos_0_to_slot_top_left_ratio_h = self.ratio_data["adv_1_chaos_0_to_slot_top_left_ratio_h"]
        self.adv_1_chaos_0_to_slot_top_left_ratio_w = self.ratio_data["adv_1_chaos_0_to_slot_top_left_ratio_w"]
        self.adv_1_order_0_to_slot_top_left_ratio_h = self.ratio_data["adv_1_order_0_to_slot_top_left_ratio_h"]
        self.adv_1_order_0_to_slot_top_left_ratio_w = self.ratio_data["adv_1_order_0_to_slot_top_left_ratio_w"]
        self.adv_1_top_left_to_slot_top_left_ratio_h = self.ratio_data["adv_1_top_left_to_slot_top_left_ratio_h"]
        self.adv_1_top_left_to_slot_top_left_ratio_w = self.ratio_data["adv_1_top_left_to_slot_top_left_ratio_w"]
        self.adv_1_bottom_right_to_slot_top_left_ratio_h = self.ratio_data["adv_1_bottom_right_to_slot_top_left_ratio_h"]
        self.adv_1_bottom_right_to_slot_top_left_ratio_w = self.ratio_data["adv_1_bottom_right_to_slot_top_left_ratio_w"]
        self.adv_2_top_left_to_slot_top_left_ratio_h = self.ratio_data["adv_2_top_left_to_slot_top_left_ratio_h"]
        self.adv_2_top_left_to_slot_top_left_ratio_w = self.ratio_data["adv_2_top_left_to_slot_top_left_ratio_w"]
        self.adv_2_bottom_right_to_slot_top_left_ratio_h = self.ratio_data["adv_2_bottom_right_to_slot_top_left_ratio_h"]
        self.adv_2_bottom_right_to_slot_top_left_ratio_w = self.ratio_data["adv_2_bottom_right_to_slot_top_left_ratio_w"]
        self.adv_3_top_left_to_slot_top_left_ratio_h = self.ratio_data["adv_3_top_left_to_slot_top_left_ratio_h"]
        self.adv_3_top_left_to_slot_top_left_ratio_w = self.ratio_data["adv_3_top_left_to_slot_top_left_ratio_w"]
        self.adv_3_bottom_right_to_slot_top_left_ratio_h = self.ratio_data["adv_3_bottom_right_to_slot_top_left_ratio_h"]
        self.adv_3_bottom_right_to_slot_top_left_ratio_w = self.ratio_data["adv_3_bottom_right_to_slot_top_left_ratio_w"]
        self.enchant_n_top_left_to_slot_top_left_ratio_h = self.ratio_data["enchant_n_top_left_to_slot_top_left_ratio_h"]
        self.enchant_n_top_left_to_slot_top_left_ratio_w = self.ratio_data["enchant_n_top_left_to_slot_top_left_ratio_w"]
        self.enchant_n_bottom_right_to_slot_top_left_ratio_h = self.ratio_data["enchant_n_bottom_right_to_slot_top_left_ratio_h"]
        self.enchant_n_bottom_right_to_slot_top_left_ratio_w = self.ratio_data["enchant_n_bottom_right_to_slot_top_left_ratio_w"]

        self.set_abs_values(self.ratio_data['fhd_left_top'], self.ratio_data['fhd_right_bottom'])

        with open(f'logs/{advice_model_name}/config.json') as f:
            advice_config = json.load(f)
        advice_model = getattr(models, advice_config['model_name'])(**advice_config['model_kwargs'])
        advice_model.load_state_dict(torch.load(f'logs/{advice_model_name}/checkpoints/best_acc.pt'))
        self.advice_model = advice_model.eval()

        with open(f'logs/{advice_one_model_name}/config.json') as f:
            advice_one_config = json.load(f)
        advice_one_model = getattr(models, advice_one_config['model_name'])(**advice_one_config['model_kwargs'])
        advice_one_model.load_state_dict(torch.load(f'logs/{advice_one_model_name}/checkpoints/best_acc.pt'))
        self.advice_one_model = advice_one_model.eval()

        with open(f'logs/{advice_two_model_name}/config.json') as f:
            advice_two_config = json.load(f)
        advice_two_model = getattr(models, advice_two_config['model_name'])(**advice_two_config['model_kwargs'])
        advice_two_model.load_state_dict(torch.load(f'logs/{advice_two_model_name}/checkpoints/best_acc.pt'))
        self.advice_two_model = advice_two_model.eval()

        with open(f'logs/{enchant_n_model_name}/config.json') as f:
            enchant_n_config = json.load(f)
        enchant_n_model = getattr(models, enchant_n_config['model_name'])(**enchant_n_config['model_kwargs'])
        enchant_n_model.load_state_dict(torch.load(f'logs/{enchant_n_model_name}/checkpoints/best_acc.pt'))
        self.enchant_n_model = enchant_n_model.eval()    

        self.messages = pd.read_csv(message_path)
        self.option_names = pd.read_csv(opt_name_path)
        with open(one_opt_adv_path,'r') as f:
            self.one_option_adv = json.load(f)
        with open(two_opt_adv_path,'r') as f:
            self.two_option_adv = json.load(f)

    def set_abs_values(self, left_top, right_bottom):
        height = right_bottom[0] - left_top[0]
        width = right_bottom[1] - left_top[1]
        abs_slot_height = int(height * self.per_slot_height_ratio)
        abs_slot_width = int(width * self.per_slot_width_ratio)
        slot_spacing_width = width/9
        slot_spacing_height = height/4
        self.opt_color_pos = np.zeros((5,10,4), dtype=int)
        for i in range(10):
            for j in range(5):
                self.opt_color_pos[j,i,0] = left_top[0]+int(j*slot_spacing_height)-abs_slot_height//2
                self.opt_color_pos[j,i,1] = left_top[0]+int(j*slot_spacing_height)+abs_slot_height//2
                self.opt_color_pos[j,i,2] = left_top[1]+int(i*slot_spacing_width)-abs_slot_width//2
                self.opt_color_pos[j,i,3] = left_top[1]+int(i*slot_spacing_width)+abs_slot_width//2
        
        self.prob_pos = np.zeros((5,4), dtype=int)
        abs_prob_height = int(self.prob_height_to_slot_height_ratio*height)
        abs_prob_width = int(self.prob_width_to_slot_width_ratio*width)
        prob_top_center = (left_top[0], left_top[1]+int(self.prob_top_center_to_left_top_ratio_w*width))
        for i in range(5):
            self.prob_pos[i,0] = prob_top_center[0]+int(i*slot_spacing_height)-abs_prob_height//2
            self.prob_pos[i,1] = prob_top_center[0]+int(i*slot_spacing_height)+abs_prob_height//2
            self.prob_pos[i,2] = prob_top_center[1]-abs_prob_width//2
            self.prob_pos[i,3] = prob_top_center[1]+abs_prob_width//2
        
        self.order_color_pos = np.zeros((3,3,4), dtype=int)
        self.chaos_color_pos = np.zeros((3,6,4), dtype=int)
        abs_adv_dist = int(self.adv_dist_to_slot_width_ratio*width)
        abs_chaos_spacing = int(self.chaos_spacing_to_slot_width_ratio*width)
        abs_order_spacing = int(self.order_spacing_to_slot_width_ratio*width)
        abs_radius = int(self.radius_to_slot_width_ratio*width)
        abs_adv_1_chaos_0_to_slot_top_left_h = int(self.adv_1_chaos_0_to_slot_top_left_ratio_h*height)
        abs_adv_1_chaos_0_to_slot_top_left_w = int(self.adv_1_chaos_0_to_slot_top_left_ratio_w*width)
        abs_adv_1_order_0_to_slot_top_left_h = int(self.adv_1_order_0_to_slot_top_left_ratio_h*height)
        abs_adv_1_order_0_to_slot_top_left_w = int(self.adv_1_order_0_to_slot_top_left_ratio_w*width)
        abs_adv_1_chaos_0 = (abs_adv_1_chaos_0_to_slot_top_left_h+left_top[0], abs_adv_1_chaos_0_to_slot_top_left_w+left_top[1])
        abs_adv_1_order_0 = (abs_adv_1_order_0_to_slot_top_left_h+left_top[0], abs_adv_1_order_0_to_slot_top_left_w+left_top[1])
        for i in range(3):
            for j in range(3):
                self.order_color_pos[i,j,0] = abs_adv_1_order_0[0]-abs_radius
                self.order_color_pos[i,j,1] = abs_adv_1_order_0[0]+abs_radius
                self.order_color_pos[i,j,2] = abs_adv_1_order_0[1]+int(i*abs_adv_dist+j*abs_order_spacing)-abs_radius
                self.order_color_pos[i,j,3] = abs_adv_1_order_0[1]+int(i*abs_adv_dist+j*abs_order_spacing)+abs_radius
            for j in range(6):
                self.chaos_color_pos[i,j,0] = abs_adv_1_chaos_0[0]-abs_radius
                self.chaos_color_pos[i,j,1] = abs_adv_1_chaos_0[0]+abs_radius
                self.chaos_color_pos[i,j,2] = abs_adv_1_chaos_0[1]+int(i*abs_adv_dist+j*abs_chaos_spacing)-abs_radius
                self.chaos_color_pos[i,j,3] = abs_adv_1_chaos_0[1]+int(i*abs_adv_dist+j*abs_chaos_spacing)+abs_radius
        self.abs_adv_1_top_left = (int(self.adv_1_top_left_to_slot_top_left_ratio_h*height + left_top[0]), 
                                   int(self.adv_1_top_left_to_slot_top_left_ratio_w*width+left_top[1]))
        self.abs_adv_1_bottom_right = (int(self.adv_1_bottom_right_to_slot_top_left_ratio_h*height + left_top[0]),
                                       int(self.adv_1_bottom_right_to_slot_top_left_ratio_w*width + left_top[1]))
        self.abs_adv_2_top_left = (int(self.adv_2_top_left_to_slot_top_left_ratio_h*height + left_top[0]),
                                   int(self.adv_2_top_left_to_slot_top_left_ratio_w*width+left_top[1]))
        self.abs_adv_2_bottom_right = (int(self.adv_2_bottom_right_to_slot_top_left_ratio_h*height + left_top[0]),
                                       int(self.adv_2_bottom_right_to_slot_top_left_ratio_w*width + left_top[1]))
        self.abs_adv_3_top_left = (int(self.adv_3_top_left_to_slot_top_left_ratio_h*height + left_top[0]),
                                   int(self.adv_3_top_left_to_slot_top_left_ratio_w*width+left_top[1]))
        self.abs_adv_3_bottom_right = (int(self.adv_3_bottom_right_to_slot_top_left_ratio_h*height + left_top[0]),
                                       int(self.adv_3_bottom_right_to_slot_top_left_ratio_w*width + left_top[1]))
        self.abs_enchant_n_top_left = (int(self.enchant_n_top_left_to_slot_top_left_ratio_h*height + left_top[0]), 
                                       int(self.enchant_n_top_left_to_slot_top_left_ratio_w*width+left_top[1]))
        self.abs_enchant_n_bottom_right = (int(self.enchant_n_bottom_right_to_slot_top_left_ratio_h*height + left_top[0]),
                                           int(self.enchant_n_bottom_right_to_slot_top_left_ratio_w*width + left_top[1]))

    def analyze(self, img:np.array):
        opt_color_average = np.zeros((5,10,3)) # For debugging
        options = np.zeros((5),dtype=int)
        for i in range(10):
            for j in range(5):
                opt_color_average[j,i] = np.mean(img[self.opt_color_pos[j,i,0]:self.opt_color_pos[j,i,1], 
                                                     self.opt_color_pos[j,i,2]:self.opt_color_pos[j,i,3]], axis=(0,1))
        for i, o in enumerate(opt_color_average):
            for opt_color in o:
                if opt_color[0] > 150:
                    options[i] += 1
                else:
                    break
        
        prob_color_average = np.zeros((5,3))
        for i in range(5):
            prob_color_average[i] = np.mean(img[self.prob_pos[i,0]:self.prob_pos[i,1], 
                                                self.prob_pos[i,2]:self.prob_pos[i,3]], axis=(0,1))
        red_blue_ratrio = prob_color_average[:,0]/prob_color_average[:,2]
        is_avail = np.ones((5), dtype=bool)
        for i, r in enumerate(red_blue_ratrio):
            if r > 1.3:
                is_avail[i] = False\
        
        order_color_average = np.zeros((3,3,3))
        chaos_color_average = np.zeros((3,6,3))
        for i in range(3):
            for j in range(3):
                order_color_average[i,j] = np.mean(img[self.order_color_pos[i,j,0]:self.order_color_pos[i,j,1], 
                                                       self.order_color_pos[i,j,2]:self.order_color_pos[i,j,3]], axis=(0,1))
            for j in range(6):
                chaos_color_average[i,j] = np.mean(img[self.chaos_color_pos[i,j,0]:self.chaos_color_pos[i,j,1], 
                                                       self.chaos_color_pos[i,j,2]:self.chaos_color_pos[i,j,3]], axis=(0,1))
        opt_gauge = np.zeros((3), dtype=int)
        for i in range(3):
            for o_c in order_color_average[i]:
                if o_c[1] > 170:
                    opt_gauge[i] += 1
                else:
                    break
            for c_c in chaos_color_average[i]:
                if c_c[0] > 170:
                    opt_gauge[i] -= 1
                else:
                    break


        adv_1_image = img[self.abs_adv_1_top_left[0]:self.abs_adv_1_bottom_right[0],
                          self.abs_adv_1_top_left[1]:self.abs_adv_1_bottom_right[1]]
        adv_2_image = img[self.abs_adv_2_top_left[0]:self.abs_adv_2_bottom_right[0],
                          self.abs_adv_2_top_left[1]:self.abs_adv_2_bottom_right[1]]
        adv_3_image = img[self.abs_adv_3_top_left[0]:self.abs_adv_3_bottom_right[0],
                          self.abs_adv_3_top_left[1]:self.abs_adv_3_bottom_right[1]]
        
        adv_1_img_tensor = torch.from_numpy(adv_1_image).permute(2,0,1).unsqueeze(0).float()/255
        adv_2_img_tensor = torch.from_numpy(adv_2_image).permute(2,0,1).unsqueeze(0).float()/255
        adv_3_img_tensor = torch.from_numpy(adv_3_image).permute(2,0,1).unsqueeze(0).float()/255
        adv_images = torch.cat((adv_1_img_tensor, adv_2_img_tensor, adv_3_img_tensor), dim=0)
        with torch.no_grad():
            adv_pred = self.advice_model(adv_images).argmax(dim=1).numpy()
            opt_one_pred = self.advice_one_model(adv_images).argmax(dim=1).numpy()
            opt_two_pred_1 = self.advice_two_model(adv_images)[:,:44].argmax(dim=1).numpy()
            opt_two_pred_2 = self.advice_two_model(adv_images)[:,44:].argmax(dim=1).numpy()
        
        enchant_n_img = img[self.abs_enchant_n_top_left[0]:self.abs_enchant_n_bottom_right[0],
                            self.abs_enchant_n_top_left[1]:self.abs_enchant_n_bottom_right[1]]
        enchant_n_img_tensor = torch.from_numpy(enchant_n_img).permute(2,0,1).unsqueeze(0).float()/255
        with torch.no_grad():
            enchant_n_pred = self.enchant_n_model(enchant_n_img_tensor).argmax(dim=1).numpy()
        
        return options, is_avail, opt_gauge, adv_pred, opt_one_pred, opt_two_pred_1, opt_two_pred_2, enchant_n_pred
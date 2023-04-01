import pandas as pd
import imgkit
import tqdm
from pathlib import Path
import multiprocessing as mp

def per_image(font_size, font, img_i, is_main):
    id_to_msg = pd.read_csv('data/id_to_message.csv')
    options = pd.read_csv('data/options.csv')
    option_names = [t for t in options['option_name']]

    save_dir = Path('data/advice_gen')

    no_option_advice_idx = []
    one_option_advice_idx = []
    two_option_advice_idx = []
    for i in range(132):
        if '{1}' in id_to_msg['Desc1'].iloc[i]:
            two_option_advice_idx.append(i)
        elif '{0}' in id_to_msg['Desc1'].iloc[i]:
            one_option_advice_idx.append(i)
        else:
            no_option_advice_idx.append(i)

    html_options = {
        'width': 300,
        'height': 70,
        'quality' : 40,
        'quiet': '',
    }

    # font_sizes = ['12','13','14']
    # fonts=['malgun gothic','batang','gulim','Gungsuh', 'nanumgothic']
    desc_types = ['Desc1', 'Desc2', 'Desc3']

    with open('formats/advice.html', 'r') as f:
        original_html = f.read()
    # if is_main:
    #     t_font_sizes =tqdm.tqdm(font_sizes, ncols=80, desc='font_sizes')
    # else:
    #     t_font_sizes = font_sizes
    # for font_size in t_font_sizes:
    # if is_main:
    #     t_fonts =tqdm.tqdm(fonts, ncols=80, leave=False, desc='fonts')
    # else:
    #     t_fonts = fonts
    # for font in t_fonts:
    if is_main:
        t_desc_types =tqdm.tqdm(desc_types, ncols=80, leave=False, desc='desc_type')
    else:
        t_desc_types = desc_types
    for desc_type in t_desc_types:
        if is_main:
            t_no_option =tqdm.tqdm(no_option_advice_idx, ncols=80, leave=False, desc='no_option')
        else:
            t_no_option = no_option_advice_idx
        for i in t_no_option:
            desc = id_to_msg[desc_type].iloc[i]
            html = original_html.replace('[font_size]', font_size)
            html = html.replace('[font]', font)
            html = html.replace('[advice_img_idx]', str(img_i))
            html = html.replace('[desc]', desc)
            img_name = f'{font_size}_{font}_{img_i}_{desc_type}_{i}_0_0.jpg'
            imgkit.from_string(
                html,
                str(save_dir / str(i) / img_name),
                options=html_options
            )
        if is_main:
            t_one_option =tqdm.tqdm(one_option_advice_idx, ncols=80, leave=False, desc='one_option')
        else:
            t_one_option = one_option_advice_idx
        for i in t_one_option:
            if is_main:
                t_option_names =tqdm.tqdm(option_names, ncols=80, leave=False, desc='option_names')
            else:
                t_option_names = option_names
            for o_i, o_n in enumerate(t_option_names):
                desc = id_to_msg[desc_type].iloc[i]
                html = original_html.replace('[font_size]', font_size)
                html = html.replace('[font]', font)
                html = html.replace('[advice_img_idx]', str(img_i))
                html = html.replace('[desc]', desc)
                html = html.replace('{0}', o_n)
                img_name = f'{font_size}_{font}_{img_i}_{desc_type}_{i}_{o_i}_0.jpg'
                imgkit.from_string(
                    html,
                    str(save_dir / str(i) / img_name),
                    options=html_options
                )
        if is_main:
            t_two_option =tqdm.tqdm(two_option_advice_idx, ncols=80, leave=False, desc='two_option')
        else:
            t_two_option = two_option_advice_idx
        for i in t_two_option:
            if is_main:
                t_option_names =tqdm.tqdm(option_names, ncols=80, leave=False, desc='option_names')
            else:
                t_option_names = option_names
            for o_i, o_n in enumerate(t_option_names):
                if is_main:
                    t_option_names2 =tqdm.tqdm(option_names, ncols=80, leave=False, desc='option_names2')
                else:
                    t_option_names2 = option_names
                for o_i2, o_n2 in enumerate(t_option_names2):
                    if o_i == o_i2:
                        continue
                    desc = id_to_msg[desc_type].iloc[i]
                    html = original_html.replace('[font_size]', font_size)
                    html = html.replace('[font]', font)
                    html = html.replace('[advice_img_idx]', str(img_i))
                    html = html.replace('[desc]', desc)
                    html = html.replace('{0}', o_n)
                    html = html.replace('{1}', o_n2)
                    img_name = f'{font_size}_{font}_{img_i}_{desc_type}_{i}_{o_i}_{o_i2}.jpg'
                    imgkit.from_string(
                        html,
                        str(save_dir / str(i) / img_name),
                        options=html_options
                    )

if __name__ == '__main__':
    save_dir = Path('data/advice_gen')
    save_dir.mkdir(exist_ok=True)
    for i in range(132):
        (save_dir / str(i)).mkdir(exist_ok=True)

    font_sizes = ['12','13','14']
    fonts=['malgun gothic','batang','gulim','Gungsuh', 'nanumgothic']
    processes = []
    for fs in font_sizes:
        for f in fonts:
            for i in range(8):
                p = mp.Process(target=per_image,
                               args=(fs,
                                     f, 
                                     i, 
                                     fs==font_sizes[0] and f==fonts[0] and i==0))
                p.start()
                processes.append(p)
    for p in processes:
        p.join()

#从文字库随机选择10个字符生成图片
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import numpy as np
import os


# 从文字库中随机选择n个字符
def read_source_txt_file(txt_file, code='utf-8-sig'):
    assert os.path.exists(txt_file)
    with open(txt_file, 'rb') as f:
        lines = [line.decode(code, 'ignore') for line in f.readlines()]

    txt_list = [line.replace(' ', '').replace('\t', '').replace('\r', '') .replace('\n', '')
                for line in lines]
    txt_list = list(set(txt_list))
    return ''.join(txt_list)


def get_str_from_source_list(txt_list, count=10):
    assert count > 0
    assert len(txt_list) > count+1

    start = random.randint(0, len(txt_list)-count-1)
    end = start + count
    return txt_list[start:end]


# 随机选取文字贴合起始的坐标, 根据背景的尺寸和字体的大小选择
def random_font_position(background_size, font_size):
    width, height = background_size

    # 为防止文字溢出图片，x，y要预留宽高
    x = random.randint(0, width-font_size*10)
    y = random.randint(0, int((height-font_size)/4))

    return x, y


def random_font_size():
    return random.randint(24, 27)


def random_font_type(font_file_path):
    font_file_list = os.listdir(font_file_path)
    random_font_file = random.choice(font_file_list)
    return font_file_path + random_font_file


def random_color():
    color_list = [[54, 54, 54], [54, 54, 54], [105, 105, 105]]
    color = random.choice(color_list)

    noise_color = np.array([random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)])
    color = (np.array(color) + noise_color).tolist()

    return tuple(color)


# 生成一张图片
def create_random_background_image(background_sample_path, width, height):
    background_file_list = os.listdir(background_sample_path)
    background_choice = random.choice(background_file_list)
    background_img = Image.open(background_sample_path+background_choice)

    x = random.randint(0, background_img.size[0]-width)
    y = random.randint(0, background_img.size[1]-height)
    return background_img.crop((x, y, x+width, y+height))


# 模糊函数
def smooth_image(img):
    # 随机选取模糊参数
    random_filter = random.choice([ImageFilter.SMOOTH,
                                  ImageFilter.SMOOTH_MORE,
                                  ImageFilter.GaussianBlur(radius=1.3)])
    return img.filter(random_filter)


def create_random_image(random_word):
    # 生成一张背景图片，已经剪裁好，宽高为32*280
    raw_image = create_random_background_image('.\\background\\', 280, 32)

    # 随机选取字体大小
    font_size = random_font_size()

    # 随机选取字体
    font_type = random_font_type('.\\font\\')

    # 随机选取字体颜色
    font_color = random_color()

    # 随机选取文字贴合的坐标 x,y
    draw_x, draw_y = random_font_position(raw_image.size, font_size)

    # 将文本贴到背景图片
    font = ImageFont.truetype(font_type, font_size)
    draw = ImageDraw.Draw(raw_image)
    draw.text((draw_x, draw_y), random_word, fill=font_color, font=font)

    # 随机选取作用函数和数量作用于图片
    return smooth_image(raw_image)


def create_data(source_txt_file, save_path, image_file, label_file):
    assert os.path.exists(source_txt_file)
    assert os.path.exists(save_path)

    # 随机选取10个字符
    txt_list = read_source_txt_file(source_txt_file)
    random_word = get_str_from_source_list(txt_list)
    random_word = ''.join(random_word)
    # print(random_word)

    raw_image = create_random_image(random_word)

    # 保存文本信息和对应图片名称
    with open(label_file, 'a+', encoding='utf-8') as f:
        f.write(image_file + '\t' + random_word + '\n')
    raw_image.save(save_path+image_file)


if __name__ == '__main__':
    source_txt_file = './/source_small.txt'
    img_file_count = 10
    img_save_path = '.\\data_set\\'
    label_file = '.\\label.txt'

    for num in range(img_file_count):
        img_file = str(num) + '.png'
        create_data(source_txt_file, img_save_path, img_file, label_file)
    print('ok')



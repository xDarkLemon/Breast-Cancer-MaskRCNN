import pydicom
import matplotlib.pyplot as plt
import scipy.misc
from skimage import io, transform, feature
from PIL import Image
import os
import re
import numpy as np

def read_files():
    dataset_dir = '/home/liujing/Desktop/mammograph/CBIS-DDSM'
    save_dir = 'dataset/mammo_all/'
    img_re = re.compile(r'(Mass|Calc)-(Training|Test).+?(CC|MLO)/')
    mask_re = re.compile(r'(Mass|Calc)-(Training|Test).+?(CC|MLO)_\d') # 1应该改成\d，匹配数字，因为有些图片对应不止一个病灶
    cnt_img = 0
    cnt_mask = 0
    for root, dirs, files in os.walk(dataset_dir, topdown=True):
        for name in files:
            path = os.path.join(root, name)
            img_mo = img_re.search(path)
            mask_mo = mask_re.search(path)
            if img_mo:
                dcm = pydicom.read_file(path)
                img = dcm.pixel_array
                img_name = img_mo.group().rstrip('/')
                if 'Mass-Training' in img_name:
                    subdir = 'mass_train/'
                elif 'Mass-Test' in img_name:
                    subdir = 'mass_test/'
                elif 'Calc-Training' in img_name:
                    subdir = 'calc_train/'
                elif 'Calc-Test' in img_name:
                    subdir = 'calc_test/'
                else:
                    print('unrecognized name:', img_name)
                    continue
                save_path = save_dir + subdir + img_name + '/full_image'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                file_path = save_path + '/' + img_name + '.png'
                cnt_img += 1
                if not os.path.exists(file_path):
                    scipy.misc.imsave(file_path, img)
                    print(cnt_img, 'img saved: ' + file_path)
                else:
                    print(cnt_img, 'img exist: ' + file_path)
            elif mask_mo:  
                # continue
                if is_mask(path):
                    dcm = pydicom.read_file(path)
                    mask = dcm.pixel_array
                    file_name = mask_mo.group()[:-2] # 去掉 _数字
                    if 'Mass-Training' in file_name:
                        subdir = 'mass_train/'
                    elif 'Mass-Test' in file_name:
                        subdir = 'mass_test/'
                    elif 'Calc-Training' in file_name:
                        subdir = 'calc_train/'
                    elif 'Calc-Test' in file_name:
                        subdir = 'calc_test/'
                    else:
                        print('unrecognized name:', file_name)
                        continue
                    save_path = save_dir + subdir + file_name + '/masks' 
                    mask_name = mask_mo.group()
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    file_path = save_path + '/' + mask_name + '.png'
                    if not os.path.exists(file_path):
                        scipy.misc.imsave(file_path, mask)
                        cnt_mask += 1
                        print(cnt_mask, 'mask saved: ' + file_path)

def is_mask(file):
    '''
    Judge if a picture is a mask or a photo. A mask is usually less than 1000 * 1000 in size.
    :param file: img file path
    :return: Boolean
    '''
    dcm = pydicom.read_file(file)
    if dcm.pixel_array.shape[0] > 1000:
        return True
    else:
        return False
            
def dcm2img(file, save_file):
    dcm = pydicom.read_file(file)
    img = dcm.pixel_array
    scipy.misc.imsave(save_file + '.jpg', img)
    return img

def show_dcm(dcm):
    # dcm = pydicom.read_file(file)
    print(dcm)
    print('shape:')
    print(dcm.pixel_array.shape)
    print(dcm.pixel_array.shape[0])
    print(dcm.pixel_array.shape[1])
    plt.imshow(dcm.pixel_array, 'gray')
    # plt.title(file)
    plt.show()

def show_img(img):
    '''
    # print(img.shape[0])  # 图片宽度
    # print(img.shape[1])  # 图片高度
    # print(img.shape[2])  # 图片通道数
    :param img: 
    :return: 
    '''
    # img = io.imread(file)
    print('''
    Type: {}
    Shape: {}
    Size: {}
    Max pixel: {}
    Min pixel: {}
    Mean pixel: {}
    '''.format(type(img), img.shape, img.size, img.max(),
                   img.min(), img.mean()))
    plt.imshow(img, cmap='gray')
    plt.show()

def gray2rgb(img):
    img = img[:, :, np.newaxis]
    img = img.tolist()
    for i in range(len(img)):
        for j in range(len(img[0])):
            img[i][j].append(img[i][j][0])
            img[i][j].append(img[i][j][0])
    img = np.array(img)
    return img

def gray2binary(img):
    if len(img.shape) == 2:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] > 0.5:
                    img[i][j] = 1
                else:
                    img[i][j] = 0
    return img

def normalize(x):
    return x / 255

def add_channel(img):
    img = img[:, :, np.newaxis]
    return img

def plot_contour(img_rgb, contour):
    '''
    Plot contour on the img. Need a contour mask.
    :param img_rgb: 3 channels
    :param contour: binary img
    :return: img with contour in red
    '''
    img_rgb = img_rgb.tolist()
    for i in range(len(img_rgb)):
        for j in range(len(img_rgb [0])):
            if contour[i][j] == 1:
                img_rgb[i][j][0] = 255 # red channel
                img_rgb[i][j][1] = 0 # green channel
                img_rgb[i][j][2] = 0 # blue channel
    img_rgb = np.array(img_rgb)
    return img_rgb

def get_contour(img):
    '''
    Use Canny filter to select the edge
    :param img: 
    :return: 
    '''
    edge = feature.canny(img, sigma=14) # default sigma = 1
    plt.imshow(edge, cmap=plt.cm.gray)
    plt.show()
    scipy.misc.imsave('data\\21_c_rgb_contour2.jpg', edge)
    return edge

def rotate_all(): 
    '''
    Rotate all jpg in train file unclockwise 90 degree
    TODO: Imcomplete
    :return: 
    '''
    for root, dirs, files in os.walk("data\\train", topdown=True):
        for name in files:
            path = os.path.join(root, name)
            img = io.imread(path)
            img = transform.rotate(img, 90, resize=True)
    return

def gray2rgb_all(dir):
    for root, dirs, files in os.walk(dir, topdown=True):
        for name in files:
            path = os.path.join(root, name)
            print(path)
            img = io.imread(path)

def get_max_lw(file):
    max_l = 0
    max_w = 0
    for root, dirs, files in os.walk(file, topdown=True):
        for name in files:
            path = os.path.join(root, name)
            img = io.imread(path)
            if img.shape[0] > img.shape[1]:
                l = img.shape[0]
                w = img.shape[1]
            else:
                l = img.shape[1]
                w = img.shape[0]
            if w > max_w:
                max_w = w
            if l > max_l:
                max_l = l
    return max_l, max_w

def padding(img, resize):
    height, width = img.shape[0:2]
    pad_w = max(resize[1] - width, 0)
    pad_h = max(resize[0] - height, 0)
    img_padded = np.lib.pad(img, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
                               'constant', constant_values=0) # 0黑色
    return img_padded

def mask_to_image(mask):
    mask_img =  (mask * 255).astype(np.uint8)
    return mask_img
    # return Image.fromarray((mask * 255).astype(np.uint8))

def prob_to_mask(mask_prob, out_threshold):
    mask = np.zeros_like(mask_prob)
    for i in range(mask_prob.shape[0]):
        for j in range(mask_prob.shape[1]):
            if mask_prob[i][j] > out_threshold:
                mask[i][j] = 1
            else:
                mask[i][j] = 0
    return mask

def resize(original_img, mask_img):
    '''
    resize mask img to original size.
    :param original_img: 
    :param mask_img: 
    :return: 
    '''
    img = original_img.squeeze()
    w = img.shape[0]
    h = img.shape[1]
    mask_img_resized = transform.resize(mask_img, (w, h))
    return mask_img_resized

if __name__ == '__main__':
    read_files()

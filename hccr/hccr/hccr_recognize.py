import tensorflow as tf
from PIL import Image, ImageFilter
from io import BytesIO
import numpy as np
import pickle
import hccr.cnn_model

model = None
response_time = 0
sess = None

def recognize_character(img_str):
    img_arr = preprocess(img_str)
    if img_arr is None:
        return None

    result = predict(img_arr)
    val = result[0][0]
    index = result[1][0]

    # 标签对应字符
    char_dict = {}
    with open("hccr/model/char_dict", "rb") as f:
        char_dict = pickle.load(f)

    character = []
    for i in range(len(index)):
        character.append(char_dict[index[i]])

    print(val)
    print(character)
    return character, val.tolist()

def preprocess(img_str):
    try:
        # 将数据解码成pil的image
        img = Image.open(BytesIO(img_str)).convert('L')
    except:
        return None

    # test
    img.save("hccr/tmp/out.bmp")

    # 找出数字所在区域的最小外接矩形
    bbox = Image.eval(img, lambda px: 255-px).getbbox()
    if bbox is None:
        return None

    # 将区域按比例缩小
    widthlen = bbox[2] - bbox[0]
    heightlen = bbox[3] - bbox[1]

    if(heightlen > widthlen):
        widthlen = int(56.0*widthlen/heightlen)
        heightlen = 56
    else:
        heightlen = int(56.0*heightlen/widthlen)
        widthlen = 56
    hstart = int((64 - heightlen)/2)
    wstart = int((64 - widthlen)/2)

    # TODO(ww): 这里nearest参数效果比默认参数好，还需探究
    img = img.crop(bbox).resize((widthlen, heightlen), Image.NEAREST)

    smallImg = Image.new('L', (64,64), 255)
    smallImg.paste(img, (wstart, hstart))
    smallImg = smallImg.filter(ImageFilter.MinFilter)
    smallImg.save("hccr/tmp/filter.bmp")

    imgdata = list(smallImg.getdata())
    # 需要改为背景为0，前景为1
    imgdata = [(255.0-x)/255.0 for x in imgdata]
    imgdata = np.array(imgdata)
    imgdata = np.reshape(imgdata, (1, 64, 64, 1))

    return imgdata


def predict(img_arr):
    global model, response_time, sess

    # 为了不让每次post都重新加载模型
    if response_time == 0:
        predict_init()
        response_time = 1

    val, index = sess.run([model.val_top_k, model.index_top_k], feed_dict={model.images: img_arr})
    return (val, index)


def predict_init():
    global model, sess
    model = hccr.cnn_model.model('hccr/model/model.bin')
    model.build()

    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)

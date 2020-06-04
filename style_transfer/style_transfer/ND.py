

from __future__ import print_function

import time

import numpy as np

from scipy.optimize import fmin_l_bfgs_b

from scipy.misc import imread, imsave,imresize

 

from keras import backend as K

from keras.layers import Input, AveragePooling2D

from keras.models import Model

from keras.preprocessing.image import load_img, img_to_array

from keras.applications import vgg19

#import pandas as pd

from config import CONFIG




# 读取/处理图像的辅助功能

def preprocess_image(image_path,img_nrows,img_ncols):

    img = load_img(image_path, target_size=(img_nrows, img_ncols))

    img = img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = vgg19.preprocess_input(img)

    return img

 

 

def deprocess_image(x,img_nrows,img_ncols):

    if K.image_data_format() == 'channels_first':

        x = x.reshape((3, img_nrows, img_ncols))

        x = x.transpose((1, 2, 0))

    else:

        x = x.reshape((img_nrows, img_ncols, 3))


    x[:, :, 0] += 103.939

    x[:, :, 1] += 116.779

    x[:, :, 2] += 123.68



    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')

    return x

 

 

def eval_loss_and_grads(x,img_nrows,img_ncols,f_outputs):

    if K.image_data_format() == 'channels_first':

        x = x.reshape((1, 3, img_nrows, img_ncols))

    else:

        x = x.reshape((1, img_nrows, img_ncols, 3))

    #print([x])


    outs = f_outputs([x])

    loss_value = outs[0]

    if len(outs[1:]) == 1:

        grad_values = outs[1].flatten().astype('float64')

    else:

        grad_values = np.array(outs[1:]).flatten().astype('float64')

    return loss_value, grad_values    



def kmeans(xs):

    #assert xs.ndim == 2

    k=CONFIG.num_labels

    try:

        from sklearn.cluster import k_means

        _, labels, _ = k_means(xs.astype('float64'), k)

    except ImportError:

        from scipy.cluster.vq import kmeans2

        _, labels = kmeans2(xs, k, missing='raise')

    return labels

 

 

def load_mask_labels(target_mask_path,style_mask_path,img_nrows,img_ncols):



    target_mask_img = load_img(target_mask_path, target_size=(img_nrows, img_ncols))#kmeans预处理

    target_mask_img = img_to_array(target_mask_img)

    style_mask_img = load_img(style_mask_path, target_size=(img_nrows, img_ncols))

    style_mask_img = img_to_array(style_mask_img)

    if K.image_data_format() == 'channels_first':

        mask_vecs = np.vstack([style_mask_img.reshape((3, -1)).T, target_mask_img.reshape((3, -1)).T])

    else:

        mask_vecs = np.vstack([style_mask_img.reshape((-1, 3)), target_mask_img.reshape((-1, 3))])

 

    labels = kmeans(mask_vecs)

    style_mask_label = labels[:img_nrows * img_ncols].reshape((img_nrows, img_ncols))#展开label

    target_mask_label = labels[img_nrows * img_ncols:].reshape((img_nrows, img_ncols))

 

    stack_axis = 0 if K.image_data_format() == 'channels_first' else -1

    style_mask = np.stack([style_mask_label == r for r in range(CONFIG.num_labels)], axis=stack_axis)#对应存入各个mask矩阵

    target_mask = np.stack([target_mask_label == r for r in range(CONFIG.num_labels)], axis=stack_axis)

 

    return (np.expand_dims(style_mask, axis=0),

            np.expand_dims(target_mask, axis=0))






def gram_matrix(x):

    assert K.ndim(x) == 3

    features = K.batch_flatten(x)

    gram = K.dot(features, K.transpose(features))

    return gram

 

 

def region_style_loss(style_image, target_image, style_mask, target_mask):



    #assert 3 == K.ndim(style_image) == K.ndim(target_image)

    #assert 2 == K.ndim(style_mask) == K.ndim(target_mask)

    if K.image_data_format() == 'channels_first':

        masked_style = style_image * style_mask

        masked_target = target_image * target_mask#元素乘积取区域

        num_channels = K.shape(style_image)[0]

    else:

        masked_style = K.permute_dimensions(style_image, (2, 0, 1)) * style_mask

        masked_target = K.permute_dimensions(target_image, (2, 0, 1)) * target_mask

        num_channels = K.shape(style_image)[-1]

    num_channels = K.cast(num_channels, dtype='float32')

    s = gram_matrix(masked_style) / K.mean(style_mask) / num_channels

    c = gram_matrix(masked_target) / K.mean(target_mask) / num_channels

    return K.mean(K.square(s - c))

 

 

def style_loss(style_image, target_image, style_masks, target_masks):


    #assert 3 == K.ndim(style_image) == K.ndim(target_image)

    #assert 3 == K.ndim(style_masks) == K.ndim(target_masks)

    loss = K.variable(0)

    for i in range(CONFIG.num_labels):

        if K.image_data_format() == 'channels_first':

            style_mask = style_masks[i, :, :]

            target_mask = target_masks[i, :, :]#按颜色取mask

        else:

            style_mask = style_masks[:, :, i]

            target_mask = target_masks[:, :, i]

        loss =loss + region_style_loss(style_image, target_image, style_mask, target_mask)

    return loss

 


 

 

def total_variation_loss(x,img_nrows,img_ncols):

    #assert 4 == K.ndim(x)

    if K.image_data_format() == 'channels_first':

        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])

        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])

    else:

        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])

        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])

    return K.sum(K.pow(a + b, 1.25))





def nerual_doodle(style_img_path,style_mask_path,target_mask_path):
    
    
    class Evaluator(object):     
        def __init__(self):        
            self.loss_value = None        
            self.grads_values = None     
        def loss(self, x):        
            assert self.loss_value is None        
            loss_value, grad_values = eval_loss_and_grads(x,img_nrows,img_ncols,f_outputs)        
            self.loss_value = loss_value        
            self.grad_values = grad_values        
            return self.loss_value     
        def grads(self, x):        
            assert self.loss_value is not None        
            grad_values = np.copy(self.grad_values)        
            self.loss_value = None        
            self.grad_values = None        
            return grad_values
      
    
    
    # 基于目标掩码确定图像尺寸
    
    ref_img = imread(target_mask_path)
    
    img_nrows, img_ncols = ref_img.shape[:2]
    
     
    #50.0
    total_variation_weight = 50.
    
    #style_weight = 1.
    
        
    # 为了获得更好的生成质量，使用更多的卷积层用于风格特征。
    
 #   style_feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

 
     
    
    
    # 建立图像张量变量
    
    if K.image_data_format() == 'channels_first':
    
        shape = (1, CONFIG.COLOR_CHANNELS, img_nrows, img_ncols)
    
    else:
    
        shape = (1, img_nrows, img_ncols, CONFIG.COLOR_CHANNELS)
    
     
    
    style_image = K.variable(preprocess_image(style_img_path,img_nrows,img_ncols))
    
    target_image = K.placeholder(shape=shape)
    
    
     
    
    images = K.concatenate([style_image, target_image], axis=0)
    
     
    
    
    # 建立掩码张量变量
    
    raw_style_mask, raw_target_mask = load_mask_labels(target_mask_path,style_mask_path,img_nrows,img_ncols)
    
    style_mask = K.variable(raw_style_mask.astype('float32'))
    
    target_mask = K.variable(raw_target_mask.astype('float32'))
    
    masks = K.concatenate([style_mask, target_mask], axis=0)
    
     
    

    
    # 图像和任务变量的索引常量
    
    STYLE, TARGET = 0, 1
    
     
    
    
    # 建立图像模型、掩模模型和使用层输出作为特征图像模型VGG19
    
    image_model = vgg19.VGG19(include_top=False, input_tensor=images)
    
     
    
    
    # 掩模模型作为一系列池化，保证张量大小一致
    
    mask_input = Input(tensor=masks, shape=(None, None, None), name='mask_input')
    
    x = mask_input
    
    for layer in image_model.layers[1:]:#每一层卷积进行一次3*3池化，每一层池化进行一次2*2池化
    
        name = 'mask_%s' % layer.name
    
        if 'conv' in layer.name:
    
            x = AveragePooling2D((3, 3), padding='same', strides=(1, 1), name=name)(x)
    
        elif 'pool' in layer.name:
    
            x = AveragePooling2D((2, 2), name=name)(x)
    
    mask_model = Model(mask_input, x)
    
     
    
    # 从图像模型和掩模模型收集特征
    
    image_features = {}
    
    mask_features = {}
    
    for img_layer, mask_layer in zip(image_model.layers, mask_model.layers):#卷积出特征
    
        if 'conv' in img_layer.name:
    
            #assert 'mask_' + img_layer.name == mask_layer.name
    
            layer_name = img_layer.name
    
            img_feat, mask_feat = img_layer.output, mask_layer.output
    
            image_features[layer_name] = img_feat
    
            mask_features[layer_name] = mask_feat

 
 
    
    loss = K.variable(0)
     
     
    
    for layer,coeff in CONFIG.STYLE_LAYERS_2:#按层算损失
    
        style_feat = image_features[layer][STYLE, :, :, :]
    
        target_feat = image_features[layer][TARGET, :, :, :]
    
        style_masks = mask_features[layer][STYLE, :, :, :]
    
        target_masks = mask_features[layer][TARGET, :, :, :]
    
        sl = style_loss(style_feat, target_feat, style_masks, target_masks)
    
        loss =loss + coeff * sl
    
     
    
    loss =loss + total_variation_weight * total_variation_loss(target_image,img_nrows,img_ncols)
    
    loss_grads = K.gradients(loss, target_image)
    
     
    
    # 计算效率的评估类
    
    outputs = [loss]
    
    if isinstance(loss_grads, (list, tuple)):
    
        outputs =outputs + loss_grads
    
    else:
    
        outputs.append(loss_grads)
    
     
    
    f_outputs = K.function([target_image], outputs)

 

    
    evaluator = Evaluator()
    
     
    
    
    # 通过迭代优化生成图像
    
    if K.image_data_format() == 'channels_first':
    
        x = np.random.uniform(0, 255, (1, 3, img_nrows, img_ncols)) - 128.
    
    else:
    
        x = np.random.uniform(0, 255, (1, img_nrows, img_ncols, 3)) - 128.
    
#    list_total=[]
#    list_time=[]
#    start_time = time.time()
    for i in range(CONFIG.num_iterations_2+1):
    
        print('Start of iteration', i)
    
        
    
        #print(evaluator.loss)
       # print(x.flatten())
        #print(fprime=evaluator.grads)
    
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
    
        #print('Current loss value:', min_val)
    
        # save current generated image
    
        # 保存生成图像
    
        img = deprocess_image(x.copy(),img_nrows,img_ncols)
    
        
    
        if i%20==0:
     #       list_total.append(min_val)
            fname = "D:\华师工程中心\style_transfer\style_transfer\generated" + '_at_iteration_%d.png' % i
            imsave(fname, img)
           # print('Image saved as', fname)
         #   end_time = time.time()
        #    iteration_time=end_time - start_time
          #  print('Iteration %d completed in %ds' % (i, iteration_time))
         #   list_time.append(iteration_time)
          #  start_time = time.time()
    

    imsave('D:\华师工程中心\style_transfer\style_transfer\generated_image.png', img)
#    dict={'total':list_total,"time":list_time}
 #   cost=pd.DataFrame(dict)
 #   cost.to_csv('D:/py/cost/cost.csv',encoding='utf-8')
    return img


#nerual_doodle("C:/Users/zmy/Desktop/sc2.png",
#              "C:/Users/zmy/Desktop/sc22.png",
#              "C:/Users/zmy/Desktop/sc23.png")
#nerual_doodle("C:/Users/zmy/Desktop/py2/Monet/style.png","C:/Users/zmy/Desktop/py2/Monet/style_mask.png","C:/Users/zmy/Desktop/py2/Monet/target_mask.png")
#nerual_doodle("C:/Users/zmy/Desktop/scenery.jpg","C:/Users/zmy/Desktop/scenery_json/label.png","C:/Users/zmy/Desktop/new.jpg")
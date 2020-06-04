# -*- coding: utf-8 -*-


import scipy.io
import scipy.misc
import tensorflow as tf
#import time
#import pandas as pd
import numpy as np

from config import CONFIG

    
def load_vgg_model(path):
 
    vgg = scipy.io.loadmat(path)

    vgg_layers = vgg['layers']
    
    def _weights(layer, expected_layer_name):

        wb = vgg_layers[0][layer][0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b

        return W, b

    def _relu(conv2d_layer):

        return tf.nn.relu(conv2d_layer)

    def _conv2d(prev_layer, layer, layer_name):

        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def _conv2d_relu(prev_layer, layer, layer_name):

        return _relu(_conv2d(prev_layer, layer, layer_name))

    def _avgpool(prev_layer):

        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Constructs the graph model.
    graph = {}
    graph['input']   = tf.Variable(np.zeros((1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)), dtype = 'float32')
    graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    
    return graph


def generate_noise_image(content_image):
    noise_image = np.random.uniform(-20, 20, (1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)).astype('float32')
    input_image = noise_image * CONFIG.NOISE_RATIO + content_image * (1 - CONFIG.NOISE_RATIO)
    return input_image


def reshape_and_normalize_image(image):
    image=scipy.misc.imresize(image,size=(CONFIG.IMAGE_HEIGHT,CONFIG.IMAGE_WIDTH))
    image = np.reshape(image, ((1,) + image.shape))
    image = image - CONFIG.MEANS
    return image


def save_image(path, image):
    image = image + CONFIG.MEANS
    image = np.clip(image[0], 0, 255).astype('uint8')#存储时选择int8，计算时选择float32
   
    
                   
    scipy.misc.imsave(path, image)

def compute_content_loss(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_C_unrolled = tf.reshape(tf.transpose(a_C, perm=[3, 2, 1, 0]), [n_C, n_H*n_W, -1])
    a_G_unrolled = tf.reshape(tf.transpose(a_G, perm=[3, 2, 1, 0]), [n_C, n_H*n_W, -1])#展开a_C和a_G
    
    content_loss = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))) / (4 * n_H * n_W * n_C)
    
    return content_loss

def gram_matrix(A): 
    GA = tf.matmul(A,tf.transpose(A)) 
    return GA

def compute_layer_style_loss(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_S = tf.reshape(tf.transpose(a_S, perm=[3, 1, 2, 0]), [n_C, n_W*n_H])
    a_G = tf.reshape(tf.transpose(a_G, perm=[3, 1, 2, 0]), [n_C, n_W*n_H])
    
    Gram_S = gram_matrix(a_S)
    Gram_G = gram_matrix(a_G)
    
    style_layer_loss = tf.reduce_sum(tf.square(tf.subtract(Gram_S, Gram_G))) / (4 * n_C**2 * (n_W * n_H)**2)
    
    return style_layer_loss

def compute_style_loss(sess,model):
    style_loss = 0
    
    
    
    for layer_name, coeff in CONFIG.STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        style_layer_loss = compute_layer_style_loss(a_S, a_G)
        style_loss += coeff * style_layer_loss
    
    
    return style_loss

def compute_total_loss(content_loss, style_loss):
    total_loss = CONFIG.alpha*content_loss + CONFIG.beta*style_loss
    return total_loss

def iteration(sess, input_image, train_step,model,total_loss, content_loss, style_loss):
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))
  #  list_content=[]
 #   list_style=[]
  #  list_total=[]
    #start_time = time.time()
    
    for i in range(CONFIG.num_iterations_1+1):
        
        
        sess.run(train_step)
        
        
        generated_image = sess.run(model['input'])
       # if i%100 == 0 and i!=0:
      #      total, content, style = sess.run([total_loss, content_loss, style_loss])
       #     list_content.append(content)
       #     list_style.append(style)
       #     list_total.append(total)
        #if i%25 == 0:
        print("Iteration " + str(i) + " :")
        save_image("D:\华师工程中心\style_transfer\style_transfer\output/" + str(i) + ".png", generated_image)
    
    
    save_image('D:\华师工程中心\style_transfer\style_transfer/output/generated_image.png', generated_image)
    #print('Iteration completed in s')
    #print(end_time - start_time)
    #dict={'content':list_content,'style':list_style,'total':list_total}
    #cost=pd.DataFrame(dict)
    #cost.to_csv('D:/py/cost/transfer_cost.csv',encoding='utf-8')
    return generated_image

def NST(content_image_path,style_image_path):
    
#    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    content_image = scipy.misc.imread(content_image_path)
    content_image = reshape_and_normalize_image(content_image)
    style_image = scipy.misc.imread(style_image_path)
    style_image = reshape_and_normalize_image(style_image)
    
    generated_image = generate_noise_image(content_image)
    
    model = load_vgg_model("D:\华师工程中心\style_transfer\style_transfer/pretrained-model/imagenet-vgg-verydeep-19.mat")
    
    sess.run(model['input'].assign(content_image))
    
    out = model['conv4_2']
    
    a_C = sess.run(out)
    a_G = out
    
    content_loss = compute_content_loss(a_C, a_G)
    sess.run(model['input'].assign(style_image))
    style_loss = compute_style_loss(sess,model)
    total_loss = compute_total_loss(content_loss, style_loss)
    
    optimizer = tf.train.AdamOptimizer(CONFIG.learning_rate)
    train_step = optimizer.minimize(total_loss)
    print(train_step)
    generated_image=iteration(sess, generated_image,train_step,model,total_loss, content_loss, style_loss)
    return generated_image


NST("D:\华师工程中心\style_transfer\style_transfer\images/cat.jpg","D:\华师工程中心\style_transfer\style_transfer\images/sandstone.jpg")
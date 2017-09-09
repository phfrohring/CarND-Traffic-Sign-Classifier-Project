import os
import csv
import re
import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage import transform as tr
from skimage.util import random_noise
from skimage.color import rgb2gray
from skimage import exposure
from sklearn import preprocessing as skpp
import tensorflow as tf
from scipy import misc







def get_data_dir():
    """Return the path of the traffic signs images."""
    return os.path.join('.','traffic-signs-data')



def get_and_split():
    """Return λ : filename → (features,labels), where filename is a file under get_data_dir()."""
    data = {} # Cache.
    
    def helper(filename):
        if filename not in data:
            with open(os.path.join(get_data_dir(),filename), mode='rb') as f:
                data[filename] = pickle.load(f) 
                
        return data[filename]['features'], data[filename]['labels']
    
    return helper
get_and_split = get_and_split()


def get_sign_name():
    """Return λ : code → label, where code is an integer synonym for a traffic sign name."""
    signames_file = os.path.join(get_data_dir(),'signnames.csv')        
    with open(signames_file, newline='') as f:
        reader = csv.reader(f)
        next(reader) # skip first line
        signames = {}
        for row in reader:
            signames[int(row[0])] = row[1]
            
    def helper(code):
        return signames[code]
    
    return helper
get_sign_name = get_sign_name()


def images_distribution(labels):
    """λ : [1,2,2,3,3,3,...] → [(1,1), (2,2), (3,3), ...] : Distribution."""
    counts = {}
    for label in labels:
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1            
    return counts.items()



def plot_barchart_of_distribution(dist,title='Title',x_label="x_label",y_label="y_label"):
    """Plot a bar chart of the Distribution `dist`."""
    sorted_dist = sorted(dist, key=lambda pair: pair[0])
    x = [label for (label,count) in sorted_dist]
    y = [count for (label,count) in sorted_dist]
    plt.figure(figsize=(16,9))
    plt.bar(x, y, align='center', alpha=0.5)
    plt.xticks(x, x)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    
    

def filler_image(ex_image):
    """λ : image → gray image of same dimensions."""
    height,width, *rest = ex_image.shape
    return np.array([[ [128.0,128.0,128.0] for j in range(width)] for i in range(height)])



def graph_images(images,labels=None,cmap=None):
    """Graph a grid of images.
    
    Fill unoccupied spots with gray images.
    """
    n_images = len(images)
    ex_image = images[0]
    height,width, *rest = ex_image.shape
    max_width = 6
    grid_width = n_images if n_images < max_width else max_width
    grid_height = 1 + (n_images-1) // max_width
    
    fig = plt.figure(figsize=(3*grid_width,2*grid_height))
    for i in range(grid_height):
        for j in range(grid_width):
            index = i*grid_width + j
            ax = plt.subplot2grid((grid_height, grid_width), (i, j))
            ax.tick_params(length=0.01,labelsize=0.01)
            
            if index < n_images:
                raw_image = np.squeeze(images[index])
                if labels is not None:
                    label = get_sign_name(labels[index])
                    ax.set_xlabel(label)
            else:
                raw_image = filler_image(ex_image) 
                plt.set_cmap('gray')
            
            if cmap is not None:
                plt.set_cmap(cmap)
            ax.imshow(raw_image,aspect='equal')

            
            
def draw_random_images(images,labels,nbr,select=None):
    """Return `nbr` random images and associated labels and indexes from images.
    
    Return ([image1,...],[label1,...],[index1,...]) all of length `nbr`.
    If `select` is a label, then draw random images from the subset of images with label == select.
    """
    images_labels = zip(images,labels)
    if select is not None:
        idx_images_labels = [(idx,(image,label)) 
                             for (idx,(image,label)) in enumerate(images_labels) 
                             if label == select]
    else:
        idx_images_labels = [(idx,(image,label)) 
                             for (idx,(image,label)) in enumerate(images_labels)]    
    indexes = [ random.randint(0, len(idx_images_labels)) for x in range(0,nbr)]
    random_indexes = np.array([idx_images_labels[i][0] for i in indexes])
    rand_images = np.array([idx_images_labels[i][1][0] for i in indexes])
    rand_labels = np.array([idx_images_labels[i][1][1] for i in indexes])
    return rand_images,rand_labels,random_indexes



def group_by_class(images,labels):
    """Return a dictionary where keys are labels and values or list of matching images.
    
    zip(images,labels) assumed to be of the form [(image,label), ...]
    Return a dictionary of the form : 
        {
              label: [image, ...],
              ...
        }
    """
    class_images = {}
    for idx,label in enumerate(labels):
        if label in class_images:
            class_images[label].append(images[idx])
        else:
            class_images[label] = [images[idx]]       
    for label in class_images.keys():
        class_images[label] = np.array(class_images[label])
    return class_images



def get_data():
    """Return all images and labels as: ([image, ...],[label,...])."""
    X_train, Y_train = get_and_split('train.p')
    X_valid, Y_valid = get_and_split('valid.p')
    X_test, Y_test = get_and_split('test.p')
    
    X = np.append(np.append(X_train,X_valid,axis=0),X_test,axis=0)
    Y = np.append(np.append(Y_train,Y_valid,axis=0),Y_test,axis=0)
    
    return X,Y



def split_data(X,Y):
    """Split X and Y into training, validation and test sets."""
    X_train, X_rest, Y_train, Y_rest = train_test_split(X, Y, test_size=0.30, random_state=42)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X_rest, Y_rest, test_size=0.33, random_state=42)
    
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test



def generate_images(images,n_images):
    """Generate n_images variations from images."""
    new_images = []
    for i in range(n_images):
        rand_img = np.copy(images[random.randint(0,len(images)-1)])
        rand_img = tr.rotate(rand_img,float(random.randint(-15,15))) 
        rand_img = tr.warp(
            rand_img, 
            tr.AffineTransform(translation=(random.randint(-2,2),random.randint(-2,2)))) 
        rand_img = random_noise(rand_img,var=(random.randint(0,99)*(0.00025/99)))
        new_images.append(rand_img)
        
    return np.array(new_images) 



def extend_images_labels(X,Y):
    """Fill X and Y with variations of existing images so that the distribution by class is even."""
    images_dist = images_distribution(Y)
    max_count = max(images_dist,key=lambda x: x[1])[1]
    images_co_dict = dict([(label,max_count - count) for (label,count) in images_dist])
    images_dict = group_by_class(X,Y)

    for label in images_dict.keys():
        n_images = images_co_dict[label]
        images = images_dict[label]
        g_images = generate_images(images, n_images)
        if g_images.shape[0] > 0:
            images_dict[label] = np.append(images,g_images,axis=0)

    X_extended = []
    Y_extended = []
    for label in images_dict.keys():
        images = images_dict[label]
        X_extended.extend(images)
        Y_extended.extend([label for i in range(len(images))])

    X_extended = np.array(X_extended)
    Y_extended = np.array(Y_extended)
    
    return X_extended, Y_extended



def preprocess_images(images,labels):
    
    # Grayscale
    X = np.array([rgb2gray(image) for image in images])
    Y = labels
    
    # Contrast
    X = np.array([exposure.equalize_hist(image) for image in X])
    Y = labels    

    # Scale
    X = np.array([skpp.scale(image,with_std=False,axis=0) for image in X])
    Y = labels
    
    # Reshape to (32,32,1)
    X = np.array([np.reshape(image,(32,32,1)) for image in X])
    Y = labels
 
    return X, Y



def collect_results(training_parameters, X, model, Y):
    """λ : X model Y → [(predicted_class,expected_class), ...]."""
    tf.reset_default_graph()
    
    image_width, image_height, image_depth = X[0].shape
    x = tf.placeholder(tf.float32,(None,image_width,image_height,image_depth))
    keep_prob = tf.placeholder(tf.float32)
    logits, tf_activations = model(x,keep_prob)

    n_classes = len(set(Y))
    y = tf.placeholder(tf.int32,(None))
    one_hot_y = tf.one_hot(y, n_classes)
    
    expected_classes = tf.argmax(one_hot_y, 1)
    predicted_classes = tf.argmax(logits, 1)
        
    predicted_expected_classes = []

    batch_size = training_parameters['batch_size']
    num_examples = len(X)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, training_parameters["save_file"])
        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = X[offset:offset+batch_size], Y[offset:offset+batch_size]
            predictions = sess.run(predicted_classes, feed_dict={x: batch_x, keep_prob: 1.})
            expectations = sess.run(expected_classes, feed_dict={y: batch_y})
            predicted_expected_classes += [(predicted,expected) 
                                           for (predicted,expected) in zip(predictions,expectations)]
    return np.array(predicted_expected_classes)



def to_true_false_positive_negative(results):
    """λ : [(predicted_class,expected_class),...] → Tensor of shape (43,4).
    
    Each row matches a class and columns:
        True Positive, True Negative, False Positive, False Negative
        
    If predicts i and expects i then:
        class i has 1 more true positive
        all other classes have 1 more true negative
    If predicts i and expects j ≠ i then:
        class i has 1 more false positive
        class j has 1 more false negative
        all other classes have 1 more true negative    
    """
    tp = 0; tn = 1; fp = 2; fn = 3
    n_expected_classes = len(set([expectation for (prediction,expectation) in results]))
    n_predicted_classes = len(set([prediction for (prediction,expectation) in results]))
    assert n_predicted_classes <= n_expected_classes
    
    M = np.zeros((n_expected_classes,4),dtype=np.int32)
    for (prediction,expectation) in results:        
        update = None        
        if prediction == expectation:
            for i in range(n_expected_classes):
                new_row = np.array([[1,0,0,0]]) if i == prediction else np.array([[0,1,0,0]]) 
                update = new_row if update is None else np.append(update,new_row,axis=0)
 
        if prediction != expectation:
            for i in range(n_expected_classes):                
                if i == prediction:
                    new_row = np.array([[0,0,1,0]])                     
                elif i == expectation:
                    new_row = np.array([[0,0,0,1]]) 
                else:
                    new_row = np.array([[0,1,0,0]]) 
                update = new_row if update is None else np.append(update,new_row,axis=0)
                    
        M += update
    return M

expected = np.array([
    [ 1,  3,  0,  0],
    [ 2,  1,  1,  0],
    [ 0,  3,  0,  1]
])
assert np.array_equal(to_true_false_positive_negative([(0,0), (1,1), (1,1), (1,2)]), expected)



def to_accuracy_precision_recall(true_false_positive_negative):
    """λ : Tensor of shape (43,4) → Tensor of shape (43,3).

    Tensor of shape (43,4) where each row matches a class and columns:
        True Positive, True Negative, False Positive, False Negative

    Tensor (43,3) where each row matches a class and columns: 
        Accuracy, Precision, Recall
    """
    height, width = true_false_positive_negative.shape
    result = None
    for i in range(height):
        row_i = true_false_positive_negative[i]
        TP = row_i[0]; TN = row_i[1]; FP = row_i[2]; FN = row_i[3]
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        new_row = np.array([[accuracy, precision, recall]])
        result = new_row if result is None else np.append(result, new_row, axis=0)
    return result



def model_accuracy(predictions_expectations):
    """λ : [(predicted_class,expected_class), ...] → Accuracy"""
    
    corrects = sum([1 if prediction == expectation else 0 
                    for (prediction,expectation) in predictions_expectations])
    trials = len(predictions_expectations)
    return corrects/trials



web_traffic_signs_dir = os.path.join('.','traffic_signs_from_web')
def get_and_split_web_traffic_signs():
    filenames = [filename 
                 for filename in os.listdir(web_traffic_signs_dir) 
                 if re.search('\.jpg$', filename)]
    
    images = np.array([misc.imread(os.path.join(web_traffic_signs_dir,filename)) for filename in filenames])              
    labels = np.array([int(int_str) for int_str in
                        [os.path.splitext(filename)[0] for filename in filenames] ])
    return images, labels



def predict(model_parameters,training_parameters,model,images):
    """Return predictions and associated probabilities for each images."""
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, (None, 32, 32, model_parameters['image_depth']))
    keep_prob = tf.placeholder(tf.float32)
    logits,tf_activations = model(x,keep_prob)
    believes = tf.nn.softmax(logits)
    prediction = tf.argmax(believes,axis = 1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, training_parameters["save_file"])
        return sess.run(prediction, feed_dict={x: images, keep_prob:1.0}), sess.run(believes, feed_dict={x: images, keep_prob:1.0 })
    
    
    
def graph_image_prob(image,probabilities):
    plt.figure(figsize=(16,4))
    ax1 = plt.subplot2grid((1, 3), (0, 0),colspan=1)
    ax1.imshow(np.squeeze(image),aspect='equal')
    ax2 = plt.subplot2grid((1, 3), (0, 1),colspan=2)
    dist = [(i,probabilities[i]) for i in range(len(probabilities))]
    counts = [count for (label,count) in dist]
    labels = [label for (label,count) in dist]
    y_pos = np.arange(len(labels))
    ax2.bar(y_pos, counts, align='center', alpha=0.5)
    plt.show()
    
    
    
def accuracy(expected_labels, predicted_labels):
    assert len(expected_labels) == len(predicted_labels)
    n_success = sum([1 if expected == predicted else 0 
                     for (expected,predicted) in zip(expected_labels,predicted_labels)])
    print("XXXXXXXXXXXXXXXXXXX")
    print("X Accurracy: {:.1%}".format(n_success/len(predicted_labels)))
    print("XXXXXXXXXXXXXXXXXXX")


    
def show_hidden_layers(model_parameters, training_parameters, stimuli_image, model):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, (None, 32, 32, model_parameters['image_depth']))
    keep_prob = tf.placeholder(tf.float32)
    logits, tf_activations = model(x,keep_prob)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, training_parameters["save_file"])

        def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
            activation = tf_activation.eval(session=sess,feed_dict={x : image_input, keep_prob: 1. })
            featuremaps = activation.shape[3]
            plt.figure(plt_num, figsize=(15,15))
            for featuremap in range(featuremaps):
                plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
                plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
                if activation_min != -1 & activation_max != -1:
                    plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
                elif activation_max != -1:
                    plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
                elif activation_min !=-1:
                    plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
                else:
                    plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
                    
        outputFeatureMap(stimuli_image,tf_activations['conv1'])
    
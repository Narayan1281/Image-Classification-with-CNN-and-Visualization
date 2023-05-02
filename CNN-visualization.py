import os
# from google.colab import drive
import tensorflow as tf
from tensorflow import keras
from keras.utils.vis_utils import plot_model
from keras.utils import load_img
import numpy as np
import pandas as pd
from keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

# Display
# from IPython.display import Image, display
import matplotlib.cm as cm
from PIL import Image as PILImage

curr_path = os.getcwd()
random_seed =42

#Adam Optimiser params
lr=1e-3
epsilon_val=1e-8
beta1=0.9
beta2=0.999

path_in = '/Group_20'
path_model = ''
print(os.listdir(path_model))

class_names = ['brain', 'butterfly', 'ewer', 'helicopter', 'ketch']
normalization_layer = tf.keras.layers.Rescaling(1./255)

## read training data
train_ds = tf.keras.utils.image_dataset_from_directory(
    path_in + '/train/',
    batch_size=32,
    seed = random_seed,
    image_size=(224, 224)
).map(lambda x, y: (normalization_layer(x), y))

# load validation dataset 
valid_ds = tf.keras.utils.image_dataset_from_directory(
    path_in + '/val/',
    batch_size=32,
    seed = random_seed,
    image_size=(224, 224),
    shuffle=False
).map(lambda x, y: (normalization_layer(x), y))

# load test dataset 
test_ds = tf.keras.utils.image_dataset_from_directory(
    path_in + '/test/',
    batch_size=32,
    seed = random_seed,
    image_size=(224, 224)
).map(lambda x, y: (normalization_layer(x), y))



image_super = [] # pairs of images and corresponding labels (will be used for plotting and other experiments)
plt.figure(figsize=(10, 10))
for i in range(5):
    label_idx = i%5
    for batch_images, batch_labels in valid_ds:
        mask = (batch_labels == label_idx)
        if tf.reduce_any(mask): #checks if any value in mask tensor is true
            indices = tf.argmax(mask)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow((batch_images[indices].numpy()*255).astype("uint8"))
            plt.title(class_names[batch_labels[indices]])
            plt.axis("off")
            image_super.append([batch_images[indices], batch_labels[indices]])
            break  

            
# Define the layers whose weights you want to save
layer_names = ['output_layer']

# Define the custom callback function to save the weights of specific layers
class SaveLayerWeightsCallback(ModelCheckpoint):
    def __init__(self, filepath, layers, **kwargs):
        self.layers = layers
        self.filepath = filepath
        super().__init__(filepath, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        for layer_name in self.layers:
            layer = self.model.get_layer(layer_name)
            weights = layer.get_weights()
            if not os.path.exists(self.filepath):
              os.mkdir(self.filepath)
            filename1 = f"{layer_name}_epoch_{epoch}_weights.csv"
            filename2 = f"{layer_name}_epoch_{epoch}_bias.csv"
            filepath1 = os.path.join(self.filepath, filename1)
            filepath2 = os.path.join(self.filepath, filename2)
            # layer.save_weights(filepath)
            pd.DataFrame(weights[0]).to_csv(filepath1, index=False)
            pd.DataFrame(np.array([weights[1]])).to_csv(filepath2, index=False)
            # pd.DataFrame(weights).to_csv(filepath) # this saves array as a string and which is problamatic

threshold_val = 0.0001
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=3,
                                                  min_delta=threshold_val,
                                                  mode='min',
                                                  restore_best_weights=True, 
                                                  verbose=1)

# Create an instance of the custom callback
save_layer_weights = SaveLayerWeightsCallback(filepath='sequential-model-weights/', layers=layer_names)

vgg19 = tf.keras.applications.VGG19(
    include_top=True,
    weights="imagenet",
    classes=1000,
    classifier_activation="softmax"
)

inp = tf.keras.layers.Input(shape=(224, 224, 3))
base_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_tensor=inp,
                                          input_shape=(224, 224, 3))

print(path_model)
best_w = pd.read_csv(path_model + '/output_layer_epoch_2_weights.csv', dtype='float32').values
best_b = pd.read_csv(path_model + '/output_layer_epoch_2_bias.csv', dtype='float32').values[0]

Flatten = tf.keras.layers.Flatten(name="flatten") # flatten output of last convolution layer
fc1 = tf.keras.layers.Dense(4096, activation='relu', name='fc1')
fc2 = tf.keras.layers.Dense(4096, activation='relu', name = 'fc2')
prediction = tf.keras.layers.Dense(5, activation='softmax', name='output')

imsize = [224, 224, 3]
inp = layers.Input(shape=(imsize[0], imsize[1], imsize[2]))
base_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_tensor=inp,
                                          input_shape=(imsize[0], imsize[1], imsize[2]))

## Creating transfer learning model using functional API
block5_pool = base_model.get_layer('block5_pool')
x = Flatten(block5_pool.output)
x = fc1(x)
x = fc2(x)
x = prediction(x)
model = tf.keras.models.Model(inputs = inp, outputs = x)

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                       optimizer=keras.optimizers.Adam(learning_rate=lr,
                                                       epsilon=epsilon_val,
                                                       beta_1=beta1,
                                                       beta_2=beta2),
                        metrics=['accuracy'])

# freeze all the convolution layers in training
for hl in model.layers:
    hl.trainable = False
model.get_layer('output').trainable = True

## Change weights of last fully connected layer to have 
## weights from best model weights from earlier training
model.get_layer('fc1').set_weights(vgg19.get_layer('fc1').get_weights())
model.get_layer('fc2').set_weights(vgg19.get_layer('fc2').get_weights())
model.get_layer('output').set_weights([best_w, best_b])

test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print("test accuracy = ", test_acc)
print("test loss ", test_loss)

_, train_acc = model.evaluate(train_ds, verbose=0)
_, val_acc = model.evaluate(valid_ds, verbose=0)
print(f"training accuracy = {train_acc :.3f}")
print(f"validation accuracy = {val_acc :.3f}" )

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=10): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # overall accuracy
  true_cls = 0
  for i in range(cm.shape[0]):
    true_cls += cm[i][i]
  acc_all = (true_cls/len(y_true))*100

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  # fig.colorbar(cax) # for now skip plotting heat index

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes 
  ax.set(title=f"Confusion Matrix\naccuracy: {acc_all}",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)

  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
             horizontalalignment="center",
             color="white" if cm[i, j] > threshold else "black",
             size=text_size)
    
    
    
def plot_confusion_matrix(data_set, label_names):
  # Get true labels
  true_labels = []
  predicted_ =[]
  for images, labels in data_set:
    true_labels += labels.numpy().tolist()
    predicted_ += model.predict(images, verbose=0).argmax(axis=1).tolist()
  make_confusion_matrix(true_labels, predicted_, label_names)


print('                    |------------------- confusion matrix for training datasets --------------|')
print()
plot_confusion_matrix(train_ds, class_names)


print('                    |------------------- confusion matrix for validation datasets --------------|')
print()
plot_confusion_matrix(valid_ds, class_names)


print('                    |------------------- confusion matrix for test datasets --------------|')
print()
plot_confusion_matrix(test_ds, class_names)
last_conv_layer_name, last_conv_layer_idx = "block5_conv4", 20

## create a submodel to calculate output of last convolution layer
receptive_model_calc = tf.keras.Model(
    inputs = model.input,
    outputs = model.get_layer(last_conv_layer_name).output)

# function to calculate argmax of a tensor or 3D matrix
def findArgmax(arr):
    flattend_argmax = np.argmax(arr)
    req_argmax = np.unravel_index(flattend_argmax, arr.shape)
    return req_argmax[1:-1]

def trace_ipnput_patch(left, right, hidd_layer, model):
    top_left = left
    bottom_right = right
    
    # trace From last layer to input layer
    for i in range(hidd_layer, 0, -1): 
        curr_layer = model.layers[i] 
        if isinstance(model.layers[i], tf.keras.layers.Conv2D):
            kernel_sz = curr_layer.kernel_size[0]
        else:
            kernel_sz = curr_layer.pool_size[0]
            
        s = curr_layer.strides[0]
        l_i, l_j = top_left
        r_i, r_j = bottom_right
        
        ## difficult case need to take care of padding (think backward)
        if curr_layer.padding == 'same':
            p = int(kernel_sz/2)
            l_ix = max(0, s * l_i - p)
            l_iy = max(0, s * l_j - p)
            r_ix = min(s * r_i + (kernel_sz-1) - p, curr_layer.output.shape[1])
            r_iy = min(s * r_j + (kernel_sz-1) - p, curr_layer.output.shape[2])
            top_left = l_ix, l_iy
            bottom_right = r_ix, r_iy
            
        else:
            top_left = s * l_i, s * l_j
            bottom_right = s * r_i + (kernel_sz-1), s * r_j + (kernel_sz-1)
    
    return top_left, bottom_right

def norm_flat_image(img):
    grads_norm = img #[:,:,0]+ img[:,:,1]+ img[:,:,2]
    grads_norm = (grads_norm - tf.reduce_min(grads_norm))/ (tf.reduce_max(grads_norm)- tf.reduce_min(grads_norm))
    return grads_norm



plt.figure(figsize=(10, 10))
count = 0
for image_ in image_super:  
    img1 = image_[0]*255 
    output = receptive_model_calc.predict(np.array([img1]), verbose=0)
    maximal_neuron = findArgmax(output)
    left_idx, right_idx = trace_ipnput_patch(maximal_neuron, maximal_neuron, last_conv_layer_idx, receptive_model_calc)

    cropped_img = img1.numpy().copy()
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            l_i, l_j = left_idx
            r_i, r_j = right_idx
            if(i < l_i or j < l_j or i > r_i or j > r_j):
                cropped_img[i,j,:] //=2
            elif(i in [l_i, l_i-1, l_i+1] or i in [r_i, r_i-1, r_i+1] or j in [l_j, l_j-1, l_j+1] or j in [r_j, r_j-1, r_j+1]):
                cropped_img[i,j,0] = 208
                cropped_img[i,j,1] = 89
                cropped_img[i,j,2] = 198
    ax = plt.subplot(3, 4, count + 1)
    plt.imshow(norm_flat_image(img1))
    plt.title(f'Normal image\n{class_names[image_[1]]}', fontsize=10)
    plt.axis('off')
    ax = plt.subplot(3, 4, count + 2)
    plt.imshow(norm_flat_image(cropped_img))
    plt.title('cropped image')
    plt.axis('off')
    count += 2
    
@tf.custom_gradient
def guidedRelu(x):
    # upstream is gradient flowing from next layer in backpropagation
    def grad(upstream):
        return tf.cast(upstream>0,tf.float32)  * tf.cast(x>0,tf.float32) * upstream
    return tf.nn.relu(x), grad

## change all the relu activations to guidedRelu 
## this will facilitate only positive gradients to flow during backpropagation
def changeActivation(model, isGuidedRelu=True):
    
    if(isGuidedRelu == True):
        for hidden_layers in model.layers:
            if hasattr(hidden_layers, 'activation'):
                if hidden_layers.activation == tf.keras.activations.relu:
                    hidden_layers.activation = guidedRelu
                    print(hidden_layers.name, " --> activation changed to guidedRelu")
    else: # change back to normal
        for hidden_layers in model.layers:
            if hasattr(hidden_layers, 'activation'):
                if hidden_layers.activation:
                    hidden_layers.activation = tf.keras.activations.relu
                    print(hidden_layers.name, " --> activation changed to relu")
    model.get_layer('output').activation = None #tf.keras.activations.softmax
    print(hidden_layers.name, " --> activation changed to softmax")       
    return

changeActivation(model, isGuidedRelu=True)

## we'll make use of above defined model
print("Model :summary guided backpropagation")
print(receptive_model_calc.summary())

def guidedBP_input_influence(img, target_neuron_idx = 0):
    # output_conv = receptive_model_calc.predict(np.array([img]), verbose=0)
     # position of neuron : target_neuron_idx 
    
    with tf.GradientTape() as tape:
        tape.watch(img)
        target_activations = receptive_model_calc(tf.expand_dims(img, axis=0))[:, :, :, target_neuron_idx]
    grads1 = tape.gradient(target_activations, img)
    
    inp_img1 = tf.expand_dims(img, axis=0)
    with tf.GradientTape() as tape:
        tape.watch(inp_img1)
        preds = model(inp_img1)
        pred_idx = tf.argmax(preds, axis=1)
        pred_val = preds[0,pred_idx[0]]
    grads2 = tape.gradient(pred_val, inp_img1)
    
    return {'grads_conv': grads1, 'grads_op': grads2 ,'neuron_pos': target_neuron_idx}



plt.figure(figsize=(10, 10))
count = 0
for image_ in image_super:  
    img1 = image_[0]*255 
    grads = guidedBP_input_influence(img1)
    ax = plt.subplot(5, 3, count + 1)
    plt.imshow(norm_flat_image(img1))
    plt.title(f'Original image\n{class_names[image_[1]]}', fontsize=10)
    plt.axis('off')
    ax = plt.subplot(5, 3, count + 2)
    plt.imshow(norm_flat_image(grads['grads_op'][0]))
    plt.title('guided-backprop\n output', fontsize=10)
    plt.axis('off')
    ax = plt.subplot(5, 3, count + 3)
    plt.imshow(norm_flat_image(grads['grads_conv']))
    plt.title('guided-backprop\nfrom conv_neuron', fontsize=10)
    plt.axis('off')
    count += 3
plt.tight_layout()

changeActivation(model, isGuidedRelu=False)


last_conv_layer_name = "block5_conv4"


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, cam_path = 'cam.jpg', alpha=0.4):
    # Load the original image
#     img = keras.preprocessing.image.load_img(img_path)
#     img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

#     # Save the superimposed image
#     superimposed_img.save(cam_path)

#     # Display Grad CAM
#     display(Image(cam_path))
    img = keras.preprocessing.image.img_to_array(superimposed_img)/255.0
    return img




def compute_all_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    if model.layers[-1].activation != None:
        model.layers[-1].activation = None
    
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape(persistent=True) as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        num_class = preds.shape[1]
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channels = [preds[:, i] for i in range(num_class)]
    
    
    grads_all = [tape.gradient(class_channels[i], last_conv_layer_output) for i in range(num_class)]

    pooled_grads_all = [tf.reduce_mean(grads_all[i], axis=(0, 1, 2)) for i in range(num_class)]
    
    heatmap_all = []
    last_conv_layer_output = last_conv_layer_output[0]
    for i in range(num_class):
        heatmap = last_conv_layer_output @ pooled_grads_all[i][..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap_all.append(heatmap)
    
    # return heatmap of all classes along with class_index of predicted class as a dict
    grad_cams = {'heatmaps': heatmap_all, 'pred_idx': pred_index}
    return grad_cams

def visualise_heat_map_all(img_, model=model, last_conv_layer_name = "block5_conv4"):
    array = img_.numpy()
    img_array = np.expand_dims(array, axis=0)
    cal_gradcams = compute_all_gradcam_heatmap(img_array, model, last_conv_layer_name)
    plt.figure(figsize=(10, 10))
    for i, heatmap in enumerate(cal_gradcams.get('heatmaps')):
        ax = plt.subplot(3, 3, i + 1)
        img = save_and_display_gradcam(array*255, heatmap)
        plt.imshow(img)
        title = ""
        if i == cal_gradcams.get('pred_idx'):
            title = f'with respect to actual\npredicted class: {class_names[i]}'
        else:
            title = f'with respect to\nclass: {class_names[i]}'
        plt.title(title, fontsize=10)
        plt.axis("off")

        
for i in range(5):
    visualise_heat_map_all(image_super[i][0], model)

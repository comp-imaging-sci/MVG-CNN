import os
import argparse
import scipy.io as sio
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K

import sklearn.metrics
from skimage.transform import resize
import matplotlib.pyplot as plt
from itertools import cycle
from utils import *
# make sure that you have the focal loss package installed
from focal_loss import SparseCategoricalFocalLoss, BinaryFocalLoss

import matplotlib

def add_arguments(parser):
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs in training')
    parser.add_argument('--lr_init', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--gamma', type=float, default=2, help='gamma factor used for focal loss')
    parser.add_argument('--model_continue', action='store_true', help='restart training by loading the most recent saved weights') 
    parser.add_argument('--model_savedir',type=str, default='/shared/anastasio-s1/MRI/xiaohui/mouse_optical/sleep-stage/paper/checkpoint', help='path to save trained weights')
    parser.add_argument('--gradcam_label', type=int, default=0, help='states users want to compute for GradCAM')


    return parser

class model:
    def __init__(self, project):
        self.project = project
        self.cp_callbacks = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.project.params.model_savedir, "ckpt_{epoch}"),
                monitor="val_loss",
                save_best_only=True, 
                save_weights_only=True, 
                verbose=0,
                save_freq='epoch')
    
        if self.project.params.loss =='binary_ce':
            self.loss = 'binary_crossentropy'
        elif self.project.params.loss == 'binary_focal':
            self.loss = BinaryFocalLoss(gamma=2)
        elif self.project.params.loss =='categorical_ce':
            self.loss = 'categorical_crossentropy'
        elif self.project.params.loss =='categorical_focal':
            self.loss = SparseCategoricalFocalLoss(gamma=self.project.params.gamma, class_weight=([0.25, 0.25, 0.25]))
        print(self.loss)
    
    def cnn2d(self):
        """2D CNN for adjacency matrix classification"""
        inputs = Input(shape=(self.project.params.num_frames, self.project.params.num_frames, len(self.project.params.brain_idx)))

        # block 1 
        x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(inputs)
        x = LeakyReLU()(x)
        x = MaxPool2D(pool_size=(2,2))(x)

        # block 2
        x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
        x = LeakyReLU()(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        
        # block 3 
        x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
        x = LeakyReLU()(x)
        x = GlobalAveragePooling2D()(x)

        if self.project.params.num_classes == 2:
            outputs = Dense(units=1, activation='sigmoid')(x)
        else:
            outputs = Dense(units=self.project.params.num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        print(model.summary())
        return model 

    def train(self, train_dataset, val_dataset, num_epochs):
        
        # Create directory to save model weights
        if not os.path.exists(self.project.params.model_savedir):
            os.makedirs(self.project.params.model_savedir)
 
        # Distribute data in multi-GPUs in training
        if self.project.params.mode == 'train':
            strategy = tf.distribute.MirroredStrategy()
            print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
            
            with strategy.scope():
                self.model = self.cnn2d()
                print(self.loss)
                self.model.compile(optimizer=tf.keras.optimizers.Adam(self.project.params.lr_init), loss=self.loss, metrics=['accuracy'])

                # Resume training from saved checkpoints
                if self.project.params.model_continue:
                    self.model.load_weights(tf.train.latest_checkpoint(self.project.params.model_savedir))
                    print('... Resume training from latest checkpoint ...')
                    self.model.trainable = True
 
        self.model.fit(train_dataset,
                epochs=num_epochs,
                verbose=1,
                validation_data=val_dataset,
                validation_freq=1,
                callbacks=[self.cp_callbacks],
                workers=4, 
                use_multiprocessing=True)
    
    def test(self, dataset):
        # Use single GPU in inference
        self.model = self.cnn2d()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.project.params.lr_init), loss=self.loss, metrics=["accuracy"])
        self.model.load_weights(tf.train.latest_checkpoint(self.project.params.model_savedir))
        self.model.trainable = False
        print('... Trained model loaded ...')

        y_trues = []
        y_scores = []
        y_preds = []

        if self.project.params.num_classes == 2:
            for x, y in dataset:
                y_trues.extend(y.numpy())
                y_score = self.model.predict(x)
                y_scores.extend(y_score)
                y_preds.extend(y_score>0.5)

        else: 
            for x, y in dataset:
                y_trues.extend(y.numpy())
                y_score = self.model.predict(x)
                y_scores.extend(y_score)
                y_preds.extend(np.argmax(y_score, axis=1)) 
        
        results = {
                'y_trues': y_trues,
                'y_preds': y_preds,
                'y_scores': y_scores
                }
        
        if self.project.params.mode == 'test_subjectwise':
            sio.savemat(os.path.join('../Results', f"{self.project.params.test_mice}_{self.project.params.mice_flist}_{self.project.params.timelen}s.mat"), results)
            print('Saved results ...')
        else:
            sio.savemat(os.path.join('../Results', self.project.params.dataset[11:] + f"_class{self.project.params.num_classes}_test.mat"), results)
            print('Saved results ...')
        plot_roc(np.squeeze(y_trues), np.squeeze(y_scores), self.project.params.loss, self.project.params.num_classes)
    
    
    def gradcam(self, dataset):
        # function to compute GradCAM
        import matplotlib.cm as cm
        from tensorflow import keras
        from IPython.display import Image, display
        import matplotlib.pyplot as plt

        self.model = self.cnn2d()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.project.params.lr_init), loss='categorical_crossentropy', metrics=[tf.keras.metrics.AUC(multi_label=True), 'accuracy'])
        self.model.load_weights(tf.train.latest_checkpoint(self.project.params.model_savedir))
        self.model.trainable = False
        print('... 2D Trained Model Loaded ...')

        last_conv_layer = self.model.get_layer("leaky_re_lu_2")
        last_conv_layer_model = keras.Model(self.model.inputs, last_conv_layer.output)

        classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        classifier_layer_names = ["global_average_pooling2d", "dense"]
        for layer_name in classifier_layer_names:
            x = self.model.get_layer(layer_name)(x)
        classifier_model = keras.Model(classifier_input, x)
        
        n = 0
        gradcams = []
        ams = []
        for idx, (x, y) in enumerate(dataset):
            with tf.GradientTape() as tape:
                last_conv_layer_output = last_conv_layer_model(x)
                tape.watch(last_conv_layer_output)

                preds = classifier_model(last_conv_layer_output)
                top_pred_index = tf.argmax(preds[0])
                top_class_channel = preds[:, top_pred_index]
            
            if top_pred_index.numpy() == self.project.params.gradcam_label and top_pred_index.numpy() == y:
    
                grads = tape.gradient(top_class_channel, last_conv_layer_output)
                
                pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
                last_conv_layer_output = last_conv_layer_output.numpy()[0]
                pooled_grads = pooled_grads.numpy()

                for i in range(pooled_grads.shape[-1]):
                        last_conv_layer_output[:,:,i] *= pooled_grads[i]

                heatmap = np.mean(last_conv_layer_output, axis=-1)
                heatmap = np.maximum(heatmap, 0)/np.max(heatmap)
                heatmap = np.uint8(255*heatmap)
                heatmap = resize(heatmap, (self.project.params.num_frames, self.project.params.num_frames))
                
                gradcams.append(heatmap)
                ams.append(x.numpy())
                
                n = n+1
                print(n)

                if n == 5: # define a number of GradCAM to save
                    sio.savemat(f"../Results/gradcam/2020_config14_{self.project.params.test_mice}_class{self.project.params.num_classes}_label{self.project.params.gradcam_label}_{self.project.params.timelen}s_area${self.project.params.brain_area}.mat",{"gradcam": gradcams, "am":ams})
                    print("Grad-CAM saved...")
                    break


            #superimposed_img = jet_heatmap * 0.4 #+ img
            #superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

            #plt.imshow(superimposed_img)
            #plt.show()



import datetime
import os
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint, TensorBoard
from datanail import *
from res_unet import *
from utils import *

# hyper parameters
model_name = "res_unet_"
input_shape = (512, 512, 1)
dataset_folder = ""
classes = []
batch_size = 1

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(batch_size,'data/train','image','label',data_gen_args,save_to_dir = 'data/train/aug')

model_file = model_name + datetime.datetime.today().strftime("_%d_%m_%y_%H:%M:%S") + ".h5"
model = build_res_unet(input_shape=input_shape)
model.summary()#输出模型各层的参数状况
optimizer = Adadelta()#梯度优化，自适应学习率调整
model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

model_checkpoint = ModelCheckpoint(os.path.join("models", model_file), monitor='loss', save_best_only=True, verbose=True)
tensorboard = TensorBoard()#计算图可视化
#train_aug = ImageDataGenerator(vertical_flip=True, horizontal_flip=True)
#train_gen = PASCALVOCIterator(directory=dataset_folder, target_file="train.txt",
                              #image_data_generator=train_aug, target_size=(input_shape[0], input_shape[1]),
                              #batch_size=batch_size, classes=classes)

model.fit_generator(myGene, steps_per_epoch=300,epochs=1,callbacks=[tensorboard, model_checkpoint])
testGene = testGenerator("data/test")
results = model.predict_generator(testGene,6,verbose=1)
saveResult("data/test",results)

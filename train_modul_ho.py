
from keras.layers import *
import keras.backend as K
import tensorflow as tf

import os
import cv2
import glob
import time
import numpy as np
from pathlib import PurePath, Path

import matplotlib.pyplot as plt
def train(ITERS):
    K.set_learning_phase(1)
    #K.set_learning_phase(0) # set to 0 in inference phase

    # Number of CPU cores
    num_cpus = os.cpu_count()

    # Input/Output resolution
    RESOLUTION = 64 # 64x64, 128x128, 256x256
    assert (RESOLUTION % 64) == 0, "RESOLUTION should be 64, 128, or 256."

    # Batch size
    # batchSize = 8
    batchSize = 2

    # Use motion blurs (data augmentation)
    # set True if training data contains images extracted from videos
    use_da_motion_blur = False

    # Use eye-aware training
    # require images generated from prep_binary_masks.ipynb
    use_bm_eyes = True
    use_bm_eyes = False
    # Probability of random color matching (data augmentation)
    prob_random_color_match = 0.5

    da_config = {
        "prob_random_color_match": prob_random_color_match,
        "use_da_motion_blur": use_da_motion_blur,
        "use_bm_eyes": use_bm_eyes
    }

    session_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    # please do not use the totality of the GPU memory
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.90

    # Path to training images
    img_dirA = './faceA'
    img_dirB = './faceB'
    #img_dirA_bm_eyes = "./binary_masks/faceA_eyes"
    #img_dirB_bm_eyes = "./binary_masks/faceB_eyes"
    img_dirA_bm_eyes = "./faceA/binary_masks_eyes"
    img_dirB_bm_eyes = "./faceB/binary_masks_eyes"

    # Path to saved model weights
    models_dir = "./models"
    Path(f"models").mkdir(parents=True, exist_ok=True)

    # Architecture configuration
    arch_config = {}
    arch_config['IMAGE_SHAPE'] = (RESOLUTION, RESOLUTION, 3)
    arch_config['use_self_attn'] = True
    arch_config['norm'] = "instancenorm" # instancenorm, batchnorm, layernorm, groupnorm, none
    arch_config['model_capacity'] = "standard" # standard, lite

    # Loss function weights configuration
    loss_weights = {}
    loss_weights['w_D'] = 0.1 # Discriminator
    loss_weights['w_recon'] = 1. # L1 reconstruction loss
    loss_weights['w_edge'] = 0.1 # edge loss
    loss_weights['w_eyes'] = 30. # reconstruction and edge loss on eyes area
    loss_weights['w_pl'] = (0.01, 0.1, 0.3, 0.1) # perceptual loss (0.003, 0.03, 0.3, 0.3)

    # Init. loss config.
    loss_config = {}
    loss_config["gan_training"] = "mixup_LSGAN" # "mixup_LSGAN" or "relativistic_avg_LSGAN"
    loss_config['use_PL'] = False
    loss_config['use_mask_hinge_loss'] = False
    loss_config['m_mask'] = 0.
    loss_config['lr_factor'] = 1.
    loss_config['use_cyclic_loss'] = False


    print('CONFIGURE DONE')

    #define models
    from networks.faceswap_gan_model import FaceswapGANModel
    global model
    model = FaceswapGANModel(**arch_config)

    print('DEFINE MODELS DONE')

    model.load_weights(path=models_dir)

    from keras_vggface.vggface import VGGFace
    # VGGFace ResNet50
    global  vggface
    vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))

    #vggface.summary()

    model.build_pl_model(vggface_model=vggface)

    model.build_train_functions(loss_weights=loss_weights, **loss_config)

    from data_loader.data_loader import DataLoader



    # Get filenames
    train_A = glob.glob(img_dirA+"/*.*")
    train_B = glob.glob(img_dirB+"/*.*")

    train_AnB = train_A + train_B

    assert len(train_A), "No image found in " + str(img_dirA)
    assert len(train_B), "No image found in " + str(img_dirB)
    print ("Number of images in folder A: " + str(len(train_A)))
    print ("Number of images in folder B: " + str(len(train_B)))

    if use_bm_eyes:
        assert len(glob.glob(img_dirA_bm_eyes+"/*.*")), "No binary mask found in " + str(img_dirA_bm_eyes)
        assert len(glob.glob(img_dirB_bm_eyes+"/*.*")), "No binary mask found in " + str(img_dirB_bm_eyes)
        assert len(glob.glob(img_dirA_bm_eyes+"/*.*")) == len(train_A), \
        "Number of faceA images does not match number of their binary masks. Can be caused by any none image file in the folder."
        assert len(glob.glob(img_dirB_bm_eyes+"/*.*")) == len(train_B), \
        "Number of faceB images does not match number of their binary masks. Can be caused by any none image file in the folder."

    def show_loss_config(loss_config):
        for config, value in loss_config.items():
            print(f"{config} = {value}")



    def reset_session(save_path):
        global model, vggface
        global train_batchA, train_batchB
        model.save_weights(path=save_path)
        del model
        del vggface
        del train_batchA
        del train_batchB
        K.clear_session()
        model = FaceswapGANModel(**arch_config)
        model.load_weights(path=save_path)
        vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))
        model.build_pl_model(vggface_model=vggface)
        train_batchA = DataLoader(train_A, train_AnB, batchSize, img_dirA_bm_eyes,
                                  RESOLUTION, num_cpus, K.get_session(), **da_config)
        train_batchB = DataLoader(train_B, train_AnB, batchSize, img_dirB_bm_eyes,
                                  RESOLUTION, num_cpus, K.get_session(), **da_config)


    print('RESET_SESSION DONE')


    # Start training
    t0 = time.time()
    gen_iterations = 0
    errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
    errGAs = {}
    errGBs = {}
    # Dictionaries are ordered in Python 3.6
    for k in ['ttl', 'adv', 'recon', 'edge', 'pl']:
        errGAs[k] = 0
        errGBs[k] = 0

    display_iters = 300
    backup_iters = 5000
    TOTAL_ITERS = ITERS//1
    # TOTAL_ITERS = 10000

    global train_batchA, train_batchB
    train_batchA = DataLoader(train_A, train_AnB, batchSize, img_dirA_bm_eyes,
                              RESOLUTION, num_cpus, K.get_session(), **da_config)
    train_batchB = DataLoader(train_B, train_AnB, batchSize, img_dirB_bm_eyes,
                              RESOLUTION, num_cpus, K.get_session(), **da_config)

    print('DATALOADER DONE')

    print("START TRAINING")
    while gen_iterations <= TOTAL_ITERS:
        print(gen_iterations)
        # Loss function automation
        if gen_iterations == (TOTAL_ITERS // 5 - display_iters // 2):
            #clear_output()
            loss_config['use_PL'] = True
            loss_config['use_mask_hinge_loss'] = False
            loss_config['m_mask'] = 0.0
            reset_session(models_dir)
            print("Building new loss funcitons...")
            show_loss_config(loss_config)
            model.build_train_functions(loss_weights=loss_weights, **loss_config)
            print("Done.")
        elif gen_iterations == (TOTAL_ITERS // 5 + TOTAL_ITERS // 10 - display_iters // 2):
            #clear_output()
            loss_config['use_PL'] = True
            loss_config['use_mask_hinge_loss'] = True
            loss_config['m_mask'] = 0.5
            reset_session(models_dir)
            print("Building new loss funcitons...")
            show_loss_config(loss_config)
            model.build_train_functions(loss_weights=loss_weights, **loss_config)
            print("Complete.")
        elif gen_iterations == (2 * TOTAL_ITERS // 5 - display_iters // 2):
            #clear_output()
            loss_config['use_PL'] = True
            loss_config['use_mask_hinge_loss'] = True
            loss_config['m_mask'] = 0.2
            reset_session(models_dir)
            print("Building new loss funcitons...")
            show_loss_config(loss_config)
            model.build_train_functions(loss_weights=loss_weights, **loss_config)
            print("Done.")
        elif gen_iterations == (TOTAL_ITERS // 2 - display_iters // 2):
            #clear_output()
            loss_config['use_PL'] = True
            loss_config['use_mask_hinge_loss'] = True
            loss_config['m_mask'] = 0.4
            reset_session(models_dir)
            print("Building new loss funcitons...")
            show_loss_config(loss_config)
            model.build_train_functions(loss_weights=loss_weights, **loss_config)
            print("Done.")
        elif gen_iterations == (2 * TOTAL_ITERS // 3 - display_iters // 2):
            #clear_output()
            loss_config['use_PL'] = True
            loss_config['use_mask_hinge_loss'] = False
            loss_config['m_mask'] = 0.
            loss_config['lr_factor'] = 0.3
            reset_session(models_dir)
            print("Building new loss funcitons...")
            show_loss_config(loss_config)
            model.build_train_functions(loss_weights=loss_weights, **loss_config)
            print("Done.")
        elif gen_iterations == (8 * TOTAL_ITERS // 10 - display_iters // 2):
            #clear_output()
            model.decoder_A.load_weights("models/decoder_B.h5")  # swap decoders
            model.decoder_B.load_weights("models/decoder_A.h5")  # swap decoders
            loss_config['use_PL'] = True
            loss_config['use_mask_hinge_loss'] = True
            loss_config['m_mask'] = 0.1
            loss_config['lr_factor'] = 0.3
            reset_session(models_dir)
            print("Building new loss funcitons...")
            show_loss_config(loss_config)
            model.build_train_functions(loss_weights=loss_weights, **loss_config)
            print("Done.")
        elif gen_iterations == (9 * TOTAL_ITERS // 10 - display_iters // 2):
            #clear_output()
            loss_config['use_PL'] = True
            loss_config['use_mask_hinge_loss'] = False
            loss_config['m_mask'] = 0.0
            loss_config['lr_factor'] = 0.1
            reset_session(models_dir)
            print("Building new loss funcitons...")
            show_loss_config(loss_config)
            model.build_train_functions(loss_weights=loss_weights, **loss_config)
            print("Done.")

        # Train dicriminators for one batch
        data_A = train_batchA.get_next_batch()
        data_B = train_batchB.get_next_batch()
        errDA, errDB = model.train_one_batch_D(data_A=data_A, data_B=data_B)
        errDA_sum += errDA[0]
        errDB_sum += errDB[0]

        # Train generators for one batch
        data_A = train_batchA.get_next_batch()
        data_B = train_batchB.get_next_batch()
        errGA, errGB = model.train_one_batch_G(data_A=data_A, data_B=data_B)
        errGA_sum += errGA[0]
        errGB_sum += errGB[0]
        for i, k in enumerate(['ttl', 'adv', 'recon', 'edge', 'pl']):
            errGAs[k] += errGA[i]
            errGBs[k] += errGB[i]
        gen_iterations += 1

        # Visualization delete
        if gen_iterations % display_iters == 0:
            # Save models
            model.save_weights(path=models_dir)

        # Backup models
        if gen_iterations % backup_iters == 0:
            bkup_dir = f"{models_dir}/backup_iter{gen_iterations}"
            Path(bkup_dir).mkdir(parents=True, exist_ok=True)
            model.save_weights(path=bkup_dir)

    print('TRAIN DONE')
import sys
if len(sys.argv) > 1:
    train(int(sys.argv[1]))
else:
    train(80000)
import os
import cv2
import math
import keras
import numpy as np
import tensorflow as tf
import efficientnet.keras as enet

from keras import backend as K
from keras.preprocessing import image
from keras.models import load_model, Model
from keras.layers import Dense, Activation, GlobalAveragePooling2D, Dropout, BatchNormalization

from efficientnet.keras import center_crop_and_resize
from google.colab.patches import cv2_imshow

# get predictions from a model
def predict(img, model):
	try:
		image_size = model.input_shape[1]
		x = center_crop_and_resize(img, image_size=image_size)
		x = np.expand_dims(x, 0)
		
		preds = model.predict(x)
		prob = np.max(preds)
		cid = np.unravel_index(preds.argmax(), preds.shape)[1]
		
		return cid, prob, x
		
	except:
		return 0, 0, 0

# generate heat map from last layer
def generate_heatmap(model, x, cid, last_conv_layer_name):
	output = model.output[:, cid]
	last_conv_layer = model.get_layer(last_conv_layer_name)
	grads = K.gradients(output, last_conv_layer.output)[0]
	pooled_grads = K.mean(grads, axis=(0, 1, 2))
	iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
	
	pooled_grads_value, conv_layer_output_value = iterate([x])
	ch_num = last_conv_layer.output_shape[-1]
	
	for i in range(ch_num):
		conv_layer_output_value[:, :, i]*=pooled_grads_value[i]
	
	heatmap = np.mean(conv_layer_output_value, axis=-1)
	heatmap = np.maximum(heatmap,0)
	heatmap /= np.max(heatmap)
	
	del output
	del last_conv_layer
	del grads
	del pooled_grads
	del pooled_grads_value
	del conv_layer_output_value
	return heatmap

# merge heat map image with original image
def merge_with_heatmap(original_img, heatmap):
	heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
	heatmap = np.uint8(255*heatmap)
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
	return cv2.addWeighted(heatmap, 0.5, original_img, 0.5, 0)

# calculate patch coordinates
def patch_coord(img, width, height, PATCH_SIZE, STEP_SIZE):
	x0, x1 = 0, PATCH_SIZE
	y0, y1 = 0, PATCH_SIZE
	coords = []
	
	while True:
		coords.append([[x0, y0], [x1, y1], []])
		x0, x1 = x0 + STEP_SIZE, x1 + STEP_SIZE
		
		if x1 > width:
			x0, x1 = width - PATCH_SIZE, width
			coords.append([[x0, y0], [x1, y1], []])
		  
		if x1 >= width:
			if y1 < height:
				x0, x1 = 0, PATCH_SIZE
				y0, y1 = y0 + STEP_SIZE, y1 + STEP_SIZE
		  
			if y1 > height:
				y0, y1 = height - PATCH_SIZE, height
		
		if x1 >= width and y1 >= height:
			break
	
	return coords

# cut patches of image and put them in a stack
def vstack_patch(img, coords):
	patches = []
	
	for c in coords:
		patch = center_crop_and_resize(img[c[0][1]:c[1][1], c[0][0]:c[1][0]], image_size=224)
		patch = np.expand_dims(patch, axis=0)
		temp = np.array(patch, dtype="float")
		patches.append(temp)
	
	return np.vstack(patches)

# add overlay to img
def add_overlay(img, mask, colormap):
	# prepare overlay
	grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	colored = cv2.applyColorMap(grey, colormap)
	colored = cv2.bitwise_and(colored, mask)
	
	# black out original image
	mask_inv = cv2.bitwise_not(mask)
	temp = cv2.bitwise_and(img, mask_inv)
	
	# add overlay
	ol = cv2.add(temp, colored)
	
	return ol

# add overlay to text
def text_overlay(img, x, y, strng, val, STEP_SIZE):
	font_scale = STEP_SIZE/15
	patch = cv2.putText(img, "{}{:.4f}".format(strng, val), (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA) 

# create and merge damage overlay
def damage_overlay(img, damage, coords):
	mask = [np.zeros(img.shape, dtype=np.uint8) for i in range(2)]
	
	for i, c in enumerate(coords):
		x0, x1, y0, y1 = c[0][0], c[1][0], c[0][1], c[1][1]
		if damage[i][0] == "1":
			cv2.rectangle(mask[0], (x0, y0), (x1, y1), (255, 255, 255), -1)
		if damage[i][1] == "1":
			cv2.rectangle(mask[1], (x0, y0), (x1, y1), (255, 255, 255), -1)
	
	temp = add_overlay(img, mask[0], cv2.COLORMAP_WINTER)
	temp = add_overlay(temp, mask[1], cv2.COLORMAP_AUTUMN)
	mask_and = cv2.bitwise_and(mask[0], mask[1])
	
	return add_overlay(temp, mask_and, cv2.COLORMAP_PLASMA)

# detect vehicle orientation 
def detect_orientation(img, ANGLE_FRONT_MAX, ANGLE_SIDE_MAX, MODEL_ORIEN, DICT_OREN):
	cid, prob, x = predict(img, MODEL_ORIEN)
	angle = DICT_OREN[str(cid)].split("_")
	angle_h = int(angle[0][1:])
	angle_v = int(angle[1][1:])
	
	if angle_h <= ANGLE_FRONT_MAX or angle_h >= (360 - ANGLE_FRONT_MAX):
		return 0, "front", prob, cid, x
	elif  angle_h <= ANGLE_SIDE_MAX or angle_h >= (360 - ANGLE_SIDE_MAX):
		return 1, "side", prob, cid, x
	else:
		return 2, "back", prob, cid, x

# detect vehicle model
def detect_vehicle(img, orien, MODEL_VEH, DICT_VEH):
	cid_k, prob_k, x_k = predict(img, MODEL_VEH['kia'][orien])
	cid_h, prob_h, x_h = predict(img, MODEL_VEH['hyn'][orien])
	
	if prob_k >= prob_h:
		mfr, cid, prob, x = 'kia', cid_k, prob_k, x_k
	else:
		mfr, cid, prob, x = 'hyn', cid_h, prob_h, x_h
	
	return mfr, DICT_VEH[mfr][orien][cid_k], prob, cid, x

# detect vehicle damage
def detect_damage(img, width, height, PATCH_SIZE, STEP_SIZE, DENT_THLD, SCRATCH_THLD, MODEL_DMG):
	coords = patch_coord(img, width, height, PATCH_SIZE, STEP_SIZE)
	patches = vstack_patch(img, coords)
	
	probs = MODEL_DMG.predict(patches)
	probs = np.array(probs)
	
	# "ds" where d = dent, s = scratch
	damage = [("1" if p[1] > DENT_THLD else "0") + ("1" if p[2] > SCRATCH_THLD else "0") for p in probs]

	return damage, probs, coords
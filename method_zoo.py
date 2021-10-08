import sys, os, copy, pickle, math,time,re, itertools
import pandas as pd
import numpy as np
import tensorflow as tf #ver.2.2.0
from scipy import stats

## Collection of basic methods for deep learning model interpretation

####################################################################################
## Saliency map, Grad*Input and Integrated gradients                              ##
## Modified A_K_Nain code (https://keras.io/examples/vision/integrated_gradients/)##
## enabled structured inputs                                                      ##
####################################################################################
def get_gradients(inputs, model):
    """Computes the gradients of outputs w.r.t input.
    Args:
        inputs: np.array(tensor of (1,*img)) or dictionary of array (structured input)
        model:  my model
    Returns:
        Gradients of the predictions w.r.t input
    """
    if isinstance(inputs, dict): #structured input
        inputs = {key: tf.cast(inputs[key], tf.float32) for key in inputs}
    else:
        inputs = tf.cast(inputs, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        preds = model(inputs)
    
    grads = tape.gradient(preds, inputs)
    return grads # same structure as inputs

## Saliency map (Simonyan et al., https://arxiv.org/pdf/1312.6034.pdf)
def get_saliency_map(inputs, model, squeeze_chdim = True, feat_keys = None):
    """Computes max(|gradients|, channelwise) of outputs w.r.t input, as Simonyan et al.
    Args:
        inputs: np.array(tensor of (1,*img)) or dictionary of array (structured input)
        model:  my model
    Returns:
        Saliency map of the predictions w.r.t input
    """
    grads        = get_gradients(inputs,model)
    if isinstance(grads, dict):
        if feat_keys is None:
            feat_keys = grads.keys()
        saliency_map  = {key:np.abs(grads[key]) for key in feat_keys}
        if squeeze_chdim:
            saliency_map = {key:np.max(saliency, axis = -1) for key,saliency in saliency_map.items()}
    else:
        saliency_map = np.abs(grads)
        if squeeze_chdim:
            saliency_map = np.max(saliency_map, axis = -1)
    return saliency_map

## Grad*Input (Baehrens et al., https://jmlr.org/papers/volume11/baehrens10a/baehrens10a.pdf)
def get_grad_input(inputs, model, squeeze_chdim = True, feat_keys = None):
    """
    Args:
        inputs: np.array(tensor of (1,*img)) or dictionary of array (structured input)
        model:  my model
    Returns:
        Grad*Input (default: channel-wise sum up)
    """
    grads = get_gradients(inputs,model)
    if isinstance(inputs, dict): #structured input
        if feat_keys is None:
            feat_keys = grads.keys()
        grad_input = {key: grads[key]*inputs[key] for key in feat_keys}
        if squeeze_chdim:
            grad_input = {key: np.sum(g_i, axis=-1) for key, g_i in grad_input.items()}
    else:
        grad_input = grads * inputs
        if squeeze_chdim:
            grad_input = np.sum(grad_input, axis=-1)
            
    return grad_input

## Integrated gradients (Sundatatajan et al., http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf)
def get_integrated_gradients(inputs, model, baseline=None, num_steps=50, squeeze_chdim = True):
    """Computes Integrated Gradients for a predicted label.
    Args:
        inputs: np.array(tensor of (1,*img)) or dictionary of array (structured input)
        model:  my model 
        baseline (same format as inputs): The baseline image to start with for interpolation
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.

    Returns:
        Integrated gradients w.r.t inputs
        (default: channel-wise sum up of IG values)
    """
    # If baseline is not provided, start with a black image (all-zeros)
    # having same size as the input image.
    if isinstance(inputs, dict): #structured input
        inputs = {key: inputs[key].astype(np.float32) for key in inputs}
        if baseline is None:
            baseline = {key: np.zeros(inputs[key].shape[1:]).astype(np.float32) for key in inputs}
        else:
            baseline = {key: baseline[key].astype(np.float32) for key in baseline}
        
        # 1. Do interpolation.
        interpolated_inputs = {}
        for key,img in inputs.items():
            interpolated_inputs[key] = [baseline[key] + (step/num_steps)*(img-baseline[key]) for step in range(num_steps+1)]
        interpolated_inputs = {key: np.concatenate(interpolated_img, axis=0).astype(np.float32) \
                               for key, interpolated_img in interpolated_inputs.items()}
        
        # 2. Get the gradients.
        grads = {key:[] for key in interpolated_inputs.keys()}
        for i in range(num_steps+1):
            tmp_itp_inputs = {key:tf.expand_dims(img[i],axis=0) for key,img in interpolated_inputs.items()}
            grad_dict = get_gradients(tmp_itp_inputs, model)
            for key,grad in grad_dict.items():
                grads[key].append(grad[0])
        grads = {key:tf.convert_to_tensor(grad, dtype=tf.float32) for key,grad in grads.items()}
        
        # 3. Approximate the integral using the trapezoidal rule.
        grads = {key: (grad[:-1] + grad[1:]) / 2.0 for key,grad in grads.items()}
        avg_grads = {key: tf.reduce_mean(grad, axis=0) for key,grad in grads.items()} 

        # 4. Calculate integrated gradients.
        integrated_grads = {key:(img-baseline[key])*avg_grads[key] for key,img in inputs.items()}
        
        if squeeze_chdim:
            integrated_grads = {key: np.sum(ig, axis=-1) for key, ig in integrated_grads.items()}

    else:
        inputs = inputs.astype(np.float32)
        if baseline is None:
            baseline = np.zeros(inputs.shape[1:]).astype(np.float32)
        else:
            baseline = baseline.astype(np.float32)
            
        # 1. Do interpolation.
        interpolated_inputs = [baseline + (step/num_steps)*(inputs-baseline) for step in range(num_steps+1)]
        interpolated_inputs = np.concatenate(interpolated_inputs,axis=0).astype(np.float32)
        
        # 2. Get the gradients.
        grads = []
        for i, img in enumerate(interpolated_inputs):
            img = tf.expand_dims(img, axis=0)
            grad = get_gradients(img, top_pred_idx=top_pred_idx)
            grads.append(grad[0])
        grads = tf.convert_to_tensor(grads, dtype=tf.float32)
        
        # 3. Approximate the integral using the trapezoidal rule.
        rads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = tf.reduce_mean(grads, axis=0)

        # 4. Calculate integrated gradients.
        integrated_grads = (img_input - baseline) * avg_grads
        
        if squeeze_chdim:
            integrated_grads = np.sum(integrated_grads, axis=-1)
    
    return integrated_grads

####################################################################################
## CAM and Grad-CAM                                                               ##
## Modified following codes:                                                      ##
##     https://jacobgil.github.io/deeplearning/class-activation-maps              ##
##     https://keras.io/examples/vision/grad_cam/                                 ##
## Enabled regression output interpretation                                       ##
## Enabled structured inputs for Grad-CAM                                         ##
####################################################################################

## CAM (Class activation map; Zhou et al., https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)
def get_cam(inputs, model, final_conv_layer_name='', output_layer_name = '',
            pred_index=None, apply_clip = "no"):
    """
    params:
    -------
    inputs: np.array(tensor of (1,*img)) (structured input: nonavailable)
    model:  my model (CAUTION: the last two layers should be GAP->Dense)
    final_conv_layer_name: layer name to visualize (should be the convolution layer before GAP)
    apply_clip: ['negative','positive','no'] When you want to focus on positive/negative contribution only
        negative-> negative relu / no-> do not apply anything (default) / positive-> relu
    return:
    cam: CAM heatmap
    """
    class_weights      = model.layers[-1].get_weights()[0]
    final_conv_layer   = model.get_layer(final_conv_layer_name)
    output_layer       = model.get_layer(output_layer_name).output
    intermediate_model = tf.keras.models.Model([model.get_input_at(0)], [final_conv_layer.output, output_layer])
    conv_outputs, predictions = intermediate_model([inputs])
    output = np.moveaxis(conv_outputs[0],-1,0) #for convenience, move channel axis into the first
    
    cam    = np.zeros(dtype = np.float32, shape = output.shape[:-1])
    #if pred_index is None: RAM (regression activation map)
    if not(pred_index is None): #CAM
        class_weights = class_weights[:, pred_index]
        
    assert len(class_weights) == output.shape[0]
    for i, w in enumerate(class_weights):
        cam += w * output[i]
                
    assert apply_clip in ['negative','positive','no'], \
    "apply_clip should be one of: negative (-inf,0], positive [0,inf), no"
    if   apply_clip == 'negative':
        cam = np.clip(cam, a_min=None, a_max=0)
    elif apply_clip == 'positive':
        cam = np.clip(cam, a_min=0,    a_max=None)
    return cam

## Grad-CAM (Gradient-weighted Class Activation Map; Selvaraju et al., https://arxiv.org/pdf/1610.02391.pdf)
def get_grad_cam(inputs, model, activation_layer_name='',output_layer_name='', pred_index=None,
                 key_list = [], apply_clip = "no"):
    """
    params:
    -------
    inputs: np.array(tensor of (1,*img)) or dictionary of array (structured input)
    model: my model
    activation_layer_name: layer name to visualize
    apply_clip: ['negative','positive','no'] When you want to focus on positive/negative contribution only
        negative-> negative relu / no-> do not apply anything (default) / positive-> relu

    return:
    grad_cam: Grad-CAM heatmap
    """
    if type(inputs) == dict:
        assert all([(key in inputs.keys()) for key in key_list])
        converted_inputs = [inputs[key] for key in key_list]
    else:
        converted_inputs = inputs
        
    activation_layer  = model.get_layer(activation_layer_name)
    cam_rank = len(activation_layer.output.shape)-1
    grad_output_layer = model.get_layer(output_layer_name).output
    
    #if pred_index is None: RAM (regression activation map)
    if not(pred_index is None): #CAM
        grad_output_layer = grad_output_layer[:,pred_index]
    grad_model = tf.keras.models.Model([model.get_input_at(0)], [activation_layer.output, grad_output_layer])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([converted_inputs])
        loss = predictions
    
    output = conv_outputs[0]
    
    grads = tape.gradient(loss, conv_outputs)[0]

    axis         = tuple([i for i in range(cam_rank-1)])
    pooled_grads = tf.reduce_mean(grads, axis=axis)
    
    grad_cam = output @ pooled_grads[..., tf.newaxis]
    grad_cam = tf.squeeze(grad_cam).numpy()
    
    #When you want to focus on positive/negative contribution only
    assert apply_clip in ['negative','positive','no'], \
    "apply_clip should be one of: negative (-inf,0], positive [0,inf), no"
    if   apply_clip == 'negative':
        grad_cam = np.clip(grad_cam,a_min=None, a_max=0)
    elif apply_clip == 'positive':
        grad_cam = np.clip(grad_cam, a_min=0,   a_max=None)
        
    return grad_cam #same shape as activation layer output


## TODO: layerwise relevance propagation
## TODO: guided backprop
## TODO: smoothgrad
## TODO: add notebook for running example
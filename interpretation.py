import sys, os, copy, pickle, math,time,re, itertools
import pandas as pd
import numpy as np
import tensorflow as tf #ver.2.2.0
from scipy import stats

## Collection of deep learning model interpretation methods

## Integrated gradients
## Modified A_K_Nain code (https://keras.io/examples/vision/integrated_gradients/),
## enabled structured input

def get_gradients(inputs, model):
    """Computes the gradients of outputs w.r.t input.

    Args:
        inputs: np.array(tensor of (1,*img)) or dictionary of array (structured input)
    Returns:
        Gradients of the predictions w.r.t input
    """
    
    if type(inputs) == dict:
        inputs = {key: tf.cast(inputs[key], tf.float32) for key in inputs}
    else:
        inputs = tf.cast(inputs, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        preds = model(inputs)
    
    grads = tape.gradient(preds, inputs)
    return grads # same structure as inputs

def get_integrated_gradients(inputs, model, baseline=None, num_steps=50):
    """Computes Integrated Gradients for a predicted label.

    Args:
        inputs: np.array(tensor of (1,*img)) or dictionary of array (structured input)
        baseline (same format as inputs): The baseline image to start with for interpolation
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.

    Returns:
        Integrated gradients w.r.t inputs
    """
    # If baseline is not provided, start with a black image
    # having same size as the input image.
    if type(inputs) == dict:
        inputs = {key: inputs[key].astype(np.float32) for key in inputs}
        if baseline is None:
            baseline = {key: np.zeros(inputs[key].shape[1:]).astype(np.float32) for key in inputs}
        else:
            baseline = {key: baseline[key].astype(np.float32) for key in baseline}
        
        # 1. Do interpolation.
        interpolated_inputs = {}
        for key,img in inputs.items():
            interpolated_inputs[key] = [baseline[key] + (step/num_steps)*(img-baseline[key]) for step in range(num_steps+1)]
        interpolated_inputs = {key: np.concatenate(interpolated_img,axis=0).astype(np.float32) for key, interpolated_img in interpolated_inputs.items()}
        
        # 2. Get the gradients
        grads = {key:[] for key in interpolated_inputs.keys()}
        for i in range(num_steps+1):
            tmp_itp_inputs = {key:tf.expand_dims(img[i],axis=0) for key,img in interpolated_inputs.items()}
            grad_dict = get_gradients(tmp_itp_inputs, model)
            for key,grad in grad_dict.items():
                grads[key].append(grad[0])
        grads = {key:tf.convert_to_tensor(grad, dtype=tf.float32) for key,grad in grads.items()}
        
        # 3. Approximate the integral using the trapezoidal rule
        grads = {key: (grad[:-1] + grad[1:]) / 2.0 for key,grad in grads.items()}
        avg_grads = {key: tf.reduce_mean(grad, axis=0) for key,grad in grads.items()} 

        # 4. Calculate integrated gradients
        integrated_grads = {key:(img-baseline[key])*avg_grads[key] for key,img in inputs.items()}

    else:
        inputs = inputs.astype(np.float32)
        if baseline is None:
            baseline = np.zeros(inputs.shape[1:]).astype(np.float32)
        else:
            baseline = baseline.astype(np.float32)
            
        # 1. Do interpolation.
        interpolated_inputs = [baseline + (step/num_steps)*(inputs-baseline) for step in range(num_steps+1)]
        interpolated_inputs = np.concatenate(interpolated_inputs,axis=0).astype(np.float32)
        
        # 2. Get the gradients
        grads = []
        for i, img in enumerate(interpolated_inputs):
            img = tf.expand_dims(img, axis=0)
            grad = get_gradients(img, top_pred_idx=top_pred_idx)
            grads.append(grad[0])
        grads = tf.convert_to_tensor(grads, dtype=tf.float32)
        
        # 3. Approximate the integral using the trapezoidal rule
        rads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = tf.reduce_mean(grads, axis=0)

        # 4. Calculate integrated gradients
        integrated_grads = (img_input - baseline) * avg_grads
    
    return integrated_grads


## Grad-CAM => Grad-RAM (regression activation map)
## Modified fchollet code (https://keras.io/examples/vision/grad_cam/),
## enabled structured input, and for regression output interpretation

def generate_grad_ram(inputs, model, activation_layer='',output_layer='', pred_index=None,
                      key_list = [], apply_clip = -1):
    """
    params:
    -------
    inputs: np.array(tensor of (1,*img)) or dictionary of array (structured input)
    model: my model
    activation_layer: layer name to visualize
    apply_clip: [-1,0,1] -1: negative relu / 0: do not apply anything / 1: relu

    return:
    grad_ram: grad_ram heatmap
    """
    if type(inputs) == dict:
        assert all([(key in inputs.keys()) for key in key_list])
        converted_inputs = [inputs[key] for key in key_list]
    else:
        converted_inputs = inputs
    
    ram_rank = len(model.get_layer(activation_layer).output.shape)-1
    grad_output_layer = model.get_layer(output_layer).output
    
    if pred_index != None:
        grad_output_layer = grad_output_layer[:,pred_index]
    grad_model = tf.keras.models.Model([model.get_input_at(0)], [model.get_layer(activation_layer).output, grad_output_layer])
    #print(grad_model.summary())
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([converted_inputs])
        loss = predictions
    #print(loss)
    #print('conv_outputs',conv_outputs.shape)
    output = conv_outputs[0]
    
    grads = tape.gradient(loss, conv_outputs)[0]

    axis         = tuple([i for i in range(ram_rank-1)])
    pooled_grads = tf.reduce_mean(grads, axis=axis)
    
    grad_ram = output @ pooled_grads[..., tf.newaxis]
    grad_ram = tf.squeeze(grad_ram).numpy()
    if   apply_clip == -1:
        grad_ram = np.clip(grad_ram,a_min=None, a_max=0)
    elif apply_clip ==  1:
        grad_ram = np.clip(grad_ram, a_min=0,   a_max=None)
        
    return grad_ram #same shape as activation layer output


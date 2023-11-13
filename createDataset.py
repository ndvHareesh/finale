import os
import numpy as np
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import os
import copy
import random
import time
from PIL import Image
import pandas as pd
import PIL
plt.ion()

def writeyourdigit():
    st.title("Create your dataset by drawing digits")
    SIZE = 400
    mode = st.checkbox("Draw Digit", True)
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=20,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=SIZE,
        height=SIZE,
        drawing_mode="freedraw" if mode else "transform",
        key='canvas')
    # if st.button("Save Image"):
    #     if canvas_result.image_data is not None:
    #         cv2.imwrite(f"test.jpg",  canvas_result.image_data)
    #     else:
    #         st.write("no image to save")
    st.write("Specify which digit you wrote, for storing it in dataset")

    col1, col2, col3,col4,col5,col6,col7,col8,col9,col10 = st.columns(10)
    with col1:
        button1 = st.button('0')
    with col2:
        button2 = st.button('1')
    with col3:
        button3 = st.button('2')
    with col4:
        button4 = st.button('3')
    with col5:
        button5 = st.button('4')
    with col6:
        button6 = st.button('5')
    with col7:
        button7 = st.button('6')
    with col8:
        button8 = st.button('7')
    with col9:
        button9 = st.button('8')
    with col10:
        button10 = st.button('9')
    def fun():
        None
    
    a = 0
    if button1:
        a += 1
        if canvas_result.image_data is not None:
            cv2.imwrite(f'./Dataset/train/0/draw_{a}.png',  canvas_result.image_data)
            fun()
        else:
            st.write("no image to save")
    if button2:
        a += 1

        if canvas_result.image_data is not None:
            cv2.imwrite(f'./Dataset/train/1/draw_{a}.png',  canvas_result.image_data)
            fun()
        else:
            st.write("no image to save")
    if button3:
        a += 1

        if canvas_result.image_data is not None:
            cv2.imwrite(f'./Dataset/train/2/draw_{a}.png',  canvas_result.image_data)
            fun()
        else:
            st.write("no image to save")
    if button4:
        a += 1

        if canvas_result.image_data is not None:
            cv2.imwrite(f'./Dataset/train/3/draw_{a}.png',  canvas_result.image_data)
            fun()
        else:
            st.write("no image to save")
    if button5:
        a += 1

        if canvas_result.image_data is not None:
            cv2.imwrite(f'./Dataset/train/4/draw_{a}.png',  canvas_result.image_data)
            fun()
        else:
            st.write("no image to save")
    if button6:
        a += 1

        if canvas_result.image_data is not None:
            cv2.imwrite(f'/./Dataset/train/5/draw_{a}.png',  canvas_result.image_data)
            fun()
        else:
            st.write("no image to save")
    if button7:
        a += 1
        if canvas_result.image_data is not None:
            cv2.imwrite(f'./Dataset/train/6/draw_{a}.png',  canvas_result.image_data)
            fun()
        else:
            st.write("no image to save")
    if button8:
        a += 1
        if canvas_result.image_data is not None:
            cv2.imwrite(f'./Dataset/train/7/draw_{a}.png',  canvas_result.image_data)
            fun()
        else:
            st.write("no image to save")
    if button9:
        a += 1
        if canvas_result.image_data is not None:
            cv2.imwrite(f'./Dataset/train/8/draw_{a}.png',  canvas_result.image_data)
            fun()
        else:
            st.write("no image to save")
    if button10:
        a += 1
        if canvas_result.image_data is not None:
            cv2.imwrite(f'./Dataset/train/9/draw_{a}.png',  canvas_result.image_data)
            fun()
        else:
            st.write("no image to save")

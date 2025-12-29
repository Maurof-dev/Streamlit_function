import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
from math import exp
import NN_represent_func as mod
from NN_represent_func import *



st.title("Build a Neural Network model to interpolate a function")





st.write('Interpolate a 1D function of choice over an interval with a Neural Network and with Python\n')
st.write('\nYou need:')
st.markdown(
    """
    - Python (Numpy, Matplotlib, Seaborn);
    - Sympy (symbolic expressions);
    - PyTorch (NN models);
    - Scikit-Learn (just the train-test splitting).
    """
)






st.sidebar.title("Input parameters")
function = st.sidebar.text_input("This is the function to be interpolated",'exp(-x**2/2.0)*sin(3.2*x)+cos(1.5*x)')
#saver = st.sidebar.radio('save trained model?',['y','n'])
#if saver == 'y':
namefile = st.sidebar.text_input("Trained model saved under","trained_NN_model")
wnoise = st.sidebar.radio('w/wo noise', ['y', 'n'])
xmin = st.sidebar.number_input("xmin",value=-3.0)
xmax = st.sidebar.number_input("xmax",value=3.0)
Npoints = st.sidebar.number_input("Number of x points",value=100)
nlayers = st.sidebar.number_input("Number of layers",value=2)
nneurons = st.sidebar.number_input("Neurons per layer",value=4)
size = st.sidebar.number_input("Size of the test set",value=0.25)
T = st.sidebar.number_input("Number of epochs (training)",value=1000)











start = st.button('Train NN model')
    
if 'result' not in st.session_state:
    st.session_state.result = None

if start:
    try:
        st.write('The model is being trained')
        model = mod.NeuralNetwork(nlayers,nneurons)
        function = sympify(function) #convert str into function
        update = mod.trainer(model,function,Npoints,T,size,xmin,xmax,wnoise) # return model, xlist, ylist <- function
        mod.tester(update[0],Npoints,update[1],update[2])
        x = 0.1
        y = x + 3 # final_res generated
    finally:
        st.session_state.result = update # update[0] should be the updated model
        


#stop = st.button('Save trained NN model') # issues when app is deployed
#if stop:
#    something = st.session_state.result[0]
#    Xl = st.session_state.result[1]
#    Yl = st.session_state.result[2]
#    torch.save(something,f'{namefile}')
#    #saver = torch.load('trained_NN_model',weights_only=True)
#    mod.tester(something,Npoints,Xl,Yl)

import io

buffer = io.BytesIO()
torch.save(model.state_dict(), buffer)
buffer.seek(0)

st.download_button(
    label="Download trained model",
    data=buffer,
    file_name=f"{namefile}.pt",
    mime="application/octet-stream",
)



count = 1
file_model = st.file_uploader(label='Upload an existing NN model and (re)plot',key=count)
if file_model:
    model = torch.load(file_model,weights_only=True)
    xlist = np.linspace(xmin,xmax,Npoints)
    x = symbols('x')
    function = sympify(function)
    ylist = np.array([function.subs(x,xi) for xi in xlist]) + np.random.normal(0,0.05,Npoints) 
    #if st.button('replot'):
    mod.tester(model,Npoints,xlist,ylist)
    




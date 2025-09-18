import streamlit as st
import pickle
import numpy as np
import math

# import the model
pipe= pickle.load(open('final_model.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))

st.title("Laptop Price Predictor")

#brand
company=st.selectbox('Brand',df['Company'].unique())

#type of laptop
type=st.selectbox('Type',df['TypeName'].unique())

#Ram
ram=st.selectbox('Ram(in Gb)',[2,4,6,8,12,16,24,32,64])

#Weight
weight=st.number_input('Weight of the laptop')

#Touchscreen
touchscreen=st.selectbox('Touchscreen',['No','Yes'])

#IPS
ips=st.selectbox('IPS',['No','Yes'])

#Screen Size
screen_size=st.number_input('Screen Size')

#Resolution
resolution=st.selectbox('Screen Resolution',['1366x768', '1920x1080', '2560x1600', '3840x2160'])

#cpu
cpu=st.selectbox('Cpu Brand',df['Cpu Brand'].unique())

#Hard drive
hdd=st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd=st.selectbox('SSD(in GB)',[0,128,256,512,1024,2048])

gpu=st.selectbox('Gpu Brand',df['Gpu Brand'].unique())

os=st.selectbox('OS',df['Os'].unique())

if st.button('Predict Price'):
    
    if touchscreen=='Yes':
        touchscreen=1
    else:
        touchscreen=0

    if ips=='Yes':
        ips=1
    else:
        ips=0

    X_res=int(resolution.split('x')[0])
    Y_res=int(resolution.split('x')[1])
    ppi=((X_res**2)+(Y_res**2))**0.5/screen_size


    query=np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query=query.reshape(1,12)
    st.title("The Predictied price of this configuration is ₹"+str(int(np.exp(pipe.predict(query)))))




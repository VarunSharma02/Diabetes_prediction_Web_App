# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 22:28:49 2024

@author: Asus Laptop
"""

import pickle
import streamlit as st
import numpy as np

loaded_model=pickle.load(open('C:/Machine_Learning/Deploy ML Model/Diabetes_Model/trained_model.sav','rb'))


def diabetes_prediction(input_data):
    input_data=(4,110,92,0,0,37.6,0.191,30)
    #changing the data to numpyt array
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    #Prediction of the species from the input vector
    prediction = loaded_model.predict(input_data_reshaped)


    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0]==0):
      return "The person is not diabetic"
    else:
      return "The person is diabetic" 
  
    
def main():
    st.title("Diabetic Prediction Model")
    
    Pregnancies=st.text_input("Enter the Pregnancies")
    Glucose=st.text_input("Enter the Glucose Level")
    BloodPressure=st.text_input("Enter the Blood Pressure")
    SkinThickness=st.text_input("Enter the SkinThickness")
    Insulin=st.text_input("Enter the Insulin level")
    BMI=st.text_input("Enter the BMI")
    DiabetesPedigreeFunction=st.text_input("Enter the  DiabetesPedigreeFunction")
    Age=st.text_input("Enter the Age")
    
    
    diagnose=''
    
    if st.button('Diabetes Test Reslt'):
         diagnose=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
         
        
    st.success(diagnose)
    
    
if __name__=='__main__':
    main()

    
    
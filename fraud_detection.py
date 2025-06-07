import streamlit as st
import pandas as pd
import joblib


model = joblib.load('deteccion_fraude_pipeline.pkl')

st.title('Deteccion de Fraude APP')


st.markdown('Por favor ingrese los detalles de la transacción y utilice el botón de predicción')

st.divider()

transaction_type = st.selectbox('Tipo de Transaccion', ['PAYMENT', 'TRANSFER', 'CASH_OUT','DEPOSIT'])
amount = st.number_input('Monto', min_value= 0.0, value = 1000.0)
oldbalanceOrg = st.number_input('saldo antiguo (remitente)', min_value= 0.0, value= 10000.0)
newbalanceOrig = st.number_input('Nuevo Balance (remitente)', min_value= 0.0, value= 9000.0)
oldbalanceDest = st.number_input('saldo antiguo (receptor)', min_value= 0.0, value=0.0)
newbalanceDest = st.number_input('Nuevo Balance (receptor)', min_value= 0.0, value=0.0)




if st.button('PREDECIR'):
    input_data = pd.DataFrame([{
        'type' : transaction_type,
        'amount' : amount,
        'oldbalanceOrg' : oldbalanceOrg,
        'newbalanceOrig' : newbalanceOrig,
        'oldbalanceDest' : oldbalanceDest,
        'newbalanceDest': newbalanceDest
    }])

    
    prediction = model.predict(input_data)[0]

    st.subheader(f"Prediccion : '{int(prediction)}'")

    if prediction == 1:
        st.error('Esta transacción puede ser un fraude')
    else:
        st.succes('Esta transacción no parece ser un fraude')


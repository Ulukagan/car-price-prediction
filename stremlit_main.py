import streamlit as st
import joblib
import pandas as pd 
import sklearn as sk
from PIL import Image

st.set_page_config(
        page_title="Car Price Prediction",
        page_icon="icon.png"
    )

image = Image.open('car_img.jpg')
st.image(image, caption='A Red Car', width=300)
st.markdown("<h1 style='text-align: right; color: red;'car_img.jpg", unsafe_allow_html=True)



#year
#year = st.number_input('Year', min_value=2014, max_value=2023, value=2019, step=1, label_visibility="visible")
year = st.selectbox("Manufacturing year of the car", (2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024))
#Present new price
price = st.slider('Price', min_value=15_000, max_value=100_000, value=3000, step=500, label_visibility="visible")

#Kms_Driven
kms = st.slider("Kms Driven", min_value=0, value=150_000, step=1000)
#Owner
owner = st.selectbox("How many owners does car have", (0,1,2,3))
#Fuel_Type_Diesel
#Fuel_Type_Petrol
fuel_type = st.radio('Fuel type', ('Diesel','Petrol'))
#Seller_Type_Individual
#Seller_Type_Dealer
seller_type = st.radio('Seller_type', ('Individual','Dealer'))
transmission = st.radio('Transmission_type', ('manual','automatic'))

columns = joblib.load('features_list.joblib')

user_input = [{
"Year":year,
"Selling_Price":price/10_000,
"Kms_Driven": kms,
"Fuel_Type":fuel_type,
"Seller_Type":seller_type,
"Transmission":transmission,
"Owner":owner
    }]

df_s = pd.DataFrame(user_input)

df_s["Year"] = 2023-df_s["Year"]
df_s = pd.get_dummies(df_s).reindex(columns=columns, fill_value=0)

scaler = joblib.load(open("scaler.joblib","rb"))
model = joblib.load(open("xgb_model.joblib","rb"))
df_s = scaler.transform(df_s)

bin_predict = st.button('Predict Price')

if bin_predict:
    pred_price = round(model.predict(df_s)[0] * 10_000)
    st.write(f"Your car's price: ${pred_price}")


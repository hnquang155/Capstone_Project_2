#------------ Import library ------------
import streamlit as st
import datetime
from datetime import timedelta

# Supress Warnings 
import warnings
warnings.filterwarnings('ignore')
import pickle

from PIL import Image
import numpy as np
import pandas as pd
import xarray as xr
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import Planetary Computer tools
import pystac_client
import planetary_computer as pc
from odc.stac import stac_load

import folium
from streamlit_folium import folium_static
#------------------------

#------------ Page ------------
st.markdown("# Page 2 ❄️")
st.sidebar.markdown("# Page 2 ❄️")
#------------------------

#------------ Add side bar ------------
st.sidebar.write("Select location, time frame and model")

# Add a selectbox to the sidebar:
selectbox = st.sidebar.selectbox('Model Selection: ', ('Sentinel-2', 'Landsat-8'))

# Latitude
lat = st.sidebar.slider('Latitude:', 9.00000000, 11.00000000)

# Longtitude
lon = st.sidebar.slider('Longtitude:', 104.00000000, 106.00000000)

# Time selection
date = st.sidebar.date_input("Time selection:", datetime.date(2022, 1, 10))
#------------------------

# (10.324181275911162, 105.28115947851899)

#------------ Functions ------------
def lat_lon(lat, lon):
    lat += np.random.randint(0, 1000000000) / 100000000000
    lon += np.random.randint(0, 1000000000) / 100000000000
    coordinates = (lat, lon) 
    return coordinates

def bounding_box(coordinates):
    box_size_deg = 0.10 # Surrounding box in degrees
    min_lon = coordinates[1]-box_size_deg/2
    min_lat = coordinates[0]-box_size_deg/2
    max_lon = coordinates[1]+box_size_deg/2
    max_lat = coordinates[0]+box_size_deg/2
    bounds = (min_lon, min_lat, max_lon, max_lat)
    return bounds

coordinates = lat_lon(lat, lon)
date_ = date + timedelta(days=1)
time_window = str(date) + '/' + str(date_) #date

#------------ Model Prediction ------------
def predict(xx):
    xx = xx.mean(dim=['latitude','longitude']).compute()
    red = xx['red'].values.tolist()[0]
    green = xx['green'].values.tolist()[0]
    blue = xx['blue'].values.tolist()[0]
    nir = xx['nir'].values.tolist()[0]

    NDVI = (nir-red)/(nir+red)
    EVI = 2.5*((nir-red)/(nir+6*red-7.5*blue+1))
    SAVI = ((nir-red)/(nir+red+0.5))*1.5
    user_input = np.array([[NDVI, EVI, SAVI]])

    # Model
    model = pickle.load(open('Sentinel2/sentinel2.pkl', "rb"))
    scaler = pickle.load(open('Sentinel2\\transform.pkl','rb'))
    scaled_input = scaler.transform(user_input)

    prediction = model.predict(scaled_input).tolist()[0]
    return prediction

def show_map(xx):
    result = predict(xx)
    st.write("Model prediction at point ({:.2f}, {:.2f}): ".format(coordinates[0], coordinates[1]), result)
    m = folium.Map(location=[coordinates[0], coordinates[1]], zoom_start=11)
    folium.Marker(location=[coordinates[0], coordinates[1]], tooltip=result).add_to(m)
    folium_static(m)
    

def predict_landsat(xx):
    xx = xx.mean(dim=['latitude','longitude']).compute()
    red = xx['red'].values.tolist()[0]
    green = xx['green'].values.tolist()[0]
    blue = xx['blue'].values.tolist()[0]
    nir = xx['nir08'].values.tolist()[0]
    lir = xx['lwir11'].values.tolist()[0]

    NDVI = (nir-red)/(nir+red)
    LIR = lir
    SWIR = (nir - (red-1*(blue-red))) / (nir + (red-1*(blue-red)))
    EVI = 2.5*((nir-red)/(nir+6*red-7.5*blue+1))
    SAVI = ((nir-red)/(nir+red+0.5))*1.5
    user_input = np.array([[NDVI, LIR, SWIR, EVI, SAVI]])

    # Model
    model = pickle.load(open('Landsat/landsatmodel.pkl', 'rb'))
    #scaler = joblib.load('transform.pkl')
    #scaled_input = scaler.transform(user_input)

    prediction = model.predict(user_input).tolist()[0]
    
    if (prediction == 1):
        prediction = 'Rice'
    else:
        prediction = 'Non Rice'
    
    return prediction

def show_map_landsat(xx):
    result = predict_landsat(xx)
    st.write("Model prediction at point ({:.2f}, {:.2f}): ".format(coordinates[0], coordinates[1]), result)
    m = folium.Map(location=[coordinates[0], coordinates[1]], zoom_start=11)
    folium.Marker(location=[coordinates[0], coordinates[1]], tooltip=result).add_to(m)
    folium_static(m)

#------------------------


# Sentinel-2
if (selectbox == 'Sentinel-2'):
    try:
        #------------ Get data ------------
        def get_sentinel_2_data(latlong, time_window):
            bbox_of_interest = bounding_box(latlong)

            stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
            search = stac.search(collections=["sentinel-2-l2a"], bbox=bbox_of_interest, datetime=time_window)
            items = list(search.get_all_items())
            
            resolution = 20  # meters per pixel 
            scale = resolution / 111320.0 # degrees per pixel for CRS:4326 
            
            xx = stac_load(
                items,
                bands=["red", "green", "blue", "nir", "SCL"],
                crs="EPSG:4326", # Latitude-Longitude
                resolution=scale, # Degrees
                chunks={"x": 2048, "y": 2048},
                dtype="uint16",
                patch_url=pc.sign,
                bbox=bbox_of_interest
            )
            return xx
        
        xx = get_sentinel_2_data(coordinates, time_window)
        #------------------------
        
        
        #------------ Color Map ------------
        scl_colormap = np.array(
            [
                [252,  40, 228, 255],  # 0  - NODATA - MAGENTA
                [255,   0,   4, 255],  # 1  - Saturated or Defective - RED
                [0  ,   0,   0, 255],  # 2  - Dark Areas - BLACK
                [97 ,  97,  97, 255],  # 3  - Cloud Shadow - DARK GREY
                [3  , 139,  80, 255],  # 4  - Vegetation - GREEN
                [192, 132,  12, 255],  # 5  - Bare Ground - BROWN
                [21 , 103, 141, 255],  # 6  - Water - BLUE
                [117,   0,  27, 255],  # 7  - Unclassified - MAROON
                [208, 208, 208, 255],  # 8  - Cloud - LIGHT GREY
                [244, 244, 244, 255],  # 9  - Definitely Cloud - WHITE
                [195, 231, 240, 255],  # 10 - Thin Cloud - LIGHT BLUE
                [222, 157, 204, 255],  # 11 - Snow or Ice - PINK
            ],
            dtype="uint8",
        )

        def colorize(xx, colormap):
            return xr.DataArray(colormap[xx.data], coords=xx.coords, dims=(*xx.dims, "band"))
        #----------------------------------------------

        #------------ Original RGB ------------
        xx[["red","green","blue"]].to_array().plot.imshow(col='time', robust=True, vmin=0, vmax=3000)
        plt.title("RGB Real Color")
        plt.axis("off")
        plt.savefig('Sample1.png', dpi=200)

        #------------ Cloud Mask Colormap ------------
        colorize(xx.SCL.compute(), scl_colormap).plot.imshow(col='time')
        plt.title("Cloud Mask Colormap")
        plt.axis("off")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.savefig('Sample2.png', dpi=200)

        #------------  Cloud / Shadows / Water Mask ------------
        cloud_mask = (xx.SCL != 0) & (xx.SCL != 1) & (xx.SCL != 3) & (xx.SCL != 6) & (xx.SCL != 8) & (xx.SCL != 9) & (xx.SCL != 10) 
        cleaned_data = xx.where(cloud_mask).astype("uint16")
        colorize(cleaned_data.SCL.compute(), scl_colormap).plot.imshow(col='time')
        plt.title("Cloud / Shadows / Water Mask")
        plt.axis("off")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.savefig('Sample3.png', dpi=200)

        #------------ Save and show images ------------
        image1 = Image.open('Sample1.png')
        image2 = Image.open('Sample2.png')
        image3 = Image.open('Sample3.png')
        st.image(image1)
        st.image(image2)
        st.image(image3)

        show_map(xx)
        
    except ValueError:
        st.write("There is no data on the selected date. Please choose another date.")

# Landsat
else:
    try:
        def get_landsat_data(latlong, time_window):
            bbox_of_interest = bounding_box(latlong)
            
            stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
            search = stac.search(
                collections=["landsat-c2-l2"], 
                bbox=bbox_of_interest, 
                datetime=time_window,
                query={"platform": {"in": ["landsat-8", "landsat-9"]},},
            )
            items = list(search.get_all_items())
            
            resolution = 30  
            scale = resolution / 111320.0 
            
            xx = stac_load(
                items,
                bands=["red", "green", "blue", "nir08", "qa_pixel", "lwir11"],
                crs="EPSG:4326", # Latitude-Longitude
                resolution=scale, # Degrees
                chunks={"x": 2048, "y": 2048},
                patch_url=pc.sign,
                bbox=bbox_of_interest
            )
            
            # Apply scaling and offsets for Landsat Collection-2 (reference below) to the spectral bands ONLY
            # https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2
            xx['red'] = (xx['red']*0.0000275)-0.2
            xx['green'] = (xx['green']*0.0000275)-0.2
            xx['blue'] = (xx['blue']*0.0000275)-0.2
            xx['nir08'] = (xx['nir08']*0.0000275)-0.2
                
            return xx

        
        xx_l = get_landsat_data(coordinates, time_window)

        def get_mask(mask, flags_list, bit_flags):
            final_mask = np.zeros_like(mask)
            for flag in flags_list:
                flag_mask = np.bitwise_and(mask, bit_flags[flag])
                final_mask = final_mask | flag_mask
            return final_mask > 0

        bit_flags = {'fill': 1<<0, 'dilated_cloud': 1<<1, 'cirrus': 1<<2, 'cloud': 1<<3, 'shadow': 1<<4, 'snow': 1<<5, 'clear': 1<<6, 'water': 1<<7}
        my_mask = get_mask(xx_l['qa_pixel'], ['fill', 'dilated_cloud', 'cirrus', 'cloud', 'shadow', 'water'], bit_flags)
        my_mask = np.array(my_mask).reshape((372, 372))

        #------------ Original RGB ------------
        plot_xx = xx_l[["red","green","blue"]].to_array()
        plt.figure(figsize=(10, 7))
        plot_xx.plot.imshow(col='time', robust=True, vmin=0, vmax=0.3)
        plt.title("RGB Real Color")
        plt.axis("off")
        plt.savefig('Sample1.png', dpi=200)

        #------------  Cloud / Shadows / Water Mask ------------
        plt.figure(figsize=(10, 7))
        plt.imshow(my_mask)
        plt.title("Cloud / Shadows / Water Mask")
        plt.savefig('Sample2.png', dpi=200)

        #------------ Save and show images ------------
        image1 = Image.open('Sample1.png')
        image2 = Image.open('Sample2.png')
        st.image(image1)
        st.image(image2)
        
        show_map_landsat(xx_l)
            
    except ValueError:
        st.write("There is no data on the selected date. Please choose another date.")

 


















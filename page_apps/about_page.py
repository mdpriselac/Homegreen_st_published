import streamlit as st
import pandas as pd
import json
from data_create import full_conn, flav_only_conn, raw_loaded_df

full_conn = full_conn
flav_only_conn = flav_only_conn
raw_loaded_df = raw_loaded_df


with open('./page_apps/modules/assets/text/about_data_process.md', 'r') as file:
    about_data_process = file.read()
with open('./page_apps/modules/assets/text/about_flavor_wheel.md', 'r') as file:
    about_flavor_wheel = file.read()
with open('./page_apps/modules/assets/images/base_flavor_wheel.jpg', 'rb') as file:
    base_flavor_wheel = file.read()
with open('./page_apps/modules/assets/images/original_flav_wheel.png', 'rb') as file:
    original_flav_wheel = file.read()
with open('./page_apps/modules/assets/text/about_home_greens.md', 'r') as file:
    about_home_greens = file.read()
with open('./page_apps/modules/assets/text/about_data_collection.md', 'r') as file:
    about_data_collection = file.read()
with open('./page_apps/modules/assets/text/about_predictive_models.md', 'r') as file:
    about_predictive_models = file.read()
with open('./page_apps/modules/assets/text/country_agg.md', 'r') as file:
    country_agg_overview = file.read()
with open('./page_apps/modules/assets/data//tf_idf_country_table.md','r') as file:
    country_tf_idf_table = file.read()
with open('./page_apps/modules/assets/text/flavor_agg.md', 'r') as file:
    flavor_agg_overview = file.read()


#wheel images for quality buckets
bucket_flavor_image_paths = [f'./page_apps/modules/assets/images/Top 15 most distinctive flavors for {i}.png' for i in range(1,8)]
bucket_images = {}
for bucket in range(1,8):
    with open(bucket_flavor_image_paths[bucket-1], 'rb') as file:
        bucket_images[bucket] = file.read()
    
def about_page():
    st.title('About the Home Greens Project')
    about_container = st.container()
    welcome, data_collection, data_processing, coffee_flavor_wheel,predictions = about_container.tabs(['Welcome', 'Data Collection', 'Data Processing', 'Flavor Wheel','Quality Predictions'])
    #Welcome tab
    welcome.subheader('Welcome to the Home Greens Project')
    welcome.markdown(about_home_greens)
    #Data Collection Tab
    data_collection.subheader('About our Data Collection')
    data_collection.markdown(about_data_collection)
    #Data Processing Tab
    data_processing.subheader('About our Data Processing')
    data_processing.markdown(about_data_process)
    #Flavor Wheel Tab
    coffee_flavor_wheel.subheader('About the Coffee Flavor Wheel')
    coffee_flavor_wheel.markdown(about_flavor_wheel)
    coffee_flavor_wheel.subheader('Our Base Flavor Wheel')
    coffee_flavor_wheel.image(base_flavor_wheel)
    coffee_flavor_wheel.subheader('SCAA Flavor Wheel')
    coffee_flavor_wheel.write('The wheel below is the wheel ours is modeled on.')
    coffee_flavor_wheel.image(original_flav_wheel)
    #Predictions Tab
    about_preds, country_agg, flavor_agg = predictions.tabs(['Predictive Model','Countries and Quality','Flavor and Quality'])
    #About Predictive Model Subtab
    about_preds.subheader('About our Predictive Models')
    about_preds.markdown(about_predictive_models)
    #Country Aggregations Subtab
    country_agg.subheader('Countries and Quality Buckets')
    country_agg.markdown(country_agg_overview)
    country_agg_table = country_agg.container()
    country_agg_table.markdown(country_tf_idf_table)
    #Flavor Aggregations Subtab
    flavor_agg.subheader('Flavors and Quality Buckets')
    flavor_agg.markdown(flavor_agg_overview)
    bucket_images_container = flavor_agg.container()
    img_1, img_2, img_3, img_4, img_5, img_6, img_7 = bucket_images_container.tabs(['Bucket 1', 'Bucket 2', 'Bucket 3', 'Bucket 4', 'Bucket 5', 'Bucket 6', 'Bucket 7'])
    #Bucket 1
    img_1.write('Top 15 most distinctive flavors for Bucket 1')
    img_1.image(bucket_images[1])
    #Bucket 2
    img_2.write('Top 15 most distinctive flavors for Bucket 2')
    img_2.image(bucket_images[2])
    #Bucket 3
    img_3.write('Top 15 most distinctive flavors for Bucket 3')
    img_3.image(bucket_images[3])
    #Bucket 4
    img_4.write('Top 15 most distinctive flavors for Bucket 4')
    img_4.image(bucket_images[4])
    #Bucket 5
    img_5.write('Top 15 most distinctive flavors for Bucket 5')
    img_5.image(bucket_images[5])
    #Bucket 6
    img_6.write('Top 15 most distinctive flavors for Bucket 6')
    img_6.image(bucket_images[6])
    #Bucket 7
    img_7.write('Top 15 most distinctive flavors for Bucket 7')
    img_7.image(bucket_images[7])
    

about_page()
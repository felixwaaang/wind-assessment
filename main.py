from utils.power_prod_bess_variation import power_bess
from utils.power_prod_height_variation import power_height
from utils.wind_analysis import wind_analysis
import scipy.special
import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from matplotlib.ticker import FuncFormatter
from utils.bilinear_interpolate import BilinearInterpolator
from jinja2 import Environment, FileSystemLoader
from utils.wind_analysis import wind_analysis
import math

######################################################

# location = 'bua_luang'
# lat = 10.5
# lng = 100.25

# location = 'namibia'
# lat = -22.5194
# lng = 15.25

# location = 'hang_tuah'
# lat = 4.15
# lng = 105.4

location = 'saudi'
lat = 28.25
lng = 34.9

######################################################

target_load = 100000
is_ocean = True

######################################################

pb = power_bess(location, lat, lng, target_load, is_ocean)
pb.process_data()

ph = power_height(location, lat, lng, target_load, is_ocean)
ph.process_data()

wa = wind_analysis(location, lat, lng, is_ocean)
wa.save_figures()   


datas = []
df = wa.prepare_data()
df_wind = df['wind_velocity']

for month in range(1, 13):
    monthly_data = []
    month_df = df_wind[df_wind.index.month == month]
    for hour in range(24):
        hour_df = month_df[month_df.index.hour == hour]
        print(hour_df)
        average = hour_df.mean()
        print(average)
        monthly_data.append(average)
    # print(monthly_data)
    datas.append(monthly_data)

df = pd.DataFrame([column for column in datas])
df = df.T
df.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

df.index.name = 'Hour\Month'
df['Average'] = (df.mean(axis=1)).round(2)

df.loc['Average'] = df.mean(axis=0).round(2)
df.loc['Minimum'] = df.min(axis=0)
df.loc['Maximum'] = df.max(axis=0)
df.loc['Range'] = df.loc['Maximum'] - df.loc['Minimum']

idx_min_list = []
idx_max_list = []

for i in range(13):
    min = df.iloc[-3,i]
    min_idx = df.loc[df.iloc[:,i] == min].index[0]
    idx_min_list.append(min_idx)

    max = df.iloc[-2,i]
    max_idx = df.loc[df.iloc[:,i] == max].index[0]
    idx_max_list.append(max_idx)

df.loc['time_of_min'] = idx_min_list
df.loc['time_of_max'] = idx_max_list

df.iloc[-1, -1] = ""
df.iloc[-2, -1] = ""
df = df.round(2)

df_html = df.to_html(classes='table table-striped', index=True)

df_copy = df.copy()
df_copy = df_copy.drop("Average", axis=1)
df_copy.drop(df_copy.tail(6).index, inplace=True)

for i in range(len(df_copy.columns)):
    col = df_copy.columns[i]
    df_copy[col] = (df_copy[col] - df.iloc[-6, i]) / df.iloc[-3, i]

df_copy *= 100
styled_df = df_copy.style.format("{:.0f}%").background_gradient(cmap="RdYlGn")

df_html_styled = styled_df.to_html(classes='table table-striped', index=True)

# Define paths
png_dir = 'pdf_elements'  # Directory where PNG files are saved
html_output_path = 'report.html'  # Output path for the HTML report

# Save the styled DataFrame as HTML
def save_styled_df_as_html(styled_df, file_path):
    html = styled_df.to_html()  # Use to_html() to get the HTML representation of the styled DataFrame
    with open(file_path, 'w') as f:
        f.write(html)

# Create the HTML report
def create_html_report(output_path, figure_1_1, ds_input, por_input, avg_ws, avg_ws_corrected, avg_density, figure_1_2, 
                       avg_ws_h_corrected, wsd_input, figure_1_3, seasonality_pct, high_season_m, figure_1_4, ws_month_peak,
                       figure_1_5, figure_1_6, df_html, df_html_styled, figure_1_7, figure_1_8, figure_1_9, figure_1_10,
                       figure_1_11, figure_1_12):
    # Load the Jinja2 template
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('report_template.html')

    # Prepare the context for the template
    context = {
        'figure_1_1': figure_1_1,
        'datasource_input': ds_input,
        'PoR_input': por_input,
        'avg_ws': avg_ws,
        'avg_ws_corrected': avg_ws_corrected,
        'avg_density': avg_density,
        'figure_1_2': figure_1_2,
        'avg_ws_h_corrected': avg_ws_h_corrected,
        'wsd_input': wsd_input,
        'figure_1_3': figure_1_3,
        'seasonality_pct': seasonality_pct,
        'high_season_m': high_season_m,
        'figure_1_4': figure_1_4,
        'ws_month_peak': ws_month_peak,
        'figure_1_5': figure_1_5,
        'figure_1_6': figure_1_6,
        'df_html': df_html,
        'df_html_styled': df_html_styled,
        'figure_1_7': figure_1_7,
        'figure_1_8': figure_1_8,
        'figure_1_9': figure_1_9,
        'figure_1_10': figure_1_10,
        'figure_1_11': figure_1_11,
        'figure_1_12': figure_1_12,
    }

    # Render the HTML report
    html_content = template.render(context)

    # Write the HTML content to a file
    with open(output_path, 'w') as f:
        f.write(html_content)

def main():
    df = wa.prepare_data()

    figure_1_1 = {
    'src': f'pdf_elements/{wa.location}/data_source.png',
    }

    figure_1_2 = {
    'src': f'pdf_elements/{wa.location}/weibull_dist.png',
    }

    figure_1_3 = {
    'src': f'pdf_elements/{wa.location}/wind_rose.png',
    }

    figure_1_4 = {
    'src': f'pdf_elements/{wa.location}/seasonality_plot.png',
    }

    figure_1_5 = {
    'src': f'pdf_elements/{wa.location}/avg_monthly_ws.png',
    }

    figure_1_6 = {
    'src': f'pdf_elements/{wa.location}/monthly_wind_rose.png',
    }

    figure_1_7 = {
    'src': f'pdf_elements/{wa.location}/2412.png',
    }

    figure_1_8 = {
    'src': f'pdf_elements/{wa.location}/avg_direction_ws.png',
    }

    figure_1_9 = {
    'src': f'pdf_elements/{wa.location}/power_bess_plots.png',
    }

    figure_1_10 = {
    'src': f'pdf_elements/{wa.location}/power_height_plots.png',
    }

    figure_1_11 = {
    'src': f'pdf_elements/{wa.location}/net_capacity.png',
    }

    figure_1_12 = {
    'src': f'pdf_elements/{wa.location}/power_wtgs_plots.png',
    }

    month_mapping = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

    # Get the month name based on the calculated month number
    month_name = month_mapping.get(math.floor(wa.get_polar_seasonality_plot(df, no_plot=True)[2]/30+1))
    # Create the HTML report
    create_html_report(html_output_path, figure_1_1, wa.get_datasource_input(), wa.get_por_input(df), wa.get_ws_input(df)[0], wa.get_ws_input(df)[1], 
                       wa.get_ws_input(df)[2], figure_1_2, wa.get_ws_input(df)[3], wa.get_wsd_input(df), figure_1_3, 
                       round(wa.get_polar_seasonality_plot(df, no_plot=True)[0]*100, 3), month_name,
                       figure_1_4, wa.get_ws_month_peak_input(df), figure_1_5, figure_1_6,
                       df_html, df_html_styled, figure_1_7, figure_1_8, figure_1_9, figure_1_10, figure_1_11, figure_1_12)

main()

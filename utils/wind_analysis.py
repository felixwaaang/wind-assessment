from utils.calculate_power_production import PowerCalculator
from utils.power_curve import PowerCurve
from utils.era5_download import DataDownloader
from utils.bilinear_interpolate import BilinearInterpolator
import scipy.stats as stats
from scipy.optimize import leastsq
import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from matplotlib.ticker import FuncFormatter
import imgkit
import windrose
from windrose import WindroseAxes, plot_windrose
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import math
import seaborn as sns

interpolator = BilinearInterpolator()

class wind_analysis:
    def __init__(self, location, latitude, longitude, is_ocean):

        self.location = location
        self.lat = latitude
        self.lng = longitude
        self.is_ocean = is_ocean

        nc_file = f'era5_data/{self.location}.nc'  # Replace with your NetCDF file path
        self.ds = nc.Dataset(nc_file, 'r')
        time_var = self.ds.variables['valid_time']
        time_units = time_var.units
        time_calendar = time_var.calendar
        time_values = time_var[:]
        times = nc.num2date(time_values, units=time_units, calendar=time_calendar)

        self.latitude_list = self.ds.variables['latitude'][:].tolist()
        self.longitude_list =  self.ds.variables['longitude'][:].tolist()
        self.times_list = [time.strftime('%Y-%m-%d %H:00:00') for time in times]

        self.bboxs = interpolator.find_bounding_box(self.lat, self.lng, self.latitude_list, self.longitude_list)


    def prepare_data(self, correction: bool=False):
        bounding_box = interpolator.find_bounding_box(self.lat, self.lng, self.latitude_list, self.longitude_list)
        wind_dfs = [interpolator.get_df(self.ds, pair[0], pair[1], self.latitude_list, self.longitude_list, self.times_list, self.is_ocean) for pair in bounding_box]

        u100_interpolated = interpolator.bilinear_interpolation(
            self.lat, self.lng, 
            bounding_box[0][0], bounding_box[0][1], 
            bounding_box[3][0], bounding_box[3][1], 
            wind_dfs[0]['u100'].to_list(), 
            wind_dfs[2]['u100'].to_list(), 
            wind_dfs[1]['u100'].to_list(), 
            wind_dfs[3]['u100'].to_list()
        )

        v100_interpolated = interpolator.bilinear_interpolation(
            self.lat, self.lng, 
            bounding_box[0][0], bounding_box[0][1], 
            bounding_box[3][0], bounding_box[3][1], 
            wind_dfs[0]['v100'].to_list(), 
            wind_dfs[2]['v100'].to_list(), 
            wind_dfs[1]['v100'].to_list(), 
            wind_dfs[3]['v100'].to_list()
        )

        sp_interpolated = interpolator.bilinear_interpolation(
            self.lat, self.lng, 
            bounding_box[0][0], bounding_box[0][1], 
            bounding_box[3][0], bounding_box[3][1], 
            wind_dfs[0]['sp'].to_list(), 
            wind_dfs[2]['sp'].to_list(), 
            wind_dfs[1]['sp'].to_list(), 
            wind_dfs[3]['sp'].to_list()
        )

        temp_interpolated = interpolator.bilinear_interpolation(
            self.lat, self.lng, 
            bounding_box[0][0], bounding_box[0][1], 
            bounding_box[3][0], bounding_box[3][1], 
            wind_dfs[0]['temperature'].to_list(), 
            wind_dfs[2]['temperature'].to_list(), 
            wind_dfs[1]['temperature'].to_list(), 
            wind_dfs[3]['temperature'].to_list()
        )

        assert len(u100_interpolated) == len(v100_interpolated) == len(sp_interpolated) == len(temp_interpolated)

        wind_velocity = [(x**2 + y**2)**0.5 for x, y in zip(u100_interpolated, v100_interpolated)]
        direction = [(180 + np.degrees(np.arctan2(u, v))) % 360 for u, v in zip(u100_interpolated, v100_interpolated)]

        #air density correction
        hws_density_corrected = [hws * ((sp / (287.058 * temp))/1.225)**(1/3) for hws, sp, temp in zip(wind_velocity, sp_interpolated, temp_interpolated)]
        print(hws_density_corrected[:10])

        #height correction (no currection at the moment)
        if self.is_ocean is True:
            hws_density_corrected = [hws * (100/100)**0.1 for hws in hws_density_corrected] #0.143 if on land
            print(hws_density_corrected[:10])
        else:
            hws_density_corrected = [hws * (100/100)**0.143 for hws in hws_density_corrected] 


        df = pd.DataFrame({
            'u100': u100_interpolated,
            'v100': v100_interpolated,
            'wind_velocity': wind_velocity,
            'direction': direction,
            'sp': sp_interpolated,
            'temperature': temp_interpolated
        }, index=self.times_list)
        df.index = pd.to_datetime(df.index)

        if correction is True:
            print('density and height corrected')
            df['wind_velocity'] = (df['wind_velocity'] * ((df['sp'] / (287.058 * df['temperature'])/1.225)**(1/3)))
            df['wind_velocity'] = (df['wind_velocity'] * (150/100)**0.1)

        return df

    def get_2412_data(self, df):
        datas = []
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

        return df

    def get_datasource_input(self):
        if self.lng % 0.25 == 0 and self.lat % 0.25 == 0:
            result = f'The potential site of the wind farm is situated at one of the ERA5 data node (as shown in Figure 1.1). The following calculations and data are all provided by this node at ({self.lat}, {self.lng}) '
            return(result)
        
        if self.lng % 0.25 == 0 and self.lat % 0.25 != 0:
            result = f'Four close ERA5 data nodes are used to interpolate the condition at the potential site of the wind farm ({self.lat}, {self.lng}). The two nodes are located at: ({self.bboxs[0][0]}, {self.lng}) and ({self.bboxs[1][0]}, {self.lng}), respectively, as shown in Figure 1.1'
            return(result)
        
        if self.lng % 0.25 != 0 and self.lat % 0.25 == 0:
            result = f'Four close ERA5 data nodes are used to interpolate the condition at the potential site of the wind farm ({self.lat}, {self.lng}). The four nodes are located at: ({self.lat}, {self.bboxs[0][1]}) and ({self.lat}, {self.bboxs[1][1]})respectively, as shown in Figure 1.1'
            return(result)
        
        else:
            result = f'Four close ERA5 data nodes are used to interpolate the condition at the potential site of the wind farm ({self.lat}, {self.lng}). The four nodes are located at: ({self.bboxs[0][0]}, {self.bboxs[0][1]}), ({self.bboxs[1][0]}, {self.bboxs[1][1]}), ({self.bboxs[2][0]}, {self.bboxs[2][1]}), ({self.bboxs[3][0]}, {self.bboxs[3][1]}) respectively, as shown in Figure 1.1'
            return result

    def get_por_input(self, df):
        start_date = str(df.index.min())
        end_date = str(df.index.max())
        years = str(df.index.year[-1] - df.index.year[0] + 1)
        result = f"{years} years from {start_date} to {end_date}"
        return result
    
    def get_ws_input(self, df):
        avg_ws = df.wind_velocity.mean()
        avg_ws_corrected = (df['wind_velocity'] * ((df['sp'] / (287.058 * df['temperature'])/1.225)**(1/3))).mean()
        avg_density = (df['sp'] / (287.058 * df['temperature'])).mean()
        avg_ws_h_corrected = (avg_ws_corrected * (150/100)**0.1).mean()

        avg_ws = round(avg_ws, 3)
        avg_ws_corrected = round(avg_ws_corrected, 3)
        avg_density = round(avg_density, 3)
        avg_ws_h_corrected = round(avg_ws_h_corrected, 3)
        return avg_ws, avg_ws_corrected, avg_density, avg_ws_h_corrected
    
    def get_wsd_input(self, df):
        ws = df.wind_velocity.values
        wd = df.direction.values

        ax = WindroseAxes.from_ax()
        ax.bar(wd, ws, bins=np.arange(0, 8, 1), lw=3, nsector=32)
        plt.close()
        
        dir_bound = sorted(ax._info['dir'])
        bin_indices = np.digitize(wd, dir_bound, right=False)
        wind_direction_counts = np.bincount(bin_indices, minlength=len(dir_bound))

        relative_main_directions_indices = []

        for i in range(0, len(wind_direction_counts) - 1):
            if (wind_direction_counts[i] > wind_direction_counts[i - 1]) and  (wind_direction_counts[i] > wind_direction_counts[i + 1]) and (wind_direction_counts[i] > wind_direction_counts.mean()):   
                relative_main_directions_indices.append(i)  

        if (wind_direction_counts[-1] > wind_direction_counts[i - 1]) and  (wind_direction_counts[-1] > wind_direction_counts[0]) and (wind_direction_counts[-1] > wind_direction_counts.mean()):
            relative_main_directions_indices.append(-1)  
        
        rel_main_dir = []
        for dir_indice in relative_main_directions_indices:
            if dir_indice == 0:
                dir = (dir_bound[0] + dir_bound[-1]) % 360
                rel_main_dir.append(dir)
            
            else:
                dir = dir_bound[dir_indice]
                rel_main_dir.append(dir)

        def degrees_to_cardinal(degrees):
            directions = [
                "North", "North-Northeast", "Northeast", "East-Northeast", "East", 
                "East-Southeast", "Southeast", "South-Southeast", "South", 
                "South-Southwest", "Southwest", "West-Southwest", "West", 
                "West-Northwest", "Northwest", "North-Northwest"
            ]
            index = int((degrees + 11.25) // 22.5) % 16
            return directions[index]

        cardinal_directions = [degrees_to_cardinal(direction) for direction in rel_main_dir]
        wd_expanded = np.expand_dims(wd, axis=1) // 10
        ws_expanded = np.expand_dims(ws, axis=1)
        data = np.hstack((ws_expanded, wd_expanded))

        wind_strength = []
        for value in rel_main_dir:
            filtered_values = data[data[:, 1] == int(value / 10)][:, 0]
            average = np.mean(filtered_values)
            wind_strength.append(average)

        direction_strengths = [f"winds from {cardinal_directions[i]} with average wind speed of {wind_strength[i]:.2f} m/s" for i in range(len(wind_strength))]
        result = (
            f"There are {len(wind_strength)} main directions represented in the wind rose, "
            f"including average winds: {', '.join(direction_strengths)}."
        )
        return result
    
    def get_ws_month_peak_input(self, df):
        df["month"] = df.index.month
        monthly_avg_wind_speed = df.groupby('month')['wind_velocity'].mean()
        mean_wind_speed = monthly_avg_wind_speed.mean()
        std_wind_speed = monthly_avg_wind_speed.std()

        threshold = mean_wind_speed + std_wind_speed
        significantly_higher_months = monthly_avg_wind_speed[monthly_avg_wind_speed > threshold]
        significantly_higher_months = significantly_higher_months.index.tolist()

        month_mapping = {1: "January",2: "February",3: "March",4: "April",5: "May",6: "June",7: "July",8: "August",9: "September",10: "October",11: "November",12: "December"}
        significantly_higher_month_names = [month_mapping[month] for month in significantly_higher_months]

        if len(significantly_higher_month_names) == 2:
            result_string = ' and '.join(significantly_higher_month_names)
        elif len(significantly_higher_month_names) > 2:
            result_string = ', '.join(significantly_higher_month_names[:-1]) + ', and ' + significantly_higher_month_names[-1]
        else:
            result_string = significantly_higher_month_names[0] if significantly_higher_month_names else ''

        print(result_string)
        return result_string

    
    def get_weibull_plot(self, df):
        file_path = f"pdf_elements/{self.location}/weibull_dist.png" 

        if os.path.exists(file_path):
            return
        
        shape, loc, scale = stats.weibull_min.fit(df['wind_velocity'], floc=0)
        plt.figure(figsize=(10, 6))
        plt.hist(df['wind_velocity'], bins=50, density=True, alpha=0.6, color='g', edgecolor='black')
        x = np.linspace(df['wind_velocity'].min(), df['wind_velocity'].max(), 100)
        pdf = stats.weibull_min.pdf(x, shape, loc, scale)

        plt.xlim(right=25)
        plt.plot(x, pdf, 'r-', lw=2, label='Weibull PDF')
        plt.title('Wind Velocity Distribution with Weibull Fit')
        plt.xlabel('Wind Velocity (m/s)')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(file_path)
        plt.close()

    def get_datasource_plot(self):
        file_path = f"pdf_elements/{self.location}/data_source.png"

        if os.path.exists(file_path):
            return

        minlon, maxlon, minlat, maxlat = (math.floor(self.lng-1), math.ceil(self.lng+1), math.floor(self.lat-1), math.ceil(self.lat+1))
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(6, 6))

        main_ax = fig.add_subplot(1, 1, 1, projection=proj)
        main_ax.set_extent([minlon, maxlon, minlat, maxlat], crs=proj)
        main_ax.gridlines(draw_labels=True)
        main_ax.coastlines()

        request = cimgt.OSM()
        main_ax.add_image(request, 12)

        if self.lng % 0.25 == 0 and self.lat % 0.25 == 0:
            main_ax.scatter(self.lng, self.lat, color='red', s=30, label='Site', zorder=5) 
            main_ax.annotate(f'({self.lat}, {self.lng})', xy=(self.lng, self.lat), xytext=(self.lng, self.lat + 0.1),
                fontsize=10, color='black',
                arrowprops=dict(facecolor='black', arrowstyle='->'))
            
        elif self.lng % 0.25 == 0 and self.lat % 0.25 != 0:
            main_ax.scatter(self.lng, self.lat, color='red', s=30, label='Site', zorder=5)
            main_ax.annotate(f'({self.lat}, {self.lng})', xy=(self.lng, self.lat), xytext=(self.lng, self.lat + 0.1),
                            fontsize=10, color='black',
                            arrowprops=dict(facecolor='black', arrowstyle='->'))
            for lat in [self.bboxs[0][0], self.bboxs[1][0]]:
                main_ax.scatter(self.lng, lat, color='blue', s=10, label=f'ERA5 node ({lat}, {self.lng})', zorder=5)

        # Case 3: Only lat is divisible by 0.25
        elif self.lng % 0.25 != 0 and self.lat % 0.25 == 0:
            main_ax.scatter(self.lng, self.lat, color='red', s=30, label='Site', zorder=5)
            main_ax.annotate(f'({self.lat}, {self.lng})', xy=(self.lng, self.lat), xytext=(self.lng, self.lat + 0.1),
                            fontsize=10, color='black',
                            arrowprops=dict(facecolor='black', arrowstyle='->'))
            for lng in [self.bboxs[0][1], self.bboxs[1][1]]:
                main_ax.scatter(lng, self.lat, color='blue', s=10, label=f'ERA5 node ({self.lat}, {lng})', zorder=5)

        else:
            i = 0
            main_ax.scatter(self.lng, self.lat, color='red', s=30, label='Site', zorder=5) 
            main_ax.annotate(f'({self.lat}, {self.lng})', xy=(self.lng, self.lat), xytext=(self.lng, self.lat + 0.1), 
                fontsize=10, color='black',
                arrowprops=dict(facecolor='black', arrowstyle='->'))
            for pair in self.bboxs:
                i += 1
                lat = pair[0]
                lng = pair[1]
                main_ax.scatter(lng, lat, color='blue', s=10, label=f'ERA5 node {i}', zorder=5)

        main_ax.legend()

        plt.savefig(file_path)
        plt.close()

    def get_polar_seasonality_plot(self, df, no_plot:str=False):
        file_path = f"pdf_elements/{self.location}/seasonality_plot.png"

        if os.path.exists(file_path) and no_plot == False:
            return

        df['hour_index'] = (df.index - df.index.to_period('Y').start_time).total_seconds() / 3600
        df['hour_index'] = df['hour_index'].astype(int)
        df['angles'] = (df['hour_index'] / 8760) * 2 * np.pi
        angles = df['angles'].values
        wind_velocity = df['wind_velocity'].values
        x = wind_velocity * np.cos(angles)
        y = wind_velocity * np.sin(angles)

        def circle_error(params, x, y):
            h, k, R = params
            return (np.sqrt((x - h)**2 + (y - k)**2) - R).tolist()

        def fit_circle(x, y):
            h0, k0 = np.mean(x), np.mean(y)
            R0 = np.mean(np.sqrt((x - h0)**2 + (y - k0)**2))
            params_initial = [h0, k0, R0]
            params_opt, _ = leastsq(circle_error, params_initial, args=(x, y))

            return params_opt

        h, k, R = fit_circle(x, y)

        displacement = np.sqrt(h**2 + k**2)
        direction = np.arctan2(k, h)  # Angle in radians
        total_hours = 8760
        angle_per_hour = 2 * np.pi / total_hours
        displacement_hour = direction / angle_per_hour

        displacement_hour = displacement_hour % total_hours
        avg_wind_speed = np.average(wind_velocity)
        seasonality = (displacement / (avg_wind_speed * (1 / (np.pi / 4)))) 
        direction_degrees = np.degrees(direction) % 360

        if no_plot == True:
            return seasonality, displacement, direction_degrees

        else:
            # Create a polar plot
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            ax.set_theta_direction(-1)  # Clockwise direction
            ax.set_theta_zero_location('N')
            ax.scatter(angles, wind_velocity, c='red', s=1)

            # Convert fitted circle to polar coordinates for plotting
            theta_fit = np.linspace(0, 2 * np.pi, 100)
            x_fit = h + R * np.cos(theta_fit)
            y_fit = k + R * np.sin(theta_fit)

            # Convert fitted circle back to polar coordinates
            angles_fit = np.arctan2(y_fit, x_fit) % (2 * np.pi)
            r_fit = np.sqrt(x_fit**2 + y_fit**2)

            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            angles_labels = np.linspace(0, 2 * np.pi, len(months), endpoint=False)  # Angles for the labels

            ax.set_xticks(angles_labels)  
            ax.set_xticklabels(months)  

            ax.plot(angles_fit, r_fit, 'b-', label='Fitted Circle')
            center_angle = np.arctan2(k, h)  # Angle of the displacement
            center_radius = np.sqrt(h**2 + k**2)
            ax.plot([0, center_angle], [0, center_radius], 'b--', label='Displacement')
            ax.plot(center_angle, center_radius, 'bo', markersize=5, label='Circle Center')

            ax.legend(bbox_to_anchor=(0.8, 0.8))
            plt.savefig(file_path)
            plt.close()

    def get_windrose(self, df):
        file_path = f"pdf_elements/{self.location}/wind_rose.png"

        if os.path.exists(file_path):
            return

        ax = WindroseAxes.from_ax()
        ax.bar(df['direction'], df['wind_velocity'], lw=3, nsector=32, normed=True)

        dir_bound = sorted(ax._info['dir'])
        bin_indices = np.digitize(df['direction'], dir_bound, right=False)
        wind_direction_counts = np.bincount(bin_indices, minlength=len(dir_bound))
        y_max = wind_direction_counts.max() / wind_direction_counts.sum() * 100

        y_ticks = range(0, math.ceil(y_max + 3), 5)
        y_tick_labels = [f'{tick}%' for tick in y_ticks]
        ax.set_rgrids(y_ticks, y_tick_labels)
        ax.set_legend()
        ax.set_title(f'{self.location} Wind Rose')
        plt.savefig(file_path)
        plt.close()

    def get_monthly_ws_avg(self, df):
        file_path = f"pdf_elements/{self.location}/avg_monthly_ws.png"

        if os.path.exists(file_path):
            return

        df["month"] = df.index.month
        monthly_avg_wind_speed = df.groupby('month')['wind_velocity'].mean()
        plt.figure(figsize=(12, 6))
        monthly_avg_wind_speed.plot(kind='bar')

        plt.ylim(0, monthly_avg_wind_speed.max()*1.25)
        plt.title('Monthly Average Wind Speed')
        plt.xlabel('Month')
        plt.ylabel('Average Wind Speed (units)')
        plt.xticks(rotation=45)
        plt.savefig(file_path)
        plt.close()

    def get_direction_ws_avg(self, df):
        file_path = f"pdf_elements/{self.location}/avg_direction_ws.png"

        if os.path.exists(file_path):
            return
        
        ax = WindroseAxes.from_ax()
        ax.bar(df['direction'], df['wind_velocity'], nsector=16, normed=True)
        table = ax._info['table']
        wd_freq = np.sum(table, axis=0)

        plt.figure()

        plt.bar(np.arange(16), wd_freq, align='center')
        xlabels = ('N','','N-E','','E','','S-E','','S','','S-W','','W','','N-W','')
        xticks=np.arange(16)
        plt.gca().set_xticks(xticks)
        plt.gca().set_xticklabels(xlabels)
        plt.ylabel('wind speed')
        plt.savefig(file_path)
        plt.close()
        

    def get_monthly_windrose(self, df):
        file_path = f"pdf_elements/{self.location}/monthly_wind_rose.png"

        if os.path.exists(file_path):
            return
        
        df["month"] = df.index.month

        def plot_windrose_subplots(data, *, direction, var, color=None, **kwargs):
            """wrapper function to create subplots per axis"""
            ax = plt.gca()
            ax = WindroseAxes.from_ax(ax=ax)
            plot_windrose(direction_or_df=data[direction], var=data[var], ax=ax, **kwargs)

        g = sns.FacetGrid(
        data=df,
        # the column name for each level a subplot should be created
        col="month",
        # place a maximum of 3 plots per row
        col_wrap=3,
        subplot_kws={"projection": "windrose"},
        sharex=False,
        sharey=False,
        despine=False,
        height=3.5,
    )

        g.map_dataframe(
            plot_windrose_subplots,
            direction="direction",
            var="wind_velocity",
            normed=True,
            # manually set bins, so they match for each subplot
            bins=(1, 4, 7, 11, 17, 21),
            calm_limit=0.1,
            kind="bar",
        )

        # make the subplots easier to compare, by having the same y-axis range
        y_ticks = range(0, 50, 8)
        y_tick_labels = [f'{tick}%' for tick in y_ticks]

        for ax in g.axes:
            ax.set_legend(
                title=r"$m \cdot s^{-1}$", bbox_to_anchor=(1.3, -0.3), loc="lower right"
            )
            ax.set_rgrids(y_ticks, y_tick_labels)

        # adjust the spacing between the subplots to have sufficient space between plots
        plt.subplots_adjust(wspace=0.2)
        plt.savefig(file_path)
        plt.close()

    def get_2412_calculation(self, df):
        file_path = f"pdf_elements/{self.location}/2412.png"

        if os.path.exists(file_path):
            return
        
        df_copy = self.get_2412_data(df)
        df_copy = df_copy.drop("Average", axis=1)
        x_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        y_values = df_copy.loc['Average']
        z_values = df_copy.loc['Minimum']
        w_values = df_copy.loc['Maximum']
        v_values = df_copy.loc['time_of_min']
        u_values = df_copy.loc['time_of_max']

        plt.figure(figsize=(14, 6))

        # Primary y-axis (left)
        ax1 = plt.gca()
        ax1.plot(x_values, y_values, label='Average HWS', color='black')
        ax1.plot(x_values, z_values, label='Minimum HWS', color='black', linestyle='--')
        ax1.plot(x_values, w_values, label='Maximum HWS', color='black', linestyle='-.')

        plt.xlabel('Months')
        plt.ylabel('Wind Speed (m/s)')

        # Secondary y-axis (right)
        ax2 = plt.gca().twinx()
        ax2.plot(x_values, v_values, label='Time of Minimum', color='red')
        ax2.plot(x_values, u_values, label='Time of Maximum', color='green')

        ax2.set_ylabel('Time of Day(hour)')


        # Combine legends from both y-axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()


        ax1.set_xlim(1,12)
        ax2.set_ylim(0, 23)

        major_locator = plt.MultipleLocator(2)
        ax2.yaxis.set_major_locator(major_locator)


        plt.legend(lines + lines2, labels + labels2, loc='upper center',bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
        plt.savefig(file_path,  bbox_inches='tight')
        plt.close()

    def save_figures(self):
        df = self.prepare_data(correction=True)

        directory = f'pdf_elements/{self.location}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 1.1 Datasource
        self.get_datasource_plot()

        # 1.2 Weibull Distribution
        self.get_weibull_plot(df)

        # 1.3 Windrose Directionality
        self.get_windrose(df)

        # 1.4 Seasonality
        self.get_polar_seasonality_plot(df)

        # 1.5 Seasonality Monthly Average
        self.get_monthly_ws_avg(df)

        # 1.6 Monthly Wind Rose
        self.get_monthly_windrose(df)

        # 1.7 2412 Calculation plots
        self.get_2412_calculation(df)

        self.get_direction_ws_avg(df)





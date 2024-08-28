import netCDF4 as nc
import numpy as np
import pandas as pd
from typing import List
from datetime import datetime

class BilinearInterpolator():
    def find_bounding_box(self, input_lat, input_long, latitude_list, longitude_list) -> List:
        """
        Summary:
            This function finds the four nearby points that form a square emcompassing the point of desired

        Args:
            input_lat (float): input latitude
            input_long (float): input longitude
            latitude_list (list): a list of latitudes with available data
            longitude_list (list): a list of the longitudes with available data

        Returns:
            List: a list that contains the four points that form a square grid around the point (input_lat, input_long). 
            pt_1 is the minimum point, pt_4 is the maximum point
        """        

        if (input_lat in latitude_list) and (input_long in longitude_list):
            print('case 0')


            pt_original = [input_lat, input_long]
            print(pt_original)
            print(f"The input location {input_lat, input_long} is already in the dataset.")
            print([pt_original, pt_original, pt_original, pt_original])

            return [pt_original, pt_original, pt_original, pt_original]


        elif (input_lat in latitude_list) and (input_long not in longitude_list):
            print('case 1')
            lower_bound_long = max([num for num in longitude_list if num < input_long])
            upper_bound_long = min([num for num in longitude_list if num > input_long])

            pt_1 = [input_lat, lower_bound_long]
            pt_2 = [input_lat, upper_bound_long]
            pt_3 = [input_lat, lower_bound_long]
            pt_4 = [input_lat, upper_bound_long]
            print([pt_1, pt_2, pt_3, pt_4])

            return [pt_1, pt_2, pt_3, pt_4]

        elif (input_lat not in latitude_list) and input_long in longitude_list:
            print('case 2')

            lower_bound_lat = max([num for num in latitude_list if num < input_lat])
            upper_bound_lat = min([num for num in latitude_list if num > input_lat])

            pt_1 = [lower_bound_lat, input_long]
            pt_2 = [upper_bound_lat, input_long]
            pt_3 = [lower_bound_lat, input_long]
            pt_4 = [upper_bound_lat, input_long]
            print([pt_1, pt_2, pt_3, pt_4])

            return [pt_1, pt_2, pt_3, pt_4]
        
        else:
            upper_bound_long = min([num for num in longitude_list if num > input_long])
            lower_bound_long = max([num for num in longitude_list if num < input_long])
            upper_bound_lat = min([num for num in latitude_list if num > input_lat])
            lower_bound_lat = max([num for num in latitude_list if num < input_lat])

            pt_1 = [lower_bound_lat, lower_bound_long]
            pt_2 = [upper_bound_lat, lower_bound_long]
            pt_3 = [lower_bound_lat, upper_bound_long]
            pt_4 = [upper_bound_lat, upper_bound_long]
            print([pt_1, pt_2, pt_3, pt_4])
            return [pt_1, pt_2, pt_3, pt_4]
    
    def bilinear_interpolation(self, input_lat, input_lng, lat1, lng1, lat2, lng2, v11, v21, v12, v22) -> List:
        """
        Summary:
            This function use bilinear interpolation to estimate values based on four nearby points
        
        Args:
            input_lat (float): input latitude, 
            input_lng (float): input longitude
            lat1 (float): latitude of the minimum point
            lng1 (float): longitude of the minimum point
            lat2 (float): latitude of the max point
            lng2 (float): longitude of the max point
            v11 (list): a list containing the associated value (i.e. wind speed, surface pressure, etc) at the point(lat1, lng1) 左下
            v21 (list): a list containing the associated value (i.e. wind speed, surface pressure, etc) at the point (lat2, lng1) 左上
            v12 (list): a list containing the associated value (i.e. wind speed, surface pressure, etc) at the point (lat1, lng2) 右下
            v22 (list): a list containing the associated value (i.e. wind speed, surface pressure, etc) at the point (lat2, lng2) 右上

        Returns:
            List: a list of the interpolated values 

        Reference:
            https://www.youtube.com/watch?v=va8vFViss90&ab_channel=EngineeringMathematicI
        """ 

        if not (len(v11) == len(v21) == len(v12) == len(v22)):
            raise ValueError("Input lists must have the same length")
        
        interpolated_values = []
        
        if (lat1 == lat2 and input_lat == lat1) and lng1 != lng2:
            print('case1')

            for i in range(len(v11)):
                value = (input_lng - lng2) / (lng1 - lng2) * v11[i] + (input_lng - lng1) / (lng2 - lng1) * v12[i]
                interpolated_values.append(value)

        elif lng1 == lng2 and input_lng == lng1 and  lat1 != lat2:
            print('case2')

            for i in range(len(v11)):
                value = (input_lat - lat2) / (lat1 - lat2) * v11[i] + (input_lat - lat1) / (lat2 - lat1) * v21[i]
                interpolated_values.append(value)

        elif (lat1 == lat2 and input_lat == lat1) and (lng1 == lng2 and input_lng == lng1):
            print('case3')

            interpolated_values = v11

        else:
            print('case4')

            for i in range(len(v11)):
                v1_lng = (input_lng - lng2) / (lng1 - lng2) * v11[i] + (input_lng - lng1) / (lng2 - lng1) * v12[i]
                v2_lng = (input_lng - lng2) / (lng1 - lng2) * v21[i] + (input_lng - lng1) / (lng2 - lng1) * v22[i]

                value = (input_lat - lat2) / (lat1 - lat2) * v1_lng + (input_lat - lat1) / (lat2 - lat1) * v2_lng

                interpolated_values.append(value)
        
        return interpolated_values
    
    def get_df(self, ds, latitude, longitude, latitude_list, longitude_list, times_list, is_ocean):
        """
        Summary:
            This function creates a dataframe of desired measurements
        
        Args:
            ds (netCDF4.Dataset): _description_
            latitude (float): _description_
            longitude (float): _description_
            latitude_list (list): _description_
            longitude_list (list): _description_
            times_list (list): _description_

        Returns:
            dataframe: _description_
        """        

        longitude_index = longitude_list.index(longitude)
        latitude_index = latitude_list.index(latitude)  

        u10 = ds.variables['u10'][:, latitude_index, longitude_index].tolist()
        v10 = ds.variables['v10'][:, latitude_index, longitude_index].tolist()

        if is_ocean is True:
            u100 = [hws * (100/10)**0.1 for hws in u10]
            v100 = [hws * (100/10)**0.1 for hws in v10]  # 100/10 is target_height / current height
        else:
            u100 = [hws * (100/10)**0.143 for hws in u10] 
            v100 = [hws * (100/10)**0.143 for hws in v10]

        sp = ds.variables['sp'][:, latitude_index, longitude_index].tolist()
        t2m = ds.variables['t2m'][:, latitude_index, longitude_index].tolist()

        u100_none_count = u100.count(None)
        v100_none_count = v100.count(None)
        sp_none_count = sp.count(None)
        temp_none_count = t2m.count(None)

        if u100_none_count > 0:
            print(f"u100 contains {u100_none_count} None values.")
        else:
            print("u100 does not contain None values.")

        if v100_none_count > 0:
            print(f"v100 contains {v100_none_count} None values.")
        else:
            print("v100 does not contain None values.")

        if sp_none_count > 0:
            print(f"sp contains {sp_none_count} None values.")
        else:
            print("sp does not contain None values.")

        if temp_none_count > 0:
            print(f"temp contains {temp_none_count} None values.")
        else:
            print("temp does not contain None values.")

        u100 = [0 if x is None else x for x in u100]
        v100 = [0 if x is None else x for x in v100]
        sp = [100000 if x is None else x for x in sp]
        t2m = [273 if x is None else x for x in t2m]

        wind_velocity = [(x**2 + y**2)**0.5 for x, y in zip(u100, v100)]
        direction = [(180 + np.degrees(np.arctan2(u, v))) % 360 for u, v in zip(u100, v100)]

        data =  {
            'u100': u100,
            'v100': v100,
            'wind_velocity': wind_velocity,
            'direction': direction,
            'temperature': t2m,
            'sp': sp,
        }
        df = pd.DataFrame(data, index=times_list)
        return df
    
    
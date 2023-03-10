# -*- coding: utf-8 -*-
"""
 SM3 Risiken durch Naturgefahren
"""
import os
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
os.environ['PROJ_LIB'] = 'C:\\Users\\morit\\Anaconda\\Library\\share\\proj'
os.environ['GDAL_DATA'] = 'C:\\Users\\morit\\Anaconda\\Library\\share'
#from mpl_toolkits.mplot3d import Axes3D



## set wd
#os.chdir("C:\\Users\\morit\\Documents\\Master\\3.Semester\\SM3_Analyse_von_Risiken_Naturgefahren\\Posterarbeit\\Daten\\cliptest")
# fuer validierung
#os.chdir("C:\\Users\\morit\\Documents\\Master\\3.Semester\\SM3_Analyse_von_Risiken_Naturgefahren\\Posterarbeit\\Daten\\Daten_validierung")
os.chdir("C:\\Users\\morit\\Documents\\Master\\3.Semester\\SM3_Analyse_von_Risiken_Naturgefahren\\Posterarbeit\\Daten\\alle_Daten")


### xyz
## read table
#tabloesch = pd.read_csv('alle_Daten_Hoehe_Rutsch.xyz', sep="\t", decimal=".", header=0)
tab_name = pd.read_csv('train_xyz.xyz', sep="\t", decimal=".", header=0)

## rename
# show names
list(tab_name.columns.values)
"""
tab_name = tab.rename(columns={"Rutschungen_clip_KOpasst_cliptest": "mask", "Zusammengef_Mo_KOpasst2m_cliptest": "Hoehe", 
                               "Hillshade_hoehe_fuss_cliptest": "hillshade", "B02_Blue_255_cliptest": "blau",
                               "B04_Green_255_cliptest": "gruen", "B04_Red_255_cliptest": "rot",
                               "B08_nir_255_cliptest": "nir", "Slope_deg_cliptest": "slope",
                               "Aspect_deg_cliptest": "aspect", "Profile Curvature30m_cliptest": "Curv_prof",
                               "Plan Curvature30m_cliptest": "Curv_plan"})
"""
#tab.head(8000)

#tab.iloc[7000:7002,:]

#tab_name = tab[tab.hillshade != -99999.0]
#tab_name.head(8000)

#tab_name.isnull().values.any()


## Matrix
# Koordinaten
# x 0
tab_pivot_x = tab_name.pivot(index='Y', columns='X', values=("X"))
tab_pivot_x = tab_pivot_x.sort_index(ascending=False)
#tab_pivot_x.head(10)
# y 1
tab_pivot_y = tab_name.pivot(index='Y', columns='X', values=("Y"))
tab_pivot_y = tab_pivot_y.sort_index(ascending=False)
#tab_pivot_y.head(10)
## z- Werte
# hoehe 2
tab_pivot_hoehe = tab_name.pivot(index='Y', columns='X', values=("hoehe"))
tab_pivot_hoehe = tab_pivot_hoehe.sort_index(ascending=False)
#tab_pivot_hoehe.head(10)
# hillshade 3
tab_pivot_hillshade = tab_name.pivot(index='Y', columns='X', values=("hillshade"))
tab_pivot_hillshade = tab_pivot_hillshade.sort_index(ascending=False)
#tab_pivot_hillshade.head(10)
# rot 4
tab_pivot_rot = tab_name.pivot(index='Y', columns='X', values=("rot"))
tab_pivot_rot = tab_pivot_rot.sort_index(ascending=False)
#tab_pivot_rot.head(10)
# gruen 5
tab_pivot_gruen = tab_name.pivot(index='Y', columns='X', values=("gruen"))
tab_pivot_gruen = tab_pivot_gruen.sort_index(ascending=False)
#tab_pivot_gruen.head(10)
# blau 6
tab_pivot_blau = tab_name.pivot(index='Y', columns='X', values=("blau"))
tab_pivot_blau = tab_pivot_blau.sort_index(ascending=False)
#tab_pivot_blau.head(10)
# nir 7
tab_pivot_nir = tab_name.pivot(index='Y', columns='X', values=("nir"))
tab_pivot_nir = tab_pivot_nir.sort_index(ascending=False)
#tab_pivot_nir.head(10)
# slope 8
tab_pivot_slope = tab_name.pivot(index='Y', columns='X', values=("slope"))
tab_pivot_slope = tab_pivot_slope.sort_index(ascending=False)
#tab_pivot_slope.head(10)
# aspect 9
tab_pivot_aspect = tab_name.pivot(index='Y', columns='X', values=("aspect"))
tab_pivot_aspect = tab_pivot_aspect.sort_index(ascending=False)
#tab_pivot_aspect.head(10)
# Curv_prof 10
tab_pivot_Curv_prof = tab_name.pivot(index='Y', columns='X', values=("Curv_prof"))
tab_pivot_Curv_prof = tab_pivot_Curv_prof.sort_index(ascending=False)
#tab_pivot_Curv_prof.head(10)
# Curv_plan 11
tab_pivot_Curv_plan = tab_name.pivot(index='Y', columns='X', values=("Curv_plan"))
tab_pivot_Curv_plan = tab_pivot_Curv_plan.sort_index(ascending=False)
#tab_pivot_Curv_plan.head(10)
# Ruggedness 12
tab_pivot_rugged = tab_name.pivot(index='Y', columns='X', values=("ruggedness"))
tab_pivot_rugged = tab_pivot_Curv_plan.sort_index(ascending=False)
#tab_pivot_rugged.head(10)

## mask extra
tab_pivot_mask = tab_name.pivot(index='Y', columns='X', values=("mask"))
tab_pivot_mask = tab_pivot_mask.sort_index(ascending=False)
#tab_pivot_mask.head(10)

#loesch = pd.read_csv('C:\\Users\\morit\\Documents\\Master\\3.Semester\\SM3_Analyse_von_Risiken_Naturgefahren\\Posterarbeit\\Daten\\cliptest\\xyz_parameter_all\\cliptest_parameter.xyz', sep="\t", decimal=".", header=0)
#list(loesch.columns.values)
#loesch.iloc[1:1000,2]

del tab_name

# plot test
plt.imshow(tab_pivot_nir)


## Array
tab_arr = np.dstack([tab_pivot_x,tab_pivot_y,tab_pivot_hoehe, tab_pivot_hillshade, tab_pivot_rot, 
                     tab_pivot_gruen, tab_pivot_blau, tab_pivot_nir, tab_pivot_slope, tab_pivot_aspect, tab_pivot_Curv_prof,
                     tab_pivot_Curv_plan, tab_pivot_rugged])

del tab_pivot_x,tab_pivot_y,tab_pivot_hoehe, tab_pivot_hillshade, tab_pivot_rot, tab_pivot_gruen, tab_pivot_blau, tab_pivot_nir, tab_pivot_slope, tab_pivot_aspect, tab_pivot_Curv_prof,tab_pivot_Curv_plan, tab_pivot_rugged

#tab_arr[:, :, 0]
#tab_arr[:, :, 1]
#tab_arr[:, :, 2]
#tab_arr[:, :, 7]

#plt.imshow(tab_arr[:, :, 7])

## Array mask
tab_arr_mask = np.dstack([tab_pivot_mask])
del tab_pivot_mask

## split in smaller arrays 512*512
#tab_arr.shape
## rows
#tab_arr.shape[0]
## columns
#tab_arr.shape[1]
## test splitz
#tab_test_split = tab_arr[0:512, 0:512, :]




# Teilgröße
part_size = 128

# Berechne die Anzahl der Teile in jeder Dimension
num_parts_x = tab_arr.shape[0] // part_size
num_parts_y = tab_arr.shape[1] // part_size

## save as .h5
#?parts = []
for i in range(num_parts_x):
    for j in range(num_parts_y):
        x_start = i * part_size
        x_end = x_start + part_size
        y_start = j * part_size
        y_end = y_start + part_size
        part = tab_arr[x_start:x_end, y_start:y_end, :]
        # Speichere den Teil in einer eigenen .h5-Datei
        filename = f'C:/Users/morit/Documents/Master/3.Semester/SM3_Analyse_von_Risiken_Naturgefahren/Posterarbeit/Daten/alle_Daten/Tiles_128x128/Exported_Teile/part_{i}_{j}.h5'
        with h5py.File(filename, 'w') as f:
            dset = f.create_dataset('img', data=part, dtype='float32')
        
# 

## Test ob funktioniert (plot .h5)
# Lade eine der exportierten h5-Dateien
filename = 'C:/Users/morit/Documents/Master/3.Semester/SM3_Analyse_von_Risiken_Naturgefahren/Posterarbeit/Daten/alle_Daten/Tiles_128x128/Exported_Teile/part_0_89.h5'
with h5py.File(filename, 'r') as f:
    data = f['img'][:]
    plt.imshow(data[:, :, 4:7].astype(int))

data[:, :, 6]
data[:, :, 10] = data[:, :, 10] + 100
# aspect
data[:, :, 9].max()
# Curv_prof
data[:, :, 10].max()
data[:, :, 10].min()
# Curv_plan
data[:, :, 11].max()
data[:, :, 11].min()



"""
part 17_3... hat 0 in rot und nir, aber es darf fuer ndvi nicht durch 0 geteilt werden, also Kanal 4 und 7
"""
#data_nir = data[:, :, 7]
#data_red = data[:, :, 4]

#data_ndvi = np.divide(data_nir - data_red,np.add(data_nir, data_red), out=np.zeros_like(data_nir), where=np.add(data_nir, data_red)!=0)
#data_ndvi = np.divide(data_nir - data_red,np.add(data_nir, data_red), where=np.add(data_nir, data_red)!=0)

    

"""
plt.imshow(data[:, :, 4:7])
rgbtest = data[:, :, 4:7]
rgbtest.dtype
rgbtest = rgbtest.astype(int)
rgbtest.dtype
plt.imshow(rgbtest.astype(int))
"""


###### selbes fuer mask
tab_arr_mask[:,:,0]

for i in range(num_parts_x):
    for j in range(num_parts_y):
        x_start = i * part_size
        x_end = x_start + part_size
        y_start = j * part_size
        y_end = y_start + part_size
        part_mask = tab_arr_mask[x_start:x_end, y_start:y_end, 0]
        # Speichere den Teil in einer eigenen .h5-Datei
        filename = f'C:/Users/morit/Documents/Master/3.Semester/SM3_Analyse_von_Risiken_Naturgefahren/Posterarbeit/Daten/alle_Daten/Tiles_128x128/Exported_Teile_mask/mask_{i}_{j}.h5'
        with h5py.File(filename, 'w') as f:
            dset_mask = f.create_dataset('mask', data=part_mask, dtype='float32')
        

## Test ob funktioniert (plot .h5)
# Lade eine der exportierten h5-Dateien
filename = 'C:/Users/morit/Documents/Master/3.Semester/SM3_Analyse_von_Risiken_Naturgefahren/Posterarbeit/Daten/alle_Daten/Tiles_128x128/Exported_Teile_mask/mask_0_0.h5'
with h5py.File(filename, 'r') as f:
    data_mask = f['mask'][:]
    
plt.imshow(data_mask[:, :])

"""
### save as .xyz
# Iteriere über jeden Teil und speichere ihn
parts = []
for i in range(num_parts_x):
    for j in range(num_parts_y):
        x_start = i * part_size
        x_end = x_start + part_size
        y_start = j * part_size
        y_end = y_start + part_size
        part = tab_arr[x_start:x_end, y_start:y_end, :]
        with open("C:/Users/morit/Documents/Master/3.Semester/SM3_Analyse_von_Risiken_Naturgefahren/Posterarbeit/Daten/alle_Daten/Exported_Teile/part_{}_{}.xyz".format(i, j), "w") as file:
            num_teile = part.shape[0] * part.shape[1]
            for raster in np.reshape(part, (num_teile, 4)):
                file.write("{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(*raster))

# raster[:3] nimmt nur dritte Dimension
# eigentlich für alle Dimensionen: nur raster
"""

#import glob
#TRAIN_PATH = r"C:/Users/morit/Documents/Master/3.Semester/SM3_Analyse_von_Risiken_Naturgefahren/Posterarbeit/Daten/alle_Daten/Tiles_128x128/Exported_Teile/*.h5"
#all_train = sorted(glob.glob(TRAIN_PATH))
#all_train = all_train[0:3000]



### Testing andere Daten
# r vor string nimmt alle zeichen in "" mit auf
path_single = r"C:/Users/morit/Documents/Master/3.Semester/SM3_Analyse_von_Risiken_Naturgefahren/Posterarbeit_nichtVerwendet/DL_landslides_Video Inder1/data/TrainData/img/image_63.h5"
path_single_mask = r'C:/Users/morit/Documents/Master/3.Semester/SM3_Analyse_von_Risiken_Naturgefahren/Posterarbeit_nichtVerwendet/DL_landslides_Video Inder1/data/TrainData/mask/mask_62.h5'

# create 4 dim array
f_data = np.zeros((1, 128,128, 3))
# check dimensions
f_data.ndim

# load hdf5 and plot
with h5py.File(path_single) as hdf:
    key = list(hdf.keys()) # get name of stored objects
    print("key", key)
    data = np.array(hdf.get('img')) # only get data with this name in array
    print("input data shape:", data.shape)
    plt.imshow(data[:, :, 3:0:-1])
    data_red = data[:, :, 3]
    data_green = data[:, :, 2]
    data_blue = data[:, :, 1]
    data_nir = data[:, :, 7]
    data_rgb = data[:, :, 3:0:-1]
    data_ndvi = np.divide(data_nir - data_red,np.add(data_nir, data_red))
    f_data[0, :, :, 0] =data_ndvi
    f_data[0, :, :, 1] = data[:, :, 12]
    f_data[0, :, :, 2] = data[:, :, 13]

    print("data ndvi shape ", data_ndvi.shape, "f_data shape: ", f_data.shape)
    plt.imshow(data_ndvi)

plt.colorbar(plt.imshow(data[:, :, 13]))
np.max(data[:, :, 3])

plt.imshow(data[:, :, 1:4])
data[:, :, 4]

with h5py.File(path_single_mask) as hdf:
    ls = list(hdf.keys())
    print("ls", ls)
    data_mask = np.array(hdf.get('mask'))
    print("input data shape:", data_mask.shape)
    plt.imshow(data_mask)
np.max(data_mask[:, :])



##### fuer eine einzelne Datei
### Check Ergebnisse
pfad_ergebnis = r"C:/Users/morit/Documents/Master/3.Semester/SM3_Analyse_von_Risiken_Naturgefahren/Posterarbeit/Daten/alle_Daten/ergebnisse/part_2_0.h5"
with h5py.File(pfad_ergebnis) as hdf:
    print("ls", list(hdf.keys()))
    data_ergebnis = np.array(hdf.get('mask'))
    print("input data shape:", data_ergebnis.shape)
plt.imshow(data_ergebnis)

## merge mit eingangsdaten fuer KO
pfad_eingangsdaten = 'C:/Users/morit/Documents/Master/3.Semester/SM3_Analyse_von_Risiken_Naturgefahren/Posterarbeit/Daten/alle_Daten/Exported_Teile/part_2_0.h5'
with h5py.File(pfad_eingangsdaten, 'r') as f:
    data_eingang = f['img'][:]
# array merged
array_ergebnis_eingang = np.dstack((data_eingang, data_ergebnis))
array_ergebnis_eingang_del = np.delete(array_ergebnis_eingang, np.s_[2:13], axis=2)
# xyz
xyz_ergbnis = pd.DataFrame(array_ergebnis_eingang_del.reshape(-1, 3), columns=['x', 'y', 'z'])
xyz_ergbnis = xyz_ergbnis.astype({"x":"int","y":"int","z":"int"})
# export als xyz
#xyz_ergbnis.to_csv('C:/Users/morit/Documents/Master/3.Semester/SM3_Analyse_von_Risiken_Naturgefahren/Posterarbeit/Daten/cliptest/ergebnisse/file_ergebnis.csv', header=True, index=False, encoding='utf-8')

## plot
# Extrahieren Sie die x-Koordinaten, y-Koordinaten und die Werte des Rasters
x_coords = array_ergebnis_eingang_del[:, :, 0]
y_coords = array_ergebnis_eingang_del[:, :, 1]
raster_values = array_ergebnis_eingang_del[:, :, 2]
# Plot des Rasters
plt.imshow(raster_values, extent=(x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()), cmap='gray')

# Rasterauflösung definieren
res = 2  # 1 Meter Auflösung
# Rastergröße aus den x-, y-Min- und Max-Werten berechnen
xmin, ymin = xyz_ergbnis[['x', 'y']].min()
xmax, ymax = xyz_ergbnis[['x', 'y']].max()
width = int((xmax - xmin) / res)+1
height = int((ymax - ymin) / res)+1
# GeoTransform erstellen
transform = from_origin(xmin, ymax, res, res)
# Numpy-Array aus z-Werten erstellen
#arr = np.zeros((height, width), dtype=np.float32)
#for index, row in xyz_ergbnis.iterrows():
#    x, y, z = row
#    x, y = int((x - xmin) / res), int((y - ymin) / res)
#    arr[y, x] = z
arr = array_ergebnis_eingang_del[:,:,2]
# GeoTIFF erstellen
with rasterio.open(
    'output_part_2_0.tif',
    'w',
    driver='GTiff',
    width=width,
    height=height,
    count=1,
    dtype=np.float32,
    transform=transform,
    crs='EPSG:26910'
) as dst:
    dst.write(arr, 1)



## fuer alle Daten
#####
# Ordnerpfad, in dem sich die Dateien befinden
ordnerpfad = "C:/Users/morit/Documents/Master/3.Semester/SM3_Analyse_von_Risiken_Naturgefahren/Posterarbeit/Daten/Daten_validierung/ergebnisse/"
# Schleife durch alle Dateien im Ordner
for datei in os.listdir(ordnerpfad):
    if datei.endswith(".h5"): # Nur für .h5-Dateien ausführen
        pfad_ergebnis = os.path.join(ordnerpfad, datei)
        with h5py.File(pfad_ergebnis) as hdf:
        #    print("ls", list(hdf.keys()))
            data_ergebnis = np.array(hdf.get('mask'))
        #    print("input data shape:", data_ergebnis.shape)
        # Pfad für Eingangsdaten konstruieren
        pfad_eingangsdaten = os.path.join("C:/Users/morit/Documents/Master/3.Semester/SM3_Analyse_von_Risiken_Naturgefahren/Posterarbeit/Daten/Daten_validierung/Exported_Teile/", datei)
        with h5py.File(pfad_eingangsdaten, 'r') as f:
            data_eingang = f['img'][:]
        #array mit beiden erstellen
        array_ergebnis_eingang = np.dstack((data_eingang, data_ergebnis))
        array_ergebnis_eingang_del = np.delete(array_ergebnis_eingang, np.s_[2:13], axis=2)
        xyz_ergbnis = pd.DataFrame(array_ergebnis_eingang_del.reshape(-1, 3), columns=['x', 'y', 'z'])
        xyz_ergbnis = xyz_ergbnis.astype({"x":"int","y":"int","z":"int"})
        print(datei)
        # Rasterauflösung definieren
        res = 2  # 1 Meter Auflösung
        # Rastergröße aus den x-, y-Min- und Max-Werten berechnen
        xmin, ymin = xyz_ergbnis[['x', 'y']].min()
        xmax, ymax = xyz_ergbnis[['x', 'y']].max()
        width = int((xmax - xmin) / res)+1
        height = int((ymax - ymin) / res)+1
        # GeoTransform erstellen
        transform = from_origin(xmin, ymax, res, res)
        # Numpy-Array aus z-Werten erstellen
        #arr = np.zeros((height, width), dtype=np.float32)
        #for index, row in xyz_ergbnis.iterrows():
        #    x, y, z = row
        #    x, y = int((x - xmin) / res), int((y - ymin) / res)
        #    arr[y, x] = z
        arr = array_ergebnis_eingang_del[:,:,2]
        # GeoTIFF erstellen
        with rasterio.open(
            os.path.join(ordnerpfad, f"ergebnisse_tiff/ergebnis_{datei}.tif"), # Speicherort des TIFFs
            'w',
            driver='GTiff',
            width=width,
            height=height,
            count=1,
            dtype=np.float32,
            transform=transform,
            crs='EPSG:26910'
        ) as dst:
            dst.write(arr, 1)
import pandas as pd
import datetime as dt
import numpy as np
import re
from collections import Counter
from operator import itemgetter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
#%%
dftrain = pd.read_csv('base_train.csv')
dftest = pd.read_csv('base_val.csv')
#%%
dftrain = dftrain.drop_duplicates()
#%%
dftrain = dftrain[dftrain['number_of_reviews'] >= 10]
dftrain = dftrain[dftrain['number_of_reviews'] <= 200]                  
#%%
dftrain = dftrain.reset_index()
dftrain['bedrooms'].fillna(dftrain['bedrooms'].mean(), inplace=True)
dftrain['bedrooms'] = dftrain['bedrooms'].astype(int)
dftrain['beds'].fillna(dftrain['beds'].mean(), inplace=True)
dftrain['beds'] = dftrain['beds'].astype(int)
#%%
dftrain.host_is_superhost = dftrain.host_is_superhost.replace({'t': 1, 'f': 0})
dftrain.host_has_profile_pic = dftrain.host_has_profile_pic.replace({'t': 1, 'f': 0})
dftrain.host_identity_verified = dftrain.host_identity_verified.replace({'t': 1, 'f': 0})
dftrain.has_availability = dftrain.has_availability.replace({'t': 1, 'f': 0})           
dftrain.instant_bookable = dftrain.instant_bookable.replace({'t': 1, 'f': 0})   
#%%
dftrain['antiquity'] = (pd.to_datetime(dt.date.today()) - pd.to_datetime(dftrain['host_since'])) / np.timedelta64(1, 'D')
dftrain['since_last_review'] = (pd.to_datetime(dt.date.today()) - pd.to_datetime(dftrain['last_review'])) / np.timedelta64(1, 'D')
#%%
for i in range(len(dftrain['neighborhood_overview'])):    
    article = dftrain['neighborhood_overview'][i]
    try:
        article = re.sub(r'<br ?/?>', '', article)
        article = re.sub(r'<?/?b>','',article)
        dftrain['neighborhood_overview'][i] = article
    except:
        pass
#%%
for i in range(len(dftrain['amenities'])):
    lista = dftrain['amenities'][i]
    try:
        lista = eval(lista)
        for j in range(len(lista)):
            lista[j] = re.sub('\u2013', '-', lista[j])
            lista[j] = re.sub('\u2019',"'",lista[j])
            dftrain['amenities'][i] = lista
    except:
        pass
#%%
ameniti = []
for i in dftrain['amenities']:
    for j in i:
        if j not in ameniti:
            ameniti.append(j)
#%%
for i in range(len(dftrain['host_response_rate'])):
    value = dftrain['host_response_rate'][i]
    try:
        dftrain['host_response_rate'][i] = float(value.replace('%',''))/100
    except:
        dftrain['host_response_rate'][i] = 0
for i in range(len(dftrain['host_acceptance_rate'])):
    value = dftrain['host_acceptance_rate'][i]
    try:
        dftrain['host_acceptance_rate'][i] = float(value.replace('%',''))/100
    except:
        dftrain['host_acceptance_rate'][i] = 0
#%%
total = []
for i in dftrain['amenities']:
    for j in i:
        total.append(j)
#%%
columna = [elemento for lista in dftrain['amenities'] for elemento in lista]

# Obtener la cuenta de apariciones de cada elemento
contador = Counter(columna)

# Ordenar el diccionario por la cantidad de apariciones en orden descendente
diccionario_ordenado = dict(sorted(contador.items(), key=itemgetter(1), reverse=True))

# Crear el DataFrame a partir del diccionario ordenado
diccionario_ordenado = pd.DataFrame(list(diccionario_ordenado.items()), columns=['Elemento', 'Apariciones'])
#%%
diccionario_ordenado = diccionario_ordenado[diccionario_ordenado['Apariciones']>=50]
#%%
diccionario_ordenado.to_clipboard(index=False)
#%%
amenities = [
    ("Essentials", 4715, "Essentials"),
    ("Smoke alarm", 4421, "Well Being"),
    ("Wifi", 4337, "Essentials"),
    ("Heating", 4101, "Heating/Cooling"),
    ("Hot water", 3990, "Heating/Cooling"),
    ("Hangers", 3949, "Miscellaneous"),
    ("Hair dryer", 3913, "Well Being"),
    ("Kitchen", 3841, "Kitchen"),
    ("Dishes and silverware", 3781, "Kitchen"),
    ("Refrigerator", 3630, "Kitchen"),
    ("Iron", 3618, "Well Being"),
    ("Long term stays allowed", 3583, "Administration"),
    ("Shampoo", 3412, "Bath & Cleaning"),
    ("Bed linens", 3215, "Bath & Cleaning"),
    ("Cooking basics", 3060, "Kitchen"),
    ("Carbon monoxide alarm", 2810, "Well Being"),
    ("Coffee maker", 2715, "Kitchen"),
    ("Fire extinguisher", 2628, "Well Being"),
    ("Dishwasher", 2435, "Kitchen"),
    ("Washer", 2432, "Bath & Cleaning"),
    ("Private entrance", 2407, "Accessibility"),
    ("First aid kit", 2376, "Well Being"),
    ("Oven", 2368, "Kitchen"),
    ("Microwave", 2319, "Kitchen"),
    ("Stove", 2100, "Kitchen"),
    ("Dedicated workspace", 1813, "Administration"),
    ("Extra pillows and blankets", 1796, "Bath & Cleaning"),
    ("TV", 1616, "Entertainment"),
    ("Host greets you", 1594, "Administration"),
    ("Luggage dropoff allowed", 1581, "Administration"),
    ("Cleaning products", 1508, "Bath & Cleaning"),
    ("Shower gel", 1502, "Bath & Cleaning"),
    ("Dryer", 1454, "Bath & Cleaning"),
    ("Wine glasses", 1429, "Kitchen"),
    ("Dining table", 1427, "Kitchen"),
    ("Paid parking off premises", 1407, "Accessibility"),
    ("Hot water kettle", 1359, "Kitchen"),
    ("Private patio or balcony", 1290, "Outdoors"),
    ("Freezer", 1261, "Kitchen"),
    ("TV with standard cable", 1228, "Entertainment"),
    ("Outdoor furniture", 1181, "Outdoors"),
    ("Coffee", 1121, "Kitchen"),
    ("Body soap", 1056, "Bath & Cleaning"),
    ("Self check-in", 1048, "Administration"),
    ("Bathtub", 1033, "Bath & Cleaning"),
    ("Room-darkening shades", 1032, "Bath & Cleaning"),
    ("Drying rack for clothing", 988, "Bath & Cleaning"),
    ("Laundromat nearby", 918, "Bath & Cleaning"),
    ("Lock on bedroom door", 895, "Safety"),
    ("Patio or balcony", 895, "Outdoors"),
    ("Waterfront", 883, "Outdoors"),
    ("Outdoor dining area", 829, "Outdoors"),
    ("Toaster", 810, "Kitchen"),
    ("Portable fans", 788, "Well Being"),
    ("Books and reading material", 760, "Entertainment"),
    ("Single level home", 757, "Miscellaneous"),
    ("Free washer - In unit", 755, "Bath & Cleaning"),
    ("Lockbox", 750, "Accessibility"),
    ("Children's books and toys", 602, "Children"),
    ("Central heating", 596, "Heating/Cooling"),
    ("Canal view", 595, "Outdoors"),
    ("High chair", 592, "Children"),
    ("Ethernet connection", 589, "Accessibility"),
    ("Baking sheet", 577, "Kitchen"),
    ("Conditioner", 573, "Bath & Cleaning"),
    ("Pets allowed", 565, "Miscellaneous"),
    ("Blender", 559, "Kitchen"),
    ("Paid street parking off premises", 557, "Accessibility"),
    ("Backyard", 548, "Outdoors"),
    ("BBQ grill", 545, "Outdoors"),
    ("Free dryer - In unit", 544, "Bath & Cleaning"),
    ("Paid parking on premises", 536, "Accessibility"),
    ("Garden view", 526, "Outdoors"),
    ("Free parking on premises", 504, "Accessibility"),
    ("Board games", 496, "Entertainment"),
    ("EV charger", 486, "Accessibility"),
    ("Coffee maker: Nespresso", 479, "Kitchen"),
    ("Crib", 469, "Children"),
    ("Children's dinnerware", 449, "Children"),
    ("Private backyard - Fully fenced", 428, "Outdoors"),
    ("Elevator", 411, "Accessibility"),
    ("Air conditioning", 400, "Heating/Cooling"),
    ("Mini fridge", 397, "Kitchen"),
    ("Free street parking", 378, "Accessibility"),
    ("Clothing storage: closet", 371, "Bath & Cleaning"),
    ("Security cameras on property", 338, "Safety"),
    ("Pack 'n play/Travel crib", 337, "Children"),
    ("Private living room", 336, "Miscellaneous"),
    ("Babysitter recommendations", 302, "Children"),
    ("Pocket wifi", 300, "Accessibility"),
    ("City skyline view", 291, "Outdoors"),
    ("Clothing storage", 289, "Bath & Cleaning"),
    ("Barbecue utensils", 275, "Kitchen"),
    ("Indoor fireplace", 272, "Well Being"),
    ("Courtyard view", 268, "Outdoors"),
    ("Changing table", 234, "Children"),
    ("Cleaning available during stay", 227, "Bath & Cleaning"),
    ("Lake access", 217, "Outdoors"),
    ("Breakfast", 216, "Well Being"),
    ("Clothing storage: wardrobe", 213, "Bath & Cleaning"),
    ("Outlet covers", 203, "Safety"),
    ("Smoking allowed", 200, "Miscellaneous"),
    ("Bikes", 200, "Outdoors"),
    ("Baby bath", 200, "Children"),
    ("Shared patio or balcony", 181, "Outdoors"),
    ("Piano", 172, "Entertainment"),
    ("Stainless steel oven", 172, "Kitchen"),
    ("Sound system", 171, "Entertainment"),
    ("Safe", 157, "Safety"),
    ("Mosquito net", 151, "Safety"),
    ("Park view", 149, "Outdoors"),
    ("Fire pit", 145, "Outdoors"),
    ("Baby safety gates", 142, "Children"),
    ("Gas stove", 138, "Kitchen"),
    ("Game console", 129, "Entertainment"),
    ("Building staff", 129, "Administration"),
    ("Rice maker", 126, "Kitchen"),
    ("Record player", 125, "Entertainment"),
    ("Hammock", 122, "Outdoors"),
    ("Bluetooth sound system", 114, "Entertainment"),
    ("Coffee maker: espresso machine", 113, "Kitchen"),
    ("Boat slip", 105, "Outdoors"),
    ("River view", 100, "Outdoors"),
    ("Resort access", 95, "Outdoors"),
    ("Window guards", 95, "Safety"),
    ("Smart lock", 90, "Safety"),
    ("Central air conditioning", 90, "Heating/Cooling"),
    ("Baby monitor", 86, "Children"),
    ("Induction stove", 85, "Kitchen"),
    ("Keypad", 82, "Safety"),
    ("Portable air conditioning", 80, "Heating/Cooling"),
    ("Gym", 77, "Well Being"),
    ("BBQ grill: charcoal", 68, "Outdoors"),
    ("Private hot tub", 67, "Well Being"),
    ("Ceiling fan", 66, "Well Being"),
    ("Sonos sound system", 66, "Entertainment"),
    ("HDTV with Netflix", 65, "Entertainment"),
    ("Free washer - In building", 64, "Bath & Cleaning"),
    ("Harbor view", 63, "Outdoors"),
    ("Free dryer - In building", 63, "Bath & Cleaning"),
    ("Sonos Bluetooth sound system", 62, "Entertainment"),
    ("Indoor fireplace: wood-burning", 62, "Well Being"),
    ("Lake view", 62, "Outdoors"),
    ("Clothing storage: dresser", 61, "Bath & Cleaning"),
    ("Paid washer - In building", 60, "Bath & Cleaning"),
    ("Private backyard - Not fully fenced", 56, "Outdoors"),
    ("Stainless steel gas stove", 54, "Kitchen"),
    ("Bread maker", 54, "Kitchen"),
    ("Paid parking garage off premises", 53, "Accessibility"),
    ("Coffee maker: drip coffee maker", 51, "Kitchen"),
    ("Game console: PS4", 51, "Entertainment"),
    ("Paid dryer - In building", 51, "Bath & Cleaning"),
    ("Exercise equipment: free weights, yoga mat", 50, "Well Being")
]

columns = ["Elemento", "Apariciones", "Clasificacion"]
df_amenities = pd.DataFrame(amenities, columns=columns)

df_amenities = df_amenities.drop(columns=['Apariciones'])
#%%
dftrain['Amenities_Kitchen'] = 0
dftrain['Amenities_Outdoors'] = 0
dftrain['Amenities_Accessibility'] = 0
dftrain['Amenities_Well Being'] = 0
dftrain['Amenities_Heating/Cooling'] = 0
dftrain['Amenities_Miscellaneous'] = 0
dftrain['Amenities_Bath & Cleaning'] = 0
dftrain['Amenities_Entertainment'] = 0
dftrain['Amenities_Children'] = 0
dftrain['Amenities_Admnistration'] = 0
dftrain['Amenities_Essentials'] = 0
#%%
for i in range(len(dftrain['price'])):
    costo = int(dftrain['price'][i].replace('$','').replace('.00','').replace(',',''))
    try:
        dftrain['price'][i] = costo
    except:
        pass
#%%       
dftrain['price'] = dftrain['price'].astype(int)
dftrain['host_response_rate'] = dftrain['host_response_rate'].astype(float)
dftrain['host_acceptance_rate'] = dftrain['host_acceptance_rate'].astype(float)
#%%       
for i in range(len(dftrain['amenities'])):
    for j in dftrain['amenities'][i]:
        for k in range(len(df_amenities['Elemento'])):
            if df_amenities['Elemento'][k] == j:
                if df_amenities['Clasificacion'][k] == 'Kitchen':
                    dftrain['Amenities_Kitchen'][i] += 1
                    
                elif df_amenities['Clasificacion'][k] == 'Outdoors':
                    dftrain['Amenities_Outdoors'][i] += 1
                    
                elif df_amenities['Clasificacion'][k] == 'Accessibility':
                    dftrain['Amenities_Accessibility'][i] += 1
                    
                elif df_amenities['Clasificacion'][k] == 'Well Being':
                    dftrain['Amenities_Well Being'][i] += 1
                    
                elif df_amenities['Clasificacion'][k] == 'Heating/Cooling':
                    dftrain['Amenities_Heating/Cooling'][i] += 1
                    
                elif df_amenities['Clasificacion'][k] == 'Bath & Cleaning':
                    dftrain['Amenities_Bath & Cleaning'][i] += 1
                    
                elif df_amenities['Clasificacion'][k] == 'Entertainment':
                    dftrain['Amenities_Entertainment'][i] += 1
                    
                elif df_amenities['Clasificacion'][k] == 'Children':
                    dftrain['Amenities_Children'][i] += 1
                    
                elif df_amenities['Clasificacion'][k] == 'Admnistration':
                    dftrain['Amenities_Admnistration'][i] += 1
                    
                elif df_amenities['Clasificacion'][k] == 'Essentials':
                    dftrain['Amenities_Essentials'][i] += 1
                else:
                    dftrain['Amenities_Miscellaneous'][i] += 1

#%%
diccionario_barrios = {
    'Centrum-Oost': 'centro',
    'Centrum-West': 'centro',
    'Oud-Oost': 'este',
    'De Pijp - Rivierenbuurt': 'sur',
    'Noord-Oost': 'noreste',
    'Oud-Noord': 'norte',
    'Noord-West': 'noroeste',
    'De Aker - Nieuw Sloten': 'oeste',
    'Geuzenveld - Slotermeer': 'oeste',
    'Bijlmer-Centrum': 'sureste',
    'Buitenveldert - Zuidas': 'sur',
    'Westerpark': 'oeste',
    'Slotervaart': 'oeste',
    'De Baarsjes - Oud-West': 'oeste',
    'Bos en Lommer': 'oeste',
    'IJburg - Zeeburgereiland': 'este',
    'Watergraafsmeer': 'este',
    'Osdorp': 'oeste',
    'Gaasperdam - Driemond': 'sureste',
    'Bijlmer-Oost': 'sureste'
}

dftrain['neighbourhood_zone'] = dftrain['neighbourhood_cleansed'].replace(diccionario_barrios) 
#%%
le = LabelEncoder()

columns = ['source','neighbourhood_zone','host_response_time', 'host_verifications', 'neighbourhood_cleansed', 'property_type', 'bathrooms_text','room_type']
for i in columns:
    dftrain[i] = le.fit_transform(dftrain[i])
#%%
# Descargar recursos adicionales de NLTK
nltk.download('vader_lexicon')

# Crear una instancia del analizador de sentimientos
sia = SentimentIntensityAnalyzer()

# Función para realizar el análisis de sentimientos
def analyze_sentiment(message):
    # Obtener el puntaje de sentimiento para el mensaje
    sentiment_score = sia.polarity_scores(message)
    
    # Determinar la etiqueta del sentimiento
    if sentiment_score['pos'] >= 0.5:
        sentiment_label = 'Muy Positivo'
    elif sentiment_score['pos'] >= 0.1 and sentiment_score['pos'] < 0.5:
        sentiment_label = 'Positivo'    
    elif sentiment_score['neg'] >= 0.1 and sentiment_score['neg'] < 0.5:
        sentiment_label = 'Negativo'      
    elif sentiment_score['neg'] >= 0.5:
        sentiment_label = 'Muy Negativo'
    else:
        sentiment_label = 'Neutro'
    
    return sentiment_label
#%%
dftrain['neighborhood_analysis'] = ''
for i in range(len(dftrain['neighborhood_overview'])):    
    article = dftrain['neighborhood_overview'][i]
    try:
        dftrain['neighborhood_analysis'][i] = analyze_sentiment(dftrain['neighborhood_overview'][i])
    except:
        pass 
#%%
dftrain = dftrain.drop(columns=['index','first_review','last_review','host_since','host_about', 'neighbourhood_group_cleansed', 'bathrooms','amenities', 'host_name', 'license', 'calendar_updated', 'calendar_last_scraped','neighbourhood','host_location','neighborhood_overview','host_neighbourhood','name','description'])
#%%
columns = ['neighborhood_analysis']
for i in columns:
    dftrain[i] = le.fit_transform(dftrain[i])
#%%
dftest = dftest.drop_duplicates()
#%%
dftest['bedrooms'].fillna(dftest['bedrooms'].mean(), inplace=True)
dftest['bedrooms'] = dftest['bedrooms'].astype(int)
dftest['beds'].fillna(dftest['beds'].mean(), inplace=True)
dftest['beds'] = dftest['beds'].astype(int)
#%%
dftest.host_is_superhost = dftest.host_is_superhost.replace({'t': 1, 'f': 0})
dftest.host_has_profile_pic = dftest.host_has_profile_pic.replace({'t': 1, 'f': 0})
dftest.host_identity_verified = dftest.host_identity_verified.replace({'t': 1, 'f': 0})
dftest.has_availability = dftest.has_availability.replace({'t': 1, 'f': 0})   
dftest.instant_bookable = dftest.instant_bookable.replace({'t': 1, 'f': 0})
#%%
dftest["first_review"] = pd.to_datetime(dftest["first_review"])
dftest["last_review"] = pd.to_datetime(dftest["last_review"])
dftest["host_since"] = pd.to_datetime(dftest["host_since"])
dftest["calendar_last_scraped"] = pd.to_datetime(dftest["calendar_last_scraped"])
#%%
dftest['antiquity'] = (pd.to_datetime(dt.date.today()) - dftest['host_since']) / np.timedelta64(1, 'D')
dftest['since_last_review'] = (pd.to_datetime(dt.date.today()) - dftest['last_review']) / np.timedelta64(1, 'D')
#%%
for i in range(len(dftest['neighborhood_overview'])):    
    article = dftest['neighborhood_overview'][i]
    try:
        article = re.sub(r'<br ?/?>', '', article)
        article = re.sub(r'<?/?b>','',article)
        dftest['neighborhood_overview'][i] = article
    except:
        pass
#%%
for i in range(len(dftest['amenities'])):
    lista = dftest['amenities'][i]
    try:
        lista = eval(lista)
        for j in range(len(lista)):
            lista[j] = re.sub('\u2013', '-', lista[j])
            lista[j] = re.sub('\u2019',"'",lista[j])
            dftest['amenities'][i] = lista
    except:
        pass
#%%
ameniti = []
for i in dftest['amenities']:
    for j in i:
        if j not in ameniti:
            ameniti.append(j)
#%%
for i in range(len(dftest['host_response_rate'])):
    value = dftest['host_response_rate'][i]
    try:
        dftest['host_response_rate'][i] = float(value.replace('%',''))/100
    except:
        dftest['host_response_rate'][i] = 0
for i in range(len(dftest['host_acceptance_rate'])):
    value = dftest['host_acceptance_rate'][i]
    try:
        dftest['host_acceptance_rate'][i] = float(value.replace('%',''))/100
    except:
        dftest['host_acceptance_rate'][i] = 0
#%%
total = []
for i in dftest['amenities']:
    for j in i:
        total.append(j)
#%%
columna = [elemento for lista in dftest['amenities'] for elemento in lista]

# Obtener la cuenta de apariciones de cada elemento
contador = Counter(columna)

# Ordenar el diccionario por la cantidad de apariciones en orden descendente
diccionario_ordenado = dict(sorted(contador.items(), key=itemgetter(1), reverse=True))

# Crear el DataFrame a partir del diccionario ordenado
diccionario_ordenado = pd.DataFrame(list(diccionario_ordenado.items()), columns=['Elemento', 'Apariciones'])
#%%
diccionario_ordenado = diccionario_ordenado[diccionario_ordenado['Apariciones']>=50]
#%%
diccionario_ordenado.to_clipboard(index=False)
#%%
amenities = [
    ("Essentials", 4715, "Essentials"),
    ("Smoke alarm", 4421, "Well Being"),
    ("Wifi", 4337, "Essentials"),
    ("Heating", 4101, "Heating/Cooling"),
    ("Hot water", 3990, "Heating/Cooling"),
    ("Hangers", 3949, "Miscellaneous"),
    ("Hair dryer", 3913, "Well Being"),
    ("Kitchen", 3841, "Kitchen"),
    ("Dishes and silverware", 3781, "Kitchen"),
    ("Refrigerator", 3630, "Kitchen"),
    ("Iron", 3618, "Well Being"),
    ("Long term stays allowed", 3583, "Administration"),
    ("Shampoo", 3412, "Bath & Cleaning"),
    ("Bed linens", 3215, "Bath & Cleaning"),
    ("Cooking basics", 3060, "Kitchen"),
    ("Carbon monoxide alarm", 2810, "Well Being"),
    ("Coffee maker", 2715, "Kitchen"),
    ("Fire extinguisher", 2628, "Well Being"),
    ("Dishwasher", 2435, "Kitchen"),
    ("Washer", 2432, "Bath & Cleaning"),
    ("Private entrance", 2407, "Accessibility"),
    ("First aid kit", 2376, "Well Being"),
    ("Oven", 2368, "Kitchen"),
    ("Microwave", 2319, "Kitchen"),
    ("Stove", 2100, "Kitchen"),
    ("Dedicated workspace", 1813, "Administration"),
    ("Extra pillows and blankets", 1796, "Bath & Cleaning"),
    ("TV", 1616, "Entertainment"),
    ("Host greets you", 1594, "Administration"),
    ("Luggage dropoff allowed", 1581, "Administration"),
    ("Cleaning products", 1508, "Bath & Cleaning"),
    ("Shower gel", 1502, "Bath & Cleaning"),
    ("Dryer", 1454, "Bath & Cleaning"),
    ("Wine glasses", 1429, "Kitchen"),
    ("Dining table", 1427, "Kitchen"),
    ("Paid parking off premises", 1407, "Accessibility"),
    ("Hot water kettle", 1359, "Kitchen"),
    ("Private patio or balcony", 1290, "Outdoors"),
    ("Freezer", 1261, "Kitchen"),
    ("TV with standard cable", 1228, "Entertainment"),
    ("Outdoor furniture", 1181, "Outdoors"),
    ("Coffee", 1121, "Kitchen"),
    ("Body soap", 1056, "Bath & Cleaning"),
    ("Self check-in", 1048, "Administration"),
    ("Bathtub", 1033, "Bath & Cleaning"),
    ("Room-darkening shades", 1032, "Bath & Cleaning"),
    ("Drying rack for clothing", 988, "Bath & Cleaning"),
    ("Laundromat nearby", 918, "Bath & Cleaning"),
    ("Lock on bedroom door", 895, "Safety"),
    ("Patio or balcony", 895, "Outdoors"),
    ("Waterfront", 883, "Outdoors"),
    ("Outdoor dining area", 829, "Outdoors"),
    ("Toaster", 810, "Kitchen"),
    ("Portable fans", 788, "Well Being"),
    ("Books and reading material", 760, "Entertainment"),
    ("Single level home", 757, "Miscellaneous"),
    ("Free washer - In unit", 755, "Bath & Cleaning"),
    ("Lockbox", 750, "Accessibility"),
    ("Children's books and toys", 602, "Children"),
    ("Central heating", 596, "Heating/Cooling"),
    ("Canal view", 595, "Outdoors"),
    ("High chair", 592, "Children"),
    ("Ethernet connection", 589, "Accessibility"),
    ("Baking sheet", 577, "Kitchen"),
    ("Conditioner", 573, "Bath & Cleaning"),
    ("Pets allowed", 565, "Miscellaneous"),
    ("Blender", 559, "Kitchen"),
    ("Paid street parking off premises", 557, "Accessibility"),
    ("Backyard", 548, "Outdoors"),
    ("BBQ grill", 545, "Outdoors"),
    ("Free dryer - In unit", 544, "Bath & Cleaning"),
    ("Paid parking on premises", 536, "Accessibility"),
    ("Garden view", 526, "Outdoors"),
    ("Free parking on premises", 504, "Accessibility"),
    ("Board games", 496, "Entertainment"),
    ("EV charger", 486, "Accessibility"),
    ("Coffee maker: Nespresso", 479, "Kitchen"),
    ("Crib", 469, "Children"),
    ("Children's dinnerware", 449, "Children"),
    ("Private backyard - Fully fenced", 428, "Outdoors"),
    ("Elevator", 411, "Accessibility"),
    ("Air conditioning", 400, "Heating/Cooling"),
    ("Mini fridge", 397, "Kitchen"),
    ("Free street parking", 378, "Accessibility"),
    ("Clothing storage: closet", 371, "Bath & Cleaning"),
    ("Security cameras on property", 338, "Safety"),
    ("Pack 'n play/Travel crib", 337, "Children"),
    ("Private living room", 336, "Miscellaneous"),
    ("Babysitter recommendations", 302, "Children"),
    ("Pocket wifi", 300, "Accessibility"),
    ("City skyline view", 291, "Outdoors"),
    ("Clothing storage", 289, "Bath & Cleaning"),
    ("Barbecue utensils", 275, "Kitchen"),
    ("Indoor fireplace", 272, "Well Being"),
    ("Courtyard view", 268, "Outdoors"),
    ("Changing table", 234, "Children"),
    ("Cleaning available during stay", 227, "Bath & Cleaning"),
    ("Lake access", 217, "Outdoors"),
    ("Breakfast", 216, "Well Being"),
    ("Clothing storage: wardrobe", 213, "Bath & Cleaning"),
    ("Outlet covers", 203, "Safety"),
    ("Smoking allowed", 200, "Miscellaneous"),
    ("Bikes", 200, "Outdoors"),
    ("Baby bath", 200, "Children"),
    ("Shared patio or balcony", 181, "Outdoors"),
    ("Piano", 172, "Entertainment"),
    ("Stainless steel oven", 172, "Kitchen"),
    ("Sound system", 171, "Entertainment"),
    ("Safe", 157, "Safety"),
    ("Mosquito net", 151, "Safety"),
    ("Park view", 149, "Outdoors"),
    ("Fire pit", 145, "Outdoors"),
    ("Baby safety gates", 142, "Children"),
    ("Gas stove", 138, "Kitchen"),
    ("Game console", 129, "Entertainment"),
    ("Building staff", 129, "Administration"),
    ("Rice maker", 126, "Kitchen"),
    ("Record player", 125, "Entertainment"),
    ("Hammock", 122, "Outdoors"),
    ("Bluetooth sound system", 114, "Entertainment"),
    ("Coffee maker: espresso machine", 113, "Kitchen"),
    ("Boat slip", 105, "Outdoors"),
    ("River view", 100, "Outdoors"),
    ("Resort access", 95, "Outdoors"),
    ("Window guards", 95, "Safety"),
    ("Smart lock", 90, "Safety"),
    ("Central air conditioning", 90, "Heating/Cooling"),
    ("Baby monitor", 86, "Children"),
    ("Induction stove", 85, "Kitchen"),
    ("Keypad", 82, "Safety"),
    ("Portable air conditioning", 80, "Heating/Cooling"),
    ("Gym", 77, "Well Being"),
    ("BBQ grill: charcoal", 68, "Outdoors"),
    ("Private hot tub", 67, "Well Being"),
    ("Ceiling fan", 66, "Well Being"),
    ("Sonos sound system", 66, "Entertainment"),
    ("HDTV with Netflix", 65, "Entertainment"),
    ("Free washer - In building", 64, "Bath & Cleaning"),
    ("Harbor view", 63, "Outdoors"),
    ("Free dryer - In building", 63, "Bath & Cleaning"),
    ("Sonos Bluetooth sound system", 62, "Entertainment"),
    ("Indoor fireplace: wood-burning", 62, "Well Being"),
    ("Lake view", 62, "Outdoors"),
    ("Clothing storage: dresser", 61, "Bath & Cleaning"),
    ("Paid washer - In building", 60, "Bath & Cleaning"),
    ("Private backyard - Not fully fenced", 56, "Outdoors"),
    ("Stainless steel gas stove", 54, "Kitchen"),
    ("Bread maker", 54, "Kitchen"),
    ("Paid parking garage off premises", 53, "Accessibility"),
    ("Coffee maker: drip coffee maker", 51, "Kitchen"),
    ("Game console: PS4", 51, "Entertainment"),
    ("Paid dryer - In building", 51, "Bath & Cleaning"),
    ("Exercise equipment: free weights, yoga mat", 50, "Well Being")
]

columns = ["Elemento", "Apariciones", "Clasificacion"]
df_amenities = pd.DataFrame(amenities, columns=columns)

df_amenities = df_amenities.drop(columns=['Apariciones'])
#%%
dftest['Amenities_Kitchen'] = 0
dftest['Amenities_Outdoors'] = 0
dftest['Amenities_Accessibility'] = 0
dftest['Amenities_Well Being'] = 0
dftest['Amenities_Heating/Cooling'] = 0
dftest['Amenities_Miscellaneous'] = 0
dftest['Amenities_Bath & Cleaning'] = 0
dftest['Amenities_Entertainment'] = 0
dftest['Amenities_Children'] = 0
dftest['Amenities_Admnistration'] = 0
dftest['Amenities_Essentials'] = 0
#%%
for i in range(len(dftest['price'])):
    costo = int(dftest['price'][i].replace('$','').replace('.00','').replace(',',''))
    try:
        dftest['price'][i] = costo
    except:
        pass
#%%       
dftest['price'] = dftest['price'].astype(int)
dftest['host_response_rate'] = dftest['host_response_rate'].astype(float)
dftest['host_acceptance_rate'] = dftest['host_acceptance_rate'].astype(float)
#%%       
for i in range(len(dftest['amenities'])):
    for j in dftest['amenities'][i]:
        for k in range(len(df_amenities['Elemento'])):
            if df_amenities['Elemento'][k] == j:
                if df_amenities['Clasificacion'][k] == 'Kitchen':
                    dftest['Amenities_Kitchen'][i] += 1
                    
                elif df_amenities['Clasificacion'][k] == 'Outdoors':
                    dftest['Amenities_Outdoors'][i] += 1
                    
                elif df_amenities['Clasificacion'][k] == 'Accessibility':
                    dftest['Amenities_Accessibility'][i] += 1
                    
                elif df_amenities['Clasificacion'][k] == 'Well Being':
                    dftest['Amenities_Well Being'][i] += 1
                    
                elif df_amenities['Clasificacion'][k] == 'Heating/Cooling':
                    dftest['Amenities_Heating/Cooling'][i] += 1
                    
                elif df_amenities['Clasificacion'][k] == 'Bath & Cleaning':
                    dftest['Amenities_Bath & Cleaning'][i] += 1
                    
                elif df_amenities['Clasificacion'][k] == 'Entertainment':
                    dftest['Amenities_Entertainment'][i] += 1
                    
                elif df_amenities['Clasificacion'][k] == 'Children':
                    dftest['Amenities_Children'][i] += 1
                    
                elif df_amenities['Clasificacion'][k] == 'Admnistration':
                    dftest['Amenities_Admnistration'][i] += 1
                    
                elif df_amenities['Clasificacion'][k] == 'Essentials':
                    dftest['Amenities_Essentials'][i] += 1
                else:
                    dftest['Amenities_Miscellaneous'][i] += 1
#%%
diccionario_barrios = {
    'Centrum-Oost': 'centro',
    'Centrum-West': 'centro',
    'Oud-Oost': 'este',
    'De Pijp - Rivierenbuurt': 'sur',
    'Noord-Oost': 'noreste',
    'Oud-Noord': 'norte',
    'Noord-West': 'noroeste',
    'De Aker - Nieuw Sloten': 'oeste',
    'Geuzenveld - Slotermeer': 'oeste',
    'Bijlmer-Centrum': 'sureste',
    'Buitenveldert - Zuidas': 'sur',
    'Westerpark': 'oeste',
    'Slotervaart': 'oeste',
    'De Baarsjes - Oud-West': 'oeste',
    'Bos en Lommer': 'oeste',
    'IJburg - Zeeburgereiland': 'este',
    'Watergraafsmeer': 'este',
    'Osdorp': 'oeste',
    'Gaasperdam - Driemond': 'sureste',
    'Bijlmer-Oost': 'sureste'
}

dftest['neighbourhood_zone'] = dftest['neighbourhood_cleansed'].replace(diccionario_barrios)   
#%%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

columns = ['source','neighbourhood_zone','host_response_time', 'host_verifications', 'neighbourhood_cleansed', 'property_type', 'bathrooms_text','room_type']
for i in columns:
    dftest[i] = le.fit_transform(dftest[i])
#%%
# Descargar recursos adicionales de NLTK
nltk.download('vader_lexicon')

# Crear una instancia del analizador de sentimientos
sia = SentimentIntensityAnalyzer()

# Función para realizar el análisis de sentimientos
def analyze_sentiment(message):
    # Obtener el puntaje de sentimiento para el mensaje
    sentiment_score = sia.polarity_scores(message)
    
    # Determinar la etiqueta del sentimiento
    if sentiment_score['pos'] >= 0.5:
        sentiment_label = 'Muy Positivo'
    elif sentiment_score['pos'] >= 0.1 and sentiment_score['compound'] < 0.5:
        sentiment_label = 'Positivo'    
    elif sentiment_score['neg'] >= 0.1 and sentiment_score['compound'] < 0.5:
        sentiment_label = 'Negativo'      
    elif sentiment_score['neg'] >= 0.5:
        sentiment_label = 'Muy Negativo'
    else:
        sentiment_label = 'Neutro'
    
    return sentiment_label
#%%
dftest['neighborhood_analysis'] = ''
for i in range(len(dftest['neighborhood_overview'])):    
    article = dftest['neighborhood_overview'][i]
    try:
        dftest['neighborhood_analysis'][i] = analyze_sentiment(dftest['neighborhood_overview'][i])
    except:
        pass
 
#%%
dftest = dftest.drop(columns=['first_review','last_review','host_since','host_about', 'neighbourhood_group_cleansed', 'bathrooms','amenities','host_name', 'license', 'calendar_updated', 'calendar_last_scraped','neighbourhood','host_location','neighborhood_overview','host_neighbourhood','name','description'])
#%%
columns = ['neighborhood_analysis']
for i in columns:
    dftest[i] = le.fit_transform(dftest[i])
#%%
dftrain = dftrain[['id','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_communication','review_scores_value']]
dftest = dftest[['id','review_scores_accuracy','review_scores_cleanliness','review_scores_communication','review_scores_value']]
#%%
Xtrain = dftrain.drop(columns=['review_scores_rating'])
Ytrain = dftrain['review_scores_rating']
#%%
Xtrain['review_scores_accuracy'].fillna(Xtrain['review_scores_accuracy'].mean(), inplace=True)
Xtrain['review_scores_cleanliness'].fillna(Xtrain['review_scores_cleanliness'].mean(), inplace=True)
#Xtrain['review_scores_checkin'].fillna(Xtrain['review_scores_checkin'].mean(), inplace=True)
Xtrain['review_scores_communication'].fillna(Xtrain['review_scores_communication'].mean(), inplace=True)
#Xtrain['review_scores_location'].fillna(Xtrain['review_scores_location'].mean(), inplace=True)
Xtrain['review_scores_value'].fillna(Xtrain['review_scores_value'].mean(), inplace=True)
#%%
Xtrain = Xtrain.drop(columns=['source','host_is_superhost', 'host_listings_count','host_total_listings_count','host_verifications', 'host_has_profile_pic', 'host_identity_verified', 'neighbourhood_cleansed', 'latitude', 'longitude', 'property_type',
                              'minimum_nights', 'maximum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'has_availability', 'availability_30', 'availability_60', 'availability_90', 'availability_365',
                              'instant_bookable', 'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms', 'host_response_rate','host_acceptance_rate','room_type','accommodates','bathrooms_text','number_of_reviews_ltm','number_of_reviews_l30d'])
#%%
dftrain = dftrain.drop(columns=['id'])
dftest = dftest.drop(columns=['id'])
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xtrain, Ytrain, test_size = 0.2, random_state = 0)
#%%
from sklearn.metrics import mean_squared_error

arbol = DecisionTreeRegressor(random_state=42)

arbol.fit(X_train, y_train)
#%%
y_pred = arbol.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
r2 = r2_score(y_test, y_pred)
print('R²:', r2)
#%%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [None, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 4, 10]
}

grid_search = GridSearchCV(estimator=DecisionTreeRegressor(random_state=32), param_grid=param_grid, scoring='r2', cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

arbol = DecisionTreeRegressor(random_state=32, **best_params)
arbol.fit(X_train, y_train)

y_pred = arbol.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
r2 = r2_score(y_test, y_pred)
print('R²:', r2)
#%%
random_forest = RandomForestRegressor(n_estimators=150, random_state=32)
random_forest.fit(X_train, y_train)

y_pred_rf = random_forest.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
print("MSE (Random Forest):", mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print('R² (Random Forest):', r2_rf)
#%%
feature_importance = arbol.feature_importances_

for feature, importance in zip(Xtrain.columns, feature_importance):
    print(feature, importance)
#%%
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

feature_importances = rf_model.feature_importances_

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances)
plt.xticks(range(len(feature_importances)), X_train.columns, rotation='vertical')
plt.xlabel('Características')
plt.ylabel('Importancia')
plt.title('Importancia de características en Random Forest')
plt.show()
#%%
linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)
r2 = r2_score(y_test, y_pred)
print('R²:', r2)
#%%
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=0.1) 
ridge_model.fit(X_train, y_train)

y_pred = ridge_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)
r2 = r2_score(y_test, y_pred)
print('R²:', r2)
#%%
from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.01) 
lasso_model.fit(X_train, y_train)

y_pred = lasso_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)
r2 = r2_score(y_test, y_pred)
print('R²:', r2)
#%%
from sklearn.linear_model import Lasso

alphas = [0.01, 0.1, 1.0, 10.0]

for alpha in alphas:
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train, y_train)
    y_pred = lasso_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Alpha: {alpha}")
    print('MSE:', mse)
    print('R²:', r2)
    print("---")

best_alpha = alphas[np.argmax(r2_score)]
best_lasso_model = Lasso(alpha=best_alpha)
best_lasso_model.fit(X_train, y_train)
y_pred = best_lasso_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mejor modelo Lasso:")
print('MSE:', mse)
print('R²:', r2)
#%%
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

ridge_model = Ridge(alpha=0.05)

ridge_model.fit(X_train, y_train)

y_pred = ridge_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)
r2 = r2_score(y_test, y_pred)
print('R²:', r2)
#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)    

rf_model = RandomForestRegressor(max_depth=5, n_estimators=100)
rf_model.fit(X_train_scaled, y_train)

y_pred = rf_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)
r2 = r2_score(y_test, y_pred)
print('R²:', r2)
#%%
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

rf_model = RandomForestRegressor(max_depth=5, n_estimators=100)

param_grid = {
    'n_estimators': [100,200, 300],
    'max_depth': [None, 5, 10]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='r2', cv=5)

grid_search.fit(X_train, y_train.ravel())

best_params = grid_search.best_params_
best_score = grid_search.best_score_

best_rf_model = RandomForestRegressor(**best_params)
best_rf_model.fit(X_train, y_train.ravel())

y_pred = best_rf_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print('R²:', r2)
#%%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

gb_model = GradientBoostingRegressor()

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.1, 0.05, 0.01],
    'max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, scoring='r2', cv=5)
grid_search.fit(X_train, y_train.ravel())

best_params = grid_search.best_params_
best_score = grid_search.best_score_

best_gb_model = GradientBoostingRegressor(**best_params)
best_gb_model.fit(X_train, y_train.ravel())

y_pred = best_gb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)
r2 = r2_score(y_test, y_pred)
print('R²:', r2)
#%%
param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'depth': [6,8,10],
    'n_estimators': [100, 150, 200, 300],
    'bagging_temperature': [0,1],
}

catboost_model = CatBoostRegressor()

# Realizar la búsqueda de hiperparámetros utilizando GridSearchCV
grid_search = GridSearchCV(estimator=catboost_model, param_grid=param_grid, cv=15, scoring='r2')
grid_search.fit(X_train, y_train)

# Obtener el mejor modelo con los hiperparámetros óptimos
best_catboost_model = grid_search.best_estimator_

# Hacer predicciones con el mejor modelo
y_pred = best_catboost_model.predict(X_test)

# Calcular métricas de evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir los resultados
print('MSE:', mse)
print('R²:', r2)
print('Mejores hiperparámetros:', grid_search.best_params_)
#%%
from lightgbm import LGBMRegressor
import lightgbm as lgb
lgb_model = lgb.LGBMRegressor()

lgb_model.fit(X_train, y_train)

y_pred = lgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)
r2 = r2_score(y_test, y_pred)
print('R²:', r2)
#%%
from sklearn.model_selection import RandomizedSearchCV
lgb_model = LGBMRegressor()

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.05, 0.01],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

random_search = RandomizedSearchCV(estimator=lgb_model, param_distributions=param_grid, scoring='r2', cv=15, n_iter=50, random_state=42)
random_search.fit(X_train, y_train.ravel())

best_params = random_search.best_params_
best_score = random_search.best_score_

best_lgb_model = LGBMRegressor(**best_params)
best_lgb_model.fit(X_train, y_train.ravel())

y_pred = best_lgb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)
r2 = r2_score(y_test, y_pred)
print('R²:', r2)
#%%
import optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300]),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_uniform('subsample', 0.8, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.8, 1.0)
    }
    
    lgb_model = LGBMRegressor(**params)
    lgb_model.fit(X_train, y_train)
    
    y_pred = lgb_model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    return r2

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

best_params = study.best_params
best_lgb_model = LGBMRegressor(**best_params)
best_lgb_model.fit(X_train, y_train)

y_pred = best_lgb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)
r2 = r2_score(y_test, y_pred)
print('R²:', r2)
#%%
X_val = dftest
y_pred = random_forest.predict(X_val)

df_results = pd.DataFrame({'id': dftest['id'], 'review_scores_rating': y_pred})

print(df_results.dtypes.unique())
print(df_results)
#%%
for i in range(len(y_pred['review_scores_rating'])):
    if y_pred['review_scores_rating'][i] > 5.0:
        y_pred['review_scores_rating'][i] = 5.0
#%%
df_results.to_csv('results_rf.csv', index=False)
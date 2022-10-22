#%%
import pandas as pd
import numpy as np
import pycountry_convert as pc

def country_to_continent(country_name):
    country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    return country_continent_name

# %%
f = open('loans_ridge.csv', 'r', encoding="utf8")
data = pd.read_csv(f)
# %% ADD 10 VARIABLES RELATEDTO TEXT ANALYSIS
# pos and neg words
pos_words = ['years', 'children', 'married', 'help', 'lives', 'income', 'old', 'husband', 'living', 'selling', 'kiva']
neg_words = ['loan', 'business', 'buy', 'family', 'work', 'house', 'store', 'improve']

data['description_texts_en'] = data['description_texts_en'].astype(str)

# add 5 variables for aspects of description
data['contains_old'] = data['description_texts_en'].apply(lambda x: 1 if  'old' in x.lower() else 0)
data['contains_improve'] = data['description_texts_en'].apply(lambda x: 1 if  'improve' in x.lower() else 0)
data['contains_help'] = data['description_texts_en'].apply(lambda x: 1 if  'help' in x.lower() else 0)
data['contains_buy'] = data['description_texts_en'].apply(lambda x: 1 if  'buy' in x.lower() else 0)
data['contains_loan'] = data['description_texts_en'].apply(lambda x: 1 if  'loan' in x.lower() else 0)

# add variable for num neg_words in description
for n in neg_words:
    data[n] = data['description_texts_en'].str.contains(n, case=False).astype(int)
data['negative_sentiment'] = data[neg_words].sum(axis=1)

# add variable for num pos_words in description
for p in pos_words:
    data[p] = data['description_texts_en'].str.contains(p, case=False).astype(int) 
data['positive_sentiment'] = data[pos_words].sum(axis=1)

# add indicator for if description is overall pos or neg
data['overall_positive'] = (data['positive_sentiment'] > data['negative_sentiment']) * 1 # cast to int

data['description_length'] = data['description_texts_en'].apply(lambda x: len(str(x).split()))

# add indicators for if description is especially pos or especially neg
# data['high_negative_sentiment'] = data['negative_sentiment'].apply(lambda x : 1 if x > 4 else 0)
# data['high_positive_sentiment'] = data['positive_sentiment'].apply(lambda x : 1 if x > 4  else 0)
# %% ADD 9 FACTOR VARIABLES (for gender, country, sector)

data['borrowers_borrower_gender'] = data['borrowers_borrower_gender'].astype(str)
data['location_country'] = data['location_country'].astype(str)
data['sector'] = data['sector'].astype(str)
data['status'] = data['status'].astype(str)

# dummy for gender
data['is_female'] = data['borrowers_borrower_gender'].apply(lambda x: 1 if x == 'F' else 0)

# dummies for country = iraq and country = bulgaria,because those countries have a much higher avg_days until funded
# (especially bulgaria)
data['is_bulgaria'] = data['location_country'].apply(lambda x: 1 if 'bulgaria' in x.lower() else 0)
data['is_iraq'] = data['location_country'].apply(lambda x: 1 if 'iraq' in x.lower() else 0)
# data =  data.assign(is_bulgaria= lambda row: 1 if row.location_country is 'Bulgaria' else 0,
#            is_iraq= lambda row: 1 if row.location_country is 'Iraq' else 0)

# dummies for country = zambia and country = thailand because those 2 have the lowest avg days_until_funded
data['is_nepal'] = data['location_country'].apply(lambda x: 1 if 'nepal' in x.lower() else 0)
data['is_timor'] = data['location_country'].apply(lambda x: 1 if 'timor' in x.lower() else 0)
# data =  data.assign(is_zambia= lambda row: 1 if row.location_country is 'Zambia' else 0,
#            is_thailand= lambda row: 1 if row.location_country is 'Thailand' else 0)

# dummies for sectors with avg days_until_funded < 1 (health, education, arts, agriculture, and food) as well
# as sector with highest avg days_until_funded (housing)
data['is_health'] = data['sector'].apply(lambda x: 1 if x == 'Health' else 0)
data['is_education'] = data['sector'].apply(lambda x: 1 if x == 'Education' else 0)
data['is_arts'] = data['sector'].apply(lambda x: 1 if x == 'Arts' else 0)
data['is_agriculture'] = data['sector'].apply(lambda x: 1 if x == 'Agriculture' else 0)
data['is_food'] = data['sector'].apply(lambda x: 1 if x == 'Food' else 0)
data['is_housing'] = data['sector'].apply(lambda x: 1 if x == 'Housing' else 0)
# data =  data.assign(is_health=lambda row: 1 if row.sector == 'Health' else 0,
#            is_ed=lambda row: 1 if row.sector == 'Education' else 0,
#            is_arts=lambda row: 1 if row.sector == 'Arts' else 0,
#            is_agriculture=lambda row: 1 if row.sector == 'Agriculture' else 0,
#            is_food=lambda row: 1 if row.sector == 'Food' else 0,
#            is_housing=lambda row: 1 if row.sector == 'Housing' else 0)

   
for index, row in data.iterrows():
    country = row['location_country']
    country_africa = ['Congo', 'Cote']
    country_asia = ['Timor', 'Myanmar', 'Lao']
    country_europe = ['Kosovo']
    if any(substring in country for substring in country_africa):
        data.at[index, 'african'] = 1
        data.at[index, 'asian'] = 0
        data.at[index, 'north_american'] = 0
        data.at[index, 'south_american'] = 0
        data.at[index, 'european'] = 0
    elif any(substring in country for substring in country_asia):
        data.at[index, 'african'] = 0
        data.at[index, 'asian'] = 1
        data.at[index, 'north_american'] = 0
        data.at[index, 'south_american'] = 0
        data.at[index, 'european'] = 0
    elif any(substring in country for substring in country_europe):
        data.at[index, 'african'] = 0
        data.at[index, 'asian'] = 0
        data.at[index, 'north_american'] = 0
        data.at[index, 'south_american'] = 0
        data.at[index, 'european'] = 1
    else: 
        continent = country_to_continent(country)
        if continent == 'Africa':
            data.at[index, 'african'] = 1
            data.at[index, 'asian'] = 0
            data.at[index, 'north_american'] = 0
            data.at[index, 'south_american'] = 0
            data.at[index, 'european'] = 0
        elif continent == 'Asia':
            data.at[index, 'african'] = 0
            data.at[index, 'asian'] = 1
            data.at[index, 'north_american'] = 0
            data.at[index, 'south_american'] = 0
            data.at[index, 'european'] = 0
        elif continent == 'North America':
            data.at[index, 'african'] = 0
            data.at[index, 'asian'] = 0
            data.at[index, 'north_american'] = 1
            data.at[index, 'south_american'] = 0
            data.at[index, 'european'] = 0
        elif continent == 'South America':
            data.at[index, 'african'] = 0
            data.at[index, 'asian'] = 0
            data.at[index, 'north_american'] = 0
            data.at[index, 'south_american'] = 1
            data.at[index, 'european'] = 0
        elif continent == 'Europe':
            data.at[index, 'african'] = 0
            data.at[index, 'asian'] = 0
            data.at[index, 'north_american'] = 0
            data.at[index, 'south_american'] = 0
            data.at[index, 'european'] = 1
        else: 
            data.at[index, 'african'] = 0
            data.at[index, 'asian'] = 0
            data.at[index, 'north_american'] = 0
            data.at[index, 'south_american'] = 0
            data.at[index, 'european'] = 0

# create dummies for different statuses (paid, refunded, defaulted, in_repayment)
data['paid'] = data['status'].apply(lambda x: 1 if x == 'paid' else 0)
data['refunded'] = data['status'].apply(lambda x: 1 if x == 'refunded' else 0)
data['defaulted'] = data['status'].apply(lambda x: 1 if x == 'defaulted' else 0)
data['in_repayment'] = data['status'].apply(lambda x: 1 if x == 'in_repayment' else 0)
# data =  data.assign(paid=lambda row: 1 if row.status == 'paid' else 0,
#            refunded=lambda row: 1 if row.status == 'refunded' else 0,
#            defaulted=lambda row: 1 if row.status == 'defaulted' else 0,
#            in_repayment=lambda row: 1 if row.status == 'in_repayment' else 0)


# %%
# CONT VARS = funded_amt, terms_disbursal_amount, loan_amount


# %%
# original features to drop not that have added new ones
# NOTE: not doing anything with location_town or activity
# NOTE: droping paid_amount bc it is null for 10714 rows which is leading to absolute nightmares with type issues in OLS
drop_list = ['description_texts_en', 'id', 'location_town', 'borrowers_borrower_gender', 'sector', 'location_country', 
            'activity', 'status', 'paid_amount']

# also drop columns for individual words (but not indicators)
drop_list += pos_words
drop_list += neg_words

data.drop(drop_list, axis=1, inplace=True)

# %%

# move days_until_funded to first column so is easier to access later
days = data.pop('days_until_funded')
data.insert(0, 'days_until_funded', days)

data.to_csv('processed_data.csv')

# %%
# code to determine which countries/sectors have higher and lowest avg days_until_funded

# country
# countries = data['location_country'].unique()
# avg_days_per_country = {}
# avg_days_per_country_lst = []
# for country in countries:
#     avg_days_per_country_lst.append(data[data["location_country"] == country]["days_until_funded"].mean())
#     avg_days_per_country[country] = data[data["location_country"] == country]["days_until_funded"].mean()

# avg_days_per_country = sorted(avg_days_per_country.items(), key=lambda kv: kv[1])

#sector
# sectors = data['sector'].unique()
# avg_days_per_sector = {}
# avg_days_per_sector_lst = []
# for sector in sectors:
#     avg_days_per_sector_lst.append(data[data["sector"] == sector]["days_until_funded"].mean())
#     avg_days_per_sector[sector] = data[data["sector"] == sector]["days_until_funded"].mean()

# avg_days_per_sector = sorted(avg_days_per_sector.items(), key=lambda kv: kv[1])
# %%

#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


file_path = 'C:\\Users\\USER\\OneDrive\\Documents\\PACIFIQUE NTETA\\Data Analytics Projects\\Maven\\GACTT_RESULTS_ANONYMIZED_v2.csv'


# In[4]:


df = pd.read_csv(file_path)


# In[5]:


print(df.head())


# In[6]:


df.head(5)


# In[7]:


print(df.info())


# In[8]:


print(df.describe())


# In[9]:


print(df.describe(include = "all"))


# In[10]:


df.columns


# In[11]:


df.isnull()


# In[12]:


missing_data = df.isnull()
df.isnull().sum()


# In[13]:


numerical_columns = df.select_dtypes(include=['int', 'float']).columns
print(numerical_columns)


# In[14]:


print(df.dtypes)


# In[15]:


float_columns = df.select_dtypes(include=['float']).columns.tolist()
print(float_columns)


# In[16]:


df["What kind of flavorings do you add?"].info()


# In[17]:


df["What kind of flavorings do you add?"].value_counts()


# In[18]:


df["Coffee A - Personal Preference"].value_counts()


# In[19]:


df["What kind of flavorings do you add?"].isnull().sum()


# In[20]:


df['Gender'].isnull().sum()


# In[21]:


most_frequent_age_group = df["What is your age?"].mode


# In[22]:


print(most_frequent_age_group)


# In[23]:


most_frequent_age_group = df["What is your age?"].mode()
most_frequent_age_group


# In[24]:


df["What is your age?"].replace(np.nan, most_frequent_age_group[0], inplace=True)


# In[25]:


df["What is your age?"].info()


# In[26]:


brewing_method_mode = df["How do you brew coffee at home?"].mode()
brewing_method_mode


# In[27]:


df["How do you brew coffee at home?"].replace(np.nan, brewing_method_mode[0], inplace=True)
df["How do you brew coffee at home?"].value_counts()


# In[28]:


df["What is your favorite coffee drink?"].value_counts()


# In[29]:


df["What is your favorite coffee drink?"].describe()


# In[30]:


mode_favorite_drink = df["What is your favorite coffee drink?"].mode()
mode_favorite_drink


# In[31]:


df["What is your favorite coffee drink?"].replace(np.nan, mode_favorite_drink[0], inplace=True)
df["What is your favorite coffee drink?"].value_counts()


# In[32]:


df["Do you usually add anything to your coffee?"].value_counts()


# In[33]:


mode_coffee_addings = df["Do you usually add anything to your coffee?"].mode()
mode_coffee_addings


# In[34]:


df["Do you usually add anything to your coffee?"].replace(np.nan, mode_coffee_addings[0], inplace=True)
df["Do you usually add anything to your coffee?"].value_counts()


# In[35]:


matching_columns = [col for col in df.columns if "What" in col and "roast" in col]

print(matching_columns)


# In[36]:


mode_roast_level = df["What roast level of coffee do you prefer?"].mode()
mode_roast_level


# In[37]:


df["What roast level of coffee do you prefer?"].replace(np.nan, mode_roast_level[0], inplace=True)
df["What roast level of coffee do you prefer?"].value_counts()


# In[38]:


df["Ethnicity/Race"].describe()


# In[39]:


df["Ethnicity/Race"].info()


# In[40]:


mode_race = df["Ethnicity/Race"].mode()
mode_race


# In[41]:


df["Ethnicity/Race"].replace(np.nan, mode_race[0], inplace=True)
df["Ethnicity/Race"].info()


# In[42]:


df["Education Level"].info()


# In[43]:


mode_education = df["Education Level"].mode()
mode_education


# In[44]:


df["Education Level"].replace(np.nan, mode_education[0], inplace=True)
df["Education Level"].describe()


# In[45]:


df["Employment Status"].info()


# In[46]:


mode_employment = df["Employment Status"].mode()
mode_employment


# In[47]:


df["Employment Status"].value_counts()


# In[48]:


df["Employment Status"].replace(np.nan, mode_employment[0], inplace=True)
df["Employment Status"].info()


# In[49]:


df["Political Affiliation"].info()


# In[50]:


mode_political = df["Political Affiliation"].mode()
mode_political


# In[51]:


df["Political Affiliation"].replace(np.nan, mode_political[0], inplace=True)
df["Political Affiliation"].info()


# In[52]:


matching_columns = [col for col in df.columns if "from" in col]

print(matching_columns)


# In[53]:


df["Do you work from home or in person?"].info()


# In[54]:


mode_work_type = df["Do you work from home or in person?"].mode()
mode_work_type


# In[55]:


df["Do you work from home or in person?"].replace(np.nan, mode_work_type[0], inplace=True)
df["Do you work from home or in person?"].info()


# In[56]:


matching_columns = [col for col in df.columns if "How" in col and "many" in col]

print(matching_columns)


# In[57]:


df["How many cups of coffee do you typically drink per day?"].info()


# In[58]:


mode_cups_of_coffee = df["How many cups of coffee do you typically drink per day?"].mode()
mode_cups_of_coffee


# In[59]:


df["How many cups of coffee do you typically drink per day?"].replace(np.nan,mode_cups_of_coffee[0], inplace=True)
df["How many cups of coffee do you typically drink per day?"].info()


# In[60]:


matching_columns = [col for col in df.columns if "much" in col or "value" in col or "cup" in col or "money" in col]

print(matching_columns)


# In[61]:


df["What is the most you've ever paid for a cup of coffee?"].info()


# In[62]:


df["What is the most you've ever paid for a cup of coffee?"].value_counts()


# In[63]:


mode_most_paid_on_cofee = df["What is the most you've ever paid for a cup of coffee?"].mode()
mode_most_paid_on_cofee


# In[64]:


df["What is the most you've ever paid for a cup of coffee?"].replace(np.nan,mode_most_paid_on_cofee[0], inplace=True)
df["What is the most you've ever paid for a cup of coffee?"].info()


# In[65]:


df["What is the most you'd ever be willing to pay for a cup of coffee?"].info()


# In[66]:


mode_most_willing_on_coffee = df["What is the most you'd ever be willing to pay for a cup of coffee?"].mode()
mode_most_willing_on_coffee


# In[67]:


df["What is the most you'd ever be willing to pay for a cup of coffee?"].replace(np.nan, mode_most_willing_on_coffee[0], inplace=True)
df["What is the most you'd ever be willing to pay for a cup of coffee?"].info()


# In[68]:


df["In total, much money do you typically spend on coffee in a month?"].info()


# In[69]:


mode_total_money_month = df["In total, much money do you typically spend on coffee in a month?"].mode()
mode_total_money_month


# In[70]:


df["In total, much money do you typically spend on coffee in a month?"].replace(np.nan,mode_total_money_month[0], inplace=True)
df["In total, much money do you typically spend on coffee in a month?"].info()


# In[71]:


df["Approximately how much have you spent on coffee equipment in the past 5 years?"].info()


# In[72]:


mode_money_equipment = df["Approximately how much have you spent on coffee equipment in the past 5 years?"].mode()
mode_money_equipment


# In[73]:


df["Approximately how much have you spent on coffee equipment in the past 5 years?"].replace(np.nan,mode_money_equipment[0],inplace=True)
df["Approximately how much have you spent on coffee equipment in the past 5 years?"].info()


# In[74]:


df["Do you feel like you’re getting good value for your money when you buy coffee at a cafe?"].info()


# In[75]:


mode_value = df["Do you feel like you’re getting good value for your money when you buy coffee at a cafe?"].mode()
mode_value


# In[76]:


df["Do you feel like you’re getting good value for your money when you buy coffee at a cafe?"].replace(np.nan,mode_value[0], inplace=True)
df["Do you feel like you’re getting good value for your money when you buy coffee at a cafe?"].info()


# In[77]:


df["Do you feel like you’re getting good value for your money with regards to your coffee equipment?"].info()


# In[78]:


mode_value_equip = df["Do you feel like you’re getting good value for your money with regards to your coffee equipment?"].mode()
mode_value_equip


# In[79]:


df["Do you feel like you’re getting good value for your money with regards to your coffee equipment?"].replace(np.nan,mode_value_equip[0], inplace=True)
df["Do you feel like you’re getting good value for your money with regards to your coffee equipment?"].info()


# In[80]:


matching_columns = [col for col in df.columns if "favorite" in col or "between" in col or "prefer" in col]

print(matching_columns)


# In[81]:


df["What is your favorite coffee drink?"].info()


# In[82]:


df["Between Coffee A, Coffee B, and Coffee C which did you prefer?"].info()


# In[83]:


mode_coffee_preference = df["Between Coffee A, Coffee B, and Coffee C which did you prefer?"].mode()
mode_coffee_preference


# In[84]:


df["Between Coffee A, Coffee B, and Coffee C which did you prefer?"].replace(np.nan, mode_coffee_preference[0], inplace=True)
df["Between Coffee A, Coffee B, and Coffee C which did you prefer?"].info()


# In[85]:


df["Lastly, what was your favorite overall coffee?"].info()


# In[86]:


mode_favorite_coffee = df["Lastly, what was your favorite overall coffee?"].mode()
mode_favorite_coffee


# In[87]:


df["Lastly, what was your favorite overall coffee?"].replace(np.nan, mode_favorite_coffee[0], inplace=True)
df["Lastly, what was your favorite overall coffee?"].info()


# In[88]:


df["Between Coffee A and Coffee D, which did you prefer?"].info()


# In[89]:


mode_AD = df["Between Coffee A and Coffee D, which did you prefer?"].mode()
mode_AD


# In[90]:


df["Between Coffee A and Coffee D, which did you prefer?"].replace(np.nan, mode_AD[0], inplace=True)
df["Between Coffee A and Coffee D, which did you prefer?"].info()


# In[91]:


matching_columns = [col for col in df.columns if "coffee" in col]
print(matching_columns)


# In[92]:


df["On the go, where do you typically purchase coffee?"].info()


# In[93]:


mode_purchase = df["On the go, where do you typically purchase coffee?"].mode()
mode_purchase


# In[94]:


df["Before today's tasting, which of the following best described what kind of coffee you like?"].info()


# In[95]:


mode_coffee_like = df["Before today's tasting, which of the following best described what kind of coffee you like?"].mode()
mode_coffee_like


# In[96]:


df["Before today's tasting, which of the following best described what kind of coffee you like?"].replace(np.nan, mode_coffee_like[0], inplace=True)
df["Before today's tasting, which of the following best described what kind of coffee you like?"].info()


# In[97]:


df["How strong do you like your coffee?"].info()


# In[98]:


mode_coffee_strong = df["How strong do you like your coffee?"].mode()
mode_coffee_strong


# In[99]:


df["How strong do you like your coffee?"].replace(np.nan,mode_coffee_strong[0], inplace=True)
df["How strong do you like your coffee?"].info()


# In[100]:


df["Lastly, how would you rate your own coffee expertise?"].info()


# In[101]:


df["Lastly, how would you rate your own coffee expertise?"]


# In[102]:


df["Lastly, how would you rate your own coffee expertise?"].value_counts()


# In[103]:


mode_coffee_expertise = df["Lastly, how would you rate your own coffee expertise?"].mode()
mode_coffee_expertise


# In[104]:


df["Lastly, how would you rate your own coffee expertise?"].replace(np.nan ,mode_coffee_expertise[0] , inplace=True)
df["Lastly, how would you rate your own coffee expertise?"].info()


# In[105]:


df["Why do you drink coffee?"].info()


# In[106]:


mode_reason_to_drink_coffee = df["Why do you drink coffee?"].mode()
mode_reason_to_drink_coffee


# In[107]:


df["Why do you drink coffee?"].replace(np.nan,mode_reason_to_drink_coffee[0],inplace=True)
df["Why do you drink coffee?"].info()


# In[108]:


df["Where do you typically drink coffee? (At home)"].info()


# In[109]:


df["Where do you typically drink coffee? (At home)"].isnull().sum()


# In[110]:


df["How do you brew coffee at home? (Pour over)"].info()


# In[111]:


df["How do you brew coffee at home? (Pour over)"].isnull().sum()


# In[112]:


df["Do you usually add anything to your coffee? (No - just black)"].info()


# In[113]:


df["Do you usually add anything to your coffee? (No - just black)"].isnull().sum()


# In[114]:


df["Where do you typically drink coffee?"].info()


# In[115]:


mode_coffee_place = df["Where do you typically drink coffee?"].mode()
mode_coffee_place


# In[116]:


df["Where do you typically drink coffee?"].replace(np.nan, mode_coffee_place[0], inplace=True)
df["Where do you typically drink coffee?"].info()


# In[117]:


df["On the go, where do you typically purchase coffee?"].isnull().sum()


# In[118]:


df["On the go, where do you typically purchase coffee?"].value_counts()


# In[119]:


df["On the go, where do you typically purchase coffee?"].info()


# In[120]:


cafe_counts = df[df['Where do you typically drink coffee?'].str.contains('Cafe', case=False)].shape[0]
print(cafe_counts)


# In[121]:


matching_columns = [col for col in df.columns if "sugar" in col]
print(matching_columns)


# In[123]:


df['What kind of sugar or sweetener do you add?'].info()


# In[124]:


df['What kind of sugar or sweetener do you add?'].describe()


# In[125]:


df['What kind of sugar or sweetener do you add?'].mode()


# In[126]:


matching_columns = [col for col in df.columns if "kind" in col]
print(matching_columns)


# In[127]:


df['What kind of dairy do you add?'].mode()


# In[128]:


df['What kind of flavorings do you add?'].mode()


# In[129]:


df['What kind of flavorings do you add?'].value_counts()


# In[131]:


matching_columns = [col for col in df.columns if "flavor" in col]
print(matching_columns)


# In[133]:


matching_columns = [col for col in df.columns if "On" in col and "go" in col]
print(matching_columns)


# In[134]:


df['On the go, where do you typically purchase coffee?'].value_counts()


# In[136]:


matching_columns = [col for col in df.columns if "Coffee" in col]
print(matching_columns)


# In[137]:


df['What other flavoring do you use?'].value_counts()


# In[122]:


df.to_csv('C:\\Users\\USER\\OneDrive\\Documents\\PACIFIQUE NTETA\\Data Analytics Projects\\Maven\\cleaned_file.csv', index=False)


import pandas as pd 
import statsmodels.api as sm 
import statsmodels.graphics.api as smg 
import statsmodels.formula.api as sm_api 
import matplotlib.pyplot as plt 
import seaborn as sns 
# Open file 
df = pd.read_csv('coffee.csv') 

# Describe score columns (To find any problems or opportunities) 
score_df = df[['Data.Scores.Aroma','Data.Scores.Flavor','Data.Scores.Aftertaste', 

               'Data.Scores.Acidity','Data.Scores.Body','Data.Scores.Balance']] 
print(score_df.describe().round(2)) 
score_df_continued = df[['Data.Scores.Uniformity','Data.Scores.Sweetness', 

                         'Data.Scores.Moisture','Data.Scores.Total']] 
print(score_df_continued.describe().round(2)) 

# Create correlation matrix (Identifies best independent variables) 
corr_matrix = df.corr(numeric_only=True).round(2) 
sns.heatmap(corr_matrix, annot=True, vmax=1, vmin=-1, cmap='icefire') 

 

# Create scatter plot & boxplot (Find which countries have best/worst score) 
df.plot.scatter(x="Location.Country", y="Data.Scores.Total") 
plt.xticks(rotation=90) 
df.boxplot(column="Data.Scores.Total",by="Location.Country") 

plt.xticks(rotation=90) 

 

# Create score scatter plots (Observe relationship between dependent and independent variables) 

df.plot.scatter(y="Data.Scores.Flavor", x="Data.Scores.Total") 

df.plot.scatter(y="Data.Scores.Balance", x="Data.Scores.Total") 

df.plot.scatter(y="Data.Scores.Aftertaste", x="Data.Scores.Total") 

df.plot.scatter(y="Data.Scores.Acidity", x="Data.Scores.Total") 

plt.xticks(rotation=90) 

 

# Run OLS regression (To find statistical significance & form predictive model) 

y = df['Data.Scores.Total'] 

x = df[['Data.Scores.Flavor', 'Data.Scores.Balance', 'Data.Scores.Aftertaste', 'Data.Scores.Acidity']] 

x = sm.add_constant(x) 

model = sm.OLS(y,x).fit() 

print(model.summary()) 

 

# Show all models 

plt.show() 
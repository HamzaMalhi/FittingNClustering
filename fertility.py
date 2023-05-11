
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Read the CSV file into a pandas dataframe
df = pd.read_csv("fertility_rate.csv")


df.head()



# Select the relevant columns and time periods to analyze
years = ['1960', '1970', '1980', '1990', '2000', '2010', '2016']
df = df[['Country Name'] + years]

# Remove missing values
df.dropna(inplace=True)

# Normalize the data using StandardScaler
scaler = StandardScaler()
df_norm = scaler.fit_transform(df[years])

# Apply DBSCAN clustering to the normalized data
dbscan = DBSCAN(eps=1, min_samples=4)
clusters = dbscan.fit_predict(df_norm)

# Add the cluster membership as a new column to the dataframe
df['Cluster'] = clusters

# Plot the clusters
plt.figure(figsize=(10, 8))
plt.scatter(df_norm[:, 0], df_norm[:, 5], c=clusters, cmap='viridis')
plt.xlabel('Most recent fertility rate')
plt.ylabel('Fertility rate change since 1960')
plt.title('Fertility rate clusters')
plt.show()


from scipy.optimize import curve_fit


# Define the err_ranges function to compute confidence intervals
def err_ranges(residuals, alpha):
    n = len(residuals)
    df = n - 2  # degrees of freedom
    t = stats.t(df).ppf(1 - alpha/2)
    std_err = np.sqrt(np.sum(residuals**2) / df)
    err_lower = t * std_err * np.sqrt(1 + 1/n + (x_pred - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    err_upper = t * std_err * np.sqrt(1 + 1/n + (x_pred - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    return err_lower, err_upper


# Define the exponential function to be fitted
def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

# Generate some sample data
x_data = np.linspace(0, 10, num=50)
y_data = 2 * np.exp(0.5 * x_data) + 1 + np.random.normal(scale=0.5, size=50)

# Fit the exponential function to the data
popt, pcov = curve_fit(exp_func, x_data, y_data, p0=(1, 1, 1))

# Plot the data and the fitted function
plt.plot(x_data, y_data, 'bo', label='data')
plt.plot(x_data, exp_func(x_data, *popt), 'r-', label='fit')
plt.legend(loc='best')

plt.xlabel('Value')
plt.ylabel('Fertility Rate')
plt.title('Fit data Fertility Rate ')
plt.show()



from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Load data
data = pd.read_csv("fertility_rate.csv")

# Filter data to keep only years from 1970 to 2016
data = data.loc[:, "1970":"2016"]

# Normalise data
data_norm = (data - data.mean()) / data.std()




# Cluster data using KMeans
kmeans = KMeans(n_clusters=3)



# Add cluster labels as a new column to the original dataframe
data["Cluster"] = kmeans

# Plot cluster membership and cluster centres
fig, ax = plt.subplots()
ax.scatter(data_norm.iloc[:, 0], data_norm.iloc[:, 1], cmap="rainbow")

ax.set_xlabel("1970 Fertility Rate")
ax.set_ylabel("2016 Fertility Rate")
ax.set_title("Fertility Rate Clusters")
plt.show()





# %%
"""
# Education Search Analysis & Finding Result
"""

# %%
"""
# Course Era
"""

# %%
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('coursera_courses.csv')

# Display the first few rows of the DataFrame
print("First few rows of the dataset:")
print(df.head())

# Display information about the DataFrame
print("\nDataFrame Information:")
print(df.info())



# %%
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read the CSV file into a DataFrame
df = pd.read_csv('coursera_courses.csv')

# Initialize the label encoder
le = LabelEncoder()

# Apply label encoding to all categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Display summary statistics for all columns, including the now-encoded categorical columns
print("\nSummary Statistics for All Columns:")
print(df.describe())


# %%
# Display summary statistics for numerical and categorical columns
print("\nSummary Statistics:")
print(df.describe(include='all'))



# %%
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read the CSV file into a DataFrame
df = pd.read_csv('coursera_courses.csv')

# Initialize the label encoder
le = LabelEncoder()

# Apply label encoding to all categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Display the first few rows of the transformed DataFrame
print("Transformed Data:")
print(df.head())


# %%
# Data Processing

# 1. Handling Missing Values
# Check for missing values
print("\nMissing Values before processing:")
print(df.isnull().sum())

# Fill missing values for numerical columns with the median
df.fillna(df.median(numeric_only=True), inplace=True)

# Fill missing values for categorical columns with the mode
df.fillna(df.mode().iloc[0], inplace=True)

# Display missing values after processing
print("\nMissing Values after processing:")
print(df.isnull().sum())



# %%
# 2. Encoding Categorical Variables
# Convert categorical columns to numeric using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# 3. Removing Duplicates
# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows before processing: {duplicates}")

# Drop duplicate rows if any
df.drop_duplicates(inplace=True)

# Check for duplicates after processing
duplicates_after = df.duplicated().sum()
print(f"\nNumber of duplicate rows after processing: {duplicates_after}")

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('coursera_courses.csv')

# Plot the distribution of course ratings
plt.figure(figsize=(10, 6))
sns.histplot(df['course_rating'].dropna(), bins=20, kde=True, color='blue')
plt.title('Distribution of Course Ratings')
plt.xlabel('Course Rating')
plt.ylabel('Frequency')
plt.show()


# %%
# Plot the number of courses by certificate type
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='course_certificate_type', order=df['course_certificate_type'].value_counts().index, palette='viridis')
plt.title('Number of Courses by Certificate Type')
plt.xlabel('Certificate Type')
plt.ylabel('Number of Courses')
plt.xticks(rotation=45)
plt.show()

# %%
# Plot average rating by course difficulty
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='course_difficulty', y='course_rating', palette='Set2')
plt.title('Average Rating by Course Difficulty')
plt.xlabel('Course Difficulty')
plt.ylabel('Course Rating')
plt.show()

# %%
# Convert enrollment count from string to numeric
df['course_students_enrolled'] = df['course_students_enrolled'].str.replace(',', '').astype(float)

# Plot the distribution of student enrollments
plt.figure(figsize=(10, 6))
sns.histplot(df['course_students_enrolled'].dropna(), bins=30, kde=True, color='green')
plt.title('Distribution of Student Enrollments')
plt.xlabel('Number of Students Enrolled')
plt.ylabel('Frequency')
plt.show()

# %%
from collections import Counter
import ast

# Convert string representation of list to actual list
df['course_skills'] = df['course_skills'].apply(lambda x: ast.literal_eval(x))

# Flatten the list of skills and count occurrences
all_skills = [skill for sublist in df['course_skills'] for skill in sublist]
skill_counts = Counter(all_skills)

# Get the top 10 most common skills
top_skills = dict(skill_counts.most_common(10))

# Plot top 10 most common skills
plt.figure(figsize=(12, 8))
sns.barplot(x=list(top_skills.values()), y=list(top_skills.keys()), palette='plasma')
plt.title('Top 10 Most Common Course Skills')
plt.xlabel('Count')
plt.ylabel('Skill')
plt.show()

# %%


# Numerical Summary for Scatter Plot
print("Numerical Summary for Course Rating vs. Number of Reviews:")
print(df[['course_reviews_num', 'course_rating']].describe())

# 2. Histogram: Distribution of Course Duration
plt.figure(figsize=(8, 6))
sns.histplot(df['course_time'], bins=10, kde=True, color='teal')
plt.title('Distribution of Course Duration')
plt.xlabel('Course Duration')
plt.ylabel('Frequency')
plt.show()

# Numerical Summary for Histogram
print("\nNumerical Summary for Course Duration:")
print(df['course_time'].describe())


# %%


# Load the dataset
df = pd.read_csv('coursera_courses.csv')

# Count the number of courses offered by each organization
courses_by_organization = df['course_organization'].value_counts().head(10)

# Plot the horizontal bar chart
plt.figure(figsize=(10, 6))
courses_by_organization.plot(kind='barh', color='skyblue')
plt.title('Top 10 Organizations by Number of Courses Offered')
plt.xlabel('Number of Courses')
plt.ylabel('Organization')
plt.show()



# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Read the CSV file into a DataFrame
df = pd.read_csv('coursera_courses.csv')

# Data Cleaning
df['course_rating'].fillna(df['course_rating'].mean(), inplace=True)
df['course_reviews_num'] = pd.to_numeric(df['course_reviews_num'].str.replace(',', ''), errors='coerce')
df['course_students_enrolled'] = pd.to_numeric(df['course_students_enrolled'].str.replace(',', ''), errors='coerce')

def convert_course_time_to_months(time_str):
    if pd.isnull(time_str):
        return np.nan
    if "Weeks" in time_str:
        weeks = int(time_str.split()[0])
        return weeks / 4
    elif "Months" in time_str:
        months = int(time_str.split()[0])
        return months
    elif "Year" in time_str:
        years = int(time_str.split()[0])
        return years * 12
    return np.nan

df['course_time'] = df['course_time'].apply(convert_course_time_to_months)
df['course_time'].fillna(df['course_time'].median(), inplace=True)

# Feature Engineering - Adding interaction terms
df['reviews_to_students'] = df['course_reviews_num'] / df['course_students_enrolled']
df['rating_to_reviews'] = df['course_rating'] / df['course_reviews_num']

X = df[['course_organization', 'course_time', 'course_reviews_num', 'course_students_enrolled', 'reviews_to_students', 'rating_to_reviews']]
y = df['course_rating']

# Define Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), ['course_organization']),
        ('num', Pipeline(steps=[
            ('scaler', StandardScaler())
        ]), ['course_time', 'course_reviews_num', 'course_students_enrolled', 'reviews_to_students', 'rating_to_reviews'])
    ]
)

# Define the pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'model__n_estimators': [100, 200, 300, 500],
    'model__max_depth': [None, 10, 20, 30, 50],
    'model__min_samples_split': [2, 5, 10, 15],
    'model__min_samples_leaf': [1, 2, 4, 6],
    'model__max_features': ['auto', 'sqrt', 'log2']  # Include max_features tuning
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train using Grid Search
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)

# Make predictions
y_pred = grid_search.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Regression plot (Actual vs Predicted)
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual Course Rating')
plt.ylabel('Predicted Course Rating')
plt.title('Actual vs Predicted Course Rating')
plt.legend()
plt.grid(True)
plt.show()

# Residuals plot
residuals = y_test - y_pred
plt.figure(figsize=(12, 6))
sns.residplot(x=y_pred, y=residuals, lowess=True, color='blue', line_kws={'color': 'red', 'lw': 2})
plt.xlabel('Predicted Course Rating')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.grid(True)
plt.show()


# %%
# Residual Plot
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
sns.scatterplot(x=y_pred, y=residuals, color='blue', alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Ratings')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid()
plt.show()


# %%
# Histogram of Residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=True, color='blue')
plt.xlabel('Residuals')
plt.title('Histogram of Residuals')
plt.grid()
plt.show()


# %%
# Box Plot of Actual vs Predicted Ratings
plt.figure(figsize=(10, 6))
sns.boxplot(data=[y_test, y_pred], palette='Set2')
plt.xticks(ticks=[0, 1], labels=['Actual Ratings', 'Predicted Ratings'])
plt.ylabel('Ratings')
plt.title('Box Plot of Actual vs Predicted Ratings')
plt.grid()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import ast

# Load the dataset
df = pd.read_csv('coursera_courses.csv')

# Data Preprocessing
# Drop rows with missing values for simplicity (you can handle this differently if needed)
df = df.dropna(subset=['course_rating', 'course_students_enrolled'])

# Convert course_students_enrolled from string to numeric
df['course_students_enrolled'] = df['course_students_enrolled'].str.replace(',', '').astype(float)

# We can use course_rating and course_students_enrolled for clustering
data_for_clustering = df[['course_rating', 'course_students_enrolled']]

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_for_clustering)

# Find the optimal number of clusters using the Elbow method
inertia = []
silhouette_scores = []

for n in range(2, 11):
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(data_scaled, kmeans.labels_))

# Plot the Elbow curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

# Plot the Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

# From the plots, choose an optimal number of clusters
optimal_clusters = 4  # Change this based on your plots

# Apply K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(data_scaled)

# Visualizing the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='course_rating', y='course_students_enrolled', hue='cluster', palette='viridis', alpha=0.7)
plt.title('K-means Clustering of Coursera Courses')
plt.xlabel('Course Rating')
plt.ylabel('Number of Students Enrolled')
plt.legend(title='Cluster')
plt.grid()
plt.show()

# Display cluster centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids, columns=['course_rating', 'course_students_enrolled'])
print("\nCluster Centroids:")
print(centroids_df)


# %%
"""
# Youtube Khan Academy
"""

# %%
import pandas as pd

# Load the dataset
df = pd.read_csv('youtube_khan_academy.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())



# %%
# Display data types of each column
print("\nData types of each column:")
print(df.dtypes)



# %%
# Display summary statistics for numerical columns
print("\nSummary statistics for numerical columns:")
print(df.describe(include=[float, int]))

# Display summary statistics for categorical columns
print("\nSummary statistics for categorical columns:")
print(df.describe(include=[object]))



# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('youtube_khan_academy.csv')

# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

# 1. Bar Graph of Video Views by Publish Year
views_per_year = df.groupby('publish_year')['view_count'].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='publish_year', y='view_count', data=views_per_year, palette='viridis')
plt.title('Total Views per Year')
plt.xlabel('Publish Year')
plt.ylabel('Total Views')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Scatter Plot of Likes vs. Views
plt.figure(figsize=(10, 6))
sns.scatterplot(x='view_count', y='like_count', data=df, alpha=0.7)
plt.title('Likes vs. Views')
plt.xlabel('View Count')
plt.ylabel('Like Count')
plt.xscale('log')  # Log scale for better visibility
plt.yscale('log')
plt.tight_layout()
plt.show()


# 4. Histogram of Title Word Count
plt.figure(figsize=(10, 6))
sns.histplot(df['title_word_count'], bins=20, kde=True, color='purple')
plt.title('Distribution of Title Word Count')
plt.xlabel('Number of Words in Title')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('youtube_khan_academy.csv')

# Convert 'published_at' to datetime format
df['published_at'] = pd.to_datetime(df['published_at'])

# Group by month and sum the view counts
views_over_time = df.resample('M', on='published_at')['view_count'].sum().reset_index()

# Create a rolling average for smoothing
views_over_time['rolling_average'] = views_over_time['view_count'].rolling(window=3).mean()

# Plotting
plt.figure(figsize=(12, 6))
sns.lineplot(data=views_over_time, x='published_at', y='rolling_average', marker='o', color='blue', label='Rolling Average (3 months)')
sns.lineplot(data=views_over_time, x='published_at', y='view_count', color='orange', alpha=0.5, label='Total Views')

# Titles and labels
plt.title('Total Views Over Time (Monthly)')
plt.xlabel('Publish Month')
plt.ylabel('Total Views')
plt.xticks(rotation=45)

# Add a legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
file_path = 'khan.csv'
data = pd.read_csv(file_path)

# Count the number of occurrences for each subject
subject_counts = data['subject'].value_counts()

# Plot the data
plt.figure(figsize=(10, 8))
sns.barplot(x=subject_counts.values, y=subject_counts.index, palette='viridis')
plt.title('Top Subjects by Count')
plt.xlabel('Count')
plt.ylabel('Subject')
plt.show()

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('youtube_khan_academy.csv')


# Data Preprocessing
# Handle missing values (example: fill with mean or drop)
df.fillna(0, inplace=True)

# Feature selection
# Select features and target variable
X = df[['title_word_count', 'like_count', 'dislike_count', 'reaction_total', 'date_diff']]
y = df['view_count']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of seaborn
sns.set(style="whitegrid")

# Create a DataFrame for actual vs predicted
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Plotting
plt.figure(figsize=(14, 7))
sns.scatterplot(data=results_df, x='Actual', y='Predicted', alpha=0.6)
plt.plot([results_df['Actual'].min(), results_df['Actual'].max()],
         [results_df['Actual'].min(), results_df['Actual'].max()],
         color='red', linestyle='--')
plt.title('Actual vs Predicted Views', fontsize=16)
plt.xlabel('Actual Views', fontsize=14)
plt.ylabel('Predicted Views', fontsize=14)
plt.xlim([0, results_df['Actual'].max() * 1.1])
plt.ylim([0, results_df['Predicted'].max() * 1.1])
plt.grid()
plt.show()


# %%
# 1. Scatter Plot of Actual vs. Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Views')
plt.ylabel('Predicted Views')
plt.grid()
plt.show()

# 2. Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='orange')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Views')
plt.ylabel('Residuals')
plt.grid()
plt.show()

# 3. Distribution of Errors
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=True, color='purple')
plt.title('Distribution of Errors (Residuals)')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('youtube_khan_academy.csv')

# Data Preprocessing
df.fillna(0, inplace=True)

# Feature selection (you can select the features relevant for clustering)
X = df[['title_word_count', 'like_count', 'dislike_count', 'reaction_total', 'date_diff']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the elbow method
inertia = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid()
plt.xticks(k_values)
plt.show()

# Fit the K-Means model with the chosen number of clusters (e.g., 3)
optimal_k = 3  # You can change this based on the elbow method
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original DataFrame
df['Cluster'] = clusters

# Visualize the clusters (using only two features for 2D visualization)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='like_count', y='view_count', hue='Cluster', palette='viridis', s=100)
plt.title('Clustering Results')
plt.xlabel('Like Count')
plt.ylabel('View Count')
plt.legend(title='Cluster')
plt.grid()
plt.show()


# %%
"""
# Udemy Academy 
"""

# %%
import pandas as pd

# Load the dataset
df_udemy = pd.read_csv('udemy_courses.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df_udemy.head())

# Display the number of columns and their data types
print("\nNumber of columns and their data types:")
print(df_udemy.info())


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df_udemy = pd.read_csv('udemy_courses.csv')

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# 1. Distribution of Course Prices
plt.figure(figsize=(10, 6))
sns.histplot(df_udemy['price'], bins=30, kde=True, color='blue')
plt.title('Distribution of Course Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# 2. Number of Subscribers by Course Level
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_udemy, x='level', y='num_subscribers', palette='Set2')
plt.title('Number of Subscribers by Course Level')
plt.xlabel('Course Level')
plt.ylabel('Number of Subscribers')
plt.grid()
plt.show()

# 3. Average Course Duration by Subject
average_duration = df_udemy.groupby('subject')['content_duration'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(data=average_duration, x='subject', y='content_duration', palette='pastel')
plt.title('Average Course Duration by Subject')
plt.xlabel('Subject')
plt.ylabel('Average Content Duration (Hours)')
plt.xticks(rotation=45)
plt.grid()
plt.show()

# 4. Number of Courses per Level
plt.figure(figsize=(8, 5))
sns.countplot(data=df_udemy, x='level', palette='husl')
plt.title('Number of Courses per Level')
plt.xlabel('Course Level')
plt.ylabel('Number of Courses')
plt.grid()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df_udemy = pd.read_csv('udemy_courses.csv')

# Calculate average content duration by subject
avg_duration_by_subject = df_udemy.groupby('subject')['content_duration'].mean()

# Plot average content duration by subject
plt.figure(figsize=(12, 7))
avg_duration_by_subject.plot(kind='bar', color='green', edgecolor='black')
plt.xlabel('Subject')
plt.ylabel('Average Content Duration (hours)')
plt.title('Average Content Duration by Subject')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df_udemy = pd.read_csv('udemy_courses.csv')

# Convert 'published_timestamp' to datetime
df_udemy['published_timestamp'] = pd.to_datetime(df_udemy['published_timestamp'], errors='coerce')

# Drop rows with NaT values in 'published_timestamp'
df_udemy = df_udemy.dropna(subset=['published_timestamp'])

# Extract year and month for plotting
df_udemy['year_month'] = df_udemy['published_timestamp'].dt.to_period('M')

# Count number of courses published per month
monthly_courses = df_udemy['year_month'].value_counts().sort_index()

# Print the numerical result
print("Number of Courses Published Per Month:")
print(monthly_courses)

# Plot number of courses published over time
plt.figure(figsize=(14, 7))
plt.plot(monthly_courses.index.astype(str), monthly_courses.values, marker='o', color='royalblue', linestyle='-', linewidth=2, markersize=8)

# Adding titles and labels
plt.title('Number of Courses Published Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Year-Month', fontsize=14)
plt.ylabel('Number of Courses Published', fontsize=14)

# Enhancing x-ticks for better readability
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# Adding grid for better readability
plt.grid(color='lightgray', linestyle='--', linewidth=0.7)

# Adding a background color for better contrast
plt.gca().set_facecolor('whitesmoke')

# Adding annotations for significant data points (optional)
for i, value in enumerate(monthly_courses.values):
    plt.text(i, value + 2, str(value), ha='center', fontsize=9, color='black')

plt.tight_layout()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df_udemy = pd.read_csv('udemy_courses.csv')

# Define price ranges
bins = [0, 20, 50, 100, 200, float('inf')]
labels = ['0-20', '21-50', '51-100', '101-200', '200+']
df_udemy['price_range'] = pd.cut(df_udemy['price'], bins=bins, labels=labels)

# Count the number of courses in each price range
price_range_counts = df_udemy['price_range'].value_counts()

# Print the numerical results
print("Number of Courses by Price Range:")
print(price_range_counts)

# Plot pie chart for price distributionac
plt.figure(figsize=(8, 8))
plt.pie(price_range_counts, labels=price_range_counts.index, autopct='%1.1f%%', colors=plt.cm.Paired(range(len(price_range_counts))))
plt.title('Distribution of Courses by Price Range')
plt.show()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df_udemy = pd.read_csv('udemy_courses.csv')

# Preprocessing
# Drop rows with missing values in 'price', 'num_reviews', or 'num_subscribers'
df_udemy = df_udemy.dropna(subset=['price', 'num_reviews', 'num_subscribers'])

# Features and target variable
X = df_udemy[['price', 'num_reviews']]
y = df_udemy['num_subscribers']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Model Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Coefficients
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Price Coefficient: {model.coef_[0]:.2f}")
print(f"Number of Reviews Coefficient: {model.coef_[1]:.2f}")

# Plotting
plt.figure(figsize=(12, 6))

# Plot Actual vs Predicted
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual Number of Subscribers')
plt.ylabel('Predicted Number of Subscribers')
plt.title('Actual vs. Predicted Number of Subscribers')
plt.legend()
plt.grid(True)
plt.show()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the dataset
df_udemy = pd.read_csv('udemy_courses.csv')

# Preprocessing
df_udemy = df_udemy.dropna(subset=['price', 'num_reviews', 'num_subscribers'])

# Features and target variable
X = df_udemy[['price', 'num_reviews']]
y = df_udemy['num_subscribers']

# Polynomial Features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Model Building with Ridge Regression
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Create residuals
residuals = y_test - y_pred

# Residual Plot
plt.figure(figsize=(12, 6))
plt.scatter(y_pred, residuals, color='blue', alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)
plt.show()

# Feature Importance Plot
plt.figure(figsize=(12, 6))
features = poly.get_feature_names_out(input_features=['price', 'num_reviews'])
coefficients = model.coef_
plt.barh(features, coefficients, color='orange')
plt.xlabel('Coefficient Value')
plt.title('Feature Importance')
plt.grid(True)
plt.show()

# Distribution of Actual vs Predicted
plt.figure(figsize=(12, 6))
plt.hist(y_test, bins=30, alpha=0.5, label='Actual', color='blue', edgecolor='black')
plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted', color='orange', edgecolor='black')
plt.xlabel('Number of Subscribers')
plt.ylabel('Frequency')
plt.title('Distribution of Actual vs Predicted Number of Subscribers')
plt.legend()
plt.grid(True)
plt.show()

# Pairplot of relevant features
sns.pairplot(df_udemy[['price', 'num_reviews', 'num_subscribers']])
plt.title('Pairplot of Features')
plt.show()

# Learning Curve (Optional)
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(model, X_poly, y, n_jobs=-1, 
                                                        train_sizes=np.linspace(0.1, 1.0, 10), 
                                                        scoring='r2')

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(12, 6))
plt.plot(train_sizes, train_scores_mean, label='Training Score', color='blue')
plt.plot(train_sizes, test_scores_mean, label='Test Score', color='orange')
plt.title('Learning Curve')
plt.xlabel('Training Size')
plt.ylabel('R-squared Score')
plt.legend()
plt.grid(True)
plt.show()


# %%
!jupyter nbconvert --to script Education_Analysis.ipynb


# %%
!jupyter nbconvert --to python Education_Analysis.ipynb


# %%
!ipynb-py-convert Complete Code Education Analysis.ipynb my_script.py


# %%

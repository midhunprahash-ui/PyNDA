import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv('PyNDA_dataset.csv')

# 1. Load and Initial Cleaning
# Drop rows where 'Text_Response' is missing or empty
df.dropna(subset=['Text_Response'], inplace=True)
df = df[df['Text_Response'].astype(str).str.strip() != '']

# Handle missing values in demographic columns by filling with 'Unknown'
demographic_cols = ['Age', 'Gender', 'Race', 'Income', 'Education']
for col in demographic_cols:
    df[col] = df[col].fillna('Unknown')

# 2. Text Normalization
df['Text_Response'] = df['Text_Response'].astype(str).str.lower() # Convert to string before applying string methods
df['Text_Response'] = df['Text_Response'].str.replace(r'[^\w\s]', '', regex=True)
df['Text_Response'] = df['Text_Response'].str.strip()

# 3. Psychometric Label Binarization (Critical Method)
psychometric_scores = {
    'HealthLiteracy_Score': 'HL_Target',
    'HealthNumeracy_Score': 'HN_Target',
    'TrustInDoctors_Score': 'TD_Target',
    'AnxietyVisiting_Score': 'AV_Target',
    'DrugExperience_Rating': 'DE_Target'
} 

processed_dfs = {}

for score_col, target_col in psychometric_scores.items():
    # Make a copy to avoid SettingWithCopyWarning
    temp_df = df.copy()

    # Calculate Q1 and Q3
    Q1 = temp_df[score_col].quantile(0.25)
    Q3 = temp_df[score_col].quantile(0.75)

    # Create the binary target column
    temp_df[target_col] = -1 # Initialize with a placeholder value

    # Assign 0 for 'Low' and 1 for 'High'
    temp_df.loc[temp_df[score_col] <= Q1, target_col] = 0
    temp_df.loc[temp_df[score_col] >= Q3, target_col] = 1

    # Discard instances where score falls strictly between Q1 and Q3
    temp_df = temp_df[temp_df[target_col] != -1]

    # Drop the original score column as it's no longer needed for the binarized task
    temp_df = temp_df.drop(columns=[score_col])

    # 4. Demographic Feature Encoding
    categorical_cols = ['Gender', 'Race', 'Income', 'Education']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(temp_df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols), index=temp_df.index)

    # Concatenate encoded features with the rest of the DataFrame and drop original categorical columns
    temp_df = pd.concat([temp_df.drop(columns=categorical_cols), encoded_df], axis=1)

    processed_dfs[score_col.replace('_Score', '').lower()] = temp_df

# Example: Print .info() and .head() of one of the resulting DataFrames (e.g., for Health Literacy)
df_health_literacy = processed_dfs['healthliteracy']
print("Info for df_health_literacy:")
df_health_literacy.info()
print("\nHead for df_health_literacy:")
df_health_literacy.head()

# You can access the five distinct DataFrames like this:
df_health_literacy = processed_dfs['healthliteracy']
df_health_numeracy = processed_dfs['healthnumeracy']
df_trust_in_doctors = processed_dfs['trustindoctors']
df_anxiety_visiting = processed_dfs['anxietyvisiting']
df_drug_experience = processed_dfs['drugexperience']
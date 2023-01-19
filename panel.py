import pandas as pd
import statsmodels.api as sm

# Load the data
df = pd.read_csv("data.csv")

# Prepare the data
df = df.set_index(["unit", "time"])
df = df.sort_index()

# Estimate the fixed effects model
fe_model = sm.OLS(df["dependent"], df[["exogenous"]])
fe_results = fe_model.fit(cov_type="clustered", cluster_entity=True)

# Print the results
print(fe_results.summary())
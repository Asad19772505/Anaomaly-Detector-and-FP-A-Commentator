import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
import os
from sklearn.ensemble import IsolationForest

# Load the dataset
file_path = "/content/synthetic_fpna_data_with_anomalies.xlsx" #change if needed and if you are using a different path or file
data = pd.read_excel(file_path)

# Convert the Date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Plot the revenue data
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Revenue', data=data, marker='o')
plt.title('Revenue Over Time')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.grid()
plt.show()

# Anomaly Detection using Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)
data['Anomaly'] = model.fit_predict(data[['Revenue']])
data['Anomaly'] = data['Anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

# Separate anomalies for easier analysis
anomalies = data[data['Anomaly'] == 'Anomaly']

# Plot anomalies
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Revenue', data=data, label='Revenue')
sns.scatterplot(
    x='Date',
    y='Revenue',
    data=anomalies,
    color='red',
    label='Anomaly',
    s=100,
    marker='o'
)
plt.title('Revenue with Anomalies Highlighted')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.grid()
plt.show()

# Prepare summary for AI Agent
anomaly_summary = f"""
Anomalies Detected in Revenue Data:
- Total Records: {len(data)}
- Anomalies Detected: {len(anomalies)}

Details of Anomalies:
{anomalies[['Date', 'Revenue']].to_string(index=False)}
"""

# AI Integration
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

fpna_chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": (
                "You are an FP&A analyst skilled in anomaly detection and financial commentary. Analyze the anomalies "
                "detected in the revenue data and provide detailed commentary. Include potential reasons for anomalies, "
                "the impact on financial performance, and actionable recommendations."
            ),
        },
        {
            "role": "user",
            "content": (
                f"The revenue data contains anomalies as described below:\n{anomaly_summary}\n"
                "Please provide your FP&A analysis and recommendations."
            ),
        }
    ],
    model="llama3-8b-8192",
)

# Print AI-generated commentary
print(fpna_chat_completion.choices[0].message.content)

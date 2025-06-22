import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# Set page configuration
st.set_page_config(
    page_title="Heart Attack Risk Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("Heart Attack risk.csv")
    
    # Clean data - extract systolic and diastolic blood pressure
    data[['Systolic BP', 'Diastolic BP']] = data['Blood Pressure'].str.split('/', expand=True).astype(int)
    
    # Categorize age groups
    bins = [0, 30, 45, 60, 75, 90, 120]
    labels = ['<30', '30-45', '45-60', '60-75', '75-90', '90+']
    data['Age Group'] = pd.cut(data['Age'], bins=bins, labels=labels).astype(str)
    
    return data

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")
selected_continents = st.sidebar.multiselect(
    "Select Continents", 
    df['Continent'].unique(), 
    df['Continent'].unique()
)
selected_age_groups = st.sidebar.multiselect(
    "Select Age Groups", 
    sorted(df['Age Group'].unique()), 
    sorted(df['Age Group'].unique())
)
selected_risk = st.sidebar.selectbox(
    "Risk Status", 
    ['All', 'At Risk', 'Not At Risk']
)

# Apply filters
filtered_df = df[
    (df['Continent'].isin(selected_continents)) &
    (df['Age Group'].isin(selected_age_groups))
]

if selected_risk == 'At Risk':
    filtered_df = filtered_df[filtered_df['Heart Attack Risk'] == 1]
elif selected_risk == 'Not At Risk':
    filtered_df = filtered_df[filtered_df['Heart Attack Risk'] == 0]

# Key Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)
total_patients = len(filtered_df)
at_risk_patients = filtered_df['Heart Attack Risk'].sum()
risk_percentage = (at_risk_patients / total_patients) * 100 if total_patients > 0 else 0

col1.metric("Total Patients", f"{total_patients:,}")
col2.metric("At Risk", f"{at_risk_patients:,}")
col3.metric("Risk Rate", f"{risk_percentage:.1f}%")
col4.metric("Avg Age", f"{filtered_df['Age'].mean():.0f}")
col5.metric("Avg Cholesterol", f"{filtered_df['Cholesterol'].mean():.0f}")

# Main Dashboard Row
col1, col2, col3 = st.columns([1, 1, 1])

# Risk by Continent
with col1:
    continent_risk = filtered_df.groupby('Continent')['Heart Attack Risk'].mean().sort_values(ascending=True)
    fig = px.bar(continent_risk, 
                 x=continent_risk.values, 
                 y=continent_risk.index,
                 orientation='h',
                 color=continent_risk.values,
                 color_continuous_scale=[[0, "#FFFFFF"], [0.2, "#FFE5E5"], [0.4, "#FFCCCC"], [0.6, "#FF9999"], [0.8, "#FF6666"], [1.0, "#CC0000"]])
    fig.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=30, b=0))
    fig.update_xaxes(title_text="Risk Rate")
    fig.update_yaxes(title_text="Continent")
    st.plotly_chart(fig, use_container_width=True)

# Age Risk Bar Chart
with col2:
    age_risk = filtered_df.groupby('Age Group')['Heart Attack Risk'].mean().reset_index()
    fig = px.bar(age_risk, 
                 x='Age Group', 
                 y='Heart Attack Risk',
                 color='Heart Attack Risk',
                 color_continuous_scale=[[0, "#FFFFFF"], [0.2, "#FFE5E5"], [0.4, "#FFCCCC"], [0.6, "#FF9999"], [0.8, "#FF6666"], [1.0, "#CC0000"]])
    fig.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=30, b=0))
    fig.update_xaxes(title_text="Age Group")
    fig.update_yaxes(title_text="Risk Rate")
    st.plotly_chart(fig, use_container_width=True)

# Lifestyle Factors Line Chart
with col3:
    lifestyle_factors = ['Smoking', 'Obesity', 'Diabetes', 'Alcohol Consumption']
    at_risk_lifestyle = filtered_df[filtered_df['Heart Attack Risk'] == 1][lifestyle_factors].mean()
    not_at_risk_lifestyle = filtered_df[filtered_df['Heart Attack Risk'] == 0][lifestyle_factors].mean()
    
    # Create DataFrame for line chart
    lifestyle_df = pd.DataFrame({
        'Factor': lifestyle_factors,
        'At Risk': at_risk_lifestyle.values,
        'Not at Risk': not_at_risk_lifestyle.values
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lifestyle_df['Factor'], y=lifestyle_df['At Risk'], 
                            mode='lines+markers', name='At Risk', 
                            line=dict(color='#CC0000', width=3), marker=dict(size=8, color='#CC0000')))
    fig.add_trace(go.Scatter(x=lifestyle_df['Factor'], y=lifestyle_df['Not at Risk'], 
                            mode='lines+markers', name='Not at Risk', 
                            line=dict(color='#FF9999', width=3), marker=dict(size=8, color='#FF9999')))
    
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
    fig.update_xaxes(title_text="Lifestyle Factors")
    fig.update_yaxes(title_text="Prevalence Rate")
    st.plotly_chart(fig, use_container_width=True)

# Bottom Row - Risk Factor Analysis
col1, col2 = st.columns([1, 1])

# Geographic Risk Map
with col1:
    geo_df = filtered_df.groupby(['Country', 'Continent']).agg({
        'Heart Attack Risk': 'mean',
        'BMI': 'mean',
        'Systolic BP': 'mean',
        'Diastolic BP': 'mean'
    }).reset_index()
    
    fig = px.choropleth(geo_df, 
                        locations="Country", 
                        locationmode='country names',
                        color="Heart Attack Risk",
                        hover_name="Country",
                        hover_data={
                            "BMI": ":.1f",
                            "Systolic BP": ":.0f", 
                            "Diastolic BP": ":.0f",
                            "Heart Attack Risk": ":.2f"
                        },
                        color_continuous_scale=[[0, "#FFFFFF"], [0.2, "#FFE5E5"], [0.4, "#FFCCCC"], [0.6, "#FF9999"], [0.8, "#FF6666"], [1.0, "#CC0000"]],
                        range_color=[0, geo_df['Heart Attack Risk'].max()])
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
    fig.update_geos(showframe=False, showcoastlines=True)
    st.plotly_chart(fig, use_container_width=True)

# Top Correlations
with col2:
    numeric_cols = filtered_df.select_dtypes(include=np.number).columns
    corr_matrix = filtered_df[numeric_cols].corr()['Heart Attack Risk'].sort_values(ascending=False)
    corr_matrix = corr_matrix.drop('Heart Attack Risk').head(8)
    
    fig = px.bar(corr_matrix, 
                 x=corr_matrix.values, 
                 y=corr_matrix.index,
                 orientation='h',
                 color=corr_matrix.values,
                 color_continuous_scale=[[0, "#FFFFFF"], [0.2, "#FFE5E5"], [0.4, "#FFCCCC"], [0.6, "#FF9999"], [0.8, "#FF6666"], [1.0, "#CC0000"]])
    fig.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=30, b=0))
    fig.update_xaxes(title_text="Correlation Coefficient")
    fig.update_yaxes(title_text="Risk Factors")
    st.plotly_chart(fig, use_container_width=True)
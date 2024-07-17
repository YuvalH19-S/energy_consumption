import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('World Energy Consumption_new.csv')
    return df

# Introduction and Title Section
st.set_page_config(layout="wide")
df = load_data()
st.markdown("<h1 style='color: green; font-weight: bold; text-align: center;'>Green Energy Impact on World Energy Consumption</h1>", unsafe_allow_html=True)
st.markdown("""
    <h2 style='text-align: center; color: darkslategray;'>Introduction</h2>
    <p style='text-align: center;'>Welcome to the Green Energy Impact Dashboard. This application provides visual insights into the world energy consumption patterns, 
    focusing on different energy types and their impact on various metrics like GDP, population, and electricity consumption per capita. 
    Navigate through the different sections to explore energy consumption maps, compare countries, and analyze trends over time.</p>
""", unsafe_allow_html=True)

# Divider
st.markdown("<hr style='border-top: 3px solid #bbb;'>", unsafe_allow_html=True)

# Visualization 1: Energy Consumption Map and Top 5 Countries
st.markdown("<h2 style='text-align: center; color: teal;'>World Energy Consumption Map </h2>", unsafe_allow_html=True)

def visualize_energy_consumption_map(df):
    energy_types = ['biofuel', 'coal', 'fossil', 'gas', 'hydro', 'nuclear', 'oil', 'renewables', 'solar', 'wind']
    
    # Filter years where relevant data is not null and within the range 2000-2022
    filtered_years = df[(df['year'] >= 2000) & (df['year'] <= 2022)].dropna(subset=[f'{etype}_consumption' for etype in energy_types] + [f'{etype}_elec_per_capita' for etype in energy_types])['year'].unique()
    filtered_years = sorted(filtered_years)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        st.subheader("Filter for Map Visualization")
        year = st.selectbox('Select Year', filtered_years, index=filtered_years.index(2022))
        energy_type = st.selectbox('Select Energy Type', energy_types, index=energy_types.index('coal'))

    with col2:
        df_filtered = df[df['year'] == year]
        continent_data = df_filtered.groupby(['continent', 'country', 'iso_code'], as_index=False).agg({
            f'{energy_type}_consumption': 'sum',
            f'{energy_type}_elec_per_capita': 'mean'
        })
        
        global_mean = df_filtered[f'{energy_type}_consumption'].mean()
        
        fig = px.choropleth(continent_data,
                            locations="iso_code",
                            color=f"{energy_type}_consumption",
                            hover_name="country",
                            hover_data={
                                "country": True,
                                f"{energy_type}_consumption": True,
                                f"{energy_type}_elec_per_capita": True
                            },
                            color_continuous_scale=px.colors.diverging.Geyser,
                            projection="natural earth")
        fig.update_geos(showcountries=True, countrycolor="Black")
        fig.update_layout(title=f'Total {energy_type.capitalize()} Consumption by Country', margin={"r":0,"t":0,"l":0,"b":0})

        top_5_countries = continent_data.nlargest(5, f'{energy_type}_consumption')
        bar_fig = px.bar(top_5_countries, x='country', y=f'{energy_type}_consumption',
                         hover_data=[f'{energy_type}_elec_per_capita'],
                         labels={f'{energy_type}_consumption': f'{energy_type.capitalize()} Consumption (TWh)'},
                         title=f'Top 5 Countries by {energy_type.capitalize()} Consumption')
        bar_fig.add_hline(y=global_mean, line_dash="dot",
                          annotation_text="Global Mean", annotation_position="top right")
        bar_fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0}, yaxis_title=None, plot_bgcolor='rgba(0,0,0,0)')
        bar_fig.update_traces(marker_color='rgb(26, 118, 255)', marker_line_color='rgb(8, 48, 107)', marker_line_width=1.5, opacity=0.6)

        sub_col1, sub_col2 = st.columns((3, 1))
        with sub_col1:
            st.plotly_chart(fig, use_container_width=True)
        with sub_col2:
            st.plotly_chart(bar_fig, use_container_width=True)

visualize_energy_consumption_map(df)

# Styled Divider
st.markdown("<hr style='border-top: 3px solid #bbb;'>", unsafe_allow_html=True)

# Visualization 2: Comparison of GDP, Population, and Energy Consumption per Capita
st.markdown("<h2 style='text-align: center; color: teal;'>Comparison of GDP, Population, and Energy Consumption per Capita</h2>", unsafe_allow_html=True)

def visualize_gdp_population_energy(df):
    energy_types_green = ['renewables', 'solar', 'wind', 'hydro']
    energy_types_other = ['coal', 'gas', 'nuclear', 'oil']
    
    df_filtered_bubble = df.dropna(subset=[f'{etype}_elec_per_capita' for etype in energy_types_green + energy_types_other])
    selected_energy_type = st.selectbox('Select Energy Type for Bubble Chart', energy_types_green + energy_types_other)
    
    bubble_data = df_filtered_bubble.groupby('country', as_index=False).agg({
        'gdp': 'mean',
        'population': 'mean',
        f'{selected_energy_type}_elec_per_capita': 'mean',
        'continent': 'first'
    })
    bubble_data = bubble_data.dropna()
    bubble_data = bubble_data[bubble_data['country'] != 'World']

    # Define population bins
    small_population = bubble_data[bubble_data['population'] < 1e7]
    medium_population = bubble_data[(bubble_data['population'] >= 1e7) & (bubble_data['population'] < 1e8)]
    large_population = bubble_data[bubble_data['population'] >= 1e8]

    # Dropdown for selecting population size category
    population_size_category = st.selectbox(
        'Select Population Size Category',
        ['Small (0-10M)', 'Medium (10M-100M)', 'Large (100M+)'],
        index=2
    )

    if population_size_category == 'Small (0-10M)':
        st.subheader("Small Population Countries")
        bubble_fig = px.scatter(small_population, x='gdp', y=f'{selected_energy_type}_elec_per_capita',
                                size='population', color='continent', hover_name='country',
                                size_max=60,
                                labels={'gdp': 'GDP (International $)',
                                        f'{selected_energy_type}_elec_per_capita': f'{selected_energy_type.capitalize()} Electricity per Capita (kWh)',
                                        'population': 'Population'},
                                title='Comparison of GDP, Population, and Energy Consumption per Capita for Small Population Countries')
        st.plotly_chart(bubble_fig)

    elif population_size_category == 'Medium (10M-100M)':
        st.subheader("Medium Population Countries")
        bubble_fig = px.scatter(medium_population, x='gdp', y=f'{selected_energy_type}_elec_per_capita',
                                size='population', color='continent', hover_name='country',
                                size_max=60,
                                labels={'gdp': 'GDP (International $)',
                                        f'{selected_energy_type}_elec_per_capita': f'{selected_energy_type.capitalize()} Electricity per Capita (kWh)',
                                        'population': 'Population'},
                                title='Comparison of GDP, Population, and Energy Consumption per Capita for Medium Population Countries')
        st.plotly_chart(bubble_fig)

    else:
        st.subheader("Large Population Countries")
        bubble_fig = px.scatter(large_population, x='gdp', y=f'{selected_energy_type}_elec_per_capita',
                                size='population', color='continent', hover_name='country',
                                size_max=60,
                                labels={'gdp': 'GDP (International $)',
                                        f'{selected_energy_type}_elec_per_capita': f'{selected_energy_type.capitalize()} Electricity per Capita (kWh)',
                                        'population': 'Population'},
                                title='Comparison of GDP, Population, and Energy Consumption per Capita for Large Population Countries')
        st.plotly_chart(bubble_fig)

visualize_gdp_population_energy(df)

# Styled Divider
st.markdown("<hr style='border-top: 3px solid #bbb;'>", unsafe_allow_html=True)

# Visualization 3: Trend Analysis
st.markdown("<h2 style='text-align: center; color: teal;'>Trend Analysis of Green and Other Energy Shares</h2>", unsafe_allow_html=True)

def visualize_trend_analysis(df):
    energy_types_green = ['renewables', 'solar', 'wind', 'hydro']
    energy_types_other = ['coal', 'gas', 'nuclear', 'oil']

    df = df[(df['year'] >= 2000) & (df['year'] <= 2023)]
    continent_options = df['continent'].dropna().unique()
    selected_continent = st.selectbox('Select Continent', continent_options)

    filtered_df = df[df['country'] == selected_continent]

    filtered_df['green_total'] = filtered_df[[f'{etype}_share_elec' for etype in energy_types_green]].sum(axis=1)
    filtered_df['other_total'] = filtered_df[[f'{etype}_share_elec' for etype in energy_types_other]].sum(axis=1)

    mean_green_total = filtered_df['green_total'].mean()
    mean_other_total = filtered_df['other_total'].mean()

    donut_labels = ['Green Energy', 'Other Energy']
    donut_values = [mean_green_total, mean_other_total]
    donut_colors = ['mediumseagreen','sandybrown']

    donut_fig = go.Figure(data=[go.Pie(labels=donut_labels, values=donut_values, hole=.3, marker=dict(colors=donut_colors))])
    donut_fig.update_layout(
        title='Mean Total Share of Green vs Other Energy',
        showlegend=True
    )

    for year in filtered_df['year'].unique():
        year_data = filtered_df[filtered_df['year'] == year]
        total_share = (year_data[[f'{etype}_share_elec' for etype in energy_types_green + energy_types_other]]
                       .sum(axis=1).values[0])
        for etype in energy_types_green + energy_types_other:
            filtered_df.loc[filtered_df['year'] == year, f'{etype}_share_elec'] = (
                filtered_df.loc[filtered_df['year'] == year, f'{etype}_share_elec'] / total_share * 100
            )

    orange_colors = px.colors.diverging.Geyser[::-1][:len(energy_types_green)]
    green_colors = px.colors.sequential.YlGn[-3::-1][:len(energy_types_other)]

    fig = go.Figure()

    for i, etype in enumerate(energy_types_green):
        fig.add_trace(go.Bar(
            x=filtered_df['year'],
            y=filtered_df[f'{etype}_share_elec'],
            name=f'{etype.capitalize()} (Green)',
            marker_color=green_colors[i]
        ))

    for i, etype in enumerate(energy_types_other):
        fig.add_trace(go.Bar(
            x=filtered_df['year'],
            y=filtered_df[f'{etype}_share_elec'],
            name=f'{etype.capitalize()} (Other)',
            marker_color=orange_colors[i]
        ))

    fig.update_layout(
        barmode='stack',
        title=f'Comparison of Green Energy vs Other Energy Share in Electricity in {selected_continent} (2000-2023)',
        xaxis=dict(title='Year'),
        yaxis=dict(title='Share of Electricity (%)', tickformat='.0f%'),
        legend_title='Energy Type',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=30, b=0)
    )

    selected_year = st.slider("Select Year", min_value=int(filtered_df['year'].min()), max_value=int(filtered_df['year'].max()))

    def update_donut(year):
        year_data = filtered_df[filtered_df['year'] == year]
        green_total = year_data['green_total'].values[0]
        other_total = year_data['other_total'].values[0]

        updated_donut_fig = go.Figure(data=[go.Pie(labels=donut_labels, values=[green_total, other_total], hole=.3, marker=dict(colors=donut_colors))])
        updated_donut_fig.update_layout(
            title=f'Total Share of Green vs Other Energy in {year}',
            showlegend=True
        )
        return updated_donut_fig

    col1, col2 = st.columns([1, 5])
    with col1:
        st.plotly_chart(update_donut(selected_year), use_container_width=True)

    with col2:
        st.plotly_chart(fig, use_container_width=True)

visualize_trend_analysis(df)

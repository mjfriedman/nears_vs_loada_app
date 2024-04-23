# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

########################################################################################################################
# Page configuration
st.set_page_config(
    page_title="Nears vs Loada visualization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")

# Adjust the width of the sidebar
st.markdown(
    """
    <style>
    .css-1l04doo { width: 280px; }
    </style>
    """,
    unsafe_allow_html=True
)
alt.themes.enable("dark")

########################################################################################################################
# Load data
nears_data = pd.read_json("data/gen_nears_annotation.json")
loada_data = pd.read_json("data/gen_loada_annotation.json")

########################################################################################################################

# Convert datetime strings to datetime objects
nears_data['start'] = pd.to_datetime(nears_data['start'], format='ISO8601')
nears_data['end'] = pd.to_datetime(nears_data['end'], format='ISO8601')

loada_data['start'] = pd.to_datetime(loada_data['start'], format='ISO8601')
loada_data['end'] = pd.to_datetime(loada_data['end'], format='ISO8601')

########################################################################################################################
# Sidebar
st.sidebar.title("")
st.sidebar.markdown("# NEARS vs LOADA")

# Section for the checkboxes
st.sidebar.markdown("---")
# st.sidebar.header("Data Selection")
show_loada_data = True
show_nears_data = True

# Section for the date inputs
st.sidebar.header("Plage de dates")
start_date = st.sidebar.date_input("Début", value=min(nears_data['start']), min_value=min(nears_data['start']),
                                   max_value=max(nears_data['end']) - pd.Timedelta(days=1))
end_date = st.sidebar.date_input("Fin", value=pd.to_datetime(start_date + pd.Timedelta(days=1)),
                                 min_value=pd.to_datetime(start_date + pd.Timedelta(days=1)),
                                 max_value=max(nears_data['end']))
st.sidebar.markdown("---")


########################################################################################################################

# Filter data
filtered_nears_data = nears_data[(nears_data['start'] >= pd.to_datetime(start_date)) &
                                 (nears_data['end'] <= pd.to_datetime(end_date))]
filtered_loada_data = loada_data[(loada_data['start'] >= pd.to_datetime(start_date)) &
                                 (loada_data['end'] <= pd.to_datetime(end_date))]

# Color mapping
color_dict = {"Tous": "grey", "Repas": "orange", "Sommeil": "darkblue", "Hygiène": "skyblue", "Sortie": "green"}


########################################################################################################################
# Dashboard Main Panel
col = st.columns((12, 4, 1), gap='large')
########################################################################################################################
with col[0]:
    st.header("Comparaison des classifications")
    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        # st.subheader("Home Filter")
        homes = ['Tous'] + list(nears_data['home'].unique())
        st.markdown("##### Selectionner une résidence")
        selected_home = st.selectbox(" ", homes)

    with col2:
        # st.subheader("Activity Filter")
        activities = ['Tous'] + list(nears_data['activity'].unique())
        st.markdown("##### Selectionner une activité")
        selected_activity = st.selectbox(" ", activities)

    # Filtering data based on selected home and activity
    if selected_home != 'Tous':
        filtered_nears_data = filtered_nears_data[nears_data['home'] == selected_home]
        filtered_loada_data = filtered_loada_data[loada_data['home'] == selected_home]

    if selected_activity != 'Tous':
        #filtered_nears_data = filtered_nears_data[filtered_nears_data['activity'] == selected_activity]
        filtered_loada_data = filtered_loada_data[filtered_loada_data['activity'] == selected_activity]

    ####################################################################################################################
    st.markdown("---")
    # Date range slider
    st.markdown("##### Glisser pour sélectionner une période")
    start_date_slider, end_date_slider = st.slider(" ", min_value=start_date, max_value=end_date,
                                                   value=(start_date, end_date))

    # Filter data based in slider data range
    filtered_nears_data = filtered_nears_data[(filtered_nears_data['start'] >= pd.to_datetime(start_date_slider)) &
                                              (filtered_nears_data['end'] <= pd.to_datetime(end_date_slider))]
    filtered_loada_data = filtered_loada_data[(filtered_loada_data['start'] >= pd.to_datetime(start_date_slider)) &
                                              (filtered_loada_data['end'] <= pd.to_datetime(end_date_slider))]

    fig = go.Figure()
    # Display the color legend
    act_label = color_dict.copy()
    act_label.pop("Tous", None)
    st.markdown(
        " ".join([f"<div style='display: inline-block; margin-right: 20px;'><div style='background-color:{color};"
                  f" width: 40px; height: 20px; display: inline-block; margin-right: 5px;'>"
                  f"</div><span style='font-size:25px;'>{activity}</span></div>"
                  for activity, color in act_label.items()]), unsafe_allow_html=True)

    # NEARS Data Visualization
    if show_nears_data:
        for activity in filtered_nears_data['activity'].unique():
            df = filtered_nears_data[filtered_nears_data['activity'] == activity]
            for i in range(len(df)):
                fig.add_trace(go.Scatter(x=[df.iloc[i]['start'], df.iloc[i]['end']], y=["NEARS", "NEARS"], mode='lines',
                                         name=activity,
                                         line=dict(color=color_dict[activity], width=450),
                                         hoverinfo='none', showlegend=False))

    # LOADA Data Visualization
    if show_loada_data:
        for activity in filtered_loada_data['activity'].unique():
            df = filtered_loada_data[filtered_loada_data['activity'] == activity]
            for i in range(len(df)):
                fig.add_trace(go.Scatter(x=[df.iloc[i]['start'], df.iloc[i]['end']], y=["LOADA", "LOADA"], mode='lines',
                                         name=activity,
                                         line=dict(color=color_dict[activity], width=450),
                                         hovertemplate=f'annot_time: {df.iloc[i]["annot_time"]}<extra></extra>',
                                         showlegend=False))

    fig.update_layout(yaxis={'title': 'Annotations'}, xaxis={'title': 'Date'}, hovermode="x unified")
    fig.update_xaxes(title_text='Date', title_font=dict(size=24), tickfont=dict(size=18))
    fig.update_yaxes(title_text='Annotations', title_font=dict(size=24), tickfont=dict(size=20))
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True, height=700)

########################################################################################################################
########################################## -------------METRICS------------#############################################
########################################################################################################################
with col[1]:
    st.header("Evaluation")
    st.markdown("---")

    # Compute Precision
    def calculate_metrics(nears_df, loada_df, classe):
        # Create a merged dataframe
        merged_data = pd.merge(nears_df, loada_df, on="id", suffixes=('_nears', '_loada'), how='inner')

        # Sort merged dataframe by start time
        merged_data = merged_data.sort_values(by='start_loada')

        # Calculate precision, recall, mAP and F1 score for "classe"
        tp_count = 0
        fp_count = 0
        fn_count = 0

        for i, row in merged_data.iterrows():
            if row['activity_nears'] == row['activity_loada'] == classe:
                tp_count += 1
            elif row['activity_loada'] != classe and row['activity_nears'] == classe:
                fp_count += 1
            elif row['activity_loada'] == classe and row['activity_nears'] != classe:
                fn_count += 1

        precision = tp_count / (tp_count + fp_count + 1e-10)
        recall = tp_count / (tp_count + fn_count + 1e-10)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        return precision, f1_score


    # Precision and F1 score calculation and display
    others_activities = activities.copy()
    others_activities.remove('Tous')
    if selected_activity == 'Tous':
        all_precision, all_f1score = 0, 0
        for activity in others_activities:
            mAP, F1_score = calculate_metrics(filtered_nears_data, filtered_loada_data, activity)
            all_precision += mAP
            all_f1score += F1_score
            st.write(f" #### <span style='color:{color_dict[activity]}'>{activity}</span>", unsafe_allow_html=True)
            with st.expander('Precision/ F1-Score', expanded=True):
                st.markdown(f"**Precision**  = {mAP * 100:.2f}%")
                st.markdown(f"**F1-score** =  {F1_score * 100:.2f}%")

        # All
        st.write(f"#### <span style='color:{color_dict['Tous']}'>{'Tous'}</span>", unsafe_allow_html=True)
        with st.expander('Precision/ F1-Score', expanded=True):
            st.markdown(f"**Precision**  = {all_precision / len(others_activities) * 100:.2f}%")
            st.markdown(f"**F1-score** =  {all_f1score / len(others_activities) * 100:.2f}%")

    else:
        st.write(f"#### <span style='color:{color_dict[selected_activity]}'>{selected_activity}</span>",
                 unsafe_allow_html=True)
        mAP, F1_score = calculate_metrics(filtered_nears_data, filtered_loada_data, selected_activity)
        with st.expander('Precision/ F1-Score', expanded=True):
            st.markdown(f"**Precision**  = {mAP * 100:.2f}%")
            st.markdown(f"**F1-score** =  {F1_score * 100:.2f}%")

########################################################################################################################
########################################---------BARPLOT & PIE CHART----------##########################################
########################################################################################################################

# Section for the tolerance and time interval inputs
st.markdown("---")
col1, _, _, col2 = st.columns([1, 0.1, 0.1, 1])
with col1:
    st.markdown("##### Sélectionner intervalle de temps")
    selected_time_interval = col1.radio(" ", options=[2, 4, 6, 12, 24], index=1,
                                        format_func=lambda x: f"{x} heures", horizontal=True)
with col2:
    st.markdown("##### Choisir une tolérance (minutes)")
    tolerance = col2.number_input(" ", min_value=0, max_value=60, value=20, step=1)


# Histogram/Barplot Data
def generate_histogram_data(nears_df, loada_df, time_interval=2):
    # Create a range of dates based on the time interval
    date_range = pd.date_range(start=start_date_slider, end=end_date_slider, freq=f"{time_interval}h")

    mAP_data = []
    F1_score_data = []

    # Cumulative mAP and F1 Score
    for j in range(1, len(date_range)+1):
        end_time = pd.to_datetime(start_date_slider) + pd.Timedelta(hours=j*time_interval)

        # Filter data
        _nears = nears_df[(nears_df['start'] >= pd.to_datetime(start_date_slider)) & (nears_df['end'] <= end_time)]
        _loada = loada_df[(loada_df['start'] >= pd.to_datetime(start_date_slider)) & (loada_df['end'] <= end_time)]

        m_precision, f1score = 0, 0
        if selected_activity == 'Tous':
            for activity in others_activities:
                pre, f1 = calculate_metrics(_nears, _loada, activity)
                m_precision += pre
                f1score += f1
            f1score /= len(others_activities)
            m_precision /= len(others_activities)
        else:
            pre, f1 = calculate_metrics(_nears, _loada, selected_activity)
            m_precision = pre
            f1score = f1

        mAP_data.append(m_precision)
        F1_score_data.append(f1score)

    return pd.DataFrame({'Date': date_range, 'mAP': mAP_data, 'F1 Score': F1_score_data})


def calculate_predictions(nears_df, loada_df, tol):
    # Merged dataframe
    merged_data = pd.merge(nears_df, loada_df, on="id", suffixes=('_nears', '_loada'), how='inner')

    # Sort merged dataframe by start time
    merged_data = merged_data.sort_values(by='start_loada')

    good, middle, bad = 0, 0, 0

    tol = pd.Timedelta(minutes=tol)

    for _, row in merged_data.iterrows():
        if row['activity_nears'] == row['activity_loada']:
            if np.abs(row['start_nears'] - row['start_loada']) <= tol and np.abs(
                    row['end_nears'] - row['end_loada']) <= tol:
                good += 1
            else:
                middle += 1
        else:
            bad += 1

    total = good + middle + bad + 1e-10

    # Percentages
    good_pct = good / total * 100
    middle_pct = middle / total * 100
    bad_pct = bad / total * 100

    return good_pct, middle_pct, bad_pct


def generate_pie_chart_data(nears_df, loada_df, tol=10):
    good, middle, bad = calculate_predictions(nears_df, loada_df, tol)
    total = good + middle + bad + 1e-10
    return pd.DataFrame({'Prediction': ['Bonne', 'Moyenne', 'Mauvaise'],
                         'Percentage': [(good / total) * 100, (middle / total) * 100, (bad / total) * 100]})


def plot_histogram_and_pie_chart():
    histogram_data = generate_histogram_data(filtered_nears_data, filtered_loada_data, selected_time_interval)

    pie_chart_data = generate_pie_chart_data(filtered_nears_data, filtered_loada_data, tolerance)

    # Display Plots
    col_1, col_2 = st.columns([2, 2])

    with col_1:
        # Plot Bar Chart
        bar_fig = go.Figure()

        # Plot Precision
        bar_fig.add_trace(
            go.Bar(x=histogram_data['Date'], y=histogram_data['mAP'] * 100, name='Precision', marker_color='brown'))
        # Plot F1 Score
        bar_fig.add_trace(
            go.Bar(x=histogram_data['Date'], y=histogram_data['F1 Score'] * 100, name='F1 Score', marker_color='beige'))

        bar_fig.update_layout(
            xaxis={'title': 'Date'},
            yaxis={'title': 'Metrics (%)'},
            hovermode="x unified",
            barmode='group'
        )

        st.plotly_chart(bar_fig, use_container_width=True)
        st.markdown("#### Evolution de la Précision et F1 Score")

    with col_2:
        # Plot Pie Chart
        pie_fig = px.pie(pie_chart_data, values='Percentage', names='Prediction', color='Prediction', hole=0.4,
                         color_discrete_map={'Bonne': 'green', 'Moyenne': 'yellow', 'Mauvaise': 'red'})
        st.plotly_chart(pie_fig, use_container_width=True)
        st.markdown("#### Qualité de la classification de NEARS")


plot_histogram_and_pie_chart()

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

np.random.seed(42)


def load_data():
    df_productivity = pd.read_csv(
        "https://docs.google.com/spreadsheets/d/10Jx_uq-YSBNm6xeTVVz3vqt0V_zwNgSit8t-1O5grfg/export?format=csv&gid=525662292"
    )

    df_emissions = pd.read_csv(
        "https://docs.google.com/spreadsheets/d/1l48OFgP5gDr_O7tYzxXJitTK1S1yHIDv0bfVFz5gsvg/export?format=csv&gid=645227798"
    )

    df_renewable = pd.read_csv(
        "https://drive.google.com/uc?export=download&id=1Q5OtzPg1MZH9YgeDB-eaM32aP6MkDXJd"
    )

    return df_productivity, df_emissions, df_renewable


def preprocess_data():
    df_productivity, df_emissions, df_renewable = load_data()

    # ambil kolom penting
    df1 = df_productivity[['REF_AREA','Reference area','TIME_PERIOD','OBS_VALUE']].rename(
        columns={'OBS_VALUE':'productivity'}
    )

    df2 = df_emissions[df_emissions['Unit of measure']=='Tonnes'][
        ['REF_AREA','Reference area','TIME_PERIOD','OBS_VALUE']
    ].rename(columns={'OBS_VALUE':'emissions'})

    df3 = df_renewable[['REF_AREA','Reference area','TIME_PERIOD','OBS_VALUE']].rename(
        columns={'OBS_VALUE':'renewable_supply'}
    )

    # merge dataset
    merged = pd.merge(df1, df2, on=['REF_AREA','Reference area','TIME_PERIOD'], how='inner')
    df = pd.merge(merged, df3, on=['REF_AREA','Reference area','TIME_PERIOD'], how='inner')

    df = df.sort_values(by=['Reference area','TIME_PERIOD']).reset_index(drop=True)

    # FEATURE ENGINEERING
    df['sustainability_category'] = pd.cut(
        df['renewable_supply'],
        bins=[-1, 20, 40, 60, 80, 101],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )

    # productivity binning
    min_val = df['productivity'].min()
    q1 = df['productivity'].quantile(0.25)
    q2 = df['productivity'].quantile(0.5)
    q3 = df['productivity'].quantile(0.75)
    max_val = df['productivity'].max()

    df['country_pattern'] = pd.cut(
        df['productivity'],
        bins=[min_val-1, q1, q2, q3, max_val+1],
        labels=['Emerging Polluters','Industrial Giants','Developing Sustainers','Green Leaders']
    )

    # scaling emissions (sesuai notebook kamu)
    scaler = RobustScaler()
    df['emissions_scaled'] = scaler.fit_transform(df[['emissions']])

    return df
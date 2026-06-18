import pandas as pd
from src.feature_engineering import add_technical_features

def load_and_align_usdjpy(h1_path, h4_path, d1_path):
    df_h1 = pd.read_csv(h1_path, parse_dates=['timestamp']).set_index('timestamp')
    df_h4 = pd.read_csv(h4_path, parse_dates=['timestamp']).set_index('timestamp')
    df_d1 = pd.read_csv(d1_path, parse_dates=['timestamp']).set_index('timestamp')

    df_h1 = add_technical_features(df_h1)
    df_h4 = add_technical_features(df_h4)
    df_d1 = add_technical_features(df_d1)

    # 將 H4 和 D1 特徵對齊到 H1（向後對齊，避免未來洩漏）
    h4_features = df_h4[['SMA_20', 'RSI_14', 'ATR_14', 'MACD']].add_suffix('_H4')
    d1_features = df_d1[['SMA_20', 'RSI_14', 'ATR_14', 'MACD']].add_suffix('_D1')

    h4_aligned = h4_features.resample('H').last().ffill()
    d1_aligned = d1_features.resample('H').last().ffill()

    df = df_h1.join(h4_aligned, how='left')
    df = df.join(d1_aligned, how='left')

    return df.dropna()

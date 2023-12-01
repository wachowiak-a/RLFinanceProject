from finrl import config_tickers
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import (
    FeatureEngineer,
)
import pandas as pd
from tqdm import t_range

# downloads stock market data in the given ranges and returns training and test splits
def download_and_process_data(
        start_date,
        end_date,
        ticker_list=config_tickers.SP_500_TICKER
):
    print("Downloading data for tickers:", ticker_list)

    df = YahooDownloader(
        start_date=start_date,
        end_date=end_date,
        ticker_list=ticker_list,
    ).fetch_data()

    fe = FeatureEngineer(
                        use_technical_indicator=True,
                        use_turbulence=True,
                        user_defined_feature = False)

    df = fe.preprocess_data(df)

    # add covariance matrix as states
    df=df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]

    cov_list = []
    return_list = []

    # look back is one year
    lookback=252
    print("Adding covariances...")
    for i in t_range(lookback,len(df.index.unique()), label="Calculating price covariances..."):
        data_lookback = df.loc[i-lookback:i,:]
        price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)

        covs = return_lookback.cov().values
        cov_list.append(covs)

    
    df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date','tic']).reset_index(drop=True)

    return df
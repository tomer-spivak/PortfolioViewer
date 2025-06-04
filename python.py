import asyncio
import streamlit as st
import yfinance as yf
import pandas as pd
import json
import pandas_ta as ta

from streamlit_lightweight_charts import renderLightweightCharts

st.set_page_config(
    layout='wide',
    page_title='My Dashboard'
)

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# Ensure event loop is set
get_or_create_eventloop()









from ib_insync import *
from ib_insync import util
ib = IB()
if not ib.isConnected():
    try:
        ib.connect('127.0.0.1', 4001, clientId=1, timeout=5)
    except (RuntimeError, TimeoutError, ConnectionRefusedError):
        st.info("Open IB Gateway and Login")
        st.button('Press here after done')
        st.stop()


def portfolio_items(ib):
    def create_portfolio_item(item):
        cds = ib.reqContractDetails(item.contract)[0]
        perc_change = (item.marketPrice / item.averageCost - 1) * 100
        if item.position < 0:
            perc_change = (item.averageCost / item.marketPrice - 1) * 100

        return {
            'contract': item.contract,
            'symbol': item.contract.symbol,
            'name': cds.longName,
            'position': item.position,
            'avg_cost': item.averageCost,
            'market_price': item.marketPrice,
            'market_value': item.marketValue,
            'perc_change': perc_change,
            'unrealized_pnl': item.unrealizedPNL,
            'realized_pnl': item.realizedPNL
        }

    _portfolio_items = ib.portfolio()
    items_dicts = [create_portfolio_item(item) for item in _portfolio_items]
    df = pd.DataFrame.from_records(items_dicts)
    return df


class PositionsTable:
    @staticmethod
    def show_table(st_obj, df):
        def highlight_up(val):
            color = 'green' if val > 0 else 'red'
            return f'background-color: {color}'

        styled_df = df.style.format(precision=2, thousands=',', decimal='.') \
            .map(highlight_up, subset=['perc_change'])
        st_obj.dataframe(
            styled_df,
            column_config={
                'contract': None,
                'symbol': 'Symbol',
                'name': 'Name',
                'avg_cost': None,
                'position': 'Size',
                'market_price': 'Price per share',
                'market_value': 'Total',
                'perc_change': 'Change%',
                'unrealized_pnl': 'Unrealized PNL',
                'realized_pnl': None
            },
            hide_index=True
        )


with st.container():
    # Create two columns: one for your custom table and one for metrics
    col1, col2 = st.columns(2)
    # Column 2: Metrics calculated from your portfolio data
    with col1:
        portfolio_df = portfolio_items(ib)

        print(portfolio_df.columns)
        print(portfolio_df.head())


        total_market_value = portfolio_df['market_value'].sum()
        st.metric(label="Current Stocks Value", value=f"{total_market_value:,.2f}$")

        # Compute cost basis for each position
        portfolio_df['cost'] = portfolio_df['avg_cost'] * portfolio_df['position']
        total_cost = portfolio_df['cost'].sum()
        st.metric(label="Stocks Bought At", value=f"{total_cost:.2f}$")

        # Retrieve cash information from IB account summary
        cash_item = next(
            (item for item in ib.accountSummary()
             if item.tag == 'TotalCashValue' and item.currency == 'USD'),
            None
        )
        current_cash = float(cash_item.value) if cash_item else 0.0

        total_cost += current_cash
        total_value = total_market_value + current_cash
        st.metric(label="Current Portfolio Value", value=f"{total_value:.2f}$")
        total_gain = float(total_value - total_cost)

        # Calculate overall portfolio return
        if total_cost != 0:
            portfolio_return = (total_value / total_cost - 1) * 100
        else:
            portfolio_return = 0
        st.metric(label="Portfolio Return", value=f"{portfolio_return:.2f}%")

    # Column 1: Your custom table (replace with your own data)
    with col2:

        st.metric(label="Total value gained", value=f"{total_gain:.2f}$")
        portfolio_percentages = ["52.5%", "12.5%", "12.5%", "12.5%", "10%"]
        gain = []
        for i in range(5):
         gain.append(round(total_gain * float(portfolio_percentages[i].split('%')[0]) / 100, 2) )
        tomer_percentage = 0.2
        my_data = pd.DataFrame({
            'Name': ['Tomer', 'Parents', 'Gily', 'Grandma', 'Ilay'],
            'Percentage of portfolio': portfolio_percentages,
            'total owned in portfolio': [str(round(total_value * float(portfolio_percentages[i].split('%')[0]) / 100, 2)) + "$" for i in range(5)],
            'Total gained': [str(gain[i]) + "$" for i in range(5)],
            'Tomer\'s share': [str(round(gain[i] * tomer_percentage, 2)) + "$" for i in range(5)],
            'Gain left': [str(round(gain[i] * (1 - tomer_percentage), 2)) + "$" for i in range(5)]
        })
        tomer_gain = 0
        for i in range(1, 5):

            tomer_gain += gain[i] * tomer_percentage
        st.metric(label="Total tomer commission", value=f"{tomer_gain:.2f}$")
        st.metric(label='Total tomer earned', value=f'{tomer_gain + gain[0]:.2f}$')

        my_data_reset = my_data.reset_index(drop=True)
        st.table(my_data_reset)





class PositionsChart:
    @staticmethod
    def show_chart(st_obj, df):
        # Your original Plotly pie chart code remains here (if you want both charts)
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(
            go.Pie(
                labels=list(df["symbol"]),
                values=list(df["market_value"].abs()),
                sort=True
            )
        )
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st_obj.plotly_chart(fig, use_container_width=True)


# Display portfolio table and pie chart
with st.container():
    col1, col2 = st.columns([5, 3])
    portfolio_df = portfolio_items(ib)
    PositionsTable.show_table(col1, portfolio_df)
    PositionsChart.show_chart(col2, portfolio_df)


# Historical data retrieval function for IB data
def historical_data(ib, stock):
    bars = ib.reqHistoricalData(
        stock,
        endDateTime='',
        durationStr='5 Y',
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    if bars:
        return util.df(bars)
    return None


# Class for rendering candlestick charts using streamlit_lightweight_charts
class StockChart:

    @staticmethod
    def show_chart(df, symbol):
        COLOR_BULL = 'rgba(38,166,154,0.9)'  # #26a69a
        COLOR_BEAR = 'rgba(239,83,80,0.9)'  # #ef5350

        df.rename(columns={'date': 'time'}, inplace=True)
        df['time'] = df['time'].astype('str')
        df.ta.sma(close='close', length=50, append=True)

        df = df.tail(200)

        candles = json.loads(df.to_json(orient='records'))
        volume = json.loads(df.rename(columns={'volume': 'value', }).to_json(orient='records'))
        sma = json.loads(df.rename(columns={'SMA_50': 'value', }).to_json(orient='records'))

        chartMultipaneOptions = [
            {
                "width": 1200,
                "height": 400,
                "layout": {
                    "background": {
                        "type": "solid",
                        "color": 'black'
                    },
                    "textColor": "white"
                },
                "grid": {
                    "vertLines": {
                        "color": "rgba(197, 203, 206, 0.5)"
                    },
                    "horzLines": {
                        "color": "rgba(197, 203, 206, 0.5)"
                    }
                },
                "crosshair": {
                    "mode": 0
                },
                "priceScale": {
                    "borderColor": "rgba(197, 203, 206, 0.8)"
                },
                "timeScale": {
                    "borderColor": "rgba(197, 203, 206, 0.8)",
                    "barSpacing": 15
                },
                "watermark": {
                    "visible": False,
                    "fontSize": 48,
                    "horzAlign": 'center',
                    "vertAlign": 'center',
                    "color": 'white',
                    "text": symbol,
                }
            }
        ]

        seriesCandlestickChart = [
            {
                "type": 'Candlestick',
                "data": candles,
                "options": {
                    "upColor": COLOR_BULL,
                    "downColor": COLOR_BEAR,
                    "borderVisible": False
                }
            }
        ]

        seriesVolumeChart = [
            {
                "type": 'Histogram',
                "data": volume,
                "options": {
                    "priceFormat": {
                        "type": 'volume',
                    },
                    "priceScaleId": ""
                },
                "priceScale": {
                    "scaleMargins": {
                        "top": 0.7,
                        "bottom": 0,
                    },
                    "alignLabels": False
                }
            }
        ]

        renderLightweightCharts([
            {
                "chart": chartMultipaneOptions[0],
                "series": seriesCandlestickChart + seriesVolumeChart
            }
        ], f'multipane{symbol}')

# ------------------------------------------------------------------------------
# Usage: For each portfolio symbol, retrieve historical data and display a chart.
for index, row in portfolio_df.iterrows():

    contract = row['contract']

    stock = Stock(symbol=contract.symbol, exchange=contract.primaryExchange)
    st.markdown(f"### {contract.symbol}")

    bars_df = historical_data(ib, stock)

    # If IB data is unavailable, fallback to Yahoo Finance.
    if bars_df is None or bars_df.empty:
        st.info(
            f"You're not subscribed to market data for exchange {contract.primaryExchange}. Data shown is obtained from Yahoo")
        yahoo_symbol = contract.symbol  # Use the same symbol or modify if needed.
        ticker = yf.Ticker(yahoo_symbol)
        bars_df = ticker.history(period='5y')
        bars_df = bars_df.reset_index()
        bars_df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)

    if bars_df is not None and not bars_df.empty:
        StockChart.show_chart(bars_df, contract.symbol)

# For each portfolio symbol, retrieve historical data and show candlestick chart

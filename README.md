# NSE Time Series Analysis
  ![image](https://github.com/user-attachments/assets/9090973b-4519-4557-8ffe-e6a6c0f3e3b4)

### Group Members
1. Brenda Mutai
2. Sharon Momanyi
3. Stephen Munyiala
4. Justin Mbugua
   
# 1. Project Overview

This project focuses on analyzing stock price movements across all publicly listed companies on the Nairobi Securities Exchange (NSE) over the years 2013 and 2024.Using daily trading data — including prices, trading volume, Sector information — the project aims to provide actionable insights into market trends, stock performance, sector dynamics, and trading patterns. The analysis aims to help investors, analysts, and researchers better understand market behavior and identify opportunities within the Kenyan stock market.

# 2. Business Understanding
Investors and financial institutions operating on the Nairobi Securities Exchange (NSE) rely on precise information to make strategic decisions regarding stock trades. By forecasting future stock prices and identifying market trends, they can optimize investment strategies, improve portfolio management, and effectively mitigate risks. Such insights empower traders to seize profitable opportunities, avoid potential losses, and enhance overall financial stability.
Additionally, predictive models enable investors to understand market behavior better, adapt to changing conditions, and maintain a competitive edge. With tools for analyzing historical trading data, stakeholders can uncover patterns, assess the impact of external factors like economic policies or global market shifts, and make data-driven choices that align with their financial goals. These advancements are essential for thriving in a dynamic stock market environment like the NSE.

# 3. Objectives
Mainly to Develop time series models to forecast future stock prices.
- To provide insights into which stocks might perform well based on historical trends and predictive models, which will allow for more informed decision-making.
- To offer short-term predictions of stock prices or trends to support timely buy/sell decisions, potentially improving their profitability.
- To develop a machine learning-based tool that provides predictive insights and visualizations for NSE market trends.

# 4. Stakeholders
1. Individual Investors: People who are actively trading or considering investments in NSE-listed companies.
2. Institutional Investors: Investment funds, banks, and pension funds looking for more structured insights into the market.
3. Stockbrokers and Analysts: Professionals who rely on historical data to guide their clients and make informed decisions.
4. NSE and Regulatory Bodies: The NSE itself and regulatory authorities that track market health and performance, using tools for market surveillance and risk assessment.
5. Financial Advisors: Professionals who can use predictive insights to guide client portfolios.

# 5. Data Understanding
This data was sourced from [Mendely](https://data.mendeley.com/datasets/ss5pfw8xnk/2) and the key features include:

o	Date: Trading date (e.g., "03-Jan-2023").

o	Code: Stock ticker symbol (e.g., "EGAD", "KUKZ").

o	Name: Company or index name (e.g., "Eaagads Ltd", "NSE 25-Share Index").

o	12m Low/High: 12-month lowest and highest prices.

o	Day Low/High: Daily trading range.

o	Day Price: Closing price for the day.

o	Previous: Previous day's closing price (missing for some entries).

o	Change/Change%: Absolute and percentage change from the previous day.

o	Volume: Trading volume (some entries missing or zero).

o	Adjusted Price: Not populated in the sample.

o	Sector: Classification (e.g., "Agricultural").


# 6. Data Cleaning
- We merged all the datasets
- Handled missing values and duplicates.
- Standardized our data. 

# 7. EXPLORATORY DATA ANALYSIS(EDA)
We explored the data using univariate, bivariate and multivariate analysis just to see the relationship between different variables. We also checked for outliers and although we identified some, we chose to keep them in the dataset because in the context of stock market data, sudden spikes or drops in price or volume are normal due to market volatility, major news events or investor sentiments. Removing them could result in loss of important information.

# 8. Feature Engineering
We created new features to ensure our models perform better.
- SMA_10 and SMA_50: Simple moving averages over 10 and 50 trading days.
- EMA_10 and EMA_50: Exponential moving averages over 10 and 50 days (more weight to recent prices).
- Relative Strength Index(RSI):The RSI measures the speed and change of price movements.
These indicators help smooth out price fluctuations and identify overall market trends, making them essential for detecting potential reversals or continuations in stock movement.

# 9. Modeling and model evaluation
We trained different models.ie, SARIMA, XGboost and LSTM. We then deployed the best performimg model.
![image](https://github.com/Brendamutai/Capstone-Project/blob/main/model%20evaluation.JPG)
Based on the evaluated metrics (MAE, MSE, and RMSE), the XGBoost model significantly outperforms both the SARIMA and LSTM models. It consistently achieves the lowest error values across all three metrics.

The SARIMA model performs better than the LSTM model but is not as effective as XGBoost according to these results.

The LSTM model appears to have performed poorly in this specific evaluation, exhibiting substantially higher errors (especially MSE and RMSE) compared to the other two models.
# 10. Conclusions 
- The XGBoost model clearly outperforms both the SARIMA and LSTM models. It consistently records the lowest error values across all three measures, indicating superior predictive accuracy and robustness. This suggests that XGBoost is particularly effective at capturing complex patterns in the data.

- Effectiveness in Handling Volatility and Non-linearity: Stock market data, including the NSE, is inherently volatile and exhibits complex non-linear relationships. The strong performance of XGBoost indicates its effectiveness in modeling these characteristics. As a tree-based ensemble method, XGBoost is well-suited to capture complex interactions between features and handle non-linear patterns that simpler linear models like ARIMA may struggle with.

- While LSTM is designed to handle sequential data and long-term dependencies effectively, its performance may have been impacted by challenges in adapting to the dataset’s structure. Additionally, recurrent neural networks like LSTM can sometimes struggle with short-term fluctuations in the NSE data which it faced potentially leading to larger errors in its predictive tasks.

# 11.Recommendation
- Focus on high-volume stocks, as they typically provide better liquidity, allowing traders to enter and exit positions with minimal price slippage. High trading volume also indicates strong investor interest and activity, which often leads to more consistent and significant price movements. These characteristics make high-volume stocks particularly attractive for short-term trading strategies, where timely execution and frequent price action are essential for capturing quick gains.

- Consider allocating a larger share of liquidity-sensitive investments to the Telecommunication and Banking sectors, as these typically exhibit higher trading volumes and market activity. This enhanced liquidity facilitates easier entry and exit from positions, reducing the likelihood of price slippage and potentially lowering transaction costs. Additionally, these sectors often include well-established companies with steady investor interest, making them more stable and predictable for managing large or frequently adjusted positions. This makes them ideal for strategies that require flexibility and quick response to market conditions.

- Diversify your portfolio by including both high-volume sectors and emerging industries to achieve a balanced risk-return profile. High-volume sectors, such as Finance and Telecommunications, offer stability, liquidity, and consistent performance due to their established market presence and strong investor participation. In contrast, emerging sectors—like Technology, Renewable Energy, or Fintech—may exhibit higher volatility but present significant growth potential as they innovate and expand. This diversification strategy helps cushion against market downturns in any single sector while positioning the portfolio to benefit from future growth opportunities in rapidly evolving industries.

# 12. Next Steps
 1.	Enhancing Feature Engineering and Integration
 2.	Expanding Data Utilization and Continuous Retraining
 3.	Improving Model Evaluation and Performance Monitoring

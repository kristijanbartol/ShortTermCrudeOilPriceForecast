# Short-Term Crude Oil Price Forecasting

Solving self-assigned crude oil short-term price forecasting problem. Dataset found online. Using XGBoost as a regressor. Data is lagged for N days (N=9 used as an optimal value).

## Dataset

This is the "crude" data of crude oil price for period from June 2012. to February 2016. in the image below. As you can see, there was been a significant price jump somewhere in 2014. This free-available data only contains prices for each working day and we will try to make a day-ahead price forecast, which is somehow ill-formed from the beginning. We can't follow price trends during the day.

![Crude oil price data (2012.-2016.)](/graphs/crude_oil_figure.png)

Also, I used prices trends of gold, silver and natural gas for the same period, which can be found in an associated folder here in repo. You may ask why those prices should be relevant. Well, as this is just a machine learning oriented practice task, I wasn't doing more advanced data mining, seeking for the information of OPEC meetings, important global political events and other related factors that could be found to be used in relevant (crude oil price forecast) papers. As a matter of a fact, this was a test for XGBoost model, as it shows some significant results on Kaggle competitions.

Let us see how it works on pretty randomised and noisy dataset...

## XGBoost configuration

Can be found in params.py (ordinary grid search with limits of my old personal computer power).

## Results

The whole thing just gone pretty wrong. As the input seems quite random, the output was following the trend. Forecasted prices vary in +/- a dolar, which sums up to a huge error, two orders of magnitude higher than in relevant papers' results.

![Feature importance (in Croatian)](/graphs/feature_importance_xgb.png)

In this particular image I'm using only crude oil and you can notice that even now the results does not make much sense. If we are lead by the image, we conclude that the most significant day is a day before the forecast, which makes sense. On the other hand, the second day is not following this rule and if we take a look at the rest of the features, they are not sorted, nor their significance shows any rule. Moreover, when run with the whole dataset, feature importance changes depending on the file position in a data/ folder, which means we predicted a bunch of NOISE.

## Conclusion

If you came all the way down to the end of this text, I should tell you that this wasn't a random task, this was my Bachelor's thesis. I first felt like this was a great failure and I was even ashamed, but as a matter of a fact, I found out that this situation happens quite often when you're a starter even with a non-bad skills and knowledge. 

Don't be mislead into thinking that the model itself will show you the pattern. You have to ask yourself does every step of your potential solution makes sense, because if you just make it and expect something reasonable, you better quit right away.

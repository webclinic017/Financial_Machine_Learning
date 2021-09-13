
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from datetime import datetime
from get_data import get_data_stocks, STOCKS
from build_model import model_building, model_interpretation
from interprate_finance import financial_interpretation

now = datetime.now()
# get the wanted starttime
start = STOCKS.start
stock = "DTE.DE"
"ORCL,LHA.DE"
ps = get_data_stocks.get_data(now, start, stock)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# the smaller the better (overfitting)
h = 0
# 0.01

# weekly the higher the better around 60 was best 31, 7
number_days = 2

days_predict_into_future = 1

# false was always better
pca = False

cumsum = False

# "a lower number is preffered as excessive partitioning would agian place in the testing set samples too similar to those ues in the testing set

# good
n_splits = 3

# in percent: 1 = 100 % all rows are excluded: No data
pct_emb = 0.2

number_cols = 10

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

best_model, y_pred_cv,y_pred_test,y_prob_cv,y_prob_test,x_train, x_test, y_train, y_test, df_out = model_building.compute_first_model(ps, [0, 0], h, number_days,days_predict_into_future, pca, number_cols, n_splits, pct_emb,cumsum)

model_interpretation.interprate_model(best_model, y_pred_cv, y_pred_test, y_prob_cv, y_prob_test, x_train, x_test, y_train, y_test)

financial_interpretation.financial_interprate_model(ps,y_pred_cv, y_pred_test, y_prob_cv, y_prob_test, x_train, x_test, y_train, y_test,df_out)
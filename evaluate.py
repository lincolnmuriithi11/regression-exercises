def plot_residuals(x, y, yhat):
    residual = y - yhat
    baseline_residual = y - y.mean()

    plt.figure(figsize = (11,5))

    plt.subplot(121)
    plt.scatter(x, baseline_residual)
    plt.axhline(y = 0, ls = '-', color = "#red")
    plt.xlabel('x')
    plt.ylabel('Residual')
    plt.title('Baseline Residuals')

    plt.subplot(122)
    plt.scatter(x, residual)
    plt.axhline(y = 0, ls = '-', color = "#blue")
    plt.xlabel('x')
    plt.ylabel('Residual')
    plt.title('OLS model residuals')

def regression_errors(y, yhat):
    '''
    this function takes in a target varable created by linear regression model. prints out:
            Sum of squared errors
            Explained sum of squares
            Total sum of squares
            Mean squared error
            Root mean squared error

    Arguments:  y - target variable
                yhat - the predicted datapoints 
    '''
    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = mean_squared_error(y, yhat, squared = False)
    ESS = sum((yhat - y.mean())**2)
    TSS = ESS + SSE

    return SSE, ESS, TSS, MSE, RMSE

def baseline_mean_errors(y):
    baseline = pd.Series(y.mean()).repeat(len(y))
    MSE = mean_squared_error(y, baseline)
    SSE = MSE * len(y)
    RMSE = mean_squared_error(y, baseline, squared = False)

    return MSE, SSE, RMSE

def better_than_baseline(y, yhat):
    model_SSE, model_ESS, model_TSS, model_MSE, model_RMSE = regression_errors(y, yhat)
    baseline_MSE, baseline_SSE, baseline_RMSE = baseline_mean_errors(y)

    return (model_SSE - baseline_MSE) > 0

import pandas as pd
import datetime as dt
import binance as bnc

'''
UTILITIES NAMESPACE:
    -Holds the utilities for trading
'''

def _check_cols(df : pd.DataFrame, *args):
    try:
        indices = []
        cols = [str.lower(elem) for elem in list(df.columns)]
        for col in args:
            indices.append(cols.index(col.lower()))
        return indices
    except ValueError:                                                               
        raise TypeError(f'DataFrame does not contains {args} column(s)')

def get_today_yesterday(today : pd.Series, yesterday : pd.Series) -> pd.Series:
    
    '''
    Function that returns two series that consists of two rows shifted by one position down and one position up respectively (today and yesterday)

    Pre:
        Series MUST BE of the same length. Else, unexpected behaviour might happen

    Parameters:

        today (pd.Series) : pandas Series object. It is the series from where we'll took the 'today' data

        yesterday (pd.Series) : pandas Series object. It is the series from where we'll took the 'yesterday' data
    
    Returns:
        Two series with the today and yesterday values (shifted rows)
    '''

    today_data = today[1:] #today series

    yesterday_data = yesterday[:-1] #yesterday series

    today_data.set_axis(list(yesterday_data.axes[0]), inplace=True) #we set the axis of the today series to the one of the yesterday series because when we opperate with other series we need to match the indices (to make 1-to-1 operations)

    return today_data , yesterday_data


def get_difference_today_yesterday(today : pd.Series, yesterday : pd.Series, invert=False, absval=False) -> pd.Series:

    '''
    Function that returns a series containing the difference of two consecutive rows (they could be from different columns)

    Parameters:
        yesterday (pd.Series) : pandas Series object. It is the series from where we'll took the 'yesterday' data

        today (pd.Series) : pandas Series object. It is the series from where we'll took the 'today' data

        invert (boolean) : if its True, then the function will return today - yesterday. If false, it returns yesterday - today 

        absval (boolean) : if True, it returns the absolute value of the series
    
    Returns:
        A Series with the difference of today with yesterday (if invert=False). Else, it will return the difference of yesterday with today. However, if absval its True, it returns the absolute value of the difference
    '''

    #THIS CAN BE BETTER IMPLEMENTED WITH THE .shift() METHOD   

    today_data, yesterday_data = get_today_yesterday(today, yesterday) 

    if absval:
        return yesterday_data.subtract(today_data, axis=0).abs()
    else:
        if invert:
            return yesterday_data.subtract(today_data, axis=0) #returns yesterday - today
        else:
            return today_data.subtract(yesterday_data, axis=0) #returns today - yesterday
    

def get_data(client : bnc.Client, currency : str, time_interval, start_str : str, end_str=dt.date.today().strftime("%d %B, %Y"), *vars : list) -> pd.DataFrame:

    '''
    Function that returns a date frame containing the historical data of a desired currency

    Parameters:
        client (binance.Client) : Binance API Client Object

        currecny (str) : Currency string from wich we desire to obtain historical data (e.g 'BNCUSDT')

        time_interval : Could be a string (e.g '1h') or a Client Kline type (e.g binance.Client.KLINE_INTERVAL_12HOUR) 

        start_str (str) : Starting date from wich we desire to retrieve historical data (e.g '75 hours ago UTC', or a formatted date)

        end_str (str) :  Same as above. This is the ending date from wich we desire to retrieve historical data (defaulted to current date)

        *vars : list of variables we wich to retrieve. If none is given, it defaults to all the variables

    Returns:
        pandas Data Frame object
    '''

    columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote asset volume', 'Number of trades', 'Taker Buy base asset volume', 'Taker buy quote asset volume', 'Ignore'] #Possible columns names
    float_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] #Possible column names that are float types
    df = pd.DataFrame(client.get_historical_klines(currency, time_interval, start_str, end_str)) #Creates the dataframe for the desired currency

    #Try to retreive desired variables. If not possible, default to all variables
    try:
        vars = vars[0] #Vars is retrieved as an embedded list. We want the interior list
        df = df[vars] #Get the variables we are interested in
        df.set_axis([columns[variables] for variables in vars], axis=1, inplace=True) #Change name of the colums
    except:
        df.set_axis(columns, axis=1, inplace=True)

    #Change the date time from unix date time to human-readable date time
    if 'Open Time' in list(df.columns):
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    if 'Close Time' in list(df.columns):
        df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')

    #Change type of the numeric columns to float
    for col in list(df.columns):
        if col in float_cols:
            df[col] = df[col].astype(float)

    return df
    

def calculate_SMA(df : pd.DataFrame, column : str, window : int, label : str, modify=False):

    '''
    Function that returns a Series with the SMA of a desired column (or modifies an existing data frame)

    Parameters:
        df (pd.DataFrame) : DataFrame object from where the information will be extracted (it could be modified)

        window (int) : Intervals we desire to take from the closing price to calculate the mean

        label (str) : Label that we wish to use for the column to be added to the new (or the existing one) data frame

        modify (boolean) : True if we wish to modify the current data frame, False if we wish to obtain a new one (False by default)
    
    Returns:
        pandas Series object if modify is False. Else, it appends a new column to the dataframe
    '''

    #If modify is set to True, it modifies the current df. Else, it returns a new one with just the closing price and the SMA as columns
    if modify:
        df[label] = df[column].rolling(window).mean()
    else:
        return df[column].rolling(window).mean()

def calculate_EMA(df : pd.DataFrame, column : str, window : int, label : str, modify=False):
    
    '''
    pydoc here
    '''

#*******************REVISAR ESTO, PORQUE PUEDE DAR ERROR EN EL CANAL DE KELTNER*******************#


    ##We calculate the first value, wich is the SMA of the first {window} data
    #first_val = df[column][:window].mean()
#
    ##We now calculate the smoothing constant
    #smoothing_constant = 2.0/(window + 1.0)
#
    #today_data = df[column][window+1:] 
    #
    #expo_mov_avg = pd.Series([first_val] + [0.0 for _ in range(window + 1, df[column].size)])
    #expo_mov_avg.set_axis(list(range(window, df[column].size)), inplace=True)
#
    #for index in range(window + 1, df[column].size):
    #    expo_mov_avg[index] = ((today_data[index] - expo_mov_avg[index-1])*smoothing_constant) + expo_mov_avg[index-1]
    #
    if modify:
        df[label] = df[column].ewm(min_periods=window, span=window, adjust=False).mean()
        #df[label] = expo_mov_avg
    else:
        return df[column].ewm(min_periods=window, span=window, adjust=False).mean()
        #return expo_mov_avg


def calculate_DMs(df : pd.DataFrame, modify=False, processing=False):

    '''
    Function that returns a dataframe with the positive and negative directional movement (or modifies an existing dataframe)

    Parameters:
        df (pd.DataFrame) : DataFrame object from where the information will be extracted (it could be modified)

        modify (boolean) : True if we wish to modify the current data frame, False if we wish to obtain a new one (False by default)

        processing (boolean) : True if we wish to process the DMs (making 0 the ones that are negative)

    Returns:
        pandas Data Frame object if modify is False
    '''
    
    #Check if the data frame has a high and low column
    high_index, low_index = _check_cols(df, 'High', 'Low')
    HIGH = list(df.columns)[high_index]
    LOW = list(df.columns)[low_index]
    
    plus_DM = df[HIGH].diff() #calculates [HIGH - PREVIOUS HIGH]
    plus_DM = plus_DM.rename('Plus DM') #Renamed, because it keeps the name of the column it was processed from (in this case, 'High')

    minus_DM = (df[LOW] * -1).diff() #calculates [PREVIOUS LOW - LOW] (The -1 is added because df.diff() does the current row minus the previous row, so we wish to invert that, thus the -1)
    minus_DM = minus_DM.rename('Minus DM')
    
    aux_df = pd.concat([plus_DM,minus_DM], axis=1) #auxiliar df. It concatenates both of the DMs (simplifies later logic)

    if processing:
        #We only take the one that is bigger (either +DM or -DM), making zero the other one
        aux_df['Plus DM'].mask(aux_df['Plus DM'] < aux_df['Minus DM'], 0.0, inplace=True)
        aux_df['Minus DM'].mask(aux_df['Minus DM'] < aux_df['Plus DM'], 0.0, inplace=True)
        
        #If one of them is negative, we turn it into zero
        aux_df['Plus DM'].mask(aux_df['Plus DM'] < 0, 0.0, inplace=True)
        aux_df['Minus DM'].mask(aux_df['Minus DM'] < 0, 0.0, inplace=True)

        #If they are the same... (***************CHEQUEAR ESTO LUEGO CON TATA************************)
        aux_df['Plus DM'].mask(aux_df['Plus DM'] == aux_df['Minus DM'], 0.0, inplace=True)
        aux_df['Minus DM'].mask(aux_df['Minus DM'] == aux_df['Plus DM'], 0.0, inplace=True)

    if modify:
        df['+DM'] = aux_df['Plus DM']
        df['-DM'] = aux_df['Minus DM']
    else:
        aux_df.columns = ['+DM', '-DM']
        return aux_df


def calculate_TR(df : pd.DataFrame, modify=False):

    '''
    Function that returns the true range (TR) indicator given a set of data (or modifies the given dataframe).

    Parameters:
        df (pd.DataFrame) : DataFrame object from where the information will be extracted (it could be modified)

        modify (boolean) : True if we wish to modify the current data frame, False if we wish to obtain a new one (False by default)

    Returns:
        pandas Series object if modify is False
    '''

    #Check if we have the desired columns
    high_index, low_index, previos_close_index = _check_cols(df, 'High', 'Low', 'Close')
    HIGH = list(df.columns)[high_index]
    LOW = list(df.columns)[low_index]
    PREV_CLOSE = list(df.columns)[previos_close_index]

    high_minus_low = df.loc[1:, HIGH].subtract(df.loc[1:, LOW], axis=0) #returns high - low
    high_minus_prevclose = get_difference_today_yesterday(df[HIGH], df[PREV_CLOSE], invert=False, absval=True) #returns abs(high - previous close)
    low_minus_prevclose = get_difference_today_yesterday(df[LOW], df[PREV_CLOSE], invert=False, absval=True) #returns abs(low - previous close)

    aux_df = pd.concat([high_minus_low, high_minus_prevclose, low_minus_prevclose], axis=1)
    aux_df.columns = ['HIGH - LOW', 'HIGH - PREV. CLOSE', 'LOW - PREV. CLOSE']

    true_range = aux_df.max(axis=1) #returns max{(high - low), abs(high - previous close), abs(low - previous close)}

    if modify:
        df['TR'] = true_range
    else:
        return true_range


def calculate_smoothed_n(df : pd.DataFrame, label : str, n : int, modify=False):

    '''
    pydoc here
    '''
    
    #check if the df has the column we are trying to smooth
    if label not in list(df.columns):
        raise TypeError(f'DataFrame does not contains the column : "{label}"')
    else:

        first_n = df[label][:n].sum() #we stop at 'n' because pandas index starts at 0

        init_content = [first_n] + [0.0 for _ in range(n, df[label].size)]

        smoothed_n = pd.Series(init_content)
        smoothed_n.set_axis(list(range(n-1, df[label].size)), inplace=True)
        
        for index in range(n, df[label].size): #We start from n+1 because we already have the first value
            smoothed_n[index] = smoothed_n[index-1] - (smoothed_n[index-1]/n) + (df[label][index]/n)

        if modify:
            df[f'{label}{n}'] = smoothed_n
        else:
            return smoothed_n


def calculate_DIs(df : pd.DataFrame, modify=False):

    '''
    pydoc here IT CALCULATES THE DIs WITH A SMOOTH OF 14 INTERVALS

    PRE:
        The Data Frame MUST have a "TR", "+DM" and "-DM" columns
    '''

    if modify:
        #Check if we have the desired columns
        tr14_index, plusdm14_index, minusdm14_index = _check_cols(df, 'TR14', '+DM14', '-DM14')
        TR14 = list(df.columns)[tr14_index]
        PLUSDM14 = list(df.columns)[plusdm14_index]
        MINUSDM14 = list(df.columns)[minusdm14_index]
    
        df['+DI'] = (df[PLUSDM14]/df[TR14])*100
        df['-DI'] = (df[MINUSDM14]/df[TR14])*100
    else:
        tr14 = calculate_smoothed_n(df, 'TR', 14, modify=False)
        plusdm14 = calculate_smoothed_n(df, '+DM', 14, modify=False)
        minusdm14 = calculate_smoothed_n(df, '-DM', 14, modify=False)
        return (plusdm14/tr14)*100, (minusdm14/tr14)*100


def calculate_DX(df : pd.DataFrame, modify=False) -> (None | pd.Series):

    '''
    pydoc here
    '''

    if modify:
        #Check if we have the desired columns
        plusdi_index, minusdi_index = _check_cols(df, '+DI', '-DI')
        PLUSDI = list(df.columns)[plusdi_index]
        MINUSDI = list(df.columns)[minusdi_index]
    
        df['DX'] = ((df[PLUSDI] - df[MINUSDI]).abs() / (df[PLUSDI] + df[MINUSDI]).abs()) * 100
    else:
        plusdi, minusdi = calculate_DIs(df, modify=False)
        return ((plusdi - minusdi).abs() / (plusdi + minusdi).abs()) * 100


def calculate_ADX(df : pd.DataFrame , modify=False):
    
    '''
    pydoc here

    PRE:
        Data Frame df MUST have the columns: "+DM", "-DM", "TR", "+DM14", "-DM14", "TR14"
    '''
    if modify:
        #Check if the data frame has a DX column 
        dx_index = _check_cols(df, 'DX')[0]
        DX = list(df.columns)[dx_index] 

        first_adx = df[DX][14:28].sum() / 14

        adx = pd.Series([first_adx] + [0.0 for _ in range(28, df[DX].size)])
        adx.set_axis(list(range(27, df[DX].size)), inplace=True)

        adx_vals = []
        adx_vals.append(first_adx)
        
        for index in range(28, df[DX].size):
            adx[index] = ((adx[index-1]*13) + df[DX][index])/14

        df['ADX'] = adx
    else:
        dx = calculate_DX(df, modify=False)

        first_adx = dx[0:14].sum() / 14 #We slice by [0:14] because, even tho the indices are [14,15,...] we need to start from 0

        adx = pd.Series([first_adx] + [0.0 for _ in range(28, dx.size)])
        adx.set_axis(list(range(27, dx.size)), inplace=True)

        adx_vals = []
        adx_vals.append(first_adx)
        
        for index in range(28, dx.size):
            adx[index] = ((adx[index-1]*13) + dx[index])/14

        return adx
    
def calculate_bollinger_bands(df : pd.DataFrame, period=20.0, multiplier=2.0, modify=False):
    '''
    pydoc here
    '''

    #Check if we have the desired columns
    high_index, low_index, close_index = _check_cols(df, 'High', 'Low', 'Close')
    HIGH = list(df.columns)[high_index]
    LOW = list(df.columns)[low_index]
    CLOSE = list(df.columns)[close_index]

    #We first calculate the tpyical price
    #typical_price = (df[HIGH] + df[LOW] + df[CLOSE]) / 3
    typical_price = df[CLOSE] #**********************THIS FORM (INSTEAD OF THE ABOVE) HAS PROVE TO FIT BETTER WITH TRADING VIEW NUMBERS

    #Now we calculate the central Bollinger Band. That is the SMA over {period}
    medium_band = typical_price.rolling(period).mean()

    #Now, we calculate the standar deviation (sigma) of the closing price
    std_dev = typical_price.rolling(period).std()

    #Now we finally calculate the upper band and lower band
    upper_band = medium_band + (multiplier*std_dev)
    lower_band = medium_band - (multiplier*std_dev)

    if modify:
        df['Medium BB'] = medium_band
        df['Upper BB'] = upper_band
        df['Lower BB'] = lower_band
    else:
        bollinger_bands = pd.concat([medium_band, upper_band, lower_band], axis=1)
        bollinger_bands.columns = ['Medium BB', 'Upper BB', 'Lower BB']
        return bollinger_bands

def calculate_keltner_channel(df : pd.DataFrame, period=20.0, multiplier=1.5, modify=False):
    '''
    pydoc here
    '''
    #Check if we have the desired columns
    close_index = _check_cols(df, 'Close')[0]
    CLOSE = list(df.columns)[close_index]

    #We first calculate the middle channel
    medium_channel = calculate_EMA(df, CLOSE, period, 'Middle KC', modify=False)

    #Now, we calculate the ATR
    tr = pd.DataFrame()
    tr['TR'] = calculate_TR(df, modify=False)
    atr = calculate_smoothed_n(tr, 'TR', period, modify=False)

    #Finally, we calculate the upper and the lower channel
    upper_channel = medium_channel + multiplier*atr
    lower_channel = medium_channel - multiplier*atr

    if modify:
        df['Medium KC'] = medium_channel
        df['Upper KC'] = upper_channel
        df['Lower KC'] = lower_channel
    else:
        keltner_channel = pd.concat([medium_channel, upper_channel, lower_channel], axis=1)
        keltner_channel.columns = ['Medium KC', 'Upper KC', 'Lower KC']
        return keltner_channel

def calculate_squeeze(df : pd.DataFrame, modify=False):
    '''
    pydoc here
    '''
    #First, we calculate the Bollinger bands and the Keltner channel
    bollinger_bands = calculate_bollinger_bands(df, period=20.0, multiplier=2.0, modify=False)
    keltner_channel = calculate_keltner_channel(df, period=20.0, multiplier=1.5, modify=False)

    #sqzOn  = (lowerBB > lowerKC) and (upperBB < upperKC)
    #sqzOff = (lowerBB < lowerKC) and (upperBB > upperKC)
    #noSqz  = (sqzOn == false) and (sqzOff == false)


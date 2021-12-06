#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
#import yfinance as yf
import matplotlib.pyplot as plt
import mlfinlab  as ml
from datetime import datetime,date
import datetime as dt
import MetaTrader5 as mt5
#from scipy.signal import savgol_filter
#from tsmoothie.utils_func import sim_randomwalk
from tsmoothie import smoother
from pyti import money_flow_index


# In[25]:


# connect to MetaTrader 5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()
 
# request connection status and parameters
print(mt5.terminal_info())
# get data on MetaTrader 5 version
print(mt5.version())


# # Read from Metatrader 

# In[33]:


period_H1=25
period_H2=25
period_H3=88
period_D1=25
period_D2=43
symbol='SAFDY00'


# In[34]:


#d1=pd.read_csv(r'C:\Users\LEnovo pc\Downloads\اپالDaily.csv', encoding = 'utf-16',names=['Date','open','high','low','close','tick_vol','vol'],index_col=0)
rat2 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 500)
d_DAILY=pd.DataFrame()
d_DAILY = pd.DataFrame(rat2)
d_DAILY['time']=pd.to_datetime(d_DAILY['time'], unit='s')
d_DAILY.set_index('time',inplace=True)
#print(d_DAILY[1:])
###########################################################################3
rat = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 1000)
d_H=pd.DataFrame()
d_H = pd.DataFrame(rat)
d_H['time']=pd.to_datetime(d_H['time'], unit='s')
d_H.set_index('time',inplace=True)
##########################################################
sp_close=d_H.close[-1000:]
sp_close.index=pd.to_datetime(sp_close.index[0:])
print(sp_close[-10:])


# In[35]:


daily_vol = ml.util.get_daily_vol(close=sp_close, lookback=int(period_H1))
#print(daily_vol)

# Apply Symmetric CUSUM Filter and get timestamps for events
# Note: Only the CUSUM filter needs a point estimate for volatility
cusum_events = ml.filters.cusum_filter(sp_close,
                                       threshold=daily_vol.mean())
#print(cusum_events)

# Compute vertical barrier
vertical_barriers = ml.labeling.add_vertical_barrier(t_events=cusum_events,
                                                     close=sp_close,
                                                     num_days=int(period_H1/2))
#print((vertical_barriers))
#fig,ax=plt.subplots(figsize=(8,6),dpi=300)
#ax.plot(sp_close)
#ax.scatter(cusum_events,sp_close[cusum_events],c='r')
#########################################################################

pt_sl = [1, 2.618]
#print(daily_vol.mean())
min_ret = daily_vol.mean()/1.618
triple_barrier_events = ml.labeling.get_events(close=sp_close,
                                               t_events=cusum_events,
                                               pt_sl=pt_sl,
                                               target=daily_vol,
                                               min_ret=min_ret,
                                               num_threads=2,
                                               vertical_barrier_times=vertical_barriers,
                                               )

print(triple_barrier_events[-10:])
#print(daily_vol[-10:])
#print(daily_vol.mean())


# In[36]:


meta_labels = ml.labeling.get_bins(triple_barrier_events, sp_close)
meta_labels[['t1']]=triple_barrier_events.t1
meta_labels.insert(4, "Exit_Price", 0, True)
meta_labels["Exit_Price"].iloc[-1]=d_H['close'].loc[meta_labels['t1'].iloc[-1]]
print(meta_labels[-3:])


# In[37]:


def EVB(price,period):
    ################################
    sp_close_D=pd.Series()
    sp_close_D=price.close
    sp_close_D.index=pd.to_datetime(sp_close_D.index[0:])
    #print(sp_close_D[-10:])
    sp_close_BD=np.log10(sp_close_D)
    #print(sp_close_BD)
    ############################3####
    data1=np.array(sp_close_BD)
    #print(data1[-10:])
    alpha1=(1-np.sin(2*np.pi/period))/np.cos(2*np.pi/period)
    a1=np.exp(-1.414*np.pi/10)
    b1=2*a1*np.cos(1.414*np.pi/10)
    c2=b1
    c3=-a1*a1
    c1=1-c2-c3
    hp=np.zeros((data1.shape[0],1))
    filt=np.zeros((data1.shape[0],1))
    wave=0
    pwr=0
    signal=np.zeros((data1.shape[0],1))
    for i in range(2,data1.shape[0]):
        hp[i]=0.5*(1+alpha1)*(data1[i]-data1[i-1])+alpha1*hp[i-1]
        filt[i]=c1*0.5*(hp[i]+hp[i-1])+c2*filt[i-1]+c3*filt[i-2]
        wave=(filt[i]+filt[i-1]+filt[i-2])/3
        pwr=np.sqrt((filt[i]*filt[i]+filt[i-1]*filt[i-1]+filt[i-2]*filt[i-2])/3)
        signal[i]=wave/pwr
    
    signal=np.nan_to_num(signal)
    price['signal']=signal
    smoother1 = smoother.KalmanSmoother(component='level_trend',component_noise={'level':1/period, 'trend':1/period})
    #ExponentialSmoother(window_len=period, alpha=0.3)
    #BinnerSmoother(n_knots=int(period))
    #SpectralSmoother(smooth_fraction=0.1, pad_len=period)
    #LowessSmoother(smooth_fraction=1/period, iterations=5)
    smoother1.smooth(signal)
    signal_SM=smoother1.smooth_data.T
    #print(signal_SM.shape)
    #print(signal_SM.shape)
    price['signal_SM']=signal_SM
    #price['signal_SM']=savgol_filter(price['signal'], int(period_s), order)
    price['signal_yestrday']=price['signal_SM'].shift(1)
    #print(price[-10:])
    return price


# In[38]:


def BPF(price,period):
    ################################
    sp_close_D=price.close
    sp_close_D.index=pd.to_datetime(sp_close_D.index[0:])
    #print(sp_close_D[-10:])
    sp_close_BD=np.log10(sp_close_D)
    #print(sp_close_BD)
    data1=sp_close_BD
    ############################3####
    bandwidth=0.33;
    alpha1= (np.cos(0.25*2*np.pi*bandwidth/period)+np.sin(0.25*2*np.pi*bandwidth/period)-1)/np.cos(0.25*2*np.pi*bandwidth/period)
    hp1=np.zeros((data1.shape[0],1))
    BP=np.zeros((data1.shape[0],1))
    peak=np.zeros((data1.shape[0],1))
    signal_BP=np.zeros((data1.shape[0],1))
    beta1=np.cos(2*np.pi/period)
    gama1=1/np.cos(2*np.pi*bandwidth/period)
    alpha2=gama1-np.sqrt(gama1**2-1)

    for i in range(2,data1.shape[0]):
        hp1[i]=(1-alpha1/2)*(data1[i]-data1[i-1])+(1-alpha1)*hp1[i-1]
    for i in range(2,data1.shape[0]):
        BP[i]=0.5*(1-alpha2/2)*(hp1[i]-hp1[i-2])+beta1*(1+alpha2)*BP[i-1]-alpha2*BP[i-2]
    for i in range(2,data1.shape[0]):
        peak[i]=0.991*peak[i-1];
        if np.abs(BP[i])>peak[i]:
            peak[i]=np.abs(BP[i])
        signal_BP[i]=BP[i]/peak[i]
        #print(signal)
    signal_BP=np.nan_to_num(signal_BP)
    
    price['signal_BP']=signal_BP
    price['BP']=BP
    if period>len(sp_close_D):
        period=len(sp_close_D)
    smoother1 = smoother.BinnerSmoother(n_knots=int(period))
    #KalmanSmoother(component='level_trend',component_noise={'level':1/period, 'trend':1/period})
    #ExponentialSmoother(window_len=period, alpha=0.3)
    #BinnerSmoother(n_knots=int(period))
    #SpectralSmoother(smooth_fraction=0.1, pad_len=period)
    #LowessSmoother(smooth_fraction=1/period, iterations=5)
    smoother1.smooth(signal_BP)
    signal_SM=smoother1.smooth_data.T
    #print(signal_SM.shape)
    #print(signal_SM.shape)
    price['signal_SM']=signal_SM
    #price['signal_SM']=savgol_filter(price['signal'], int(period_s), order)
    price['signal_yestrday']=price['signal_SM'].shift(1)
    #print(price[-10:])
    return price


# In[39]:


is_NaN = triple_barrier_events.isnull()
row_has_NaN = is_NaN. any(axis=1)
Trade_signal= triple_barrier_events[row_has_NaN].copy()
#print(Trade_signal)
Trade_signal.insert(4, "Action", 0, True)
Trade_signal.insert(5, "Price", 0, True)
Trade_signal.insert(6, "TP", 0, True)
Trade_signal.insert(7, "SL", 0, True)

#print(meta_labels[-32:])
for index1,row  in Trade_signal.iterrows():
    #print(index1)
    index_d=index1.replace(hour=0)
    dR_H2=pd.DataFrame()
    dR_H3=pd.DataFrame()
    dR_D1=pd.DataFrame()
    dR_H3_BPF=pd.DataFrame()
    dR_D2=pd.DataFrame()
    dR_H2=d_H.loc[:index1].copy()
    dR_H2=EVB(dR_H2,int(period_H2))
    dR_H3=d_H.loc[:index1].copy()
    dR_H3=EVB(dR_H3,int(period_H3))
    dR_H_BPF=d_H.loc[:index1].copy()
    dR_H_BPF=BPF(dR_H_BPF,int(period_H3))
    dR_D1=d_DAILY.loc[:index_d].copy()
    dR_D1.loc[index_d,['close']]=dR_H2.loc[index1,['close']]
    dR_D1=EVB(dR_D1,int(period_D1))
    dR_D2=d_DAILY.loc[:index_d].copy()
    dR_D2.loc[index_d,['close']]=dR_H2.loc[index1,['close']]
    dR_D2=EVB(dR_D2,int(period_D2))
    
    #print(index1)
    #index_next=index1+dt.timedelta( hours=1)
    #index_bef=index1+dt.timedelta( hours=-1)
    
    
    if dR_D1['signal_SM'].loc[index_d]>=0.9 and dR_D2['signal_SM'].loc[index_d]>=0.9:
        
        if dR_H2['signal_SM'].loc[index1]>= 0.9 and dR_H3['signal_SM'].loc[index1]>= 0.9 and dR_H_BPF['signal_SM'].loc[index1]>= 0.9  : 
            Trade_signal['Action'].loc[index1]=1
        if dR_H2['signal_SM'].loc[index1]>= 0.9 and dR_H3['signal_SM'].loc[index1]>dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= 0.9 :
            Trade_signal['Action'].loc[index1]=1
        if dR_H3['signal_SM'].loc[index1]>= 0.9 and dR_H2['signal_SM'].loc[index1]>dR_H2['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= 0.9 :
            Trade_signal['Action'].loc[index1]=1
        if dR_H2['signal_SM'].loc[index1]>dR_H2['signal_yestrday'].loc[index1]  and dR_H3['signal_SM'].loc[index1]>dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= 0.9:
            Trade_signal['Action'].loc[index1]=1
                  
        if dR_H2['signal_SM'].loc[index1]>= 0.9 and dR_H3['signal_SM'].loc[index1]>= 0.9 and dR_H_BPF['signal_SM'].loc[index1]>= dR_H_BPF['signal_yestrday'].loc[index1]  : 
            Trade_signal['Action'].loc[index1]=1
        if dR_H2['signal_SM'].loc[index1]>= 0.9 and dR_H3['signal_SM'].loc[index1]>dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= dR_H_BPF['signal_yestrday'].loc[index1] :
            Trade_signal['Action'].loc[index1]=1
        if dR_H3['signal_SM'].loc[index1]>= 0.9 and dR_H2['signal_SM'].loc[index1]>dR_H2['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= dR_H_BPF['signal_yestrday'].loc[index1] :
            Trade_signal['Action'].loc[index1]=1
        if dR_H2['signal_SM'].loc[index1]>dR_H2['signal_yestrday'].loc[index1]  and dR_H3['signal_SM'].loc[index1]>dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= dR_H_BPF['signal_yestrday'].loc[index1]:
            Trade_signal['Action'].loc[index1]=1          
    if dR_D1['signal_SM'].loc[index_d]>=0.9 and dR_D2['signal_SM'].loc[index_d]>dR_D2['signal_yestrday'].loc[index_d] :
        if dR_H2['signal_SM'].loc[index1]>= 0.9 and dR_H3['signal_SM'].loc[index1]>= 0.9 and dR_H_BPF['signal_SM'].loc[index1]>= 0.9  : 
            Trade_signal['Action'].loc[index1]=1
        if dR_H2['signal_SM'].loc[index1]>= 0.9 and dR_H3['signal_SM'].loc[index1]>dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= 0.9 :
            Trade_signal['Action'].loc[index1]=1
        if dR_H3['signal_SM'].loc[index1]>= 0.9 and dR_H2['signal_SM'].loc[index1]>dR_H2['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= 0.9 :
            Trade_signal['Action'].loc[index1]=1
        if dR_H2['signal_SM'].loc[index1]>dR_H2['signal_yestrday'].loc[index1]  and dR_H3['signal_SM'].loc[index1]>dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= 0.9:
            Trade_signal['Action'].loc[index1]=1
                  
        if dR_H2['signal_SM'].loc[index1]>= 0.9 and dR_H3['signal_SM'].loc[index1]>= 0.9 and dR_H_BPF['signal_SM'].loc[index1]>= dR_H_BPF['signal_yestrday'].loc[index1]  : 
            Trade_signal['Action'].loc[index1]=1
        if dR_H2['signal_SM'].loc[index1]>= 0.9 and dR_H3['signal_SM'].loc[index1]>dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= dR_H_BPF['signal_yestrday'].loc[index1] :
            Trade_signal['Action'].loc[index1]=1
        if dR_H3['signal_SM'].loc[index1]>= 0.9 and dR_H2['signal_SM'].loc[index1]>dR_H2['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= dR_H_BPF['signal_yestrday'].loc[index1] :
            Trade_signal['Action'].loc[index1]=1
        if dR_H2['signal_SM'].loc[index1]>dR_H2['signal_yestrday'].loc[index1]  and dR_H3['signal_SM'].loc[index1]>dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= dR_H_BPF['signal_yestrday'].loc[index1]:
            Trade_signal['Action'].loc[index1]=1   
    if dR_D1['signal_SM'].loc[index_d]>dR_D1['signal_yestrday'].loc[index_d]  and dR_D2['signal_SM'].loc[index_d]>=0.9 :
        if dR_H2['signal_SM'].loc[index1]>= 0.9 and dR_H3['signal_SM'].loc[index1]>= 0.9 and dR_H_BPF['signal_SM'].loc[index1]>= 0.9  : 
            Trade_signal['Action'].loc[index1]=1
        if dR_H2['signal_SM'].loc[index1]>= 0.9 and dR_H3['signal_SM'].loc[index1]>dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= 0.9 :
            Trade_signal['Action'].loc[index1]=1
        if dR_H3['signal_SM'].loc[index1]>= 0.9 and dR_H2['signal_SM'].loc[index1]>dR_H2['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= 0.9 :
            Trade_signal['Action'].loc[index1]=1
        if dR_H2['signal_SM'].loc[index1]>dR_H2['signal_yestrday'].loc[index1]  and dR_H3['signal_SM'].loc[index1]>dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= 0.9:
            Trade_signal['Action'].loc[index1]=1
                  
        if dR_H2['signal_SM'].loc[index1]>= 0.9 and dR_H3['signal_SM'].loc[index1]>= 0.9 and dR_H_BPF['signal_SM'].loc[index1]>= dR_H_BPF['signal_yestrday'].loc[index1]  : 
            Trade_signal['Action'].loc[index1]=1
        if dR_H2['signal_SM'].loc[index1]>= 0.9 and dR_H3['signal_SM'].loc[index1]>dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= dR_H_BPF['signal_yestrday'].loc[index1] :
            Trade_signal['Action'].loc[index1]=1
        if dR_H3['signal_SM'].loc[index1]>= 0.9 and dR_H2['signal_SM'].loc[index1]>dR_H2['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= dR_H_BPF['signal_yestrday'].loc[index1] :
            Trade_signal['Action'].loc[index1]=1
        if dR_H2['signal_SM'].loc[index1]>dR_H2['signal_yestrday'].loc[index1]  and dR_H3['signal_SM'].loc[index1]>dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= dR_H_BPF['signal_yestrday'].loc[index1]:
            Trade_signal['Action'].loc[index1]=1  
    if dR_D1['signal_SM'].loc[index_d]>dR_D1['signal_yestrday'].loc[index_d]  and dR_D2['signal_SM'].loc[index_d]>dR_D2['signal_yestrday'].loc[index_d] :
        if dR_H2['signal_SM'].loc[index1]>= 0.9 and dR_H3['signal_SM'].loc[index1]>= 0.9 and dR_H_BPF['signal_SM'].loc[index1]>= 0.9  : 
            Trade_signal['Action'].loc[index1]=1
        if dR_H2['signal_SM'].loc[index1]>= 0.9 and dR_H3['signal_SM'].loc[index1]>dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= 0.9 :
            Trade_signal['Action'].loc[index1]=1
        if dR_H3['signal_SM'].loc[index1]>= 0.9 and dR_H2['signal_SM'].loc[index1]>dR_H2['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= 0.9 :
            Trade_signal['Action'].loc[index1]=1
        if dR_H2['signal_SM'].loc[index1]>dR_H2['signal_yestrday'].loc[index1]  and dR_H3['signal_SM'].loc[index1]>dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= 0.9:
            Trade_signal['Action'].loc[index1]=1
                  
        if dR_H2['signal_SM'].loc[index1]>= 0.9 and dR_H3['signal_SM'].loc[index1]>= 0.9 and dR_H_BPF['signal_SM'].loc[index1]>= dR_H_BPF['signal_yestrday'].loc[index1]  : 
            Trade_signal['Action'].loc[index1]=1
        if dR_H2['signal_SM'].loc[index1]>= 0.9 and dR_H3['signal_SM'].loc[index1]>dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= dR_H_BPF['signal_yestrday'].loc[index1] :
            Trade_signal['Action'].loc[index1]=1
        if dR_H3['signal_SM'].loc[index1]>= 0.9 and dR_H2['signal_SM'].loc[index1]>dR_H2['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= dR_H_BPF['signal_yestrday'].loc[index1] :
            Trade_signal['Action'].loc[index1]=1
        if dR_H2['signal_SM'].loc[index1]>dR_H2['signal_yestrday'].loc[index1]  and dR_H3['signal_SM'].loc[index1]>dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]>= dR_H_BPF['signal_yestrday'].loc[index1]:
            Trade_signal['Action'].loc[index1]=1  
###############################################################################################################################
    if dR_D1['signal_SM'].loc[index_d]<=-0.9 and dR_D2['signal_SM'].loc[index_d]<=-0.9 :
        if dR_H2['signal_SM'].loc[index1]<= -0.9 and dR_H3['signal_SM'].loc[index1]<= -0.9 and dR_H_BPF['signal_SM'].loc[index1]<=-0.9: #or d1['signal'].loc[index1]<d1['signal'].loc[index1]
            Trade_signal['Action'].loc[index1]=-1
        if dR_H2['signal_SM'].loc[index1]<= -0.9 and dR_H3['signal_SM'].loc[index1]<dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=-0.9 :
            Trade_signal['Action'].loc[index1]=-1
        if dR_H3['signal_SM'].loc[index1]<= -0.9 and dR_H2['signal_SM'].loc[index1]<dR_H2['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=-0.9 :
            Trade_signal['Action'].loc[index1]=-1
        if dR_H2['signal_SM'].loc[index1]<dR_H2['signal_yestrday'].loc[index1] and dR_H3['signal_SM'].loc[index1]<dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=-0.9 :
            Trade_signal['Action'].loc[index1]=-1
                  
        if dR_H2['signal_SM'].loc[index1]<= -0.9 and dR_H3['signal_SM'].loc[index1]<= -0.9 and dR_H_BPF['signal_SM'].loc[index1]<=dR_H_BPF['signal_yestrday'].loc[index1]: #or d1['signal'].loc[index1]<d1['signal'].loc[index1]
            Trade_signal['Action'].loc[index1]=-1
        if dR_H2['signal_SM'].loc[index1]<= -0.9 and dR_H3['signal_SM'].loc[index1]<dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=dR_H_BPF['signal_yestrday'].loc[index1] :
            Trade_signal['Action'].loc[index1]=-1
        if dR_H3['signal_SM'].loc[index1]<= -0.9 and dR_H2['signal_SM'].loc[index1]<dR_H2['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=dR_H_BPF['signal_yestrday'].loc[index1] :
            Trade_signal['Action'].loc[index1]=-1
        if dR_H2['signal_SM'].loc[index1]<dR_H2['signal_yestrday'].loc[index1] and dR_H3['signal_SM'].loc[index1]<dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=dR_H_BPF['signal_yestrday'].loc[index1] :
            Trade_signal['Action'].loc[index1]=-1
    if dR_D1['signal_SM'].loc[index_d]<=-0.9 and dR_D2['signal_SM'].loc[index_d]<dR_D2['signal_yestrday'].loc[index_d]:
        if dR_H2['signal_SM'].loc[index1]<= -0.9 and dR_H3['signal_SM'].loc[index1]<= -0.9 and dR_H_BPF['signal_SM'].loc[index1]<=-0.9: #or d1['signal'].loc[index1]<d1['signal'].loc[index1]
            Trade_signal['Action'].loc[index1]=-1
        if dR_H2['signal_SM'].loc[index1]<= -0.9 and dR_H3['signal_SM'].loc[index1]<dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=-0.9 :
            Trade_signal['Action'].loc[index1]=-1
        if dR_H3['signal_SM'].loc[index1]<= -0.9 and dR_H2['signal_SM'].loc[index1]<dR_H2['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=-0.9 :
            Trade_signal['Action'].loc[index1]=-1
        if dR_H2['signal_SM'].loc[index1]<dR_H2['signal_yestrday'].loc[index1] and dR_H3['signal_SM'].loc[index1]<dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=-0.9 :
            Trade_signal['Action'].loc[index1]=-1
                  
        if dR_H2['signal_SM'].loc[index1]<= -0.9 and dR_H3['signal_SM'].loc[index1]<= -0.9 and dR_H_BPF['signal_SM'].loc[index1]<=dR_H_BPF['signal_yestrday'].loc[index1]: #or d1['signal'].loc[index1]<d1['signal'].loc[index1]
            Trade_signal['Action'].loc[index1]=-1
        if dR_H2['signal_SM'].loc[index1]<= -0.9 and dR_H3['signal_SM'].loc[index1]<dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=dR_H_BPF['signal_yestrday'].loc[index1] :
            Trade_signal['Action'].loc[index1]=-1
        if dR_H3['signal_SM'].loc[index1]<= -0.9 and dR_H2['signal_SM'].loc[index1]<dR_H2['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=dR_H_BPF['signal_yestrday'].loc[index1] :
            Trade_signal['Action'].loc[index1]=-1
        if dR_H2['signal_SM'].loc[index1]<dR_H2['signal_yestrday'].loc[index1] and dR_H3['signal_SM'].loc[index1]<dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=dR_H_BPF['signal_yestrday'].loc[index1] :
            Trade_signal['Action'].loc[index1]=-1
    
    if dR_D1['signal_SM'].loc[index_d]<dR_D1['signal_yestrday'].loc[index_d] and dR_D2['signal_SM'].loc[index_d]<=-0.9 :
        if dR_H2['signal_SM'].loc[index1]<= -0.9 and dR_H3['signal_SM'].loc[index1]<= -0.9 and dR_H_BPF['signal_SM'].loc[index1]<=-0.9: #or d1['signal'].loc[index1]<d1['signal'].loc[index1]
            Trade_signal['Action'].loc[index1]=-1
        if dR_H2['signal_SM'].loc[index1]<= -0.9 and dR_H3['signal_SM'].loc[index1]<dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=-0.9 :
            Trade_signal['Action'].loc[index1]=-1
        if dR_H3['signal_SM'].loc[index1]<= -0.9 and dR_H2['signal_SM'].loc[index1]<dR_H2['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=-0.9 :
            Trade_signal['Action'].loc[index1]=-1
        if dR_H2['signal_SM'].loc[index1]<dR_H2['signal_yestrday'].loc[index1] and dR_H3['signal_SM'].loc[index1]<dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=-0.9 :
            Trade_signal['Action'].loc[index1]=-1
                  
        if dR_H2['signal_SM'].loc[index1]<= -0.9 and dR_H3['signal_SM'].loc[index1]<= -0.9 and dR_H_BPF['signal_SM'].loc[index1]<=dR_H_BPF['signal_yestrday'].loc[index1]: #or d1['signal'].loc[index1]<d1['signal'].loc[index1]
            Trade_signal['Action'].loc[index1]=-1
        if dR_H2['signal_SM'].loc[index1]<= -0.9 and dR_H3['signal_SM'].loc[index1]<dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=dR_H_BPF['signal_yestrday'].loc[index1] :
            Trade_signal['Action'].loc[index1]=-1
        if dR_H3['signal_SM'].loc[index1]<= -0.9 and dR_H2['signal_SM'].loc[index1]<dR_H2['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=dR_H_BPF['signal_yestrday'].loc[index1] :
            Trade_signal['Action'].loc[index1]=-1
        if dR_H2['signal_SM'].loc[index1]<dR_H2['signal_yestrday'].loc[index1] and dR_H3['signal_SM'].loc[index1]<dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=dR_H_BPF['signal_yestrday'].loc[index1] :
            Trade_signal['Action'].loc[index1]=-1
    if dR_D1['signal_SM'].loc[index_d]<dR_D1['signal_yestrday'].loc[index_d]  and dR_D2['signal_SM'].loc[index_d]<dR_D2['signal_yestrday'].loc[index_d]  :
        if dR_H2['signal_SM'].loc[index1]<= -0.9 and dR_H3['signal_SM'].loc[index1]<= -0.9 and dR_H_BPF['signal_SM'].loc[index1]<=-0.9: #or d1['signal'].loc[index1]<d1['signal'].loc[index1]
            Trade_signal['Action'].loc[index1]=-1
        if dR_H2['signal_SM'].loc[index1]<= -0.9 and dR_H3['signal_SM'].loc[index1]<dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=-0.9 :
            Trade_signal['Action'].loc[index1]=-1
        if dR_H3['signal_SM'].loc[index1]<= -0.9 and dR_H2['signal_SM'].loc[index1]<dR_H2['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=-0.9 :
            Trade_signal['Action'].loc[index1]=-1
        if dR_H2['signal_SM'].loc[index1]<dR_H2['signal_yestrday'].loc[index1] and dR_H3['signal_SM'].loc[index1]<dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=-0.9 :
            Trade_signal['Action'].loc[index1]=-1
                  
        if dR_H2['signal_SM'].loc[index1]<= -0.9 and dR_H3['signal_SM'].loc[index1]<= -0.9 and dR_H_BPF['signal_SM'].loc[index1]<=dR_H_BPF['signal_yestrday'].loc[index1]: #or d1['signal'].loc[index1]<d1['signal'].loc[index1]
            Trade_signal['Action'].loc[index1]=-1
        if dR_H2['signal_SM'].loc[index1]<= -0.9 and dR_H3['signal_SM'].loc[index1]<dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=dR_H_BPF['signal_yestrday'].loc[index1] :
            Trade_signal['Action'].loc[index1]=-1
        if dR_H3['signal_SM'].loc[index1]<= -0.9 and dR_H2['signal_SM'].loc[index1]<dR_H2['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=dR_H_BPF['signal_yestrday'].loc[index1] :
            Trade_signal['Action'].loc[index1]=-1
        if dR_H2['signal_SM'].loc[index1]<dR_H2['signal_yestrday'].loc[index1] and dR_H3['signal_SM'].loc[index1]<dR_H3['signal_yestrday'].loc[index1] and dR_H_BPF['signal_SM'].loc[index1]<=dR_H_BPF['signal_yestrday'].loc[index1] :
            Trade_signal['Action'].loc[index1]=-1
for index2,row1  in Trade_signal.iterrows():
    if row1.Action!=0:
        Trade_signal.loc[index2,['Price']]=d_H['close'].loc[index2]
        #print("Price="+str(d1['close'].loc[index2]))
        #print(pt_sl[0])
        Trade_signal.loc[index2,['TP']]=round((d_H['close'].loc[index2]+d_H['close'].loc[index2]*pt_sl[1]*row1.trgt*row1.Action*1)/100)*100
        #print("TP="+str(round(TP1)))
        Trade_signal.loc[index2,['SL']]=round((d_H['close'].loc[index2]-d_H['close'].loc[index2]*pt_sl[0]*row1.trgt*row1.Action*1)/100)*100
print(d_H['close'].iloc[-1:])
print(Trade_signal[-3:])
print(meta_labels[-3:])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





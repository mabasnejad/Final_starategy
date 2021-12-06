#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import mlfinlab  as ml
from datetime import datetime,date
import datetime as dt
import MetaTrader5 as mt5
from scipy.signal import savgol_filter


# In[2]:


# connect to MetaTrader 5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()
# request connection status and parameters
print(mt5.terminal_info())
# get data on MetaTrader 5 version
print(mt5.version())


# # Read from Metatrader 

# In[3]:


period=44
period_d=int(27)
order=3
order_EV=2
order_BP=2
symbol='SAFDY00'
#d1=pd.read_csv(r'C:\Users\LEnovo pc\Downloads\اپالDaily.csv', encoding = 'utf-16',names=['Date','open','high','low','close','tick_vol','vol'],index_col=0)


# In[4]:


rat2 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 500)
d2 = pd.DataFrame(rat2)
d2['time']=pd.to_datetime(d2['time'], unit='s')
d2.set_index('time',inplace=True)
#print(d2[-10:])
#######################################33
sp2_close_D=d2.close[-500:]
sp_close_D=sp2_close_D[-500:]
sp_close_D.index=pd.to_datetime(sp_close_D.index[0:])
sp2_close_D.index=pd.to_datetime(sp2_close_D.index[0:])
#sp_close=np.log10(sp_close)
print(sp2_close_D[-10:])
########################################################################################
#d1=pd.read_csv(r'C:\Users\LEnovo pc\Downloads\اپالDaily.csv', encoding = 'utf-16',names=['Date','open','high','low','close','tick_vol','vol'],index_col=0)
rat = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 500)
d1 = pd.DataFrame(rat)
d1['time']=pd.to_datetime(d1['time'], unit='s')
d1.set_index('time',inplace=True)
##########################################################
#d1=pd.read_csv(r'C:\Users\LEnovo pc\Downloads\SAFSH00H1.csv', encoding = 'utf-16',names=['Date','open','high','low','close','tick_vol','vol'],index_col=0)
#print(d1[-10:])
sp2_close=d1.close[-500:]
sp_close=sp2_close[-500:]
sp_close.index=pd.to_datetime(sp_close.index[0:])
sp2_close.index=pd.to_datetime(sp2_close.index[0:])
print(sp2_close[-10:])
#print(d1.index[-1])


# In[5]:


daily_vol = ml.util.get_daily_vol(close=sp_close, lookback=period)
#print(daily_vol)

# Apply Symmetric CUSUM Filter and get timestamps for events
# Note: Only the CUSUM filter needs a point estimate for volatility
cusum_events = ml.filters.cusum_filter(sp_close,
                                       threshold=daily_vol.mean())
#print(cusum_events)
# Compute vertical barrier
vertical_barriers = ml.labeling.add_vertical_barrier(t_events=cusum_events,
                                                     close=sp_close,
                                                     num_days=int(period/2))
#print((vertical_barriers))
#ax.scattter(cusum_events,sp_close[cusum_events])
#fig,ax=plt.subplots(figsize=(8,6),dpi=300)
#ax.plot(sp_close)
#ax2=ax.twinx()
#ax.scatter(cusum_events,sp_close[cusum_events],c='r')
##############################################################################################
pt_sl = [1, 1]
print(daily_vol.mean())
min_ret = daily_vol.mean()/1.618
triple_barrier_events = ml.labeling.get_events(close=sp_close,
                                               t_events=cusum_events,
                                               pt_sl=pt_sl,
                                               target=daily_vol,
                                               min_ret=min_ret,
                                               num_threads=2,
                                               vertical_barrier_times=vertical_barriers,
                                               )

print(triple_barrier_events[-20:])

#print(daily_vol[-10:])
print(daily_vol.mean())
###############################################################################################################
meta_labels = ml.labeling.get_bins(triple_barrier_events, sp_close)
meta_labels[['t1']]=triple_barrier_events.t1
#print(meta_labels[-32:])
meta_labels=meta_labels[meta_labels.index>d1.index[period_d*5]]
print(meta_labels[-12:])


# # EVB

# In[6]:


def EVB(price,period,order,period_s):
    ################################
    sp_close_D=price.close[-500:]
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
    if (period_s%2==0):
        period_s=period_s+1
    price['signal']=signal
    price['signal_SM']=savgol_filter(price['signal'], int(period_s), order)
    price['signal_yestrday']=price['signal_SM'].shift(1)
    #print(price[-10:])
    return price


# # EVB_Daily

# In[7]:


d2=EVB(d2,int(period_d),order_EV,int(period_d))
#print(d2[-10:])
##############################################
fig , ax1 =plt.subplots(figsize=(20,15),dpi=100)
#print(d1.index)
ax1.plot(d2['signal_SM'])            
ax1.set_ylabel("Even_Better_Sin")
plt.axhline(0.9, color='g',ls='--')
plt.axhline(-0.9, color='g',ls='--')
ax2 = ax1.twinx()
ax2.plot(d2['close'],'r')
ax2.set_ylabel("Price")
ax2.set_yscale('log')


# # EVB_H1

# In[8]:


d1=EVB(d1,period,order,int(period))
#print(d1[-10:])
#########################################################
fig , ax1 =plt.subplots(figsize=(20,15),dpi=100)
#print(d1.index)
ax1.plot(d1['signal_SM'])            
ax1.set_ylabel("Even_Better_Sin")
plt.axhline(0.9, color='g',ls='--')
plt.axhline(-0.9, color='g',ls='--')
ax2 = ax1.twinx()
ax2.plot(d1['close'],'r')
ax2.set_ylabel("Price")
ax2.set_yscale('log')


# # BPF_DAILY

# In[9]:


def BPF(price,period,order_BP,period_s):
    sp_close_D=price.close[-500:]
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
    if (period_s%2==0):
        period_s=period_s+1
    price['signal_BP']=signal_BP
    price['BP']=BP
    price['signal_BP_SM']=savgol_filter(price['signal_BP'], int(period_s), order)
    
    price['BP_yestrday']=price['signal_BP_SM'].shift(1)
    return price


# In[10]:


d2=BPF(d2,period_d,order_BP,int(period_d))
#print(d2[-10:])
############################################
#print(d1.iloc[BP_Peak])
#print(BP_Valley)
#print(BP_Peak)
#####################################################################
fig , ax1 =plt.subplots(figsize=(25,20),dpi=100)
ax1.plot(d2['signal_BP_SM'])            
ax1.set_ylabel("Band pass filter")
plt.axhline(1.01, color='g',ls='--')
plt.axhline(-1.01, color='g',ls='--')
plt.axhline(0.0, color='g',ls='--')
ax2 = ax1.twinx()
ax2.plot(d2['close'],'r')
ax2.set_ylabel("Price")
ax2.set_yscale('log')
locs, labels=plt.xticks()
#print(locs)
n_locs=locs[::50]
plt.xticks(n_locs,rotation='vertical')
ax1.tick_params(axis='x',direction='out', length=4, width=1,
               grid_alpha=0.1,labelrotation=90)

######################################################
fig2 , ax3 =plt.subplots(figsize=(25,20),dpi=100)
ax3.plot(d2['BP'])
#ax3.plot(BP_Peak,BP[BP_Peak],"v",color='r',markersize=15)
#ax3.plot(BP_Valley,BP[BP_Valley],"^",color='g',markersize=15)
ax3.set_ylabel("Band pass filter")
#ax3.plot(d1.index[],d1[BP_Peak])
#plt.axhline(0.9, color='g',ls='--')
#plt.axhline(-0.9, color='g',ls='--')
#plt.axhline(0.9, color='g',ls='--')
plt.axhline(0.0, color='g',ls='--')
ax4 = ax3.twinx()
#ax4.plot(d1['close'],'r')
#ax4.plot(d1.close.iloc[BP_Valley],"^",color='g',markersize=15)
#ax4.plot(d1.close.iloc[BP_Peak],"^",color='r',markersize=15)
#ax4.set_ylabel("Price")
#ax4.set_yscale('log')


# In[11]:


is_NaN = triple_barrier_events.isnull()
row_has_NaN = is_NaN. any(axis=1)
Trade_signal= triple_barrier_events[row_has_NaN]
#print(Trade_signal)
Trade_signal.insert(4, "Action", 0, True)
Trade_signal.insert(5, "Price", 0, True)
Trade_signal.insert(6, "TP", 0, True)
Trade_signal.insert(7, "SL", 0, True)
#print(Trade_signal)
#print(meta_labels[-32:])
for index1,row  in Trade_signal.iterrows():
    #print(index1)
    #index_next=index1+dt.timedelta( hours=1)
    #index_bef=index1+dt.timedelta( hours=-1)
    index_d=index1.replace(hour=0)
    
    if d2['signal_SM'].loc[index_d]>=0.9 and d2['signal_BP_SM'].loc[index_d]>=0.9:
        if d1['signal_SM'].loc[index1]>= 0.9 : #or d1['signal'].loc[index1]<d1['signal'].loc[index1]
            Trade_signal['action'].loc[index1]=1
        if d1['signal_SM'].loc[index1]>d1['signal_yestrday'].loc[index1] and d1['signal_SM'].loc[index1]>-0.9:
            Trade_signal['action'].loc[index1]=1
    if d2['signal_SM'].loc[index_d]>=0.9 and d2['signal_BP_SM'].loc[index_d]>d2['BP_yestrday'].loc[index_d] and d2['signal_BP_SM'].loc[index_d]>-0.9:
        if d1['signal_SM'].loc[index1]>= 0.9 : #or d1['signal'].loc[index1]<d1['signal'].loc[index1]
            Trade_signal['action'].loc[index1]=1
        if d1['signal_SM'].loc[index1]>d1['signal_yestrday'].loc[index1] and d1['signal_SM'].loc[index1]>-0.9:
            Trade_signal['action'].loc[index1]=1
    if d2['signal_SM'].loc[index_d]>d2['signal_yestrday'].loc[index_d] and d2['signal_SM'].loc[index_d]>-0.9 and d2['signal_BP_SM'].loc[index_d]>=0.9 :
        if d1['signal_SM'].loc[index1]>= 0.9 : #or d1['signal'].loc[index1]<d1['signal'].loc[index1]
            Trade_signal['action'].loc[index1]=1
        if d1['signal_SM'].loc[index1]>d1['signal_yestrday'].loc[index1] and d1['signal_SM'].loc[index1]>-0.9:
            Trade_signal['action'].loc[index1]=1
    if d2['signal_SM'].loc[index_d]>d2['signal_yestrday'].loc[index_d] and d2['signal_SM'].loc[index_d]>-0.9 and d2['signal_BP_SM'].loc[index_d]>d2['BP_yestrday'].loc[index_d] and d2['signal_BP_SM'].loc[index_d]>-0.9 :
        if d1['signal_SM'].loc[index1]>= 0.9 : #or d1['signal'].loc[index1]<d1['signal'].loc[index1]
            Trade_signal['action'].loc[index1]=1
        if d1['signal_SM'].loc[index1]>d1['signal_yestrday'].loc[index1] and d1['signal_SM'].loc[index1]>-0.9:
            Trade_signal['action'].loc[index1]=1
###############################################################################################################################
    if d2['signal_SM'].loc[index_d]<=-0.9 and d2['signal_BP_SM'].loc[index_d]<=-0.9 :
        if d1['signal_SM'].loc[index1]<=-0.9 : #or d1['signal'].loc[index1]<d1['signal'].loc[index1]
            Trade_signal['action'].loc[index1]=-1 
        if d1['signal_SM'].loc[index1]<d1['signal_yestrday'].loc[index1] and d1['signal_SM'].loc[index1]< 0.9:
            Trade_signal['action'].loc[index1]=-1 
    if d2['signal_SM'].loc[index_d]<=-0.9 and d2['signal_BP_SM'].loc[index_d]<d2['BP_yestrday'].loc[index_d] and d2['signal_BP_SM'].loc[index_d]<0.9 :
        if d1['signal_SM'].loc[index1]<=-0.9 : #or d1['signal'].loc[index1]<d1['signal'].loc[index1]
            Trade_signal['action'].loc[index1]=-1 
        if d1['signal_SM'].loc[index1]<d1['signal_yestrday'].loc[index1] and d1['signal_SM'].loc[index1]< 0.9:
            meta_labels['action'].loc[index1]=-1 
    
    if d2['signal_SM'].loc[index_d]<d2['signal_yestrday'].loc[index_d] and d2['signal_SM'].loc[index_d]<0.9 and d2['signal_BP_SM'].loc[index_d]<=-0.9 :
        if d1['signal_SM'].loc[index1]<=-0.9 : #or d1['signal'].loc[index1]<d1['signal'].loc[index1]
            Trade_signal['action'].loc[index1]=-1 
        if d1['signal_SM'].loc[index1]<d1['signal_yestrday'].loc[index1] and d1['signal_SM'].loc[index1]< 0.9:
            Trade_signal['action'].loc[index1]=-1 
    if d2['signal_SM'].loc[index_d]<d2['signal_yestrday'].loc[index_d] and d2['signal_SM'].loc[index_d]<0.9 and d2['signal_BP_SM'].loc[index_d]<d2['BP_yestrday'].loc[index_d] and d2['signal_BP_SM'].loc[index_d]<0.9 :
        if d1['signal_SM'].loc[index1]<=-0.9 : #or d1['signal'].loc[index1]<d1['signal'].loc[index1]
            Trade_signal['action'].loc[index1]=-1 
        if d1['signal_SM'].loc[index1]<d1['signal_yestrday'].loc[index1] and d1['signal_SM'].loc[index1]< 0.9:
            Trade_signal['action'].loc[index1]=-1 
            
#meta_labels['flag']=meta_labels['action']-meta_labels['bin']
#print(meta_labels[-32:])
#RE_AC0=meta_labels[meta_labels['action']==0]
#print(ABP[-30:])
#meta_labels=meta_labels.drop(index=RE_AC0.index)
#print(meta_labels[-32:])
#ac=meta_labels['flag'].value_counts()
#print(meta_labels[-32:])
#print(ac)
#for index,a in ac.iteritems():
    #if index==8:
        #ac=ac.drop(index=8)
#print(ac)
#print(ac2)
#print(ac_2)
#print("accuracy="+str(ac[0]/(np.sum(ac))))
for index2,row1  in Trade_signal.iterrows():
    if row1.Action!=0:
        Trade_signal['Price'].loc[index2]=d1['close'].loc[index2]
        #print("Price="+str(d1['close'].loc[index2]))
        #print(pt_sl[0])
        Trade_signal['TP'].loc[index2]=round((d1['close'].loc[index2]+d1['close'].loc[index2]*pt_sl[0]*row1.trgt*1*1.271)/100)*100
        #print("TP="+str(round(TP1)))
        Trade_signal['SL'].loc[index2]=round((d1['close'].loc[index2]-d1['close'].loc[index2]*pt_sl[0]*row1.trgt*1*1)/100)*100
print(d1['close'].iloc[-1:])
print(Trade_signal[-3:])
print(meta_labels[-3:])


# In[12]:


print(d1[-10:])


# In[ ]:





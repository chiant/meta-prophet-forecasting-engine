#!/usr/bin/env python3 

# --------------------------------------------------------------------------
# Developer:    Brian Sun
# Created:      2024-01-15
# Last Updated: 2025-10-07
# Version:      1.2.0
# Contact:      sun.gsm@hotmail.com
# License:      MIT
# --------------------------------------------------------------------------

"""
Forecast Engine

This module implements a lightweight batch forecasting engine built on top of
Meta Prophet (fbprophet / prophet) for monthly demand forecasting. It was
designed to automate training, cross-validation and prediction for many time
series grouped by arbitrary dimension columns (for example, geography and
product lines). The engine supports grid search over Prophet hyperparameters,
time-based cross validation (rolling window), optional event/holiday handling,
and utilities for plotting and aggregating results.

Key features and assumptions:
- Input time series must be a DataFrame with columns ['ds', 'y'] plus any
	grouping dimensions listed in `ts_dim` (ds = monthly period start, freq = MS).
- Forecast horizon and test_size are expressed in months.
- Negative forecasts are clipped to zero.
- Event handling: if `event_df` is provided it must contain columns matching
	grouping dims and Prophet holiday columns ['holiday','ds','lower_window','upper_window'].

This file augments the original implementation with clearer docstrings and
inline comments to improve maintainability and ease of handoff.
"""

import os
import time
import math
import itertools
import copy
import logging
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from json import dumps, loads as jsonloads
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta

from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation, performance_metrics

# region & country mapping
region_dict = {'ASP':['Australia','Bangladesh','Hong Kong SAR','Hong Kong SAR - BBPM','Hong Kong SAR - HASE','Hong Kong SAR - HASE - BBPM','India',
				 'India - Gift City','India - HSCI','Indonesia','Japan','Macau SAR','Macau SAR - HASE','mainland China','mainland China - HASE',
				 'mainland China - IBCN','Malaysia','Mauritius','New Zealand','Philippines','Singapore','Singapore - HASE','South Korea - SEL',
				 'South Korea - SLS','Sri Lanka and Maldives','Taiwan','Taiwan - HCTW','Thailand','Vietnam'],
		  'MENAT':['Algeria','Bahrain','Egypt','Kuwait','Oman','Qatar','Saudi Arabia','Turkey','UAE'],
		  'Europe':['Armenia','Austria','Belgium','Bermuda','Bulgaria','CIIOM','Cyprus','Czech Republic','Estonia','France','Germany','Greece',
					'Ireland','Israel','Italy','Luxembourg','Malta','Netherlands','Poland','South Africa','Spain','Sweden','Switzerland','UK NRFB'],
		  'UK RFB':['UK - Innovation Banking','UK RFB RM','UK RFB SBB','UK RFB SBB Kinetic'],
		  'USA':['US'],
		  'Canada':['Canada'],
		  'LAM':['Argentina','Brazil','Cayman Islands','Chile','Colombia','Mexico','Uruguay']
}

def get_china_time():
	"""Return current time in Asia/Shanghai timezone as a formatted string.

	Returns:
		str: timestamp in format "YYYY-MM-DD HH:MM:SS" (Asia/Shanghai timezone)
	"""
	SHA_TZ = timezone(
		timedelta(hours=8),
		name='Asia/Shanghai',
	)

	utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
	china_now = utc_now.astimezone(SHA_TZ)
	return china_now.strftime("%Y-%m-%d %H:%M:%S")

#get region from country
def get_region(country, mapping=region_dict):
	"""Map a country name to a region using `mapping`.

	Args:
		country (str): country name to map.
		mapping (dict): mapping from region -> list of countries.

	Returns:
		str|None: region key if found, otherwise None.
	"""
	for key, values in mapping.items():
		if country in values:
			return key
	return None



def ts_summary(tsdata, tsdim):
	"""Compute summary statistics for grouped time series.

	Args:
		tsdata (pd.DataFrame): time series dataframe with columns ['ds','y'] plus grouping dims.
		tsdim (list): list of column names used to group time series (e.g. ['geo','type']).

	Returns:
		pd.DataFrame: grouped summary including y-aggregates and ds range in months.
	"""

	ts_stat = pd.DataFrame(
		columns=tsdim
		+ ["ymean", "ysum", "ymin", "ymax", "dsmin", "dsmax", "dsn", "ds_range"]
	)
	if len(tsdim) > 0:
		ts_stat = (
			tsdata.groupby(tsdim)
			.agg(
				ymean=pd.NamedAgg(column="y", aggfunc="mean"),
				ysum=pd.NamedAgg(column="y", aggfunc="sum"),
				ymin=pd.NamedAgg(column="y", aggfunc="min"),
				ymax=pd.NamedAgg(column="y", aggfunc="max"),
				dsmin=pd.NamedAgg(column="ds", aggfunc="min"),
				dsmax=pd.NamedAgg(column="ds", aggfunc="max"),
				dsn=pd.NamedAgg(column="ds", aggfunc="count"),
			)
			.reset_index()
		)
	else:
		ts_stat = pd.DataFrame(
			{
				"ymean": tsdata.y.mean(),
				"ysum": tsdata.y.sum(),
				"ymin": tsdata.y.min(),
				"ymax": tsdata.y.max(),
				"dsmin": tsdata.ds.min(),
				"dsmax": tsdata.ds.max(),
				"dsn": tsdata.ds.count(),
			},
			index=[0],
		)
	ts_stat["ds_range"] = (
		(ts_stat["dsmax"].dt.year - ts_stat["dsmin"].dt.year) * 12
		+ (ts_stat["dsmax"].dt.month - ts_stat["dsmin"].dt.month)
		+ 1
	)
	return ts_stat

def plot_ts(data,datalabel,datelimit):
	"""Plot a single time series with point labels.

	Args:
		data (pd.DataFrame): dataframe with columns ['ds','y'] where 'ds' is datetime indexable.
		datalabel (str): legend label for the series.
		datelimit (tuple): (start_date, end_date) for x-axis limits.
	"""
	plotdata = data.set_index("ds")
	plotdata.plot(kind='line', 
				  marker='o', 
				  linestyle='-', 
				  figsize=(12,3), 
				  grid=True, 
				  xlabel='',
				  xlim=datelimit,
				  ylim=(0,max(data.y)*1.2))
	for x,y in zip(plotdata.index, plotdata.values):
		label = '{:.0f}'.format(y[0])
		plt.annotate(label, # this is the text
			 (x,y), # these are the coordinates to position the label
			 textcoords="offset points", # how to position the text
			 xytext=(0,5), # distance from text to points (x,y)
			 fontsize=8,
			 ha='center') 
	plt.title("Monthly service Demand Volumn Trend",fontsize=20)
	plt.legend([datalabel],loc='upper left',fontsize=16, facecolor='bisque')
	plt.show()

def plot_all_ts(tsdata, tsdim=[]):
	"""Plot all series in `tsdata` either aggregated or by groups in `tsdim`.

	If `tsdim` is empty, plots the overall series. Otherwise, iterates groups and
	plots each group's series separately.
	"""
	ds_min = tsdata.ds.min().date()
	ds_max = tsdata.ds.max().date()
    
	if (len(tsdim)==0):
		plotdata = tsdata[['ds','y']]
		plotlabel = ''
		plot_ts(data=plotdata, datalabel=plotlabel, datelimit=(ds_min,ds_max))
	else:
		ts_stat = ts_summary(tsdata, tsdim)
		for index, row in ts_stat.iterrows():
			plotdata = tsdata.copy()
			plotlabel = ''
			for dim in tsdim:
				plotdata = plotdata[(plotdata[dim]==row[dim])]
				plotlabel = ','.join([plotlabel, row[dim]])
			plotdata = plotdata[['ds','y']]
			plotlabel = row.geo + ', '+row.type
			plot_ts(data=plotdata, datalabel=plotlabel, datelimit=(ds_min,ds_max))
            

def plot_forecast(data,ds,tslabel,start=None,end=None):
    
    
	# Determine x axis start/end limits. Accepts either None or 'YYYY-MM-DD' strings.
	if (start is None):
		xstart = np.min(data[ds])
	else:
		xstart = datetime.strptime(start, '%Y-%m-%d').date()

	if (end is None):
		xend = np.max(data[ds])
	else:
		xend = datetime.strptime(end, '%Y-%m-%d').date()

	plotdata = data.set_index(ds)

	# Simple line plot containing actuals and forecasts (multiple columns allowed)
	plotdata.plot(kind='line', 
				  marker='o', 
				  linestyle='-', 
				  figsize=(12,3), 
				  grid=True, 
				  xlabel='',
				  xlim=(xstart,xend),
				  ylim=(0,np.nanmax(plotdata.values)*1.2))

	for k in range(plotdata.values.shape[1]):        
		for x,y in zip(plotdata.index, plotdata.values):
			label = '{:.0f}'.format(y[k])
			plt.annotate(label, # this is the text
				 (x,y[k]), # these are the coordinates to position the label
				 textcoords="offset points", # how to position the text
				 xytext=(0,5), # distance from text to points (x,y)
				 fontsize=7,
				 ha='center') 
            
	plt.yticks([])  
	plt.grid(axis ='y', which='both')
	plt.title(tslabel + " - Monthly Service Demand",fontsize=16)
	plt.legend(facecolor='bisque')
	plt.show()
    
def forecast_result_agg(resultdata, aggdim=['trainmonth','forecastmonth','n_window']):
	"""Aggregate forecast results by provided dimensions and compute error metrics.

	Args:
		resultdata (pd.DataFrame): detailed forecast results containing 'forecast' and 'actual'.
		aggdim (list): list of columns to aggregate by (default ['trainmonth','forecastmonth','n_window']).

	Returns:
		pd.DataFrame: aggregated results with error and percent error columns.
	"""
	agg_cols = aggdim

	resultdata = resultdata.groupby(agg_cols).agg(n_rec=pd.NamedAgg(column='forecastmonth',aggfunc='count'), 
												 actual=pd.NamedAgg(column='actual',aggfunc='sum'),
												 forecast=pd.NamedAgg(column='forecast',aggfunc='sum'))

	# Avoid divide-by-zero when calculating percent errors by replacing 0 actuals with NaN.
	resultdata['actual'].replace(0, np.nan, inplace=True)

	resultdata['error'] = resultdata['forecast']-resultdata['actual']

	def cal_error_percent(forecast,actual):
		if actual>0:
			return (forecast-actual)/actual

	resultdata['error_percent'] = resultdata.apply(lambda row: cal_error_percent(row['forecast'],row['actual']),axis=1)
	resultdata['abs_error_percent'] = resultdata['error_percent'].abs() 
	return resultdata.reset_index()


class prophet_engine:
	"""A batch forecasting engine wrapper around Prophet.

	Responsibilities:
	- Run grid search over Prophet hyperparameters for many grouped time series.
	- Use rolling time-window cross validation to evaluate parameter sets.
	- Persist best parameters and run batch predictions for all saved models.

	Important constructor args:
	- ts_data: pandas DataFrame containing all time series (must include `ds` and `y`).
	- param_grid: dictionary mapping Prophet argument names to lists of candidate values.
	- ts_dim: list of columns that uniquely identify each time series (used to group data).
	- event_df: optional DataFrame of holiday/event rows to pass into Prophet as 'holidays'.
	"""

	def __init__(
		self,
		ts_data,
		param_grid={'changepoint_prior_scale':[0.001, 0.01, 0.1, 0.5]},
		ts_dim=[],
		event_df=None,
		test_size=3,
		forecast_end_month="2024-12-1",
		workpath="model_result",
		select_criteria="M",
		engine_para = None,
		last_run_result = None,
		predict_m_result = None,
		predict_agg_result = None
	):
		# Core inputs and configuration
		self.ts_data = ts_data
		self.param_grid = param_grid
		self.ts_dim = ts_dim
		self.event_df = event_df
		self.test_size = test_size
		self.forecast_end_month = forecast_end_month
		self.select_criteria = select_criteria

		# Columns used for persisting engine parameter summaries
		self.para_cols= self.ts_dim + ['params', 'event_included', 'mape_month', 'mape_period', 'exclude_period', 
			  'engine_run_timestamp', 'dsmin', 'dsn', 'dsmax', 'ymin', 'ds_range', 'ysum', 'ymax', 'ymean']

		# Optional persisted artifacts (can be loaded from disk)
		self.engine_para = engine_para
		self.last_run_result = last_run_result
		self.predict_m_result = predict_m_result
		self.predict_agg_result = predict_agg_result

		# Prepare file-system workspace and load any saved engine parameters
		self.set_workpath(workpath)
		self.load_engine_para()

		# Pre-warm Prophet and quiet verbose stan logs to avoid noisy output during mass runs
		self.disable_annoying_log()

	def set_workpath(self, workpath):
		isExist = os.path.exists(workpath)
		if not isExist:
			os.makedirs(workpath)
		self.workpath = workpath
        
	def load_engine_para(self, filepath=''):
		if filepath == '':
			filepath = self.workpath + '/' + 'engine_para.csv'
		if os.path.isfile(filepath):
			self.engine_para = pd.read_csv(filepath)
			print('Engine parameter data (self.engine_para) is successfully loaded from '+ filepath)
		else:
			print('Sorry, engine parameter file is NOT found from '+ filepath)
            
	def save_engine_para(self, output=''):
		if output == '':
			output = self.workpath + '/' + 'engine_para.csv'
		else:
			output = self.workpath + '/' + output
		if (self.engine_para is not None):
			self.engine_para.to_csv(output, index=False)
			print('engine parameter data is saved in '+ output)
		else:
			print('engine parameter data is not loaded')
            
	def refresh_engine_para(self, criteria='', mode = 'U'):
		# U: update record ONLY if the model error is smaller than existing one, 
		# O: overwrite the record anyway, 
		# R: replace the entire data
        
		if (criteria == ''):
			criteria = self.select_criteria
            
		if (self.last_run_result is None):
			print('No latest model running result available for refreshing')        
		elif (self.engine_para is None or mode == 'R'):
			self.engine_para = self.last_run_result
			print('engine parameter data is successfully created/replaced!')
		elif mode == 'U':
			self.engine_para = pd.concat([self.engine_para, self.last_run_result])
			if criteria == 'M':
				sortvar = 'mape_month'
			else:
				sortvar = 'mape_period'
			self.engine_para = self.engine_para.sort_values(self.ts_dim + [sortvar]).drop_duplicates(subset=self.ts_dim, keep='first') 
			print('engine parameter data is successfully updated!')
		elif mode == 'O':
			self.engine_para = pd.concat([self.engine_para, self.last_run_result])
			sortvar = 'engine_run_timestamp'
			self.engine_para = self.engine_para.sort_values(self.ts_dim + [sortvar]).drop_duplicates(subset=self.ts_dim, keep='last') 
			print('engine parameter data is successfully overwritten!')
		else:
			print('wrong value of the mode parameter, please check!')
            
        
	def engine_timer(func):
		def innerfunc(self, *args, **kwargs):
			logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
			run_start = time.perf_counter()

			func(self, *args, **kwargs)

			run_end = time.perf_counter()
			total_run_time = (run_end - run_start) / 3600
			print(f"Total Run Hours: {total_run_time:0.1f} hours")

		return innerfunc
    
	def disable_annoying_log(self):
		import logging
		logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
		df_test = pd.DataFrame({'ds': pd.date_range(start=datetime(2020, 1, 1), periods=12, freq="MS"), 
								'y': [251, 186, 241, 267, 307, 349, 478, 264, 245, 234, 258, 215]})
		test_model = Prophet()
		test_model.fit(df_test)

	def get_exclude_period(self, ts_info):
		excludelist = []
		if (self.event_df is not None):
			event_sel = self.event_df[(self.event_df.exclude_assess == "Y")]
			for dimvar in self.ts_dim:
				event_sel = event_sel[event_sel[dimvar] == ts_info[dimvar]].reset_index(
					drop=True
				)
			if len(event_sel) > 0:
				excludelist.append(pd.to_datetime(event_sel.ds.unique().tolist()))
				excludelist = excludelist[0]
		return excludelist

	def param_generator(self, ts_info):
		"""Generate list of Prophet parameter dictionaries to evaluate for a single series.

		Removes parameters that don't make sense for the series (e.g. seasonality priors
		when the series only has a few months).
		"""
		param_config = self.param_grid.copy()

		# If event dataframe exists, filter it down to the current series' events.
		if (self.event_df is not None):
			event_sel = self.event_df
			for dimvar in self.ts_dim:
				event_sel = event_sel[event_sel[dimvar] == ts_info[dimvar]]

		# If no events are present for this series, drop holiday-related hyperparams
		if (len(event_sel) == 0) and ("holidays_prior_scale" in param_config.keys()):
			del param_config["holidays_prior_scale"]

		# Series with very few data points may not support seasonality hyperparams
		if (ts_info.dsn <= 12) and ("seasonality_prior_scale" in param_config.keys()):
			del param_config["seasonality_prior_scale"]

		# Build Cartesian product of all hyperparameter choices
		params = [
			dict(zip(param_config.keys(), v))
			for v in itertools.product(*param_config.values())
		]

		# If there are events for this series, attach them to each param dict as holidays
		if len(event_sel) > 0:
			for param in params:
				param["holidays"] = event_sel[
					["holiday", "ds", "lower_window", "upper_window"]
				].reset_index(drop=True)

		return params

	def ts_model_predict(self, modelobj, start_month, end_month):
		"""Generate predictions for a fitted Prophet model between two months.

		Args:
			modelobj (Prophet): fitted Prophet model instance.
			start_month (datetime.date|Timestamp): first month to predict (inclusive).
			end_month (datetime.date|Timestamp): last month to predict (inclusive).

		Returns:
			pd.DataFrame: DataFrame with columns ['ds','yhat'] containing monthly predictions.
		"""
		month_periods = (
			end_month.month
			- start_month.month
			+ 12 * (end_month.year - start_month.year)
			+ 1
		)
		predict_months = pd.date_range(
			start=start_month, periods=month_periods, freq="MS"
		)
		predict_months = pd.DataFrame({"ds": predict_months})

		# Use Prophet to predict and clip negative forecasts to zero.
		predict_result = modelobj.predict(predict_months)
		predict_result.loc[predict_result.yhat < 0, "yhat"] = 0
		return predict_result[["ds", "yhat"]]

	def cal_forecast_result(self, forecast_data, df):
		"""Join predictions with actuals and compute error metrics per month and in aggregate.

		Args:
			forecast_data (pd.DataFrame): output of Prophet.predict with columns ['ds','yhat'].
			df (pd.DataFrame): original timeseries containing 'ds' and 'y' (actuals).

		Returns:
			tuple(pd.DataFrame, pd.DataFrame): (detailed monthly result, aggregated result)
		"""
		def cal_error_percent(forecast, actual):
			if actual > 0:
				return (forecast - actual) / actual

		# Align forecast with actuals on 'ds' and rename columns to more explicit names
		result = pd.merge(forecast_data[["ds", "yhat"]], df, on="ds", how="left")
		result["ds"] = pd.to_datetime(result["ds"])
		result = result.rename(
			columns={"ds": "forecastmonth", "yhat": "forecast", "y": "actual"}
		)

		# Per-month error columns
		result["error"] = result["forecast"] - result["actual"]
		result["error_percent"] = result.apply(
			lambda row: cal_error_percent(row["forecast"], row["actual"]), axis=1
		)
		result["abs_error_percent"] = result["error_percent"].abs()

		# Aggregate across the whole forecast window
		agg_result = pd.DataFrame(
			{
				"forecast_start_month": result["forecastmonth"].min(),
				"forecast_end_month": result["forecastmonth"].max(),
				"actual": result["actual"].sum(),
				"forecast": result["forecast"].sum(),
			},
			index=[0],
		)
		agg_result["error"] = agg_result["forecast"] - agg_result["actual"]

		agg_result["error_percent"] = agg_result.apply(
			lambda row: cal_error_percent(row["forecast"], row["actual"]), axis=1
		)
		agg_result["abs_error_percent"] = agg_result["error_percent"].abs()

		return result, agg_result

	# Meta Prophet model cross valuation process
	def prophet_model_cv_monthly(
		self, tsdata, mparams, initial=12, period=1, exclude_periods=[]
	):
		rolling_result = pd.DataFrame(
			columns=[
				"runid",
				"trainmonth",
				"forecastmonth",
				"n_window",
				"forecast",
				"actual",
				"error",
				"error_percent",
				"abs_error_percent",
			]
		)
		rolling_agg_result = pd.DataFrame(
			columns=[
				"runid",
				"trainmonth",
				"forecast_start_month",
				"forecast_end_month",
				"n_window",
				"forecast",
				"actual",
				"error",
				"error_percent",
				"abs_error_percent",
			]
		)

		# Number of months in the CV horizon (how many months to hold out at each step)
		horizon = self.test_size

		for runid in range(initial, len(tsdata), period):
			rollingdata = tsdata[: runid + 1]
			train, test = (
				rollingdata.iloc[:-horizon].copy(),
				rollingdata.iloc[-horizon:].copy(),
			)

			# Train cut-off date (last training month)
			traindate = train[-1:]["ds"][runid - horizon]

			# Fit the model with the training data
			m = Prophet(**mparams)
			m.fit(train)

			# Generate forecast values for the next k months (the test months)
			predict_months = pd.DataFrame({"ds": test["ds"]})
			predict_result = m.predict(predict_months)

			result, agg_result = self.cal_forecast_result(
				forecast_data=predict_result, df=tsdata
			)

			# Annotate with CV metadata
			result["runid"] = runid
			result["trainmonth"] = traindate

			result["n_window"] = result.forecastmonth.dt.to_period("M").astype(
				int
			) - result.trainmonth.dt.to_period("M").astype(int)

			rolling_result = pd.concat([rolling_result, result])

			agg_result["runid"] = runid
			agg_result["trainmonth"] = traindate
			agg_result["n_window"] = (
				agg_result.forecast_end_month.dt.to_period("M").astype(int)
				- agg_result.forecast_start_month.dt.to_period("M").astype(int)
				+ 1
			)

			rolling_agg_result = pd.concat([rolling_agg_result, agg_result])

		# Optionally exclude certain forecast months from performance assessment
		if len(exclude_periods) > 0:
			rolling_result_sel = rolling_result[
				~rolling_result.forecastmonth.isin(exclude_periods)
			]

			rolling_result_unsel = rolling_result[
				rolling_result.forecastmonth.isin(exclude_periods)
			]
			q_exclude_traindate = list(
				pd.to_datetime(rolling_result_unsel.trainmonth.unique())
			)
			rolling_agg_result_sel = rolling_agg_result[
				~rolling_agg_result.trainmonth.isin(q_exclude_traindate)
			]

			cur_map_m = rolling_result_sel["abs_error_percent"].mean()
			cur_map_q = rolling_agg_result_sel["abs_error_percent"].mean()
		else:
			cur_map_m = rolling_result["abs_error_percent"].mean()
			cur_map_q = rolling_agg_result["abs_error_percent"].mean()

		exclude_date_string = ""
		for i, exclude_date in enumerate(exclude_periods):
			if i == 0:
				exclude_date_string = exclude_date.strftime("%Y-%m-%d")
			else:
				exclude_date_string = ",".join(
					[exclude_date_string, exclude_date.strftime("%Y-%m-%d")]
				)

		# Prepare a human-readable MAPE record for this parameter set
		paras_text = mparams.copy()
		if "holidays" in paras_text.keys():
			# Remove heavy DataFrame objects from the textual record
			del paras_text["holidays"]
			event_flag = 1
		else:
			event_flag = 0

		paras_text = dumps(paras_text)

		mape = pd.DataFrame(
			{
				"params": paras_text,
				"exclude_period": exclude_date_string,
				"event_included": event_flag,
				"mape_month": cur_map_m,
				"mape_period": cur_map_q,
				"engine_run_timestamp": get_china_time(),
			},
			index=[0],
		)

		return rolling_result, rolling_agg_result, mape, m
    
	def set_initial_size(self, tsdata):
		"""Heuristic for selecting the initial training window size for CV.

		Returns number of months to use as the 'initial' window for rolling CV
		depending on the total length of the series.
		"""
		if len(tsdata) <= 12:
			size = 6
		elif len(tsdata) <= 15:
			size = 9
		elif len(tsdata) <= 24:
			size = 12
		elif len(tsdata) <= 36:
			size = 18
		else:
			size = 24
		return size

	# single model run by market & lob
	def single_model_run(self, ts_info, ts_all):

		current_model_label = ""
		for var in self.ts_dim:
			current_model_label = ", ".join([current_model_label, ts_info[var]])
            
		# Decide how big the initial training window should be.
		INITIAL_SIZE = self.set_initial_size(ts_all)

		# Some events may be excluded from CV error calculation; build that list.
		exclude_period_list = self.get_exclude_period(ts_info=ts_info)

		# Build grid of parameter dictionaries to evaluate for this series
		params_set = self.param_generator(ts_info=ts_info)

		mapes_m = []
		mapes_q = []
		for i, param in enumerate(params_set):
			# Lightweight progress logging for long grid searches
			if i == 0 or (i + 1) % 10 == 0 or i == len(params_set) - 1:
				print(
					"******************{time}: {ts_name}: Grid Search Round {k} Start******************".format(
						time=get_china_time(), ts_name=current_model_label, k=i + 1
					)
				)

			rolling_result, rolling_agg_result, mape_df, latest_model = (
				self.prophet_model_cv_monthly(
					tsdata=ts_all,
					initial=INITIAL_SIZE,
					period=1,
					mparams=param,
					exclude_periods=exclude_period_list,
				)
			)

			mapes_m.append(mape_df["mape_month"].values[0])
			mapes_q.append(mape_df["mape_period"].values[0])

			mape_m_label = "{0:.1%}".format(mapes_m[-1])
			mape_q_label = "{0:.1%}".format(mapes_q[-1])

			if i == 0 or (i + 1) % 10 == 0:
				print(
					"......{time}:{parameter}>>>>> mape_m:{mape_m}, mape_q:{mape_q}".format(
						time=get_china_time(),
						parameter=param,
						mape_m=mape_m_label,
						mape_q=mape_q_label,
					)
				)

		# select the best MAPE on monthly average level
		if self.select_criteria == "M":
			sel_criteria = mapes_m
		else:
			sel_criteria = mapes_q

		best_params = params_set[np.argmin(sel_criteria)]
		best_map_label = "{:.1%}".format(np.min(sel_criteria))

		print("\n")
		print("********************Best Parameter******************")
		print(best_params, best_map_label)
		best_result, best_agg_result, best_mape, best_model = (
			self.prophet_model_cv_monthly(
				tsdata=ts_all,
				initial=INITIAL_SIZE,
				period=1,
				mparams=best_params,
				exclude_periods=exclude_period_list,
			)
		)

		return best_mape, best_result, best_agg_result

	@engine_timer
	def batch_fit(self):
		# Compute grouped series metadata (counts, ranges etc.) for the whole dataset
		ts_stat = ts_summary(self.ts_data, self.ts_dim)

		prophet_model_summary = pd.DataFrame(columns=self.para_cols)
		prophet_model_m_forecast = pd.DataFrame(
			columns=self.ts_dim
			+ [
				"runid",
				"trainmonth",
				"forecastmonth",
				"n_window",
				"actual",
				"forecast",
				"error",
				"error_percent",
				"abs_error_percent",
			]
		)
		prophet_model_q_forecast = pd.DataFrame(
			columns=self.ts_dim
			+ [
				"runid",
				"trainmonth",
				"forecast_start_month",
				"forecast_end_month",
				"n_window",
				"actual",
				"forecast",
				"error",
				"error_percent",
				"abs_error_percent",
			]
		)

		runtime = []
		for index, row in ts_stat.iterrows():
			tic = time.perf_counter()
			print("\n")

			# Extract the time series for this group
			df_cur = self.ts_data.copy()
			for dimvar in self.ts_dim:
				df_cur = df_cur[df_cur[dimvar] == row[dimvar]].reset_index(drop=True)

			current_model_label = ""
			for var in self.ts_dim:
				current_model_label = ", ".join([current_model_label, row[var]])

			print(
				"Model {k} Training Start: {model_label}...............................".format(
					model_label=current_model_label, k=index + 1
				)
			)
			print("\n")

			# params_set = copy.deepcopy(all_params)
			# Run grid search + CV for this single series and return the best summary
			summary, result, agg_result = self.single_model_run(
				ts_info=row, ts_all=df_cur
			)
			summary = pd.concat(
				[pd.DataFrame(row.to_dict(), index=[0]), summary], axis=1
			)

			for dimvar in self.ts_dim:
				# summary[dimvar] = row[dimvar]
				result[dimvar] = row[dimvar]
				agg_result[dimvar] = row[dimvar]

			prophet_model_summary = pd.concat([prophet_model_summary, summary])
			prophet_model_m_forecast = pd.concat([prophet_model_m_forecast, result])
			prophet_model_q_forecast = pd.concat([prophet_model_q_forecast, agg_result])

			self.last_run_result = prophet_model_summary

			prophet_model_summary.to_csv(
				self.workpath + "/" + "model_result.csv", index=False
			)
			prophet_model_m_forecast.to_csv(
				self.workpath + "/" + "model_train_month_result.csv", index=False
			)
			prophet_model_q_forecast.to_csv(
				self.workpath + "/" + "model_train_agg_result.csv", index=False
			)

			toc = time.perf_counter()
			runtime.append((toc - tic) / 60)
			est_time_left = np.mean(runtime) * (len(ts_stat) - index - 1)

			print(
				f"Model {index+1} Training completed:{current_model_label} completed in {runtime[-1]:0.1f} minutes, estimated {est_time_left:0.1f} minutes left"
			)

	def prophet_cv_monthly_predict(self, tsdata, mparams, initial=12, horizon=0, period=1, end_month='2024-12-1'):
		rolling_result = pd.DataFrame(columns=['runid','trainmonth','forecastmonth','n_window','forecast','actual','error','error_percent'])
		#rolling_agg_result = pd.DataFrame(columns=['runid','trainmonth','forecast_start_month','forecast_end_month','n_window','forecast','actual','error','error_percent'])
		# Predict forward from the earliest observed month to 'end_month'
		start_month = np.min(tsdata.ds)
		end_month = datetime.strptime(end_month, '%Y-%m-%d')

		for runid in range(initial, len(tsdata), period):
			rollingdata = tsdata[: runid + 1]
			if (horizon > 0):
				train = rollingdata.iloc[:-horizon].copy()
			else:
				train = rollingdata.copy()

			traindate = train[-1:]['ds'][runid-horizon]

			m = Prophet(**mparams)
			m.fit(train)

			predict_result = self.ts_model_predict(m, start_month, end_month)

			result, agg_result = self.cal_forecast_result(predict_result, df=tsdata)

			result['runid'] = runid
			result['trainmonth'] = traindate

			result['n_window'] = result.forecastmonth.dt.to_period('M').astype(int) - result.trainmonth.dt.to_period('M').astype(int)

			rolling_result = pd.concat([rolling_result, result])

		return rolling_result
            
	def single_model_predict(self, ts_all, model_meta):
        
		# Prepare parameters and event holidays for prediction based on stored metadata
		ts_info = model_meta

		INITIAL_SIZE = self.set_initial_size(ts_all)

		param = jsonloads(ts_info.params)

		if self.event_df is not None:
			event_sel = self.event_df
			for dimvar in self.ts_dim:
				event_sel = event_sel[event_sel[dimvar] == model_meta[dimvar]]

			if (len(event_sel)==0) and ('holidays_prior_scale' in param.keys()):
				del param['holidays_prior_scale']

			if (len(event_sel)>0):
				param['holidays'] = event_sel[['holiday','ds','lower_window','upper_window']].reset_index(drop=True)
		else:
			if ('holidays_prior_scale' in param.keys()):
				del param['holidays_prior_scale']            

		forecast_result = self.prophet_cv_monthly_predict(tsdata=ts_all, initial=INITIAL_SIZE, horizon=0, period=1, mparams=param, end_month=self.forecast_end_month)

		# Reattach grouping dimension values so caller can identify the series
		for dimvar in self.ts_dim:
			forecast_result[dimvar] = ts_info[dimvar]

		return forecast_result            

	@engine_timer
	def batch_predict(self):
		if (self.engine_para is not None):
			runtime=[]
			model_m_forecast = pd.DataFrame(columns=self.ts_dim + ['runid', 'trainmonth', 'forecastmonth', 'n_window',
																   'actual', 'forecast', 'error','error_percent', 'abs_error_percent'])

			for index, row in self.engine_para.iterrows():
					# Human-readable label for logs
					current_model_label = ""
					for var in self.ts_dim:
						current_model_label = ", ".join([current_model_label, row[var]])  
                        
					# Extract series for this model
					df_cur = self.ts_data.copy()
					for dimvar in self.ts_dim:
						df_cur = df_cur[df_cur[dimvar] == row[dimvar]].reset_index(drop=True)

					tic = time.perf_counter()
					print('\n')
					print("Model {k} Prediction Start: {model_label}...............................".format(model_label=current_model_label, k=index+1))
					print('\n')

					# Run prediction using stored best-parameter metadata
					result = self.single_model_predict(model_meta=row, ts_all=df_cur)

					model_m_forecast = pd.concat([model_m_forecast, result])
                    
					# Persist intermediate results so long-running batches can be inspected
					model_m_forecast.to_csv(self.workpath + "/" + "model_predict_m_result.csv", index=False)

					toc = time.perf_counter()
					runtime.append((toc - tic)/60)
					est_time_left = np.mean(runtime)*(len(self.engine_para)-index-1)

					print('\n')
					print(f"Model {index+1} Prediction Completed: {current_model_label} completed in {runtime[-1]:0.1f} minutes, estimated {est_time_left:0.1f} minutes left")
                    
			self.predict_m_result = model_m_forecast
			self.predict_agg_result = forecast_result_agg(self.predict_m_result)
			self.predict_agg_result.to_csv(self.workpath + "/" + "model_predict_agg_result.csv", index=False)
            

#            return model_m_forecast
		else:
			print('engine parameter data is NOT loaded')
#            return None
    
	def single_model_plot(self, ts_all, ts_info, plot_title='', horizon=3, start=None, end=None):
        
		param = jsonloads(ts_info.params)

		if self.event_df is not None:
			event_sel = self.event_df
			for dimvar in self.ts_dim:
				event_sel = event_sel[event_sel[dimvar] == ts_info[dimvar]]
                
			if (len(event_sel)==0) and ('holidays_prior_scale' in param.keys()):
				del param['holidays_prior_scale']

			if (len(event_sel)>0):
				param['holidays'] = event_sel[['holiday','ds','lower_window','upper_window']].reset_index(drop=True)
		else:
			if ('holidays_prior_scale' in param.keys()):
				del param['holidays_prior_scale']            
        
		param['yearly_seasonality'] = True
        
            
		plotdata = ts_all.copy()
        
		#ds_min = plotdata.ds.min().date()
		#ds_max = plotdata.ds.max().date()        
        
		traindata = ts_all.iloc[:-horizon].copy()

		m = Prophet(**param)    
		m.fit(traindata)
		predict_months = pd.DataFrame({"ds": ts_all["ds"]})
		# print(predict_months)
		predict_result = m.predict(predict_months)
		result = pd.merge(predict_result[["ds", "yhat"]], plotdata[['ds','y']], on="ds", how="left")
		result.rename(columns = {'y':'actual', 'yhat':'forecast'}, inplace = True)
		#print(result.head(5))
		plot_forecast(result, ds='ds', tslabel=plot_title, start=start, end=end)
		#m.plot(predict_result)
		m.plot_components(predict_result)

	@engine_timer
	def batch_predict_plot(self, horizon=3, start=None, end=None):
		if (self.engine_para is not None):

			for index, row in self.engine_para.iterrows():
					for i,var in enumerate(self.ts_dim): 
						if i==0:
							current_model_label = row[var]
						else:
							current_model_label = ", ".join([current_model_label, row[var]])  
                        
					df_cur = self.ts_data.copy()
					for dimvar in self.ts_dim:
						df_cur = df_cur[df_cur[dimvar] == row[dimvar]].reset_index(drop=True)

					self.single_model_plot(ts_info=row, 
										   ts_all=df_cur, 
										   plot_title=current_model_label, 
										   horizon=horizon,
										   start=start,
										   end=end)
		else:
			print('engine parameter data is NOT loaded')

if __name__ == "__main__":
	print("Forecast Engine is running.")

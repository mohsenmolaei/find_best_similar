import tehran_stocks
import pandas as pd
import numpy as np
# import darts
from tehran_stocks import Stocks, update_group

import nest_asyncio 
nest_asyncio.apply()
from tehran_stocks.download import get_all_price
import os
import shutil
import darts
from darts.models.forecasting.rnn_model import RNNModel
from darts.timeseries import TimeSeries
import torch
from darts.metrics.metrics import quantile_loss
from pytorch_forecasting.metrics.quantile import QuantileLoss
from torch.nn.modules import L1Loss
from darts.metrics.metrics import mse, mae, r2_score, rmse
from IPython.display import clear_output
import pickle




# get_all_price()



class get_Namad:
    def __init__(self) -> None:
        self.Namads_data = {"names":[],"namads":[], "namads_df_raw":[], "namads_df":[]}
    
    @property
    def update_allGroups(self):
        print("\nupdating started... \n")
        groups = Stocks.get_group() # to see list of group codes
        _ = [update_group(group[0]) for group in groups]
        print("\ndone\n")

    
    def update_group(self, group_number):
        update_group(group_number)
            
    
    def get_NamadsByGroupNumber(self, group_number ):
        return Stocks.query.filter_by(group_code = group_number).all()
    
    
    def get_NamadByName(self, name):
        return Stocks.query.filter_by(name = name).all()
    
    @property
    def get_allGroups(self):
        return Stocks.get_group()
        
    
    @property
    def get_allNamads(self):        
        groups = self.get_allGroups
        all_namads = [namad for group in groups for namad in self.get_NamadsByGroupNumber(group[0]) ]
        return all_namads
        
    
    def filter_Namads(self, Namads_withRows_moreThan = 2000, maxNull_percent = 50,
                      inplace = True, fillnan_method = "bfill" ):
        filtered_namads = []
        
        for namad in self.get_allNamads:
            if namad.df.empty: continue
            max_df_len =  len(pd.date_range(namad.df.index[0], namad.df.index[-1]))
            
            if len(namad.df) > Namads_withRows_moreThan and \
            not namad.df.empty and \
            max_df_len-len(namad.df) < (maxNull_percent/100)*(max_df_len): 
                filtered_namads.append(namad)
     
        filtered_namads = sorted(filtered_namads, key = lambda x: len(x.df), reverse = True ) 
                   
        if inplace: 
            self.Namads_data["names"] = [f"{self.get_namad_name(namad)}" for namad in filtered_namads]
            self.Namads_data["namads"] = filtered_namads
            self.Namads_data["namads_df_raw"] = [namad.df for namad in filtered_namads]
            self.get_NamadsDataset(fillnan_method = fillnan_method)
            
        return filtered_namads
    
    
    
    
    def get_NamadDatasetByName(self, Name:str):
        index = [self.Namads_data["names"].index(name) for name in self.Namads_data["names"]
                if Name in name][0]
        return self.Namads_data["namads_df"][index] 
    
    
    def get_NamadsDataset(self, fillnan_method = "bfill"):
        if self.Namads_data["namads"] == []: raise Exception("no namads filtered yet. first filter them by filter_Namads")
        namads_df = []
        for namad in self.Namads_data["namads"]:
            df:pd.DataFrame = namad.df[['close', 'vol']].copy()
            df["change"] = df["close"].pct_change()
            df["volume"] = df['vol'] / df['vol'].max()
            df.drop(['close', 'vol'], axis = 1, inplace = True)
            df.fillna(0)
            df = self.fillnans(df, method = fillnan_method)
            namads_df.append(df)
        self.Namads_data["namads_df"] = namads_df
        return namads_df
    
    
    def get_Namad_Dataset(self, namad):
        df:pd.DataFrame = namad.df[['close', 'vol']].copy()
        df["change"] = df["close"].pct_change()
        df["volume"] = df['vol'] / df['vol'].max()
        df.drop(['close', 'vol'], axis = 1, inplace = True)
        df.fillna(0)
        df = self.fillnans(df, method = "bfill")
        return df
    
    
    def skip_firstYear_ifNoChange(self, df):
        df_ = df.copy()
        zeros_df = df_[df_["change"] == 0]
        if zeros_df.index[0].year == df_.index[0].year:
            df2_ = df_.loc[f"{df_.index[0].year+1}":].copy()
        return df2_
    
    
    
    def fillnans(self, df:pd.DataFrame, method = "bfill"):
        df_ = df.copy()
        new_ind = pd.date_range(df_.index[0], df_.index[-1])
        df_ = df_.reindex(new_ind)
        match method.lower():
            case "bfill": df_.bfill(inplace = True)
            case "ffill": df_.ffill(inplace = True)
            case "interpolate": df_ = df_.astype("float64").interpolate(axis=0, limit_direction = "both")
            case _: raise ValueError(f"{method.lower()} fillnan method not found")
        return df_

    
    def save_namads2CSV(self, dir_name = None, remove_previous = False, which_data = "namads_df"):
        cwd = os.getcwd()
        if dir_name == None: dir_name = 'dataset'
        if  remove_previous and dir_name in os.listdir(): shutil.rmtree(dir_name)
        if not dir_name in os.listdir(): os.mkdir(dir_name)
        os.chdir(dir_name)
        for name, namad in zip(self.Namads_data["names"], self.Namads_data[which_data]): 
            namad.df.to_csv(f"{name}.csv") if which_data=="namads" else namad.to_csv(f"{name}.csv")
        os.chdir(cwd)
        
        
        
    def readNamads_fromCSVfiles(self, dir:str):
        pwd = os.getcwd()
        os.chdir(dir)
        Namad_names = os.listdir()
        self.Namads_data["names"] = Namad_names
        self.Namads_data["namads"] = []
        self.Namads_data["namads_df_raw"] = []
        self.Namads_data["namads_df"] = []
        
        for namad_name in Namad_names: # time as index
            df = pd.read_csv(namad_name).rename(columns={"Unnamed: 0":"time"})
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace = True)
            self.Namads_data["namads_df"].append(df)
            
        os.chdir(pwd)
        return self.Namads_data["names"]
    
    
    def get_Namad_fullname_bySome(self, partOfName:str):
        return [namad_name for namad_name in self.Namads_data["names"] if partOfName in namad_name][0]
        
        
    def get_namad_name(self, namad_obj:object) -> str:
        namad_title = namad_obj.title.removesuffix("',FaraDesc ='")
        return f"{namad_obj.group_name}-->{namad_title}"
    
        
class train_model:
    def __init__(self, model_type:str = "LSTM"
                 , layer_size = 20, numOf_hiddenLayers = 2, loss_obj = None, 
                 input_BatchSize = 100, train_len = 50, dropout = 0 ) -> None:
        
        model_type = model_type.upper()
        assert model_type in ["LSTM", "GRU", "RNN"], "model can be 'LSTM' or 'GRU'"
        assert train_len < input_BatchSize, "train len must be smaller than input batch size"
        
        self.model = RNNModel(hidden_dim = layer_size,
                         model = model_type,
                         training_length = input_BatchSize ,
                         n_rnn_layers = numOf_hiddenLayers,
                         dropout = dropout,
                         input_chunk_length = train_len,
                         loss_fn = loss_obj,
                        )
        self.ts_train = None
        self.ts_test = None
    
    
    def train_on_Namad(self, Namad_df, epochs = 3, train_test_ratio = 0.2):
        df = Namad_df.copy()
        ts = TimeSeries.from_dataframe(df, freq='1D', fill_missing_dates = False)
        self.ts_train, self.ts_test = ts.split_after(1-train_test_ratio)
        self.model.fit(self.ts_train, epochs = epochs)
        
        
        
    def predict_and_plot(self, nTimestemps = None, plot_train_data = False, which_column = None,
                         inplace = True, plot_results = True):
        
        y_hat = self.model.predict(nTimestemps if nTimestemps else len(self.ts_test))
        if plot_train_data: 
            self.ts_train[which_column].plot(label=f"{which_column} train") \
            if which_column else self.ts_train.plot(label=f"train")
        
        if plot_results:
            y_hat[which_column].plot(label=f"{which_column} predicted") if which_column \
            else y_hat.plot(label='predicted')
            
            self.ts_test[which_column].plot(label=f"{which_column} test") if which_column \
            else self.ts_test.plot(f"test")
        
        if inplace: self.y_hat = y_hat
        return y_hat         
    
    
    def evaluate_metrics(self, metrics_list:list):
        return {metric.__name__: metric(self.ts_test, self.y_hat) for metric in metrics_list}
    
    
    def _bulk_train_on_Namad(Namad_name, namads_CSV_dir, losses:list, *, epochs = 2, train_test_ratio = 0.1,
                   metrics:list = [mse, mae, r2_score], model_types = ["LSTM", "GRU"], 
                   layerSizes = [20], numOfHiddenLayers = [2], input_BatchSize = [100],
                   train_len = [70], dropouts = [0]):
        
        columns = ["namad_name", "loss", "epochs", "train_test_ratio", "metrics", "model", "layerSize",
                   "numOfHiddenLayers", "input_batchSize", "train_len", "train_obj"]
        results_df = pd.DataFrame(columns = columns)
        
        for model in model_types:
            for nlayer in numOfHiddenLayers:
                for layer_size in layerSizes:
                    for batchsize in input_BatchSize:
                        for trainlen in train_len:
                            for dropout in dropouts:
                                for loss in losses:
                                    train_obj = train_model(model, layer_size, nlayer, loss, batchsize, 
                                                            trainlen, dropout)
                                    namad_obj = get_Namad()
                                    namad_obj.readNamads_fromCSVfiles(namads_CSV_dir)
                                    namad_full_name = namad_obj.get_Namad_fullname_bySome(Namad_name)
                                    namad_df = namad_obj.get_NamadDatasetByName(Namad_name)
                                    print(f"\n\ntraining on: {namad_full_name}\n\n")
                                    train_obj.train_on_Namad(namad_df,
                                                             epochs, train_test_ratio)
                                    train_obj.predict_and_plot(plot_results = False)
                                    metric_results = train_obj.evaluate_metrics(metrics)
                                    row_data = [namad_full_name , 
                                                {loss.__class__.__name__: loss.__dict__},
                                                epochs, train_test_ratio, metric_results, model,
                                                layer_size, nlayer, batchsize, trainlen, train_obj]
                                    results_df.loc[len(results_df)] = row_data
                                    clear_output()
                                    del train_obj, namad_obj
                                    print("\n\ndone\n\n")
        return results_df
    
    
    def train_on_AllNamads(get_Namad_object:get_Namad, save_as_pickle = True, 
                           filename2save = "BulkRunResult_Namads",
                           saveAsCSV = True ,**kwargs):
        
        columns = ["namad_name", "loss", "epochs", "train_test_ratio", "metrics", "model", "layerSize",
                   "numOfHiddenLayers", "input_batchSize", "train_len", "train_obj"]
        
        total_results_df = pd.DataFrame(columns = columns)
        for namad_name in get_Namad_object.Namads_data["names"]:
            df = train_model._bulk_train_on_Namad(namad_name, **kwargs)
            total_results_df = pd.concat([total_results_df, df], axis=0, ignore_index = True)
        
        if save_as_pickle: 
            filename = filename2save+".txt"  
            if filename in os.listdir(): os.remove(filename)
            with open(filename, "wb") as file:
                pickle.dump(total_results_df, file)
                file.close()    
        if saveAsCSV: total_results_df.to_csv(filename2save+".csv")
        return total_results_df 
    
    
    def get_namads_ts_TrainTestdata(all_namads_df_list:list, train_test_ratio = 0.2):
        ts_train_list = ts_test_list = []
        for namad_df in all_namads_df_list:
            ts = TimeSeries.from_dataframe(namad_df, freq='1D',fill_missing_dates = False)
            ts_train, ts_test = ts.split_after(1-train_test_ratio)
            ts_train_list.append(ts_train)
            ts_test_list.append(ts_test)
        return ts_train_list, ts_test_list


    
    def fit_nModel_onAll_Namads(models_list:list,
                                all_namads_df_list:list,
                                train_test_ratio = 0.2, epochs = 3,
                                inplace:bool = True):
        
        ts_list_train, ts_list_test = train_model.get_namads_ts_TrainTestdata(
                                                                all_namads_df_list,
                                                                train_test_ratio=train_test_ratio)

        fitted_models = [obj.model.fit(ts_list_train, epochs = epochs) for obj in models_list]
        
        if inplace: 
            train_model.ts_list_train, train_model.ts_list_test = ts_list_train, ts_list_test
            train_model.fitted_models = fitted_models
        return fitted_models, ts_list_train, ts_list_test
    
    
    def _evaluate_metrics(metrics_list:list, ts_test, y_hat):
        return {metric.__name__: metric(ts_test, y_hat) for metric in metrics_list}
    
    
        
    def predict_plot_namad(self,targetNamadName:str ,  namadNames_list:list[str], nTimestemps = None, which_column:str = None,
                      plot_train_data:bool = False, plot_results:bool = True, inplace:bool = True,
                      metrics_list:list = [mse, mae, r2_score, rmse ]):
        y_hats = []
        metreics_val = []
        
        for i, fitted_model in enumerate(train_model.fitted_models):
            for namad_name, ts_train, ts_test in zip(namadNames_list, 
                                                    train_model.ts_list_train, 
                                                    train_model.ts_list_test):
                if targetNamadName:
                    if targetNamadName in namad_name:
                        y_hat:darts.TimeSeries = fitted_model.predict(nTimestemps if nTimestemps else len(self.ts_test),
                                                                      series = ts_train)
                        y_hats.append([y_hat, train_model._evaluate_metrics(metrics_list,ts_test, 
                                                                            ts_train) ])
                        if plot_results:
                            y_hat.plot(new_plot = True, label = f"{namad_name} estimated- model{i}")
                            ts_test.plot(label = f"{namad_name} test - model{i}")
                            if plot_train_data: 
                                ts_train.plot(label = f"{namad_name} train- model{i}")
                        break
        return y_hats

        
        
                
                        
            
            
            
        
        
        
            
            
        
                                
                                
        
        
        
class PinBall_Loss(torch.nn.Module):

    def __init__(self, tau) -> None:
        assert 0 < tau < 1, "tau must be between 0 and 1"
        super().__init__()
        self.tau = tau
        
    def forward(self, y_hat, y): 
        return torch.where(y >= y_hat, self.tau*(y-y_hat), (1-self.tau)*(y_hat-y) ).mean()
         
        
    def __call__(self, y_hat, y): 
        return torch.where(y >= y_hat, self.tau*(y-y_hat), (1-self.tau)*(y_hat-y) ).mean()
    
    
    
        
        

        
        
        
        
        
    
        
                      
                    
        
            
    
        

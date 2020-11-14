import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

# Dickey-fuller test for stationarity of a series
from statsmodels.tsa.stattools import adfuller

# seasonal decomposition of a signal (trend, seasonal, residuals)
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

if not __name__ == "__main__":
    from utils.visualization import *
    from utils.utils import open_file, save_files


class Dataset:
    def __init__(
        self,
        path,
        sep=";",
        date_col=None,
        excl_cols=[],
        include_pct_change=False,
        encode_categorical=True,
    ):
        self.data = pd.read_csv(path, sep=sep)
        self._params = {
            "include_pct_change": include_pct_change,
            "date_col": date_col,
            "excl_cols": excl_cols,
        }
        self.preprocess(
            path=path,
            sep=sep,
            include_pct_change=include_pct_change,
            encode_categorical=encode_categorical,
        )
        if self.date_col:
            # not before preprocessing as date_col must be in datetime format
            self._params["start_date"] = self.data[self.date_col].iloc[0]
            self._params["end_date"] = self.data[self.date_col].iloc[-1]

    @property
    def include_pct_change(self):
        return self._params["include_pct_change"]

    @property
    def date_col(self):
        return self._params["date_col"]

    def drop(self, columns: list, inplace=False):
        return self.data.drop(columns=columns, inplace=inplace)

    def del_rows(self, start_date, end_date, inplace=True):
        if not self.date_col:
            raise ValueError(
                "Not date column in data was defined. Please define one first"
            )
        new_data = self.data.loc[
            (self.data[self.date_col] >= start_date)
            & (self.data[self.date_col] <= end_date)
        ]
        if start_date > self._params["start_date"]:
            self._params["start_date"] = start_date
        if end_date < self._params["end_date"]:
            self._params["end_date"] = end_date
        if not inplace:
            return new_data
        self.data = new_data

    def __check_cols(self, columns: list, excl_cols: list) -> list:
        if not isinstance(columns, list):
            raise TypeError(
                "Invalid input: columns must be a list of columns names in data"
            )
        if not columns:
            columns = self.data.columns
        columns = list(
            filter(lambda x: x != self.date_col and x not in excl_cols, columns)
        )
        if not columns:
            raise ValueError(
                "Invalid input: no remaining columns to analyse or visualize"
            )
        return columns

    def preprocess(
        self, path, sep=";", include_pct_change=False, encode_categorical=False
    ):
        """
        Categorical to numbers
        create X and Y sets
        remove unuseful columns
        """
        if self.date_col:
            self.data[self.date_col] = pd.to_datetime(self.data[self.date_col])

        if include_pct_change:
            data_pct_change = pd.read_csv(
                path.replace(".csv", "") + "_pct_change.csv", sep=sep
            )
            if self.date_col:
                data_pct_change[self.date_col] = pd.to_datetime(
                    data_pct_change[self.date_col]
                )
            self.data = pd.merge(self.data, data_pct_change, on=self.date_col)

        self._features_info = open_file("./config/features_info.yaml")

        # handling categorical features
        if encode_categorical:
            self.encoders = dict()
            for feature in self._features_info["categorical features"]:
                idx = self.data[feature].notna()
                enc = LabelEncoder().fit(self.data[feature][idx])
                self.data[feature][idx] = enc.transform(self.data[feature][idx])
                self.encoders[feature] = enc

        # drop excluded columns
        self.data.drop(columns=self._params["excl_cols"], inplace=True)

    def visualize_features(
        self,
        columns=[],
        excl_cols=[],
        ncols=3,
        height_per_ax=2,
        subplot_titles_suffix=None,
    ):
        columns = self.__check_cols(columns=columns, excl_cols=excl_cols)
        visualize_features(
            data=self.data,
            date_col=self.date_col,
            columns=columns,
            excl_cols=excl_cols,
            ncols=ncols,
            height_per_ax=height_per_ax,
            subplot_titles_suffix=subplot_titles_suffix,
        )

    def dickey_fuller_test(
        self,
        columns=[],
        excl_cols=[],
    ):
        """ Perfoms Dickey-Fuller test for feature stationarity """
        columns = self.__check_cols(columns=columns, excl_cols=excl_cols)

        res_tests = pd.DataFrame(
            columns=[
                "Test Statistic",
                "p-value",
                "# Lags used",
                "# Obs used",
                "Critical Value 1%",
                "Critical Value 5%",
                "Critical Value 10%",
            ]
        )
        for col_name in columns:
            res = adfuller(self.data[col_name].dropna(), autolag="AIC")
            res_tests.loc[col_name] = list(res[:4]) + list(res[4].values())
        return res_tests

    def visualize_stationarity(
        self,
        columns=[],
        excl_cols=[],
        addfuller_results=None,
        plot_test_results=False,
        ncols=3,
        height_per_ax=2,
        subplot_titles_suffix=None,
    ):
        columns = self.__check_cols(columns=columns, excl_cols=excl_cols)
        visualize_stationarity(
            data=self.data,
            date_col=self.date_col,
            columns=columns,
            excl_cols=excl_cols,
            ncols=ncols,
            height_per_ax=height_per_ax,
            subplot_titles_suffix=subplot_titles_suffix,
            addfuller_results=addfuller_results,
            plot_test_results=plot_test_results,
        )

    def remove_non_stationary_features(
        self, addfuller_results, stat_conf_level="1%", columns=[], excl_cols=[]
    ):
        """
        Remove non stationary columns and keep one among a feature and its % change
        Input: data,
            stat_results: dict with boolean for each column giving stationarity
        Ouput: data with either col_name if feature was stationary, or col_name_pct_change otherwise
        """
        if stat_conf_level is not None and stat_conf_level not in ("1%", "5%", "10%"):
            raise ValueError(
                "Invalid Argument: stat_conf_level must either None or one of '1%', '5%', '10%'"
            )
        if not columns:
            columns = list(addfuller_results.index)
        else:
            columns = list(filter(lambda x: x in addfuller_results.index, columns))
        columns = self.__check_cols(columns=columns, excl_cols=excl_cols)

        non_existing_cols = []
        dropped_cols = []
        kept_cols = []
        conf_level = "Critical Value {:s}".format(stat_conf_level)
        stat_results = addfuller_results.apply(
            lambda x: x["Test Statistic"] < x[conf_level], axis=1
        )
        for col_name in columns:
            try:
                if col_name + "_pct_change" in addfuller_results.keys():
                    if not stat_results[col_name]:
                        self.data.drop(columns=[col_name], inplace=True)
                        kept_cols.append(col_name + "_pct_change")
                        dropped_cols.append(col_name)
                    else:
                        self.data.drop(columns=[col_name + "_pct_change"], inplace=True)
                        kept_cols.append(col_name)
                        dropped_cols.append(col_name + "_pct_change")
                else:
                    if not stat_results[col_name]:
                        self.data.drop(columns=[col_name], inplace=True)
                        dropped_cols.append(col_name)
            except KeyError:
                non_existing_cols.append(col_name)
        if len(dropped_cols) > 0:
            print(" Non-stationarity: following features dropped ".center(120, "-"))
            print("\n".join(dropped_cols))
        if len(non_existing_cols) > 0:
            print(" Following features not found in dataset ".center(120, "-"))
            print("\n".join(non_existing_cols))
        self._params["drop non stationary at " + stat_conf_level] = dropped_cols
        self._params["stat_conf_level"] = stat_conf_level
        return kept_cols, dropped_cols

    def test_stationarity(
        self,
        columns=[],
        excl_cols=[],
        plot_test_results=False,
        ncols=3,
        height_per_ax=2,
        subplot_titles_suffix=None,
    ):
        # columns = self.__check_cols(columns=columns, excl_cols=excl_cols)

        res_tests = self.dickey_fuller_test(columns=columns)
        self.visualize_stationarity(
            columns=columns,
            excl_cols=excl_cols,
            addfuller_results=res_tests,  # will plot text on graphs
            plot_test_results=plot_test_results,
            ncols=ncols,
            height_per_ax=height_per_ax,
            subplot_titles_suffix=subplot_titles_suffix,
        )
        return res_tests

    def seasonal_decomposition(self, columns=[], excl_cols=[]):
        columns = self.__check_cols(columns=columns, excl_cols=excl_cols)

        df_trend = pd.DataFrame(index=self.data[self.date_col])
        df_seas = pd.DataFrame(index=self.data[self.date_col])
        df_resid = pd.DataFrame(index=self.data[self.date_col])
        df = self.data.set_index(self.date_col)
        for col_name in columns:
            col = df[col_name].dropna()
            decomposition = seasonal_decompose(col)
            df_trend[col_name + "_trend"] = decomposition.trend
            df_seas[col_name + "_seasonal"] = decomposition.seasonal
            df_resid[col_name + "_residual"] = decomposition.resid
        df_trend.reset_index(level=[self.date_col], inplace=True)
        df_seas.reset_index(level=[self.date_col], inplace=True)
        df_resid.reset_index(level=[self.date_col], inplace=True)
        return df_trend, df_seas, df_resid

    def visualize_seasonality(
        self,
        df_trend,
        df_seas,
        df_resid,
        columns=[],
        excl_cols=[],
        ncols=3,
        height_per_ax=2,
        subplot_titles_suffix=None,
    ):
        columns = self.__check_cols(columns=columns, excl_cols=excl_cols)

        visualize_seasonality(
            data=self.data,
            df_trend=df_trend,
            df_seas=df_seas,
            df_resid=df_resid,
            date_col=self.date_col,
            columns=columns,
            excl_cols=excl_cols,
            ncols=ncols,
            height_per_ax=height_per_ax,
            subplot_titles_suffix=subplot_titles_suffix,
        )

    def remove_seasonality(self, data_seas, threshold=0.2, columns=[], excl_cols=[]):
        if not columns:
            columns = [c.replace("_seasonal", "") for c in data_seas.columns]
        columns = self.__check_cols(columns=columns, excl_cols=excl_cols)
        if threshold is not None and not 0 < threshold <= 1:
            raise ValueError("Invalid Argument: threshold must in ]0, 1].")
        non_existing_cols = []
        modified_cols = []
        for col_name in columns:
            try:
                if (
                    data_seas[col_name + "_seasonal"].max()
                    > threshold * self.data[col_name].std()
                ):
                    self.data[col_name] = (
                        self.data[col_name] - data_seas[col_name + "_seasonal"]
                    )
                    modified_cols.append(col_name)
            except KeyError:
                non_existing_cols.append(col_name)
        if len(modified_cols) > 0:
            print(" Seasonality removed from the following features ".center(120, "-"))
            print("\n".join(modified_cols))
        if len(non_existing_cols) > 0:
            print(
                " Following features not found in dataset or seasonal data ".center(
                    120, "-"
                )
            )
            print("\n".join(non_existing_cols))
        self._params["removed seasonality"] = modified_cols
        self._params["seasonality_threshold"] = threshold
        return modified_cols

    def test_seasonality(
        self,
        columns=[],
        excl_cols=[],
        plot_graphs=False,
        ncols=3,
        height_per_ax=2,
        subplot_titles_suffix=None,
    ):
        # columns = self.__check_cols(columns=columns, excl_cols=excl_cols)

        df_trend, df_seas, df_resid = self.seasonal_decomposition(
            columns=columns, excl_cols=excl_cols
        )
        if plot_graphs:
            self.visualize_seasonality(
                df_trend=df_trend,
                df_seas=df_seas,
                df_resid=df_resid,
                columns=columns,
                excl_cols=excl_cols,
                ncols=ncols,
                height_per_ax=height_per_ax,
                subplot_titles_suffix=subplot_titles_suffix,
            )
        return df_trend, df_seas, df_resid

    def vif_analysis(self, columns=[], excl_cols=[], threshold=10):
        columns = self.__check_cols(columns=columns, excl_cols=excl_cols)

        excl_columns_vif = {}
        data_vif = self.data[columns]
        data_vif = data_vif.dropna()
        n_cols = len(data_vif.columns)
        print(" VIF analysis starting: {:d} features ".format(n_cols).center(120, "-"))

        while True:
            if n_cols <= 1:
                print(
                    " VIF analysis stopped: only one feature remaining ".center(
                        120, "-"
                    )
                )
                vif_final = pd.DataFrame({"features": data_vif.columns, "vif": [None]})
                break
            vif_values = [
                variance_inflation_factor(data_vif.values, i)
                for i in range(data_vif.shape[1])
            ]
            max_vif_idx = np.argmax(vif_values)
            max_vif_value = vif_values[max_vif_idx]
            if max_vif_value <= threshold:
                vif_final = pd.DataFrame(
                    {"features": data_vif.columns, "vif": vif_values}
                )
                break
            max_vif_feature = data_vif.columns[max_vif_idx]
            excl_columns_vif[max_vif_feature] = max_vif_value
            data_vif.drop(columns=[max_vif_feature], inplace=True)
            n_cols -= 1

        print(" VIF analysis succesfully completed ".center(120, "-"))
        print("remaining features: {:d}".format(n_cols))
        print("excluded features: {:d}".format(len(excl_columns_vif)))
        return vif_final, excl_columns_vif

    def shift_features(self, row_shifts=(1), columns=[], excl_cols=[], inplace=False):
        columns = self.__check_cols(columns=columns, excl_cols=excl_cols)

        data_shifted = pd.DataFrame(index=self.data.index)
        for col_name in columns:
            for t in row_shifts:
                data_shifted[col_name + "_t-{:d}".format(t)] = self.data[
                    col_name
                ].shift(t)
        if not inplace:
            return data_shifted
        self.data = pd.concat([self.data, data_shifted], axis=1)
        self._params["row_shifts"] = list(row_shifts)

    def train_test_split_dates(
        self, X_cols=[], Y_cols=[], test_date=None, test_size=0.2, standardize=True
    ):
        if not self.date_col:
            raise ValueError(
                "Not date column in data was defined. Please define one first"
            )
        if not X_cols and not Y_cols:
            raise ValueError(
                "Invalid input: X_cols, y_cols are empty. Please input at least one of the two"
            )
        if Y_cols:
            self.Y = self.data[[self.date_col] + Y_cols]
        else:  # not Y_cols means X_cols is not None
            self.Y = self.data.drop(columns=X_cols)
        if X_cols:
            self.X = self.data[[self.date_col] + X_cols]
        else:  # not X_cols means Y_cols is not None
            self.X = self.data.drop(columns=Y_cols)
        if not test_date:
            test_date_idx = int(len(self.X) * test_size)
            test_date = self.X[self.date_col].iloc[test_date_idx]
        X_train = self.X.loc[self.X[self.date_col] < test_date]
        X_test = self.X.loc[self.X[self.date_col] >= test_date]
        Y_train = self.Y.loc[self.Y[self.date_col] < test_date]
        Y_test = self.Y.loc[self.Y[self.date_col] >= test_date]
        if standardize:
            scaler = StandardScaler()
            cols = [
                c
                for c in self.X.columns
                if c in self._features_info["numeric features"]
            ]
            X_train.loc[:, cols] = scaler.fit_transform(X_train[cols])
            X_test.loc[:, cols] = scaler.transform(X_test[cols])
        self._params["X_cols"] = list(self.X.columns)
        self._params["Y_cols"] = list(self.Y.columns)
        self._params["standardize"] = standardize
        self._params["test_size"] = test_size
        return X_train, Y_train, X_test, Y_test

    def save_dataset(
        self, X_train, Y_train, X_test, Y_test, path="./Models", replace=False
    ):
        files = [X_train, Y_train, X_test, Y_test, self._params]
        file_names = ["X_train.csv", "Y_train.csv", "X_test.csv", "Y_test.csv"]
        file_names.append("params.yaml")
        save_files(path=path, files=dict(zip(file_names, files)), replace=replace)

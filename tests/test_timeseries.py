from unittest import TestCase
import pandas as pd
from pandas._testing import assert_frame_equal
from data_science_utils.timeseries import get_rolling_timeseries


class Test(TestCase):

    def get_test_dataframe(self):
        data = {
            'A': list(range(10)),
            'B': [0.1*i for i in range(10,20)],
            'C': list(range(20, 30)),
            'D': [str(i) for i in range(30, 40)]
        }
        return pd.DataFrame(data)

    def test_get_rolling_timeseries__window_size_one__same_dataframe_is_returned(self):
        test_df = self.get_test_dataframe()
        expected_df = test_df
        rolled_df = get_rolling_timeseries(
            df_x=test_df,
            start_index=0,
            lag=0,
            window_start=0,
            window_end=1)
        print(rolled_df)
        assert len(rolled_df) == len(expected_df)
        assert_frame_equal(expected_df, rolled_df[rolled_df.columns[:-1]])

    def test_get_rolling_timeseries__lag_one_window_size_one__same_dataframe_except_last_row_is_returned_and_window_id_starts_at_1(self):
        test_df = self.get_test_dataframe()
        expected_df = test_df[:-1]
        rolled_df = get_rolling_timeseries(
            df_x=test_df,
            start_index=0,
            lag=1,
            window_start=0,
            window_end=1)
        print(rolled_df)
        assert len(rolled_df) == len(expected_df)
        assert_frame_equal(expected_df, rolled_df[rolled_df.columns[:-1]])
        assert (list(range(1, len(expected_df)+1)) == rolled_df['window_id']).all()

    def test_get_rolling_timeseries__different_lag_window_size_one__same_dataframe_except_last_row_is_returned_and_window_id_starts_at_1(self):
        test_df = self.get_test_dataframe()
        for lag in range(1, len(test_df)):
            expected_df = test_df[:-lag]
            rolled_df = get_rolling_timeseries(
                df_x=test_df,
                start_index=0,
                lag=lag,
                window_start=0,
                window_end=1)
            print(rolled_df)
            assert len(rolled_df) == len(expected_df)
            assert_frame_equal(expected_df, rolled_df[rolled_df.columns[:-1]])
            assert (list(range(1, len(expected_df)+1)) == rolled_df['window_id']).all()

    def test_get_rolling_timeseries__window_size_one_with_different_offsets__dataframe_minus_offset_is_returned(self):
        test_df = self.get_test_dataframe()
        for i in range(1, len(test_df)):
            rolled_df = get_rolling_timeseries(
                df_x=test_df,
                start_index=i,
                lag=0,
                window_start=i,
                window_end=i+1)
            assert len(test_df) - i == len(rolled_df)
            assert (test_df.index[:-i] == rolled_df.window_id).all()
            assert (test_df.A[:-i].values == rolled_df.A.values).all()
            assert (test_df.B[:-i].values == rolled_df.B.values).all()
            assert (test_df.C[:-i].values == rolled_df.C.values).all()

    def test_get_rolling_timeseries__changing_start_index_over_zero__sub_dataframe_is_returned(self):
        test_df = self.get_test_dataframe()
        for i in range(1, len(test_df)):
            rolled_df = get_rolling_timeseries(
                df_x=test_df,
                start_index=i,
                lag=0,
                window_start=0,
                window_end=1)
            print(rolled_df)
            print(test_df.index[:-i])
            assert len(test_df) - i == len(rolled_df)
            assert (test_df.index[:-i] == rolled_df.window_id).all()
            assert (test_df.A[i:].values == rolled_df.A.values).all()
            assert (test_df.B[i:].values == rolled_df.B.values).all()
            assert (test_df.C[i:].values == rolled_df.C.values).all()

    def test_get_rolling_timeseries__changing_start_index_zero__sub_dataframe_is_returned(self):
        test_df = self.get_test_dataframe()
        rolled_df = get_rolling_timeseries(
            df_x=test_df,
            start_index=0,
            lag=0,
            window_start=0,
            window_end=0)
        print(rolled_df)
        print(test_df.index[:])
        assert len(test_df) == len(rolled_df)
        assert (test_df.index[:] == rolled_df.window_id).all()
        assert (test_df.A[:].values == rolled_df.A.values).all()
        assert (test_df.B[:].values == rolled_df.B.values).all()
        assert (test_df.C[:].values == rolled_df.C.values).all()

    def test_get_rolling_timeseries__changing_lag_over_zero__sub_dataframe_is_returned(self):
        test_df = self.get_test_dataframe()
        for i in range(1, len(test_df)):
            rolled_df = get_rolling_timeseries(
                df_x=test_df,
                start_index=0,
                lag=i,
                window_start=0,
                window_end=0)
            assert len(test_df) - i == len(rolled_df)
            assert (test_df.index[:-i] == rolled_df.window_id).all()
            assert (test_df.A[:-i].values == rolled_df.A.values).all()
            assert (test_df.B[:-i].values == rolled_df.B.values).all()
            assert (test_df.C[:-i].values == rolled_df.C.values).all()

    def test_get_rolling_timeseries__changing_lag_zero__sub_dataframe_is_returned(self):
        test_df = self.get_test_dataframe()
        rolled_df = get_rolling_timeseries(
            df_x=test_df,
            start_index=0,
            lag=0,
            window_start=0,
            window_end=0)
        assert len(test_df) == len(rolled_df)
        assert (test_df.index[:] == rolled_df.window_id).all()
        assert (test_df.A[:].values == rolled_df.A.values).all()
        assert (test_df.B[:].values == rolled_df.B.values).all()
        assert (test_df.C[:].values == rolled_df.C.values).all()

    def test_get_rolling_timeseries__increasing_window_size(self):
        test_df = self.get_test_dataframe()
        for i in range(1, len(test_df)):
            rolled_df = get_rolling_timeseries(
                df_x=test_df,
                start_index=i,
                lag=0,
                window_start=0,
                window_end=i)
            assert (i+1)*(len(test_df)-i) == len(rolled_df)
            assert (rolled_df.groupby('window_id').size() == i+1).all()

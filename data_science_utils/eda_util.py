import pandas as pd

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def get_possible_join_columns(df1, df2):
    for dtype in set(df1.dtypes):
        test_df = df1.select_dtypes(include=dtype)
        check_df = df2.select_dtypes(include=dtype)
        max_sim = 0
        column_pair = [None, None]
        for test_col in test_df.columns:
            for check_col in check_df.columns:
                sim = jaccard_similarity(df1[test_col], df2[check_col])
                if max_sim<sim:
                    max_sim = sim
                    column_pair = [test_col, check_col]
    print("Max similarity:", max_sim, "with", column_pair)
    return column_pair

def auto_join(df1, df2):
    join_columns = get_possible_join_columns(df1, df2)
    return pd.merge(df1, df2, left_on=join_columns[0], right_on=join_columns[1], how='inner')

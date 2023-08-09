#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: xgboost2sql.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-05-17
'''

import codecs
import json
import math
import warnings

import xgboost
from decimal import Decimal


class XGBoost2Sql:
    sql_str = '''
    select {0},1 / (1 + exp(-(({1})+({2})))) as score
    from (
    select {0},
    {3}
    from {4})
    '''
    code_str = ''

    def transform(self, xgboost_model, keep_columns=['key'], table_name='data_table', sql_is_format=True,
                  round_decimal=-1):
        """

        Args:
            xgboost_model:训练好的xgboost模型
            keep_columns:使用sql语句进行预测时，需要保留的列。默认主键保留
            table_name:待预测数据集的表名

        Returns:xgboost模型的sql预测语句

        """

        strs = self.get_dump_model(xgboost_model)
        ### 读模型保存为txt的文件
        # with open('xgboost_model', 'r') as f:
        #     strs = f.read()
        ### 读模型保存为txt的文件

        logit = self.get_model_config(xgboost_model)

        # 解析的逻辑
        columns_l = []
        tree_list = strs.split('booster')
        for i in range(1, len(tree_list)):
            tree_str = tree_list[i]
            lines = tree_str.split('\n')
            v_lines = lines[1:-1]

            self.code_str += '--tree' + str(i) + '\n'
            is_right = False
            self.pre_tree(v_lines, is_right, 1, sql_is_format, round_decimal)
            columns_l.append('tree_{}_score'.format(i))
            if i == len(tree_list) - 1:
                self.code_str += '\t\tas tree_{}_score'.format(i)
            else:
                self.code_str += '\t\tas tree_{}_score,\n'.format(i)
        columns = ' + '.join(columns_l)
        self.sql_str = self.sql_str.format(','.join(keep_columns), columns, logit, self.code_str, table_name)
        return self.sql_str

    def get_dump_model(self, xgb_model):
        """

        Args:
            xgb_model:xgboost模型

        Returns:

        """
        if isinstance(xgb_model, xgboost.XGBClassifier):
            xgb_model = xgb_model.get_booster()
        # joblib.dump(xgb_model, 'xgb.ml')
        # xgb_model.dump_model('xgb.txt')
        # xgb_model.save_model('xgb.json')
        ret = xgb_model.get_dump()
        tree_strs = ''
        for i, _ in enumerate(ret):
            tree_strs += 'booster[{}]:\n'.format(i)
            tree_strs += ret[i]
        return tree_strs

    def get_model_config(self, xgb_model):
        """

        Args:
            xgb_model:xgboost模型

        Returns:

        """
        if isinstance(xgb_model, xgboost.XGBClassifier):
            xgb_model = xgb_model.get_booster()

        try:
            ###-math.log((1 / x) - 1)
            x = float(json.loads(xgb_model.save_config())['learner']['learner_model_param']['base_score'])
            return -math.log((1 - x) / x)
        except:
            warnings.warn(
                'xgboost model version less than :: 1.0.0, If the base_score parameter is not 0.5 when developing the model, Insert the base_score value into the formula "-math.log((1-x)/x)" and replace the -0.0 value at +(-0.0) in the first sentence of the generated sql statement with the calculated value')
            warnings.warn(
                'xgboost 模型的版本低于1.0.0，如果开发模型时， base_score 参数不是0.5，请将base_score的参数取值带入"-math.log((1 - x) / x)"公式，计算出的值，替换掉生成的sql语句第1句中的+(-0.0)处的-0.0取值')
            return -0.0

    def pre_tree(self, lines, is_right, n, sql_is_format=True, round_decimal=-1):
        """

        Args:
            lines:二叉树行
            is_right:是否右边
            n:第几层

        Returns:

        """
        n += 1
        res = ''
        if len(lines) <= 1:
            str_row = lines[0].strip()
            if 'leaf=' in str_row:
                tmp = str_row.split('leaf=')
                if len(tmp) > 1:
                    v_value = str(round(Decimal(tmp[1].strip()), round_decimal)) if round_decimal != -1 else tmp[1].strip()
                    if is_right:
                        if sql_is_format:
                            format = '\t' * (n - 1)
                            res = format + 'else\n' + format + '\t' + v_value + '\n' + format + ' end'
                        else:
                            res = ' else ' + v_value + ' end'
                    else:
                        res = '\t' * n + v_value if sql_is_format else v_value
            self.code_str += res + '\n' if sql_is_format else res
            return
        v = lines[0].strip()
        start_index = v.find('[')
        median_index = v.find('<')
        end_index = v.find(']')
        v_name = v[start_index + 1:median_index].strip()
        v_value = str(round(Decimal(v[median_index+1:end_index]), round_decimal)) if round_decimal != -1 else v[median_index+1:end_index]
        ynm = v[end_index + 1:].strip().split(',')
        yes_v = int(ynm[0].replace('yes=', '').strip())
        no_v = int(ynm[1].replace('no=', '').strip())
        miss_v = int(ynm[2].replace('missing=', '').strip())
        z_lines = lines[1:]

        if is_right:
            res = res + '\t' * (n - 1) + 'else' + '\n' if sql_is_format else res + ' else '
        if miss_v == yes_v:
            res = res + '\t' * n + 'case when (' + v_name + ' < ' + v_value + ' or ' + v_name + ' is null' + ') then' if sql_is_format else res + 'case when (' + v_name + ' < ' + v_value + ' or ' + v_name + ' is null' + ') then '
        else:
            res = res + '\t' * n + 'case when (' + v_name + ' < ' + v_value + ' and ' + v_name + ' is null' + ') then' if sql_is_format else res + 'case when (' + v_name + ' < ' + v_value + ' and ' + v_name + ' is null' + ') then '
        self.code_str += res + '\n' if sql_is_format else res
        left_right = self.get_tree_str(z_lines, yes_v, no_v)

        left_lines = left_right[0]
        right_lines = left_right[1]
        self.pre_tree(left_lines, False, n, sql_is_format, round_decimal)
        self.pre_tree(right_lines, True, n, sql_is_format, round_decimal)
        if is_right:
            self.code_str += '\t' * (n - 1) + 'end\n' if sql_is_format else ' end'

    def get_tree_str(self, lines, yes_flag, no_flag):
        """

        Args:
            lines:二叉树行
            yes_flag:左边
            no_flag:右边

        Returns:

        """
        res = []
        left_n = 0
        right_n = 0
        for i in range(len(lines)):
            tmp = lines[i].strip()
            f_index = tmp.find(':')
            next_flag = int(tmp[:f_index])
            if next_flag == yes_flag:
                left_n = i
            if next_flag == no_flag:
                right_n = i
        if right_n > left_n:
            res.append(list(lines[left_n:right_n]))
            res.append(list(lines[right_n:]))
        else:
            res.append(lines[left_n:])
            res.append(lines[right_n:left_n])
        return res

    def save(self, filename='xgb_model.sql'):
        """

        Args:
            filename:sql语句保存的位置

        Returns:

        """
        with codecs.open(filename, 'w', encoding='utf-8') as f:
            f.write(self.sql_str)


if __name__ == '__main__':
    ###训练1个xgboost二分类模型
    import xgboost as xgb
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=10000,
                               n_features=10,
                               n_informative=3,
                               n_redundant=2,
                               n_repeated=0,
                               n_classes=2,
                               weights=[0.7, 0.3],
                               flip_y=0.1,
                               random_state=1024)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1024)

    ###训练模型
    model = xgb.XGBClassifier(n_estimators=3)
    model.fit(X_train, y_train)
    xgb.to_graphviz(model)

    ###使用xgboost2sql包将模型转换成的sql语句
    xgb2sql = XGBoost2Sql()
    # sql_str = xgb2sql.transform(model)
    # sql_str = xgb2sql.transform(model, sql_is_format=False)
    sql_str = xgb2sql.transform(model, sql_is_format=False, round_decimal=4)
    #print(sql_str)
    #xgb2sql.save()

    import pandas as pd

    pd.set_option('display.float_format', lambda x: '%.9f' % x)
    ###使用模型对测试数据集进行预测
    test_pred = model.predict_proba(X_test)[:, 1]
    test_pred = pd.DataFrame(test_pred, columns=['python_pred_res'])
    test_pred.reset_index(inplace=True)
    test_pred.rename(columns={'index': 'key'}, inplace=True)
    print(test_pred)

    ###使用模型转换成的sql语句对测试数据集进行预测
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    X_test_df = pd.DataFrame(X_test)
    X_test_df.columns = X_test_df.columns.map(lambda x: "f" + str(x))
    X_test_df.reset_index(inplace=True)
    X_test_df.rename(columns={'index': 'key'}, inplace=True)
    values = X_test_df.values.tolist()
    columns = X_test_df.columns.tolist()
    spark_df = spark.createDataFrame(values, columns)
    spark_df.createOrReplaceTempView('data_table')
    sql_pred_pysdf = spark.sql(sql_str)
    sql_pred_df = sql_pred_pysdf.toPandas()
    sql_pred_df.head(2)

    ###对比python模型预测出来的结果和sql语句预测出来的结果是否一致
    test_pred_sql_pred_df = test_pred.merge(sql_pred_df, on='key')
    test_pred_sql_pred_df['diff'] = test_pred_sql_pred_df['python_pred_res'] - test_pred_sql_pred_df['score']
    print(test_pred_sql_pred_df['diff'].describe())

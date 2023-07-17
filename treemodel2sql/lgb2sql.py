#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: lgb2sql.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2023-07-14
'''
import codecs

import lightgbm


class Lgb2Sql:
    sql_str = '''
        select {0},1 / (1 + exp(-(({1})+({2})))) as score
        from (
        select {0},
        {3}
        from {4})
        '''

    feature_names = []

    def transform(self, lgb_model, keep_columns=['key'], table_name='data_table'):
        """

        Args:
            lightgbm_model:训练好的lightgbm模型
            keep_columns:使用sql语句进行预测时，需要保留的列。默认主键保留
            table_name:待预测数据集的表名

        Returns:lightgbm模型的sql预测语句

        """

        json_object = self.get_dump_model(lgb_model)
        self.feature_names = json_object['feature_names']
        trees = json_object['tree_info']

        # logit = self.get_model_config(lgb_model)
        logit = -0.0

        # 解析的逻辑
        trees_func_code_l = []
        columns_l = []

        for i, tree in enumerate(trees):
            one_tree_code = self.pre_tree(tree['tree_structure'], 1)
            if i == len(trees) - 1:
                trees_func_code_l.append(
                    '--tree' + str(i) + '\n' + one_tree_code + '\n' + '\t\tas tree_{}_score'.format(i))
            else:
                trees_func_code_l.append(
                    '--tree' + str(i) + '\n' + one_tree_code + '\n' + '\t\tas tree_{}_score,\n'.format(i))

            columns_l.append('tree_{}_score'.format(i))

        columns = ' + '.join(columns_l)
        trees_func_code = ''.join(trees_func_code_l)
        self.sql_str = self.sql_str.format(','.join(keep_columns), columns, logit, trees_func_code, table_name)
        return self.sql_str

    def get_dump_model(self, lgb_model):
        """

        Args:
            lgb_model:lightgbm模型

        Returns:

        """
        if isinstance(lgb_model, lightgbm.LGBMClassifier):
            lgb_model = lgb_model._Booster
        return lgb_model.dump_model()

    def pre_tree(self, node, n):
        """

        Args:
            node:树结构
            n:第几层

        Returns:

        """
        n += 1
        if 'leaf_index' in node:
            return '\t' * n + str(node['leaf_value'])

        condition = []
        split_feature = self.feature_names[node['split_feature']]
        is_default_left = node['default_left']

        if node['decision_type'] == '<=' or node['decision_type'] == 'no_greater':
            threshold = node['threshold']
            if 'e' in str(threshold):
                threshold = format(threshold, 'f')
            condition.append(
                f'{split_feature} is null and {str(is_default_left).lower()}==true or {split_feature}<={threshold}')
        else:
            threshold = node['threshold']
            if 'e' in str(threshold):
                threshold = format(threshold, 'f')
            condition.append(
                f'{split_feature} is null and {str(is_default_left).lower()}==false or {split_feature}=={threshold}')

        left = self.pre_tree(node['left_child'], n)
        right = self.pre_tree(node['right_child'], n)

        strformat = '\t' * n
        return f'{strformat}case when ({condition[0]}) then\n{left}\n{strformat}else\n{right}\n{strformat}end'

    def save(self, filename='lgb_model.sql'):
        """

        Args:
            filename:sql语句保存的位置

        Returns:

        """
        with codecs.open(filename, 'w', encoding='utf-8') as f:
            f.write(self.sql_str)


if __name__ == '__main__':
    ###训练1个lightgbm二分类模型
    import lightgbm as lgb
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
    model = lgb.LGBMClassifier(n_estimators=3)
    model.fit(X_train, y_train)
    lgb.create_tree_digraph(model._Booster, tree_index=0)

    ###使用treemodel2sql包将模型转换成的sql语句
    lgb2sql = Lgb2Sql()
    sql_str = lgb2sql.transform(model)
    print(sql_str)
    lgb2sql.save()

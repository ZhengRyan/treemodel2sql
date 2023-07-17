# XGBoost和LightGBM模型转sql语句工具包
现在是大数据量的时代，我们开发的模型要应用在特别大的待预测集上，使用单机的python，需要预测2、3天，甚至更久，中途很有可能中断。因此需要通过分布式的方式来预测。这个工具包就是实现了将训练好的python模型，转换成sql语句。将生成的sql语句可以放到大数据环境中进行分布式执行预测，能比单机的python预测快好几个量级


## 思想碰撞

|  微信 |  微信公众号 |
| :---: | :----: |
| <img src="https://github.com/ZhengRyan/autotreemodel/blob/master/images/%E5%B9%B2%E9%A5%AD%E4%BA%BA.png" alt="RyanZheng.png" width="50%" border=0/> | <img src="https://github.com/ZhengRyan/autotreemodel/blob/master/images/%E9%AD%94%E9%83%BD%E6%95%B0%E6%8D%AE%E5%B9%B2%E9%A5%AD%E4%BA%BA.png" alt="魔都数据干饭人.png" width="50%" border=0/> |
|  干饭人  | 魔都数据干饭人 |


> 仓库地址：https://github.com/ZhengRyan/treemodel2sql
> 
> 微信公众号文章：
> 
> pipy包：https://pypi.org/project/treemodel2sql/



## 环境准备
可以不用单独创建虚拟环境，因为对包的依赖没有版本要求

### `treemodel2sql` 安装
pip install（pip安装）

```bash
pip install treemodel2sql # to install
pip install -U treemodel2sql # to upgrade
```

Source code install（源码安装）

```bash
python setup.py install
```

## 运行样例
###【注意：：：核验对比python模型预测出来的结果和sql语句预测出来的结果是否一致请查看教程代码】"https://github.com/ZhengRyan/treemodel2sql/tree/master/examples"路径处的tutorial_code_lightgbm_model_transform_sql.ipynb和tutorial_code_xgboost_model_transform_sql.ipynb

## xgboost model transform sql
+ 导入相关依赖
```python
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
```

+ 训练1个xgboost二分类模型
```python
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
#xgb.to_graphviz(model)
```
+ 使用xgboost2sql工具包将模型转换成的sql语句
```python
from xgboost2sql import XGBoost2Sql
xgb2sql = XGBoost2Sql()
sql_str = xgb2sql.transform(model)
```

+ 将sql语句保存
```python
xgb2sql.save()
```

+ 将sql语句打印出来
```python
print(sql_str)
```

```sql
select key,1 / (1 + exp(-((tree_0_score + tree_1_score + tree_2_score)+(-0.0)))) as score
    from (
    select key,
    --tree0
    case when (Column_9 is null and true==true or Column_9<=-1.6632177658322267) then
        case when (Column_3 is null and true==true or Column_3<=-1.5495347315228287) then
            -0.8762269835062461
        else
            case when (Column_3 is null and true==true or Column_3<=1.3809327191663714) then
                case when (Column_7 is null and true==true or Column_7<=-0.9295802204706084) then
                    -0.5897045622608377
                else
                    -0.45043755217654896
                end
            else
                -0.8259039361137315
            end
        end
    else
        case when (Column_9 is null and true==true or Column_9<=1.6094262296522934) then
            case when (Column_3 is null and true==true or Column_3<=0.5872311864187175) then
                case when (Column_5 is null and true==true or Column_5<=0.6614392939376948) then
                    case when (Column_7 is null and true==true or Column_7<=-0.7719746626173067) then
                        case when (Column_3 is null and true==true or Column_3<=-0.47247338144095047) then
                            case when (Column_8 is null and true==true or Column_8<=1.366961884973301) then
                                -0.4786799717862057
                            else
                                -0.7246405208927198
                            end
                        else
                            case when (Column_5 is null and true==true or Column_5<=0.33583497995253037) then
                                -0.8576174849758683
                            else
                                case when (Column_8 is null and true==true or Column_8<=0.725660478148893) then
                                    -0.45255653873335094
                                else
                                    -0.7268525857883245
                                end
                            end
                        end
                    else
                        case when (Column_3 is null and true==true or Column_3<=-1.1870271596234325) then
                            -0.8614608311023392
                        else
                            case when (Column_3 is null and true==true or Column_3<=0.3450499438974393) then
                                case when (Column_9 is null and true==true or Column_9<=0.719805683840674) then
                                    -0.452301345707117
                                else
                                    case when (Column_3 is null and true==true or Column_3<=0.24246683106041103) then
                                        -0.5283016889821918
                                    else
                                        -0.7127023928846948
                                    end
                                end
                            else
                                case when (Column_8 is null and true==true or Column_8<=-0.41161510713991156) then
                                    -0.4296985348121034
                                else
                                    case when (Column_5 is null and true==true or Column_5<=0.10191127467897389) then
                                        -0.8660786096722881
                                    else
                                        -0.7213351365659543
                                    end
                                end
                            end
                        end
                    end
                else
                    case when (Column_7 is null and true==true or Column_7<=0.1335285190014809) then
                        case when (Column_8 is null and true==true or Column_8<=0.3151386351229311) then
                            -0.7813601336005307
                        else
                            case when (Column_8 is null and true==true or Column_8<=0.797889562545179) then
                                case when (Column_7 is null and true==true or Column_7<=-0.6119464230811894) then
                                    -0.47541454265459887
                                else
                                    case when (Column_9 is null and true==true or Column_9<=1.036435633602195) then
                                        -0.5335985526359568
                                    else
                                        -0.7562414479727859
                                    end
                                end
                            else
                                -0.4911879020856353
                            end
                        end
                    else
                        case when (Column_8 is null and true==true or Column_8<=-0.2260123021962532) then
                            -0.4495750599610143
                        else
                            case when (Column_5 is null and true==true or Column_5<=1.3421825678767703) then
                                -0.8610329200059168
                            else
                                case when (Column_7 is null and true==true or Column_7<=0.9404699950203693) then
                                    -0.8282483467723211
                                else
                                    -0.6811365779458289
                                end
                            end
                        end
                    end
                end
            else
                case when (Column_7 is null and true==true or Column_7<=2.2029620939306103) then
                    case when (Column_8 is null and true==true or Column_8<=-0.20972527484810635) then
                        -0.8620295284631452
                    else
                        -0.6406954940851597
                    end
                else
                    -0.49244599655670496
                end
            end
        else
            case when (Column_3 is null and true==true or Column_3<=1.7659981730252496) then
                case when (Column_7 is null and true==true or Column_7<=-0.709846883702808) then
                    -0.44913417585419935
                else
                    -0.7039945818670766
                end
            else
                -0.8609816276658351
            end
        end
    end
    as tree_0_score,
--tree1
    case when (Column_9 is null and true==true or Column_9<=-1.6632177658322267) then
        case when (Column_3 is null and true==true or Column_3<=-1.5495347315228287) then
            -0.1304300334944157
        else
            case when (Column_3 is null and true==true or Column_3<=1.084128912605023) then
                0.23707714508016445
            else
                -0.0285157252239888
            end
        end
    else
        case when (Column_9 is null and true==true or Column_9<=1.6094262296522934) then
            case when (Column_3 is null and true==true or Column_3<=0.5872311864187175) then
                case when (Column_5 is null and true==true or Column_5<=0.6614392939376948) then
                    case when (Column_7 is null and true==true or Column_7<=-0.9295802204706084) then
                        case when (Column_3 is null and true==true or Column_3<=-0.47247338144095047) then
                            case when (Column_9 is null and true==true or Column_9<=0.6831935640096413) then
                                0.19417988489655402
                            else
                                -0.02440020590879623
                            end
                        else
                            case when (Column_9 is null and true==true or Column_9<=1.036435633602195) then
                                -0.11286120415008301
                            else
                                0.08687819192210895
                            end
                        end
                    else
                        case when (Column_3 is null and true==true or Column_3<=-1.1870271596234325) then
                            -0.11564799847528345
                        else
                            case when (Column_9 is null and true==true or Column_9<=0.3004389492550987) then
                                0.2405259765006961
                            else
                                case when (Column_3 is null and true==true or Column_3<=0.2158166423973093) then
                                    case when (Column_7 is null and true==true or Column_7<=-0.6547653704267197) then
                                        0.11054725461728983
                                    else
                                        0.2269735323256574
                                    end
                                else
                                    case when (Column_5 is null and true==true or Column_5<=0.10191127467897389) then
                                        -0.1208387319652229
                                    else
                                        case when (Column_1 is null and true==true or Column_1<=0.6151636033969191) then
                                            case when (Column_8 is null and true==true or Column_8<=-0.018071874612147206) then
                                                0.0016586271011360867
                                            else
                                                0.1332964744950624
                                            end
                                        else
                                            -0.04300639955361431
                                        end
                                    end
                                end
                            end
                        end
                    end
                else
                    case when (Column_7 is null and true==true or Column_7<=0.1335285190014809) then
                        case when (Column_8 is null and true==true or Column_8<=0.27824194846895633) then
                            -0.0613742224255702
                        else
                            case when (Column_3 is null and true==true or Column_3<=-0.37785910715343846) then
                                0.23646592246665976
                            else
                                case when (Column_7 is null and true==true or Column_7<=-0.3752232334694377) then
                                    case when (Column_8 is null and true==true or Column_8<=1.2531231065059638) then
                                        0.20734413117883457
                                    else
                                        0.08650249777206469
                                    end
                                else
                                    case when (Column_6 is null and true==true or Column_6<=0.1864449772206759) then
                                        0.0014253268716679143
                                    else
                                        0.19695058122999737
                                    end
                                end
                            end
                        end
                    else
                        case when (Column_8 is null and true==true or Column_8<=-0.2260123021962532) then
                            0.23847961174940435
                        else
                            case when (Column_5 is null and true==true or Column_5<=1.3421825678767703) then
                                -0.115222500730401
                            else
                                case when (Column_8 is null and true==true or Column_8<=0.1800136296190439) then
                                    0.060703267253890685
                                else
                                    -0.08603393258665121
                                end
                            end
                        end
                    end
                end
            else
                case when (Column_7 is null and true==true or Column_7<=2.0654353175567177) then
                    case when (Column_8 is null and true==true or Column_8<=-0.20972527484810635) then
                        -0.11660912729818569
                    else
                        0.08556190442726179
                    end
                else
                    0.19286044838453295
                end
            end
        else
            case when (Column_3 is null and true==true or Column_3<=1.7659981730252496) then
                case when (Column_7 is null and true==true or Column_7<=-0.709846883702808) then
                    0.23881782889986833
                else
                    0.031196179358762684
                end
            else
                -0.11517151290097437
            end
        end
    end
    as tree_1_score,
--tree2
    case when (Column_9 is null and true==true or Column_9<=-1.6632177658322267) then
        case when (Column_3 is null and true==true or Column_3<=-1.5495347315228287) then
            -0.1246790381676725
        else
            case when (Column_3 is null and true==true or Column_3<=1.3809327191663714) then
                case when (Column_3 is null and true==true or Column_3<=0.7880730732430219) then
                    0.20644395671242038
                else
                    0.1189732213687583
                end
            else
                -0.0788791406092519
            end
        end
    else
        case when (Column_9 is null and true==true or Column_9<=1.6094262296522934) then
            case when (Column_3 is null and true==true or Column_3<=0.6038877970616275) then
                case when (Column_5 is null and true==true or Column_5<=0.6614392939376948) then
                    case when (Column_7 is null and true==true or Column_7<=-0.9295802204706084) then
                        case when (Column_3 is null and true==true or Column_3<=-0.47247338144095047) then
                            case when (Column_7 is null and true==true or Column_7<=-1.6012773225012003) then
                                case when (Column_9 is null and true==true or Column_9<=0.5166566313259823) then
                                    0.14609631025061118
                                else
                                    -0.08519567041375777
                                end
                            else
                                0.20775467159820815
                            end
                        else
                            case when (Column_9 is null and true==true or Column_9<=1.013669661144344) then
                                -0.10698219135389433
                            else
                                0.06871478156256401
                            end
                        end
                    else
                        case when (Column_3 is null and true==true or Column_3<=-1.3968501345470952) then
                            -0.11575647257046306
                        else
                            case when (Column_3 is null and true==true or Column_3<=0.3381305759653203) then
                                case when (Column_9 is null and true==true or Column_9<=0.6711065070274237) then
                                    0.2055124396523362
                                else
                                    case when (Column_3 is null and true==true or Column_3<=0.08416177325543671) then
                                        0.18486979450427765
                                    else
                                        case when (Column_5 is null and true==true or Column_5<=0.4068544493018646) then
                                            -0.04785536117613759
                                        else
                                            0.11879752654423877
                                        end
                                    end
                                end
                            else
                                case when (Column_8 is null and true==true or Column_8<=-0.5159476948660279) then
                                    0.22110765206038635
                                else
                                    case when (Column_9 is null and true==true or Column_9<=0.7486693174493504) then
                                        -0.07426890908854177
                                    else
                                        0.12667195625568606
                                    end
                                end
                            end
                        end
                    end
                else
                    case when (Column_7 is null and true==true or Column_7<=0.1335285190014809) then
                        case when (Column_8 is null and true==true or Column_8<=0.24202650482335777) then
                            -0.07289970951054063
                        else
                            case when (Column_8 is null and true==true or Column_8<=0.797889562545179) then
                                case when (Column_7 is null and true==true or Column_7<=-0.6119464230811894) then
                                    0.19832855719431475
                                else
                                    0.05525269245438189
                                end
                            else
                                0.17963780553305242
                            end
                        end
                    else
                        case when (Column_8 is null and true==true or Column_8<=-0.25745431183810524) then
                            0.20767489164465192
                        else
                            case when (Column_5 is null and true==true or Column_5<=1.3421825678767703) then
                                -0.10914096073415368
                            else
                                case when (Column_9 is null and true==true or Column_9<=1.1239281866391186) then
                                    0.04590138464892762
                                else
                                    -0.08877584425107066
                                end
                            end
                        end
                    end
                end
            else
                case when (Column_7 is null and true==true or Column_7<=2.2516563501714586) then
                    case when (Column_8 is null and true==true or Column_8<=-0.20972527484810635) then
                        -0.11032211105487942
                    else
                        0.06868549293845214
                    end
                else
                    0.18271175459060623
                end
            end
        else
            case when (Column_3 is null and true==true or Column_3<=1.7659981730252496) then
                case when (Column_7 is null and true==true or Column_7<=-0.709846883702808) then
                    0.2062125352630051
                else
                    0.027771830564502133
                end
            else
                -0.10919849076791249
            end
        end
    end
    as tree_2_score
    from data_table)
```

## lightgbm model transform sql
+ 导入相关依赖
```python
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
```

+ 训练1个lightgbm二分类模型
```python
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
```
+ 使用lgb2sql工具包将模型转换成的sql语句
```python
from lgb2sql import Lgb2Sql
lgb2sql = Lgb2Sql()
sql_str = lgb2sql.transform(model)
```

+ 将sql语句保存
```python
lgb2sql.save()
```

+ 将sql语句打印出来
```python
print(sql_str)
```

```sql
select key,1 / (1 + exp(-((tree_0_score + tree_1_score + tree_2_score)+(-0.0)))) as score
    from (
    select key,
    --tree0
    case when (Column_9 is null and true==true or Column_9<=-1.6632177658322267) then
        case when (Column_3 is null and true==true or Column_3<=-1.5495347315228287) then
            -0.8762269835062461
        else
            case when (Column_3 is null and true==true or Column_3<=1.3809327191663714) then
                case when (Column_7 is null and true==true or Column_7<=-0.9295802204706084) then
                    -0.5897045622608377
                else
                    -0.45043755217654896
                end
            else
                -0.8259039361137315
            end
        end
    else
        case when (Column_9 is null and true==true or Column_9<=1.6094262296522934) then
            case when (Column_3 is null and true==true or Column_3<=0.5872311864187175) then
                case when (Column_5 is null and true==true or Column_5<=0.6614392939376948) then
                    case when (Column_7 is null and true==true or Column_7<=-0.7719746626173067) then
                        case when (Column_3 is null and true==true or Column_3<=-0.47247338144095047) then
                            case when (Column_8 is null and true==true or Column_8<=1.366961884973301) then
                                -0.4786799717862057
                            else
                                -0.7246405208927198
                            end
                        else
                            case when (Column_5 is null and true==true or Column_5<=0.33583497995253037) then
                                -0.8576174849758683
                            else
                                case when (Column_8 is null and true==true or Column_8<=0.725660478148893) then
                                    -0.45255653873335094
                                else
                                    -0.7268525857883245
                                end
                            end
                        end
                    else
                        case when (Column_3 is null and true==true or Column_3<=-1.1870271596234325) then
                            -0.8614608311023392
                        else
                            case when (Column_3 is null and true==true or Column_3<=0.3450499438974393) then
                                case when (Column_9 is null and true==true or Column_9<=0.719805683840674) then
                                    -0.452301345707117
                                else
                                    case when (Column_3 is null and true==true or Column_3<=0.24246683106041103) then
                                        -0.5283016889821918
                                    else
                                        -0.7127023928846948
                                    end
                                end
                            else
                                case when (Column_8 is null and true==true or Column_8<=-0.41161510713991156) then
                                    -0.4296985348121034
                                else
                                    case when (Column_5 is null and true==true or Column_5<=0.10191127467897389) then
                                        -0.8660786096722881
                                    else
                                        -0.7213351365659543
                                    end
                                end
                            end
                        end
                    end
                else
                    case when (Column_7 is null and true==true or Column_7<=0.1335285190014809) then
                        case when (Column_8 is null and true==true or Column_8<=0.3151386351229311) then
                            -0.7813601336005307
                        else
                            case when (Column_8 is null and true==true or Column_8<=0.797889562545179) then
                                case when (Column_7 is null and true==true or Column_7<=-0.6119464230811894) then
                                    -0.47541454265459887
                                else
                                    case when (Column_9 is null and true==true or Column_9<=1.036435633602195) then
                                        -0.5335985526359568
                                    else
                                        -0.7562414479727859
                                    end
                                end
                            else
                                -0.4911879020856353
                            end
                        end
                    else
                        case when (Column_8 is null and true==true or Column_8<=-0.2260123021962532) then
                            -0.4495750599610143
                        else
                            case when (Column_5 is null and true==true or Column_5<=1.3421825678767703) then
                                -0.8610329200059168
                            else
                                case when (Column_7 is null and true==true or Column_7<=0.9404699950203693) then
                                    -0.8282483467723211
                                else
                                    -0.6811365779458289
                                end
                            end
                        end
                    end
                end
            else
                case when (Column_7 is null and true==true or Column_7<=2.2029620939306103) then
                    case when (Column_8 is null and true==true or Column_8<=-0.20972527484810635) then
                        -0.8620295284631452
                    else
                        -0.6406954940851597
                    end
                else
                    -0.49244599655670496
                end
            end
        else
            case when (Column_3 is null and true==true or Column_3<=1.7659981730252496) then
                case when (Column_7 is null and true==true or Column_7<=-0.709846883702808) then
                    -0.44913417585419935
                else
                    -0.7039945818670766
                end
            else
                -0.8609816276658351
            end
        end
    end
    as tree_0_score,
--tree1
    case when (Column_9 is null and true==true or Column_9<=-1.6632177658322267) then
        case when (Column_3 is null and true==true or Column_3<=-1.5495347315228287) then
            -0.1304300334944157
        else
            case when (Column_3 is null and true==true or Column_3<=1.084128912605023) then
                0.23707714508016445
            else
                -0.0285157252239888
            end
        end
    else
        case when (Column_9 is null and true==true or Column_9<=1.6094262296522934) then
            case when (Column_3 is null and true==true or Column_3<=0.5872311864187175) then
                case when (Column_5 is null and true==true or Column_5<=0.6614392939376948) then
                    case when (Column_7 is null and true==true or Column_7<=-0.9295802204706084) then
                        case when (Column_3 is null and true==true or Column_3<=-0.47247338144095047) then
                            case when (Column_9 is null and true==true or Column_9<=0.6831935640096413) then
                                0.19417988489655402
                            else
                                -0.02440020590879623
                            end
                        else
                            case when (Column_9 is null and true==true or Column_9<=1.036435633602195) then
                                -0.11286120415008301
                            else
                                0.08687819192210895
                            end
                        end
                    else
                        case when (Column_3 is null and true==true or Column_3<=-1.1870271596234325) then
                            -0.11564799847528345
                        else
                            case when (Column_9 is null and true==true or Column_9<=0.3004389492550987) then
                                0.2405259765006961
                            else
                                case when (Column_3 is null and true==true or Column_3<=0.2158166423973093) then
                                    case when (Column_7 is null and true==true or Column_7<=-0.6547653704267197) then
                                        0.11054725461728983
                                    else
                                        0.2269735323256574
                                    end
                                else
                                    case when (Column_5 is null and true==true or Column_5<=0.10191127467897389) then
                                        -0.1208387319652229
                                    else
                                        case when (Column_1 is null and true==true or Column_1<=0.6151636033969191) then
                                            case when (Column_8 is null and true==true or Column_8<=-0.018071874612147206) then
                                                0.0016586271011360867
                                            else
                                                0.1332964744950624
                                            end
                                        else
                                            -0.04300639955361431
                                        end
                                    end
                                end
                            end
                        end
                    end
                else
                    case when (Column_7 is null and true==true or Column_7<=0.1335285190014809) then
                        case when (Column_8 is null and true==true or Column_8<=0.27824194846895633) then
                            -0.0613742224255702
                        else
                            case when (Column_3 is null and true==true or Column_3<=-0.37785910715343846) then
                                0.23646592246665976
                            else
                                case when (Column_7 is null and true==true or Column_7<=-0.3752232334694377) then
                                    case when (Column_8 is null and true==true or Column_8<=1.2531231065059638) then
                                        0.20734413117883457
                                    else
                                        0.08650249777206469
                                    end
                                else
                                    case when (Column_6 is null and true==true or Column_6<=0.1864449772206759) then
                                        0.0014253268716679143
                                    else
                                        0.19695058122999737
                                    end
                                end
                            end
                        end
                    else
                        case when (Column_8 is null and true==true or Column_8<=-0.2260123021962532) then
                            0.23847961174940435
                        else
                            case when (Column_5 is null and true==true or Column_5<=1.3421825678767703) then
                                -0.115222500730401
                            else
                                case when (Column_8 is null and true==true or Column_8<=0.1800136296190439) then
                                    0.060703267253890685
                                else
                                    -0.08603393258665121
                                end
                            end
                        end
                    end
                end
            else
                case when (Column_7 is null and true==true or Column_7<=2.0654353175567177) then
                    case when (Column_8 is null and true==true or Column_8<=-0.20972527484810635) then
                        -0.11660912729818569
                    else
                        0.08556190442726179
                    end
                else
                    0.19286044838453295
                end
            end
        else
            case when (Column_3 is null and true==true or Column_3<=1.7659981730252496) then
                case when (Column_7 is null and true==true or Column_7<=-0.709846883702808) then
                    0.23881782889986833
                else
                    0.031196179358762684
                end
            else
                -0.11517151290097437
            end
        end
    end
    as tree_1_score,
--tree2
    case when (Column_9 is null and true==true or Column_9<=-1.6632177658322267) then
        case when (Column_3 is null and true==true or Column_3<=-1.5495347315228287) then
            -0.1246790381676725
        else
            case when (Column_3 is null and true==true or Column_3<=1.3809327191663714) then
                case when (Column_3 is null and true==true or Column_3<=0.7880730732430219) then
                    0.20644395671242038
                else
                    0.1189732213687583
                end
            else
                -0.0788791406092519
            end
        end
    else
        case when (Column_9 is null and true==true or Column_9<=1.6094262296522934) then
            case when (Column_3 is null and true==true or Column_3<=0.6038877970616275) then
                case when (Column_5 is null and true==true or Column_5<=0.6614392939376948) then
                    case when (Column_7 is null and true==true or Column_7<=-0.9295802204706084) then
                        case when (Column_3 is null and true==true or Column_3<=-0.47247338144095047) then
                            case when (Column_7 is null and true==true or Column_7<=-1.6012773225012003) then
                                case when (Column_9 is null and true==true or Column_9<=0.5166566313259823) then
                                    0.14609631025061118
                                else
                                    -0.08519567041375777
                                end
                            else
                                0.20775467159820815
                            end
                        else
                            case when (Column_9 is null and true==true or Column_9<=1.013669661144344) then
                                -0.10698219135389433
                            else
                                0.06871478156256401
                            end
                        end
                    else
                        case when (Column_3 is null and true==true or Column_3<=-1.3968501345470952) then
                            -0.11575647257046306
                        else
                            case when (Column_3 is null and true==true or Column_3<=0.3381305759653203) then
                                case when (Column_9 is null and true==true or Column_9<=0.6711065070274237) then
                                    0.2055124396523362
                                else
                                    case when (Column_3 is null and true==true or Column_3<=0.08416177325543671) then
                                        0.18486979450427765
                                    else
                                        case when (Column_5 is null and true==true or Column_5<=0.4068544493018646) then
                                            -0.04785536117613759
                                        else
                                            0.11879752654423877
                                        end
                                    end
                                end
                            else
                                case when (Column_8 is null and true==true or Column_8<=-0.5159476948660279) then
                                    0.22110765206038635
                                else
                                    case when (Column_9 is null and true==true or Column_9<=0.7486693174493504) then
                                        -0.07426890908854177
                                    else
                                        0.12667195625568606
                                    end
                                end
                            end
                        end
                    end
                else
                    case when (Column_7 is null and true==true or Column_7<=0.1335285190014809) then
                        case when (Column_8 is null and true==true or Column_8<=0.24202650482335777) then
                            -0.07289970951054063
                        else
                            case when (Column_8 is null and true==true or Column_8<=0.797889562545179) then
                                case when (Column_7 is null and true==true or Column_7<=-0.6119464230811894) then
                                    0.19832855719431475
                                else
                                    0.05525269245438189
                                end
                            else
                                0.17963780553305242
                            end
                        end
                    else
                        case when (Column_8 is null and true==true or Column_8<=-0.25745431183810524) then
                            0.20767489164465192
                        else
                            case when (Column_5 is null and true==true or Column_5<=1.3421825678767703) then
                                -0.10914096073415368
                            else
                                case when (Column_9 is null and true==true or Column_9<=1.1239281866391186) then
                                    0.04590138464892762
                                else
                                    -0.08877584425107066
                                end
                            end
                        end
                    end
                end
            else
                case when (Column_7 is null and true==true or Column_7<=2.2516563501714586) then
                    case when (Column_8 is null and true==true or Column_8<=-0.20972527484810635) then
                        -0.11032211105487942
                    else
                        0.06868549293845214
                    end
                else
                    0.18271175459060623
                end
            end
        else
            case when (Column_3 is null and true==true or Column_3<=1.7659981730252496) then
                case when (Column_7 is null and true==true or Column_7<=-0.709846883702808) then
                    0.2062125352630051
                else
                    0.027771830564502133
                end
            else
                -0.10919849076791249
            end
        end
    end
    as tree_2_score
    from data_table)
```

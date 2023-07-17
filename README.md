# XGBoost和LightGBM模型转sql语句工具包
现在是大数据量的时代，我们开发的模型要应用在特别大的待预测集上，使用单机的python，需要预测2、3天，甚至更久，中途很有可能中断。因此需要通过分布式的方式来预测。这个工具包就是实现了将训练好的python模型，转换成sql语句。将生成的sql语句可以放到大数据环境中进行分布式执行预测，能比单机的python预测快好几个量级


## 思想碰撞

|  微信 |  微信公众号 |
| :---: | :----: |
| <img src="https://github.com/ZhengRyan/autotreemodel/blob/master/images/%E5%B9%B2%E9%A5%AD%E4%BA%BA.png" alt="RyanZheng.png" width="50%" border=0/> | <img src="https://github.com/ZhengRyan/autotreemodel/blob/master/images/%E9%AD%94%E9%83%BD%E6%95%B0%E6%8D%AE%E5%B9%B2%E9%A5%AD%E4%BA%BA.png" alt="魔都数据干饭人.png" width="50%" border=0/> |
|  干饭人  | 魔都数据干饭人 |


> 仓库地址：https://github.com/ZhengRyan/treemodel2sql
> 
> 微信公众号文章：https://mp.weixin.qq.com/s/u8Nsp5M93WIGL2M0tU4U_g
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
+ 使用treemodel2sql工具包将模型转换成的sql语句
```python
from treemodel2sql import XGBoost2Sql
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
select key,1 / (1 + exp(-(tree_1_score + tree_2_score + tree_3_score)+(-0.0))) as score
    from (
    select key,
    --tree1
		case when (f9<-1.64164519 or f9 is null) then
			case when (f3<-4.19117069 or f3 is null) then
				case when (f2<-1.31743848 or f2 is null) then
					-0.150000006
				else
					-0.544186056
				end
			else
				case when (f3<1.23432565 or f3 is null) then
					case when (f7<-2.55682254 or f7 is null) then
						-0.200000018
					else
						case when (f5<0.154983491 or f5 is null) then
							0.544721723
						else
							case when (f3<0.697217584 or f3 is null) then
								-0.150000006
							else
								0.333333373
							end
						end
					end
				else
					case when (f5<-1.0218116 or f5 is null) then
						case when (f0<-0.60882163 or f0 is null) then
							case when (f2<0.26019755 or f2 is null) then
								0.0666666701
							else
								-0.300000012
							end
						else
							-0.520000041
						end
					else
						0.333333373
					end
				end
			end
		else
			case when (f9<1.60392439 or f9 is null) then
				case when (f3<0.572542191 or f3 is null) then
					case when (f5<0.653370142 or f5 is null) then
						case when (f7<-0.765973091 or f7 is null) then
							case when (f3<-0.432390809 or f3 is null) then
								0.204000011
							else
								-0.485454559
							end
						else
							case when (f3<-1.20459461 or f3 is null) then
								-0.5104478
							else
								0.441509455
							end
						end
					else
						case when (f7<0.133017987 or f7 is null) then
							case when (f8<0.320554674 or f8 is null) then
								-0.290322572
							else
								0.368339777
							end
						else
							case when (f8<-0.211985052 or f8 is null) then
								0.504000008
							else
								-0.525648415
							end
						end
					end
				else
					case when (f7<2.22314501 or f7 is null) then
						case when (f8<-0.00532855326 or f8 is null) then
							case when (f8<-0.204920739 or f8 is null) then
								-0.533991575
							else
								-0.200000018
							end
						else
							0.428571463
						end
					else
						case when (f3<1.33772755 or f3 is null) then
							case when (f0<-0.975171864 or f0 is null) then
								0.163636371
							else
								0.51818186
							end
						else
							-0
						end
					end
				end
			else
				case when (f3<1.77943277 or f3 is null) then
					case when (f7<-0.469875157 or f7 is null) then
						case when (f3<-0.536645889 or f3 is null) then
							case when (f9<1.89841866 or f9 is null) then
								-0
							else
								0.333333373
							end
						else
							case when (f4<-2.43660188 or f4 is null) then
								0.150000006
							else
								0.551020443
							end
						end
					else
						case when (f1<-0.0788691565 or f1 is null) then
							0.150000006
						else
							-0.375
						end
					end
				else
					case when (f4<-1.73232496 or f4 is null) then
						-0.150000006
					else
						case when (f6<-1.6080606 or f6 is null) then
							-0.150000006
						else
							case when (f7<-0.259483218 or f7 is null) then
								-0.558620751
							else
								-0.300000012
							end
						end
					end
				end
			end
		end
		as tree_1_score,
--tree2
		case when (f9<-1.64164519 or f9 is null) then
			case when (f3<-4.19117069 or f3 is null) then
				case when (f0<0.942570388 or f0 is null) then
					-0.432453066
				else
					-0.128291652
				end
			else
				case when (f3<1.23432565 or f3 is null) then
					case when (f7<-2.55682254 or f7 is null) then
						-0.167702854
					else
						case when (f5<0.154983491 or f5 is null) then
							case when (f1<2.19985676 or f1 is null) then
								0.41752997
							else
								0.115944751
							end
						else
							0.115584135
						end
					end
				else
					case when (f5<-1.0218116 or f5 is null) then
						case when (f0<-0.60882163 or f0 is null) then
							-0.119530827
						else
							-0.410788596
						end
					else
						0.28256765
					end
				end
			end
		else
			case when (f9<1.60392439 or f9 is null) then
				case when (f3<0.460727394 or f3 is null) then
					case when (f5<0.653370142 or f5 is null) then
						case when (f7<-0.933565617 or f7 is null) then
							case when (f3<-0.572475374 or f3 is null) then
								0.182491601
							else
								-0.377898693
							end
						else
							case when (f3<-1.20459461 or f3 is null) then
								-0.392539263
							else
								0.352721155
							end
						end
					else
						case when (f7<0.207098693 or f7 is null) then
							case when (f8<0.498489976 or f8 is null) then
								-0.193351224
							else
								0.29298231
							end
						else
							case when (f8<-0.117464997 or f8 is null) then
								0.400667101
							else
								-0.402199954
							end
						end
					end
				else
					case when (f7<1.98268723 or f7 is null) then
						case when (f8<-0.00532855326 or f8 is null) then
							case when (f7<1.36281848 or f7 is null) then
								-0.408002198
							else
								-0.236123681
							end
						else
							case when (f5<1.14038813 or f5 is null) then
								0.404326111
							else
								-0.110877581
							end
						end
					else
						case when (f3<1.56952488 or f3 is null) then
							case when (f5<2.14646816 or f5 is null) then
								0.409404457
							else
								0.0696995854
							end
						else
							-0.32059738
						end
					end
				end
			else
				case when (f3<1.77943277 or f3 is null) then
					case when (f7<-0.469875157 or f7 is null) then
						case when (f3<-0.536645889 or f3 is null) then
							case when (f9<1.89841866 or f9 is null) then
								-0
							else
								0.28256765
							end
						else
							0.419863999
						end
					else
						case when (f3<0.444227457 or f3 is null) then
							-0.34664312
						else
							0.0693304539
						end
					end
				else
					case when (f4<-1.10089087 or f4 is null) then
						case when (f3<2.3550868 or f3 is null) then
							0.0147894565
						else
							-0.331404865
						end
					else
						-0.421277165
					end
				end
			end
		end
		as tree_2_score,
--tree3
		case when (f9<-1.64164519 or f9 is null) then
			case when (f3<-4.19117069 or f3 is null) then
				case when (f4<-1.30126143 or f4 is null) then
					-0.0772174299
				else
					-0.374165356
				end
			else
				case when (f3<1.23432565 or f3 is null) then
					case when (f7<-2.55682254 or f7 is null) then
						-0.142005175
					else
						case when (f5<0.154983491 or f5 is null) then
							case when (f7<3.59379435 or f7 is null) then
								0.352122813
							else
								0.132789165
							end
						else
							0.0924336985
						end
					end
				else
					case when (f5<-1.0218116 or f5 is null) then
						case when (f0<-0.60882163 or f0 is null) then
							-0.0954768136
						else
							-0.351594836
						end
					else
						0.245992288
					end
				end
			end
		else
			case when (f9<1.60392439 or f9 is null) then
				case when (f3<0.347133756 or f3 is null) then
					case when (f5<0.661561131 or f5 is null) then
						case when (f7<-0.933565617 or f7 is null) then
							case when (f3<-0.472413659 or f3 is null) then
								0.116336405
							else
								-0.313245147
							end
						else
							case when (f3<-1.5402329 or f3 is null) then
								-0.352897167
							else
								0.311400592
							end
						end
					else
						case when (f7<0.275665522 or f7 is null) then
							case when (f8<0.403402805 or f8 is null) then
								-0.292606086
							else
								0.220064178
							end
						else
							case when (f8<-0.0442957953 or f8 is null) then
								0.350784421
							else
								-0.336107522
							end
						end
					end
				else
					case when (f7<1.77503061 or f7 is null) then
						case when (f8<0.196157426 or f8 is null) then
							case when (f7<1.36281848 or f7 is null) then
								-0.3376683
							else
								-0.0711223111
							end
						else
							case when (f7<-0.661211252 or f7 is null) then
								0.434363276
							else
								-0.219307661
							end
						end
					else
						case when (f3<1.37940335 or f3 is null) then
							case when (f6<1.34894884 or f6 is null) then
								0.367155522
							else
								0.124757253
							end
						else
							-0.293739736
						end
					end
				end
			else
				case when (f3<1.77943277 or f3 is null) then
					case when (f7<-0.469875157 or f7 is null) then
						case when (f3<-0.536645889 or f3 is null) then
							case when (f9<1.89841866 or f9 is null) then
								-0
							else
								0.245992288
							end
						else
							case when (f0<1.60565615 or f0 is null) then
								0.357973605
							else
								0.193993196
							end
						end
					else
						case when (f9<1.89456153 or f9 is null) then
							-0.276471078
						else
							0.111896731
						end
					end
				else
					case when (f1<1.35706067 or f1 is null) then
						case when (f4<-1.10089087 or f4 is null) then
							case when (f3<2.3550868 or f3 is null) then
								0.0119848112
							else
								-0.284813672
							end
						else
							-0.376859784
						end
					else
						case when (f2<-0.25748384 or f2 is null) then
							0.0723158419
						else
							-0.253415495
						end
					end
				end
			end
		end
		as tree_3_score
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
+ 使用treemodel2sql工具包将模型转换成的sql语句
```python
from treemodel2sql import Lgb2Sql
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

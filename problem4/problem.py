import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from show_pic import *


def datacut(df):
    # 筛选出性别为男+汉族+已婚+高学历
    class1 = df[(df['性别'] == 1) & (df['民族'] == 1) &
                (df['婚姻状况'] > 1) & (df['文化程度'] > 4)]
    # 筛选出性别为男+汉族+已婚+低学历
    class2 = df[(df['性别'] == 1) & (df['民族'] == 1) &
                (df['婚姻状况'] > 1) & (df['文化程度'] < 4)]
    # 筛选出性别为男+汉族+未婚+高学历
    class3 = df[(df['性别'] == 1) & (df['民族'] == 1) &
                (df['婚姻状况'] == 1) & (df['文化程度'] > 4)]
    # 筛选出性别为男+汉族+未婚+低学历
    class4 = df[(df['性别'] == 1) & (df['民族'] == 1) &
                (df['婚姻状况'] == 1) & (df['文化程度'] < 4)]
    # 筛选出性别为男+少数民族+已婚+高学历
    class5 = df[(df['性别'] == 1) & (df['民族'] > 1) &
                (df['婚姻状况'] > 1) & (df['文化程度'] > 4)]
    # 筛选出性别为男+少数民族+已婚+低学历
    class6 = df[(df['性别'] == 1) & (df['民族'] > 1) &
                (df['婚姻状况'] > 1) & (df['文化程度'] < 4)]
    # 筛选出性别为男+少数民族+未婚+高学历
    class7 = df[(df['性别'] == 1) & (df['民族'] > 1) &
                (df['婚姻状况'] == 1) & (df['文化程度'] > 4)]
    # 筛选出性别为男+少数民族+未婚+低学历
    class8 = df[(df['性别'] == 1) & (df['民族'] > 1) &
                (df['婚姻状况'] == 1) & (df['文化程度'] < 4)]
    # 筛选出性别为女+汉族+已婚+高学历
    class9 = df[(df['性别'] == 2) & (df['民族'] == 1) &
                (df['婚姻状况'] > 1) & (df['文化程度'] > 4)]
    # 筛选出性别为女+汉族+已婚+低学历
    class10 = df[(df['性别'] == 2) & (df['民族'] == 1) &
                 (df['婚姻状况'] > 1) & (df['文化程度'] < 4)]
    # 筛选出性别为女+汉族+未婚+高学历
    class11 = df[(df['性别'] == 2) & (df['民族'] == 1) &
                 (df['婚姻状况'] == 1) & (df['文化程度'] > 4)]
    # 筛选出性别为女+汉族+未婚+低学历
    class12 = df[(df['性别'] == 2) & (df['民族'] == 1) &
                 (df['婚姻状况'] == 1) & (df['文化程度'] < 4)]
    # 筛选出性别为女+少数民族+已婚+高学历
    class13 = df[(df['性别'] == 2) & (df['民族'] > 1) &
                 (df['婚姻状况'] > 1) & (df['文化程度'] > 4)]
    # 筛选出性别为女+少数民族+已婚+低学历
    class14 = df[(df['性别'] == 2) & (df['民族'] > 1) &
                 (df['婚姻状况'] > 1) & (df['文化程度'] < 4)]
    # 筛选出性别为女+少数民族+未婚+高学历
    class15 = df[(df['性别'] == 2) & (df['民族'] > 1) &
                 (df['婚姻状况'] == 1) & (df['文化程度'] > 4)]
    # 筛选出性别为女+少数民族+未婚+低学历
    class16 = df[(df['性别'] == 2) & (df['民族'] > 1) &
                 (df['婚姻状况'] == 1) & (df['文化程度'] < 4)]
    return class1, class2, class3, class4, class5, class6, class7, class8, class9, class10, class11, class12, class13, class14, class15, class16


def vegetable(class_name, df):
    # 定位需要处理的列
    index_relate = df.columns.get_loc('是否吃新鲜蔬菜')
    select_data = df.iloc[:, index_relate:index_relate+10].values
    # 缺失值处理
    select_data = np.nan_to_num(select_data)
    # print(df)
    # 计算每人平均每日蔬菜摄入量
    pre1 = select_data[:, 0]*(select_data[:, 1]*select_data[:, 4]+(
        select_data[:, 2]*select_data[:, 4])/7+(select_data[:, 3]*select_data[:, 4])/30)*50
    # # 计算深色蔬菜的平均每日摄入量
    # pre2=(select_data[:,5]*(select_data[:,6]*select_data[:,9]+(select_data[:,7]*select_data[:,9])/7+(select_data[:,8]*select_data[:,9])/30)*50)/pre1
    # 绘图
    plot_show1(pre1, 'vegetable', class_name)


def fruit(df):
    # 定位需要处理的列
    index_relate = df.columns.get_loc('是否吃水果')
    select_data = df.iloc[:, index_relate:index_relate+5].values
    # 缺失值处理
    select_data = np.nan_to_num(select_data)
    # 计算每人平均每日水果摄入量
    pre1 = select_data[:, 0]*(select_data[:, 1]*select_data[:, 4]+(
        select_data[:, 2]*select_data[:, 4])/7+(select_data[:, 3]*select_data[:, 4])/30)*50
    plot_show2(pre1, 'fruit', 'all')


def milk(df):
    # 定位需要处理的列
    index_relate = df.columns.get_loc('是否吃鲜奶')
    select_data = df.iloc[:, index_relate:index_relate+15].values
    # 缺失值处理
    select_data = np.nan_to_num(select_data)
    # 计算每人平均每日奶制品摄入量
    pre1 = select_data[:, 0]*(select_data[:, 1]*select_data[:, 4]+(select_data[:, 2]*select_data[:, 4])/7+(select_data[:, 3]*select_data[:, 4])/30)*50 +\
        select_data[:, 5]*(select_data[:, 6]*select_data[:, 9]+(select_data[:, 7]*select_data[:, 9])/7+(select_data[:, 8]*select_data[:, 9])/30)*10 +\
        select_data[:, 10]*(select_data[:, 11]*select_data[:, 14]+(select_data[:, 12]
                            * select_data[:, 14])/7+(select_data[:, 13]*select_data[:, 14])/30)*50
    plot_show2(pre1, 'milk', 'all')


def cereals(df):
    # 定位需要处理的列
    index_relate = df.columns.get_loc('是否吃杂粮')
    select_data = df.iloc[:, index_relate:index_relate+15].values
    # 缺失值处理
    select_data = np.nan_to_num(select_data)
    # 计算每人平均每日杂粮摄入量
    pre1 = select_data[:, 0]*(select_data[:, 1]*select_data[:, 4]+(select_data[:, 2]*select_data[:, 4])/7+(select_data[:, 3]*select_data[:, 4])/30)*50 +\
        select_data[:, 5]*(select_data[:, 6]*select_data[:, 9]+(select_data[:, 7]*select_data[:, 9])/7+(select_data[:, 8]*select_data[:, 9])/30)*10 +\
        select_data[:, 10]*(select_data[:, 11]*select_data[:, 14]+(select_data[:, 12]
                            * select_data[:, 14])/7+(select_data[:, 13]*select_data[:, 14])/30)*50
    plot_show2(pre1, 'cereals', 'all')


def protein(df):
    # 定位需要处理的列
    index_relate = df.columns.get_loc('是否吃豆腐')
    select_data = df.iloc[:, index_relate:index_relate+20].values
    # 缺失值处理
    select_data = np.nan_to_num(select_data)
    # 计算每人平均每日肉蛋摄入量
    pre1 = select_data[:, 0]*(select_data[:, 1]*select_data[:, 4]+(select_data[:, 2]*select_data[:, 4])/7+(select_data[:, 3]*select_data[:, 4])/30)*50 +\
        select_data[:, 5]*(select_data[:, 6]*select_data[:, 9]+(select_data[:, 7]*select_data[:, 9])/7+(select_data[:, 8]*select_data[:, 9])/30)*50 +\
        select_data[:, 10]*(select_data[:, 11]*select_data[:, 14]+(select_data[:, 12] * select_data[:, 14])/7+(select_data[:, 13]*select_data[:, 14])/30)*50 +\
        select_data[:, 15]*(select_data[:, 16]*select_data[:, 19]+(select_data[:, 17]
                            * select_data[:, 19])/7+(select_data[:, 18]*select_data[:, 19])/30)*50
    plot_show2(pre1, 'protein', 'all')


def meat(df):
    # 定位需要处理的列
    index_relate1 = df.columns.get_loc('是否吃猪肉')
    index_relate2 = df.columns.get_loc('是否吃蛋类')
    select_data = df.iloc[:, index_relate1:index_relate1 +
                          20].join(df.iloc[:, index_relate2:index_relate2+5]).values
    # 缺失值处理
    select_data = np.nan_to_num(select_data)
    # 计算每人平均每日肉蛋摄入量
    pre1 = select_data[:, 0]*(select_data[:, 1]*select_data[:, 4]+(select_data[:, 2]*select_data[:, 4])/7+(select_data[:, 3]*select_data[:, 4])/30)*50 +\
        select_data[:, 5]*(select_data[:, 6]*select_data[:, 9]+(select_data[:, 7]*select_data[:, 9])/7+(select_data[:, 8]*select_data[:, 9])/30)*50 +\
        select_data[:, 10]*(select_data[:, 11]*select_data[:, 14]+(select_data[:, 12] * select_data[:, 14])/7+(select_data[:, 13]*select_data[:, 14])/30)*50 +\
        select_data[:, 15]*(select_data[:, 16]*select_data[:, 19]+(select_data[:, 17] * select_data[:, 19])/7+(select_data[:, 18]*select_data[:, 19])/30)*50 +\
        select_data[:, 20]*(select_data[:, 21]*select_data[:, 24]+(select_data[:, 22]
                            * select_data[:, 24])/7+(select_data[:, 23]*select_data[:, 24])/30)*60
    plot_show2(pre1, 'meat', 'all')


def oil(df, yes_show=1):
    # 定位需要处理的列
    index_relate = df.columns.get_loc('植物油')
    select_data = df.iloc[:, index_relate:index_relate+2].values
    # 缺失值处理
    select_data = np.nan_to_num(select_data)
    # 计算每户平均每月油摄入量
    pre1 = select_data[:, 0]+select_data[:, 1]
    if yes_show == 1:
        plot_show2(pre1, 'oil', 'all')
    return pre1


def sugar(df):
    # 定位需要处理的列
    index_relate = df.columns.get_loc('是否吃果汁饮料')
    select_data = df.iloc[:, index_relate:index_relate+10].values
    # 缺失值处理
    select_data = np.nan_to_num(select_data)
    # 计算每人平均每日糖摄入量
    pre1 = (select_data[:, 0]*(select_data[:, 1]*select_data[:, 4]+(select_data[:, 2]*select_data[:, 4])/7+(select_data[:, 3]*select_data[:, 4])/30)*250 +
            select_data[:, 5]*(select_data[:, 6]*select_data[:, 9]+(select_data[:, 7]*select_data[:, 9])/7+(select_data[:, 8]*select_data[:, 9])/30)*250)*0.1
    plot_show1(pre1, 'sugar', 'all')


def salt(df):
    # 定位需要处理的列
    index_relate = df.columns.get_loc('盐')
    select_data = df.iloc[:, index_relate:index_relate+1].values
    # 缺失值处理
    select_data = np.nan_to_num(select_data)
    # 计算每人平均每日糖摄入量
    pre1 = select_data[:, 0]*50
    plot_show2(pre1, 'salt', 'all')


def wine(df, yes_show=1):
    # 定位需要处理的列
    index_relate = df.columns.get_loc('是否饮用高度白酒')
    select_data = df.iloc[:, index_relate:index_relate+15].values
    # 缺失值处理
    select_data = np.nan_to_num(select_data)
    # 计算每人平均每日糖摄入量
    pre1 = (select_data[:, 0]*select_data[:, 1]*select_data[:, 2]*0.45 +
            select_data[:, 3]*select_data[:, 4]*select_data[:, 5]*0.25 +
            select_data[:, 6]*select_data[:, 7]*select_data[:, 8]*0.05 +
            select_data[:, 9]*select_data[:, 10]*select_data[:, 11]*0.15 +
            select_data[:, 12]*select_data[:, 13]*select_data[:, 14]*0.1)/7*50
    if yes_show == 1:
        plot_show1(pre1, 'wine', 'all')
    return pre1


def breakfast(df):
    # 定位需要处理的列
    index_relate = df.columns.get_loc('不吃早餐')
    select_data = df.iloc[:, index_relate:index_relate+1].values
    # 缺失值处理
    select_data = np.nan_to_num(select_data)
    plot_show1(select_data, 'breakfast', 'all')


def lunch(df):
    # 定位需要处理的列
    index_relate = df.columns.get_loc('不吃中餐')
    select_data = df.iloc[:, index_relate:index_relate+1].values
    # 缺失值处理
    select_data = np.nan_to_num(select_data)
    plot_show1(select_data, 'lunch', 'all')


def dinner(df):
    # 定位需要处理的列
    index_relate = df.columns.get_loc('不吃晚餐')
    select_data = df.iloc[:, index_relate:index_relate+1].values
    # 缺失值处理
    select_data = np.nan_to_num(select_data)
    plot_show1(select_data, 'dinner', 'all')


def cigar(df, class_name, yes_show=1):
    if class_name == 'man':
        df = df[(df['性别'] == 1)]
    elif class_name == 'women':
        df = df[(df['性别'] == 2)]
    df['是否吸烟'] = np.where(df['是否吸烟'] == 1, df['是否吸烟'], 0)
    df['开始吸烟年龄'].fillna(0, inplace=True)
    df['开始吸烟年龄'] = df['开始吸烟年龄'].replace({99: 19})
    df['烟龄'] = 2023-df['出生年']-df['开始吸烟年龄']
    df['平均每周吸烟天数'].fillna(0, inplace=True)
    df['一天吸烟支数'].fillna(0, inplace=True)
    select_data = df['是否吸烟']*df['平均每周吸烟天数']*df['一天吸烟支数']/7*df['烟龄'].values
    if yes_show == 1:
        if class_name == 'man':
            plot_show2(select_data, 'cigar', class_name)
        elif class_name == 'women':
            plot_show2(select_data, 'cigar', class_name)
    return select_data


def sport(df):
    # 定位需要处理的列
    index_relate = df.columns.get_loc('体育锻炼的强度')
    select_data = df.iloc[:, index_relate:index_relate+2].values
    # 缺失值处理
    select_data = np.nan_to_num(select_data)
    # 计算每人平均每日糖摄入量
    pre1 = select_data[:, 0]*select_data[:, 1]
    return pre1


def cor_factor(df):
    # 得到月均每户食用油摄入量
    oil_data = oil(df, 0)
    # 吸烟指数
    cigar_data = cigar(df, 0)
    # 日均饮酒量
    achole_data = wine(df, 0)
    # 日均运动量
    sport_data = sport(df)
    # 高血压患病情况
    blood_data = df['有没有被社区或以上医院的医生诊断过患有高血压'].values
    blood_data[np.isnan(blood_data)] = 1
    # 生成df_new数据
    df_new = pd.DataFrame({'blood': blood_data, 'oil': oil_data,
                           'cigar': cigar_data, 'achole': achole_data,
                           'sport': sport_data})

    # 计算相关系数
    corr = df_new.corr()
    # 绘制热力图
    plt.close('all')
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    # 输出相关系数
    # 保存相关系数到CSV文件
    corr.to_csv('./corr.csv')
    # 保存热力图到./fig文件夹
    plt.savefig('./fig/heatmap.png')


def solve1(df):
    class1, class2, class3, class4, class5, class6, class7, class8, class9, class10, class11, class12, class13, class14, class15, class16 = datacut(
        df)
    vegetable('class1', class1)
    vegetable('class2', class2)
    vegetable('class3', class3)
    vegetable('class4', class4)
    vegetable('class5', class5)
    vegetable('class6', class6)
    vegetable('class7', class7)
    vegetable('class8', class8)
    vegetable('class9', class9)
    vegetable('class10', class10)
    vegetable('class11', class11)
    vegetable('class12', class12)
    vegetable('class13', class13)
    vegetable('class14', class14)
    vegetable('class15', class15)
    vegetable('class16', class16)
    fruit(df)
    milk(df)
    cereals(df)
    protein(df)
    meat(df)
    oil(df)
    sugar(df)
    salt(df)
    wine(df)
    breakfast(df)
    lunch(df)
    dinner(df)
    cigar(df, 'man')
    cigar(df, 'women')
    cor_factor(df)

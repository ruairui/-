import matplotlib.pyplot as plt
import os
import pandas as pd
from problem import *

if __name__ == '__main__':
    # 读取数据
    df = pd.read_csv('data.csv', encoding='gbk',low_memory=False)
    # 调用problem1.py中的函数解决问题1
    # print(df)
    df = solve1(df)

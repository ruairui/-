import matplotlib.pyplot as plt
import numpy as np


def plot_show1(pre1, name, class_name):
    if (pre1.shape[0] == 0):
        return
    if name == 'vegetable':
        range_low = 0
        range_high = 5000
        step = 100
        threshold = 300
    elif name == 'sport':
        range_low = 0
        range_high = 1950
        step = 150
        threshold = 150
    elif name == 'sugar':
        range_low = 0
        range_high = 1850
        step = 25
        threshold = 25
    elif name == 'wine':
        range_low = 0
        range_high = 100
        step = 5
        threshold = 15
    elif name == 'breakfast':
        range_low = 0
        range_high = 7
        step = 1
        threshold = 1
    elif name == 'lunch':
        range_low = 0
        range_high = 7
        step = 1
        threshold = 1
    elif name == 'dinner':
        range_low = 0
        range_high = 7
        step = 1
        threshold = 1

    bin_edges = [(i, i+step) for i in range(range_low, range_high, step)]

    # 根据区间范围生成标签
    labels = [f"{i}-{i+step}" for i in range(range_low, range_high, step)]

    # 统计各个区间的数量
    counts1 = [np.sum((pre1 >= lower_bound) & (pre1 < upper_bound))
               for lower_bound, upper_bound in bin_edges]

    # print(counts1)
    # 绘制柱状图，并设置颜色
    plt.figure(figsize=(20, 12))
    # 用于记录绘制过的柱子的索引
    drawn_bars = []
    for i, count in enumerate(counts1):
        # 跳过统计数值为0的柱子
        if count == 0:
            continue
        bar = plt.bar(labels[i], count)
        drawn_bars.append(bar[0])

        # 低于这区间的标为黄色，高于这区间的标为绿色
        if bin_edges[i][1] <= threshold:
            bar.patches[0].set_facecolor('yellow')
        else:
            bar.patches[0].set_facecolor('lightgreen')

        # 在每个柱子上方添加计数的文本标签
        plt.text(labels[i], count, f'{count}', ha='center', va='bottom')

    plt.xlabel('Group')
    plt.ylabel('Count')
    plt.title(f'Distribution of {name}',)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # plt.show()
    fig_name = './fig/'+name+'1'+class_name+'.png'
    plt.savefig(fig_name)

    # 绘制饼状图，并显示计数和比例
    plt.figure()
    labels = ['Enough', 'Inadequate']
    sizes = [np.sum(pre1 >= threshold), np.sum(pre1 < threshold)]
    sizes = np.nan_to_num(sizes)
    # print(sizes)
    colors = ['lightgreen', 'yellow']
    _, _, autotexts = plt.pie(sizes, labels=labels,
                              colors=colors, autopct='%1.1f%%', startangle=90)

    # 设置文本显示格式
    for i, t in enumerate(autotexts):
        t.set_text(f'{sizes[i]} - {t.get_text()}')
    fig_name = './fig/'+name+'2'+class_name+'.png'
    plt.savefig(fig_name)


def plot_show2(pre1, name, class_name):
    if (pre1.shape[0] == 0):
        return
    if name == 'fruit':
        range_low = 0
        range_high = 28600
        step = 50
        threshold_low = 200
        threshold_high = 350
    elif name == 'milk':
        range_low = 0
        range_high = 17900
        step = 100
        threshold_low = 300
        threshold_high = 500
    elif name == 'cereals':
        range_low = 0
        range_high = 2000
        step = 50
        threshold_low = 50
        threshold_high = 150
    elif name == 'protein':
        range_low = 0
        range_high = 840
        step = 5
        threshold_low = 30
        threshold_high = 50
    elif name == 'meat':
        range_low = 0
        range_high = 1000
        step = 40
        threshold_low = 120
        threshold_high = 200
    elif name == 'oil':
        range_low = 0
        range_high = 30
        step = 1
        threshold_low = 3
        threshold_high = 6
    elif name == 'salt':
        range_low = 0
        range_high = 2500
        step = 100
        threshold_low = 300
        threshold_high = 4500
    elif name == 'cigar':
        range_low = 0
        range_high = 1000
        step = 100
        threshold_low = 200
        threshold_high = 400
    # 设置区间范围和标签
    bin_edges = [(i, i+step) for i in range(range_low, range_high, step)]
    labels = [f"{i}-{i+step}" for i in range(range_low, range_high, step)]

    # 统计各个区间的数量
    counts = [np.sum((pre1 >= lower_bound) & (pre1 < upper_bound))
              for lower_bound, upper_bound in bin_edges]

    plt.figure(figsize=(20, 12))
    # 用于记录绘制过的柱子的索引
    drawn_bars = []
    for i, count in enumerate(counts):
        # 跳过统计数值为0的柱子
        if count == 0:
            continue
        bar = plt.bar(labels[i], count)
        drawn_bars.append(bar[0])

        # 中间的柱子标为绿色，低于这区间的标为黄色，高于这区间的标为红色
        if range_low <= bin_edges[i][0] < threshold_low:
            bar[0].set_color('yellow')
        elif threshold_low <= bin_edges[i][0] < threshold_high:
            bar[0].set_color('lightgreen')
        elif threshold_high <= bin_edges[i][0] < range_high:
            bar[0].set_color('salmon')

        # 在每个柱子上方添加计数的文本标签
        plt.text(labels[i], count, f'{count}', ha='center', va='bottom')
    plt.xlabel('Group')
    plt.ylabel('Count')
    plt.title(f'Distribution of {name}',)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # plt.show()
    fig_name = './fig/'+name+'1'+class_name+'.png'
    plt.savefig(fig_name)

    # 绘制饼状图，并显示计数和比例
    plt.figure()
    labels = ['Low', 'Normal', 'High']

    sizes = [np.sum((pre1 >= range_low) & (pre1 <= threshold_low)), np.sum(
        (pre1 > threshold_low) & (pre1 < threshold_high)), np.sum((pre1 >= threshold_high) & (pre1 <= range_high))]
    sizes = np.nan_to_num(sizes)
    # print(sizes)
    colors = ['yellow', 'lightgreen', 'salmon']
    _, _, autotexts = plt.pie(sizes, labels=labels,
                              colors=colors, autopct='%1.1f%%', startangle=90)
    # 设置文本显示格式
    for i, t in enumerate(autotexts):
        t.set_text(f'{sizes[i]} - {t.get_text()}')
    fig_name = './fig/'+name+'2'+class_name+'.png'
    plt.savefig(fig_name)

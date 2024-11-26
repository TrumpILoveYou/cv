import pandas as pd
import numpy as np

def txttocsv():
    # 假设你的图像路径和标签如下
    image_paths = list()
    labels = list()

    # 打开文件，按行读取
    file_path = "../SplitAndIngreLabel/IngreLabel.txt"

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # .strip() 去除每行末尾的换行符
            parts = line.split()  # 按空格分割
            image_path = parts[0]  # 第一部分是图像路径
            label = list(map(int, parts[1:]))  # 后面的部分是标签，转换为整数列表
            image_paths.append(image_path)  # 将图像路径添加到列表
            labels.append(label)  # 将标签添加到列表

    # 将数据转化为DataFrame
    df = pd.DataFrame({
        'Image Path': image_paths,
        'Label': [str(label) for label in labels]  # 将标签转换为字符串，以便存储
    })

    # 保存为CSV文件
    df.to_json('image_labels.json', index=False)

    print("CSV文件已保存！")

def split_TR():
    df=pd.read_csv("../SplitAndIngreLabel/image_labels.csv")
    # 打开文件，按行读取
    file_path = "../SplitAndIngreLabel/TR.txt"

    # 读取文件，将每行作为一个数据项，并设置列名为 'Image'
    d= pd.read_csv(file_path, header=None, names=['Image Path'],sep=" ")
    # 使用 merge 按照 'Image' 列外连接（左外连接）
    df_merged = pd.merge(d, df, on='Image Path', how='left')
    df_merged.to_csv("TR_ingre.csv")
def split_TE():
    df=pd.read_csv("../SplitAndIngreLabel/image_labels.csv")
    # 打开文件，按行读取
    file_path = "../SplitAndIngreLabel/TE.txt"

    # 读取文件，将每行作为一个数据项，并设置列名为 'Image'
    d= pd.read_csv(file_path, header=None, names=['Image Path'],sep=" ")
    # 使用 merge 按照 'Image' 列外连接（左外连接）
    df_merged = pd.merge(d, df, on='Image Path', how='left')
    df_merged.to_csv("TE_ingre.csv")
def split_VAL():
    df=pd.read_csv("../SplitAndIngreLabel/image_labels.csv")
    # 打开文件，按行读取
    file_path = "../SplitAndIngreLabel/VAL.txt"

    # 读取文件，将每行作为一个数据项，并设置列名为 'Image'
    d= pd.read_csv(file_path, header=None, names=['Image Path'],sep=" ")
    # 使用 merge 按照 'Image' 列外连接（左外连接）
    df_merged = pd.merge(d, df, on='Image Path', how='left')
    df_merged.to_csv("VAL_ingre.csv")

split_TR()
split_TE()
split_VAL()


import json
import pandas
import numpy
import string
import xlsxwriter

# 导入数据
file_name = './第一批数据.xlsx'
opened_file = pandas.read_excel(file_name)

# 处理数据 对单元格内数据取均值
height, width = opened_file.shape

result = []
j = 0
while j < width:
    i = 0
    while i < height:
        s = opened_file.iloc[i, j]
        if type(s) == str:
            data = s.split(',')
            data = list(map(float, data))
            data = numpy.mean(data)
        elif type(s):
            if numpy.isnan(s):
                data = 0
            else:
                data = s
        result.append(data)
        i += 1
    j += 1

# 对数据用列均值替换0
j = 0
while j < width:
    i = 0
    flag = 0
    index = []
    while i < height:
        if result[i + j * height] == 0:
            flag += 1
            index.append(i)
        i += 1
    d_sum = numpy.sum(result[j * height:(j + 1) * height])
    d_average = d_sum / (height - flag)
    # print(d_sum)
    # print(d_average)
    i = 0
    while i < height:
        if result[i + j * height] == 0:
            result[i + j * height] = d_average
        i += 1
    j += 1
# print(result)

# 将数据写入xlsx文件
workbook = xlsxwriter.Workbook('./Mean_Cope2.xlsx', {'nan_inf_to_errors': True})
worksheet = workbook.add_worksheet(u'sheet1')

x = 0
index = 0
while x < width:
    y = 0
    while y < height:
        worksheet.write(y, x, result[index])
        index += 1
        y += 1
    x += 1

workbook.close()

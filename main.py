import json
import pandas
import numpy
import string
import xlsxwriter

# 导入数据
file_name = './第一批数据.xlsx'
opened_file = pandas.read_excel(file_name)

# 处理数据
height, width = opened_file.shape

result = []
i = 0
j = 0
while j < width:
    i = 0
    while i < height:
        s = opened_file.iloc[i, j]
        if type(s) == str:
            data = s.split(',')
            data = list(map(float, data))
            data = numpy.mean(data)
            #data = numpy.around(data, 1)
        elif type(s):
            data = float(s)
        result.append(data)
        i += 1
    j += 1
# print(result)

workbook = xlsxwriter.Workbook('./均值处理.xlsx', {'nan_inf_to_errors': True})
worksheet = workbook.add_worksheet(u'sheet1')

x = 0
index = 0
while x < width:
    y = 0
    while y < height:
        worksheet.write(y,x,result[index])
        index += 1
        y += 1
    x += 1

workbook.close()
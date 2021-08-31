import pandas
import numpy
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
            dic = {}
            for d in data:
                if d in dic:
                    dic[d] += 1
                else:
                    dic[d] = 1
            sk = dic.keys()
            sv = dic.values()
            re = []
            for d in sk:
                if dic[d] == (max(sv)):
                    re.append(d)
            if len(re) == 1:
                data = re[0]
            else:
                data = numpy.mean(re)
        elif type(s):
            data = float(s)
        result.append(data)
        i += 1
    j += 1
# print(result)

workbook = xlsxwriter.Workbook('./众数处理.xlsx', {'nan_inf_to_errors': True})
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
import random
from prettytable import PrettyTable

# варіант - 310 (min(Y))

a0 = 3; a1 = 1; a2 = 0; a3 = 2

# generating x1, x2, x3
x1,x2,x3 = [[random.randint(0, 20) for i in range(8)] for i in range(3)]

# calculating Y
y_list = [a0 + a1*x1[i] + a2*x2[i] + a3*x3[i] for i in range(8)]

# calculating x01, x02, x03
x0 = [(max(i) + min(i))/2 for i in [x1,x2,x3]]

# calculating dx1, dx2, dx3
dx = [x0[i]-min(x) for i, x in enumerate([x1,x2,x3])]

# calculating xn1, xn2, xn3
xn=[[round((x[i]-x0[j])/dx[j], 3) for j, x in enumerate([x1,x2,x3])] for i in range(8)]

# getting min(Y)
minY = min(y_list)

# calculating Yet
Yet = a0 + (a1 * x0[0]) + (a2 * x0[1]) + (a3 * x0[2])

# generating table
tabl_rows=[]
th = ['№','X1','X2','X3','Y','Xn1','Xn2','Xn3']
for i in range(8):tabl_rows.append([i+1,x1[i],x2[i],x3[i],y_list[i],xn[i][0],xn[i][1],xn[i][2]])
for i in range(4): x0.append(''); dx.append('')
tabl_rows.append(['x0']+x0)
tabl_rows.append(['dx']+dx)

table = PrettyTable(th)
for i in tabl_rows: table.add_row(i)

print("Варіант 310")
print(table)
print("Yэт:", Yet)
print("minY:", minY, "\n")

"""----------------------Варіант 327------------------------------"""
Y_v2 = [(i- Yet)*(i- Yet) for i in y_list]
print("Варіант 327\nmin((Y-Yэт)^2):", min(Y_v2))

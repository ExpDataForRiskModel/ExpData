For the expdata is large, we store it at:

	Websit: https://pan.baidu.com/s/1HxlYG8xVtUEuAxkIba8pgw 
	Download Code: 4rf2

In expdata.zip, exp1_1~exp3_3 are the artificial data with matrix $Y$ (without the preprocessing of multiplying $\lambda$) and matrix $C_1$ (dtype=numpy.float64), which is the output of numpy.tofile(), (when loaded by numpy.fromfile(), we should reshape the vector to a matrix):

​	exp1_1 ($\delta=5$) and exp1_2 ($\delta=10$) are the data for the first experiment;

​	exp2_1 ($\delta=3$), exp2_2 ($\delta=5 $) and exp2_3 ($\delta=10 $) are the data for the second experiment;

​	exp3_1 ($\delta=3 $), exp3_2 ($ \delta=5$) and exp3_3 ($ \delta=10$) are the data for the third experiment.

exp4 contains the stocks and fields of basic infomation:

​	stocks.txt is the stock used in the real data experiment;

​	basic_fields.txt is the basic infomation used in real data experiment.

Besides, we provide a example program for the exp2 at folder codes, use command 'python3 matrix.py' to run the exp2. Remind the Y.dat and C1.dat of either exp2_* should be unziped at the folder.
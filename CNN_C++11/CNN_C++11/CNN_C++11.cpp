// CNN_C++11.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <string>
#include "Loader.h"
#include "conv1.h"


using namespace  std;
float lreaning_point = 0.5;//学习系数
int ***picture;

float**** c1_core;//卷积核1 9*3*7*7
float* c1_b;//偏置
int* r_size_1;//第一层卷积结果维数
float ***re_1;//第一层卷积结果
float ***error_1;//第一层误差（反向传递时用到）



//读取图像的函数：
int*** getPicture(string url) {
	return Loader::getPicture(url);
};



int main()
{
	conv1 con_1;
	int *size = new int[2];
	picture= getPicture("1.jpg");
	size = Loader::getSize("1.jpg");
	cout << picture[0][400][400] <<  "  " << size[0]<< endl;

	cout << re_1[1][101][100] << endl;

	system("pause");
    return 0;
}


// CNN_C++11.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <string>
#include "Loader.h"
#include "conv1.h"
#include "pool1.h"
#include "conv2.h"
#include "pool2.h"
#include "conv3.h"
#include "pool3.h"
#include "FC1.h"
#include "FC2.h"
#include "FC3.h"

using namespace  std;
float lreaning_point = 0.5;//学习系数
int ***picture;
int pic_size[2];

float*** c1_result;
float*** p1_result;
float*** c2_result;
float*** p2_result;
float*** c3_result;
float*** p3_result;
float* FC1_result;
float* FC2_result;
float* FC3_result;


int main()
{
	
	//cout << picture[0][400][400] <<  "  " << pic_size[0]<< endl;
	picture= Loader::getPicture("1.jpg",pic_size);

	Conv_level con1;
	Pool pool1;

	con1.setSize(pic_size);
	//std::future<float**> f1 = std::async(std::launch::async, std::bind(&Conv_level::cal_cnn, &con1,picture,1));//通过bind，future实现多线程
	//c1_result = new float**[2];
	//c1_result[1] = f1.get();
	c1_result = con1.train_cnn(picture ,pic_size);

	pool1.setSize(con1.getSize());
	p1_result = pool1.max_pooling(c1_result);
	cout << "con1_Done" << endl;
	Conv_level2 con2;
	Pool2 pool2;

	con2.setSize(pool1.r_size);
	c2_result = con2.train_cnn(p1_result, pool1.r_size);
	pool2.setSize(con2.getSize());
	p2_result = pool2.max_pooling(c2_result);
	cout << "con2_Done" << endl;
	Conv_level3 con3;
	Pool3 pool3;

	con3.setSize(pool2.r_size);
	c3_result = con3.train_cnn(p2_result, pool2.r_size);
	pool3.setSize(con3.getSize());
	p3_result = pool3.max_pooling(c3_result);
	cout << "con3_Done" << endl;
	FC1 fc1(pool3.r_size);
	FC1_result = fc1.cal_result(p3_result);
	cout << "FC1_Done" << endl;
	FC2 fc2;
	FC2_result = fc2.cal_result(FC1_result);
	cout << "FC2_Done" << endl;
	FC3 fc3;
	FC3_result = fc3.cal_result(FC2_result);
	cout << "FC3_Done" << endl;
	for (int i = 0; i < 50; i++) { cout << FC3_result[i] << endl; }
	system("pause");
    return 0;
}


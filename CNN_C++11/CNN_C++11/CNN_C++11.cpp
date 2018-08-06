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
#include "CNN_Activitor.h"
#include "ImageLoader.h"


using namespace  std;
int ***picture;
int pic_size[2];

float*** c1_result;
float*** p1_result;
float*** c2_result;
float*** p2_result;
float*** c3_result;
float* p3_result;
float* FC1_result;
float* FC2_result;
float* FC3_result;


//std::future<float**> f1 = std::async(std::launch::async, std::bind(&Conv_level::cal_cnn, &con1,picture,1));//通过bind，future实现多线程
//c1_result = new float**[2];
//c1_result[1] = f1.get();

 int main()
{
	//初始化，调用各自的构造函数从文件中恢复权值
	 CNN_Activitor cnn(0.25);

	 double ans[2] = {1.0,0.0 };
	 cnn.Train_CNN("1.jpg", ans);
	 
	 system("pause");
     return 0;
}


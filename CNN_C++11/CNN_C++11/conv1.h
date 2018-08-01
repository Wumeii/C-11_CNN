#pragma once
#include<iostream>
#include <iostream>
#include <future>
#include <thread>
#include <cstdint>
#include <stdexcept>
#include <limits>

class conv1
{
public:
	static void init_c1_core(float**** c1_core, float* c1_b, int* r_size_1);
	static float*** train_c1(int*** picture, float**** c1_core, int* r_size_1, int* size);
	static float** cal_c1(int*** picture,float**** c1_core,int* r_size_1,int n);

private:
	
};


void conv1::init_c1_core(float**** c1_core,float* c1_b,int* r_size_1) {//初始化9个3*7*7的core
	c1_core = new float***[9];
	c1_b = new float[9];
	r_size_1 = new int[2];
	for (int i = 0; i < 9; i++) {
		c1_core[i] = new float**[3];
		for (int j = 0; j < 3; j++) {
			c1_core[i][j] = new float*[7];
			for (int k = 0; k < 7; k++) {
				c1_core[i][j][k] = new float[7];
			}
		}
	}
	for (int i = 0; i < 9; i++) {//不知道怎么设置初始值，全部置为0.1
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 7; k++) {
				for (int l = 0; l < 7; l++) {
					c1_core[i][j][k][l] = 0.1;
				}
			}
		}
	}
	for (int i = 0; i < 9; i++) { c1_b[i] = 1; }
	//初始化完成
}

float*** conv1::train_c1(int*** picture,float**** c1_core,int* r_size_1,int* size) {//可以使用多线程

	r_size_1[0] = size[0] - 7 + 1;
	r_size_1[1] = size[1] - 7 + 1;

	float*** result_c1 = new float**[9];
	for (int i = 0; i < 9; i++) { 
		result_c1[i] = new float*[r_size[0]];
		for (int j = 0; j < r_size[0]; j++) {
			result_c1[i][j] = new float[r_size[1]];
		}
	}

	for (int i = 0; i < 9; i++) {
		
		result_c1[i] = cal_c1(picture,c1_core,r_size_1,i);
		//result_c1[i] = cal_c1(picture,i);
	}
	return result_c1;
}
//用于多线程的计算核
float** conv1::cal_c1(int*** picture,float**** c1_core,int* r_size_1, int n) {
	float** re = new float*[r_size_1[0]];
	float sum=0;
	for (int i = 0; i < r_size_1[0]; i++) { re[i] = new float[r_size_1[1]]; }
	for (int i = 0; i < r_size_1[0]; i++) {
		for (int j = 0; j < r_size_1[1]; j++) {
			sum = 0;
			for (int k = i; k < i+7; k++) {
				for (int l = j; l < j+7; l++) {
					sum += picture[0][k][l] * c1_core[n][0][k-i][l-j] + picture[1][k][l] * c1_core[n][1][k-i][l-j] + picture[2][k][l] * c1_core[n][2][k-i][l-j];
				}
			}
			re[i][j] = sum+c1_b[n];
			//激励Relu
			if (re[i][j] < 0) { re[i][j] = 0; }
		}
	}
	return re;
}

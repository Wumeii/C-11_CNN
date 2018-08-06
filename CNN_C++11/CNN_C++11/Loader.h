#pragma once


#include <iostream>
#include <fstream>
#include <string>
#include <windows.h>
#include <gdiplus.h>
#pragma comment(lib, "gdiplus.lib")

using namespace std;
using namespace Gdiplus;

class Loader
{
public:
	Loader();
	~Loader();
	static int*** getPicture(string url,int* size);
	static wstring StringToWString(string &str);
	static int* getSize(string url);
private:
	static int size[2];
};

int*** Loader::getPicture(string url,int* size) {
	int ***p = new int **[3];
	GdiplusStartupInput gdiplusstartupinput;
	ULONG_PTR gdiplustoken;
	GdiplusStartup(&gdiplustoken, &gdiplusstartupinput, NULL);
	//wstring infilename(L"1.jpg");
	wstring wstr(url.length(), L' ');
	copy(url.begin(), url.end(), wstr.begin());
	Bitmap* bmp = new Bitmap(wstr.c_str());
	UINT height = bmp->GetHeight();
	UINT width = bmp->GetWidth();
	cout << height << "  " << width << endl;
	size[0] = height; size[1] = width;
	//初始化三维数组：1Red；2Green；3Blue；
	for (int i = 0; i < 3; i++)
	{
		p[i] = new int*[height];
		for (int j = 0; j < height; j++)
		{
			p[i][j] = new int[width];
		}
	}
	Color color;
	for (UINT y = 0; y < height; y++) {
		for (UINT x = 0; x < width; x++) {
			bmp->GetPixel(x, y, &color);
			p[0][y][x] = (int)color.GetRed();
			p[1][y][x] = (int)color.GetGreen();
			p[2][y][x] = (int)color.GetBlue();
		}
	}
	delete[] bmp;
    GdiplusShutdown(gdiplustoken);
	cout << url << "  load_Done." << endl;
	return p;
}

wstring Loader::StringToWString(string &str)
{
	wstring wstr(str.length(), L' ');
	copy(str.begin(), str.end(), wstr.begin());
	return wstr;
}

int* Loader::getSize(string url) {
	int* size = new int[2];
	GdiplusStartupInput gdiplusstartupinput;
	ULONG_PTR gdiplustoken;
	GdiplusStartup(&gdiplustoken, &gdiplusstartupinput, NULL);
	Bitmap* bmp2 = new Bitmap(StringToWString(url).c_str());
	size[0] = bmp2->GetHeight();
	size[1] = bmp2->GetWidth();
	GdiplusShutdown(gdiplustoken);
	return size;
}

Loader::Loader()
{
}

Loader::~Loader()
{
}


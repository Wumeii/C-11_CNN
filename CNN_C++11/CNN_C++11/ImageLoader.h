//
// Created by 章程 on 2018/8/5.
//

#ifndef CNN_PIC_LOADER_H
#define CNN_PIC_LOADER_H

#include <string>

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

using namespace std;

class ImageLoader {
public:

    explicit ImageLoader(const char* url, int* result);

    ~ImageLoader();

    int*** getPicture();

    int* getSize();

private:
    ImageLoader();
    const char* url{};
    unsigned char *data{nullptr};
    int*** arr;
    int size[3]{-1, -1, -1};
};

ImageLoader::ImageLoader(const char* url, int* result) {
  this->url = url;

  int x, y, n;
  data = stbi_load(this->url, &x, &y, &n, 0);

  if (nullptr == data) {
    *result = 0;
    return;
  }

  *result = 1;
  this->size[0] = x;
  this->size[1] = y;
  this->size[2] = n;

  arr = new int**[size[2]];
  for (int i = 0; i < size[2]; i++) {
    arr[i] = new int*[size[0]];
    for (int j = 0; j < size[0]; j++) {
      arr[i][j] = new int[size[1]];
      for (int k = 0; k < size[1]; k++) {
        arr[i][j][k] = *(data + size[2]*(k*size[0] + j) + i);
      }
    }
  }
}

ImageLoader::~ImageLoader() {
  stbi_image_free(this->data);
  delete[] arr;
}

ImageLoader::ImageLoader() = default;


int*** ImageLoader::getPicture() {
  return arr;
}

int* ImageLoader::getSize() {
  return this->size;
}

#endif //CNN_PIC_LOADER_H

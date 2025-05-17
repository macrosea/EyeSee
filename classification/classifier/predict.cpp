#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
using namespace cv::dnn;
static string class_name[] = {"non", "wm"};

int main() {
  Mat frame = imread(
      "/Users/macrosea/ws/work/watermark/classifier/images/no/"
      "xy_street-performer_N2Q9SO2IQT.bmp",
      0);
  imshow("1", frame);
  //     cout<<frame.channels()<<endl;
  string path = "/tmp/frozen_models/frozen_graph.pb";
  Net net     = readNetFromTensorflow(path);
  printf("模型加载成功");
  Mat frame_32F;
  frame.convertTo(frame_32F, CV_32FC1);
  //   cout<<1-frame_32F/255.0<<endl;
  Mat blob =
      blobFromImage(frame_32F / 255.0, 1.0, Size(28, 28), Scalar(0, 0, 0));
  //   cout<<(blob.channels());
  net.setInput(blob);
  Mat out = net.forward();
  //     cout<<out.cols<<endl;
  Point maxclass;
  minMaxLoc(out, NULL, NULL, NULL, &maxclass);
  cout << "预测结果为：" << class_name[maxclass.x] << endl;
  waitKey(0);
}

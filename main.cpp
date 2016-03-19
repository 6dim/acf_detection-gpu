#include <opencv2/opencv.hpp>
//#include <boost/filesystem.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <iterator>
#include <sys/time.h>
#include <sys/stat.h>
#include <math.h>

#include "img_process.hpp"
#include "acf_detect.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{

	VideoCapture capture(argv[1]);

	if (!capture.isOpened())
		cout << "fail to open!" << endl;

	int counter = 0;
	Mat vframe;
	while (true)
	{
		capture >> vframe;
		if (vframe.empty())
		{
			break;
		}
		counter++;
	}

	cout << "frames in total = " << counter << endl;
	long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
	int frame_height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	int frame_width = capture.get(CV_CAP_PROP_FRAME_WIDTH);

	cout << "Total Frame Number:" << totalFrameNumber << endl;


	long frameToStart = 3 * totalFrameNumber / 4.0;
	capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
	cout << "Start with frame" << frameToStart << endl;


	long frameToStop = totalFrameNumber;

	if (frameToStop < frameToStart)
	{
		cout << "Frame Number is wrong!" << endl;
		return -1;
	}
	else
	{
		cout << "End with frame" << frameToStop << endl;
	}

	double rate = capture.get(CV_CAP_PROP_FPS);
	cout << "Frame Ratio FPS:" << rate << endl;


	bool stop = false;

	int delay = 1;          // 1000 / rate;
	long currentFrame = frameToStart;

	namedWindow("pedestrian detector", 1);


	unsigned img_height = 480;
	unsigned img_width  = 640;
	acf_detect acf(Size(img_width, img_height));
	img_process im_proc;
	Mat frame;

	while (!stop)
	{
		Mat img, img_luv;
		vector<bb_xma> bbs;   /// bb stored the detected bb
		if (!capture.read(frame))
		{
			cout << "Video Reading Failure" << endl;
			return -1;
		}

		cout << endl << "Now detect frame " << currentFrame << endl;

		double t = (double)getTickCount();

		/////////////////////////////////////////////////////////////////////////////////////

		img = frame;
		if (!img.data)
		{
			cout << "Image  is not loaded properly" << endl;  //handle failing images
			continue;
		}
		im_proc.rgb2luv(img, img_luv);
		acf(img_luv, bbs);

		///////////////////////////////////////////////////////////////////////////////////////

		t = (double)getTickCount() - t;
		printf("detection time = %gms\n", t*1000. / cv::getTickFrequency());
		size_t i, j;
		for(unsigned int j = 0; j < bbs.size(); ++j )
		{
			///@xma updated to include detection score (which will be used to sort the detection result)

			// cout < "boundingboxes ," << bbs[j].x << "," << bbs[j].y << "," <<  bbs[j].wd << "," << bbs[j].ht << "," << bbs[j].wt << endl;
			Rect r(bbs[j].x,bbs[j].y, bbs[j].wd ,bbs[j].ht);
			cout << "rectangle topleft " << r.tl() << " bottom right " << r.br() << endl;
			rectangle(img,r.tl() , r.br(),  cv::Scalar(0, 255, 0), 1);

			//rectangle(Mat& img, Rect rec, const Scalar& color, int thickness=1, int lineType=8, int shift=0 )
		}
		imshow("pedestrian detector", frame);

		//waitKey(int delay=0)
		int c = waitKey(delay);

		if ((char)c == 27 || currentFrame > frameToStop)
		{
			stop = true;
		}

		if (c >= 0)
		{
			waitKey(0);
		}
		currentFrame++;

	}

	im_proc.free_gpu();

	capture.release();
	waitKey(0);
	return 0;
}

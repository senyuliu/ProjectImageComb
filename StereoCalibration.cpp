/* This is sample from the OpenCV book. The copyright notice is below */

/* *************** License:**************************
Oct. 3, 2008
Right to use this code in any way you want without warranty, support or any guarantee of it working.

BOOK: It would be nice if you cited it:
Learning OpenCV: Computer Vision with the OpenCV Library
by Gary Bradski and Adrian Kaehler
Published by O'Reilly Media, October 3, 2008

AVAILABLE AT:
http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
Or: http://oreilly.com/catalog/9780596516130/
ISBN-10: 0596516134 or: ISBN-13: 978-0596516130

OPENCV WEBSITES:
Homepage:      http://opencv.org
Online docs:   http://docs.opencv.org
Q&A forum:     http://answers.opencv.org
GitHub:        https://github.com/opencv/opencv/
************************************************** */

#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <sstream>

#include "shellCommand.h"

using namespace cv;
using namespace std;

static int print_help()
{
	cout <<
		" Given a list of chessboard images, the number of corners (nx, ny)\n"
		" on the chessboards, and a flag: useCalibrated for \n"
		"   calibrated (0) or\n"
		"   uncalibrated \n"
		"     (1: use stereoCalibrate(), 2: compute fundamental\n"
		"         matrix separately) stereo. \n"
		" Calibrate the cameras and display the\n"
		" rectified results along with the computed disparity images.   \n" << endl;
	cout << "Usage:\n ./stereo_calib -w=<board_width default=9> -h=<board_height default=6> -s=<square_size default=1.0> <image list XML/YML file default=stereo_calib.xml>\n" << endl;
	return 0;
}


static void
StereoCalib(const vector<string>& imagelist, Size boardSize, float squareSize, bool displayCorners = true, bool useCalibrated = true, bool showRectified = true)
{
	if (imagelist.size() % 1 != 0)
	{
		cout << "Error: the image list contains odd (non-even) number of elements\n";
		return;
	}

	const int maxScale = 2;
	// ARRAY AND VECTOR STORAGE:

	vector<vector<Point2f> > imagePoints[2];
	vector<vector<Point3f> > objectPoints;
	Size imageSize;

	int i, j, k, nimages = (int)imagelist.size();

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	vector<string> goodImageList;

	for (i = j = 0; i < nimages; i++)
	{
		const string& filename = imagelist[i * 1];
		Mat img_o = imread(filename, 0);
		if (img_o.empty())
			break;
		if (imageSize == Size())
			imageSize = img_o.size();
		else if (img_o.size() != imageSize)
		{
			cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
			break;
		}
                cv::Rect rect_l(0,0,img_o.cols/2, img_o.rows);
                cv::Rect rect_r(img_o.cols/2,0, img_o.cols/2, img_o.rows);

                Mat mat_l, mat_r; 
                
                img_o(rect_l).copyTo(mat_l); 
                img_o(rect_r).copyTo(mat_r);

		for (k = 0; k < 2; k++)
		{
                        Mat img; 
                        if(k == 0)
                        {
                            mat_l.copyTo(img);
                        }
                        else 
                        {
                            mat_r.copyTo(img);                                    
                        }   
                        //imshow("mat",img);
                        //waitKey(0);                        
                           
			bool found = false;
			vector<Point2f>& corners = imagePoints[k][j];
			for (int scale = 1; scale <= maxScale; scale++)
			{
				Mat timg;
				if (scale == 1)
					timg = img;
				else
					resize(img, timg, Size(), scale, scale, INTER_LINEAR);
				found = findChessboardCorners(timg, boardSize, corners,
					CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
				if (found)
				{
					if (scale > 1)
					{
						Mat cornersMat(corners);
						cornersMat *= 1. / scale;
					}
					break;
				}
			}
			if (displayCorners)
			{
				cout << filename << endl;
				Mat cimg, cimg1;
				cvtColor(img, cimg, COLOR_GRAY2BGR);
				drawChessboardCorners(cimg, boardSize, corners, found);
				double sf = 1280. / MAX(img.rows, img.cols);
				resize(cimg, cimg1, Size(), sf, sf, INTER_LINEAR);
				imshow("corners", cimg1);
				char c = (char)waitKey(500);
				if (c == 27 || c == 'q' || c == 'Q') //Allow ESC to quit
					exit(-1);
			}
			else
				putchar('.');
			if (!found)
				break;
			cornerSubPix(img, corners, Size(11, 11), Size(-1, -1),
				TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,
					30, 0.01));
		}
		if (k == 2)
		{
			goodImageList.push_back(imagelist[i]);
			//goodImageList.push_back(imagelist[i * 2 + 1]);
			j++;
		}
	}
	cout << j << " pairs have been successfully detected.\n";
	nimages = j;
	if (nimages < 2)
	{
		cout << "Error: too little pairs to run the calibration\n";
		return;
	}

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	objectPoints.resize(nimages);

	for (i = 0; i < nimages; i++)
	{
		for (j = 0; j < boardSize.height; j++)
			for (k = 0; k < boardSize.width; k++)
				objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
	}

	cout << "Running stereo calibration ...\n";

	Mat cameraMatrix[2], distCoeffs[2];
	cameraMatrix[0] = initCameraMatrix2D(objectPoints, imagePoints[0], imageSize, 0);
	cameraMatrix[1] = initCameraMatrix2D(objectPoints, imagePoints[1], imageSize, 0);
	Mat R, T, E, F;

        imageSize.width = 1280;
        std::cout<< "imageSize: " << imageSize.width << std::endl; 
        std::cout<< "imageSize: " << imageSize.height << std::endl; 


	double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
		cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, E, F,
		0,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,800, 1e-5));
	cout << "done with RMS error=" << rms << endl;

	// CALIBRATION QUALITY CHECK
	// because the output fundamental matrix implicitly
	// includes all the output information,
	// we can check the quality of calibration using the
	// epipolar geometry constraint: m2^t*F*m1=0
	double err = 0;
	int npoints = 0;
	vector<Vec3f> lines[2];
	for (i = 0; i < nimages; i++)
	{
		int npt = (int)imagePoints[0][i].size();
		Mat imgpt[2];
		for (k = 0; k < 2; k++)
		{
			imgpt[k] = Mat(imagePoints[k][i]);
			undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
			computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);
		}
		for (j = 0; j < npt; j++)
		{
			double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
				imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
				fabs(imagePoints[1][i][j].x*lines[0][j][0] +
					imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
			err += errij;
		}
		npoints += npt;
	}
	cout << "average epipolar err = " << err / npoints << endl;

	// save intrinsic parameters
	FileStorage fs("intrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
			"M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
		fs.release();
	}
	else
		cout << "Error: can not save the intrinsic parameters\n";

	fs.open("_M1.xml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs <<"_M1" << cameraMatrix[0]; 
		fs.release();
	}
	fs.open("_D1.xml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs <<"_D1" << distCoeffs[0];
		fs.release();
	}
	fs.open("_M2.xml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs <<"_M2" << cameraMatrix[1];
		fs.release();
	}
	fs.open("_D2.xml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs <<"_D2" << distCoeffs[1];
		fs.release();
	}
	Mat R1, R2, P1, P2, Q;
	Rect validRoi[2];

	stereoRectify(cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, R1, R2, P1, P2, Q,
		CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

	fs.open("extrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
		fs.release();
	}
	else
		cout << "Error: can not save the extrinsic parameters\n";

	fs.open("_R.xml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs <<"_R" << R;
		fs.release();
	}
	fs.open("_T.xml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs <<"_T" << T;
		fs.release();
	}

}


static bool readStringList(const string& filename, vector<string>& l)
{
	l.resize(0);
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	FileNode n = fs.getFirstTopLevelNode();
	if (n.type() != FileNode::SEQ)
		return false;
	FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
		l.push_back((string)*it);
	return true;
}

int main(int argc, char** argv)
{
	Size boardSize;
	string imagelistfn;
	bool showRectified;
	cv::CommandLineParser parser(argc, argv, "{w|12|}{h|16|}{s|5.0|}{nr||}{help||}{@input|stereo_calib.xml|}");
	if (parser.has("help"))
		return print_help();
	showRectified = !parser.has("nr");
	//imagelistfn = samples::findFile(parser.get<string>("@input"));

	boardSize.width = parser.get<int>("w");
	boardSize.height = parser.get<int>("h");
	float squareSize = parser.get<float>("s");
	if (!parser.check())
	{
		parser.printErrors();
		return 1;
	}
	vector<string> imagelist;
        //bool getformatList(std::string &cmd, std::vector<std::string> &fileList);
        //string path  = "/home/ibd01/DVISION/src/tools/calibration/data_org/right/*.jpg";
        //string pathr  = "/home/ibd01/DVISION/src/tools/calibration/data_org/left/";        
        string path  = "/media/ibd01/智能感知业务部张腾波2号盘/标尺/0027-1-10-0001-20191104-173131000-L-1.jpg/0027-1-10-0001-20191104-173131000-L-1.jpg/*.jpg";
        string pathr  = "/media/ibd01/智能感知业务部张腾波2号盘/标尺/0027-1-10-0001-20191104-173131000-L-1.jpg/0027-1-10-0001-20191104-173131000-R-1.jpg/";

        string patho = "/media/ibd01/智能感知业务部张腾波2号盘/标尺/0027-1-10-0001-20191104-173131000-L-1.jpg/combine/"; 
        string cmd   = "ls " + path; 

        IBD_SD::CShellCommand commandObj;
        bool ok = commandObj.getformatList(cmd, imagelist);
        std::cout<< "imagelist.size: " << imagelist.size(); 
        for(int i = 0 ; i < imagelist.size() ; i++)
        {
            //std::cout<< imagelist[i] << std::endl; 
            string files = imagelist[i];
          
            int    pos     = files.find_last_of("/");
            string name    = files.substr(pos + 1); 
          
            int    pos_n = name.find_last_of("-");
            string name_n= name.substr(0,pos_n-1) + "R" + name.substr(pos_n);
            //string name_n= name.substr(0,pos_n + 1) + "0.jpg.jpg";
            //string  name_n = name; 

            //std::cout<<"name_n: " << name_n <<std::endl; 
            //continue; 

            string filesr= pathr + name_n; 
            //std::cout<< filesr << std::endl; 

            //image 
            Mat matLeft = imread(files , 1);
            Mat matRight= imread(filesr, 1);

            if(matRight.rows < 32 || matLeft.rows < 32)
            {
                std::cout<<"bad news" <<filesr << std::endl;
                //remove(imagelist[i].c_str()); 
                continue; 
            }

            //namedWindow("matl", 2);
            //namedWindow("matr", 2);

            //imshow("matl", matLeft);
            //imshow("matr", matRight);

            //waitKey(0);

            int rows = matLeft.rows; 
            int cols = matLeft.cols; 
            //std::cout<<"rows: "<<rows << "cols: " <<cols <<std::endl; 
            
            Mat matOut(rows , cols * 2 , CV_8UC3, Scalar::all(0));

            cv::Rect rect_l(0,0, cols, rows);
            cv::Rect rect_r(cols,0, cols,rows);

            matLeft.copyTo(matOut(rect_l)); 
            matRight.copyTo(matOut(rect_r));
            for(int i = 0 ; i < matOut.rows; i = i + (matOut.rows/10) )
            {
                Point pt1(0, i); 
                Point pt2(matOut.cols - 1, i);
                line(matOut, pt1, pt2, Scalar(0,0,255), 2, 8);
            }
            
            //imshow("mat", matOut);
            //waitKey(0);
            string pathout = patho + name; 
            imwrite(pathout, matOut);
        }

	//StereoCalib(imagelist, boardSize, squareSize, true, true, showRectified);
	return 0;
}

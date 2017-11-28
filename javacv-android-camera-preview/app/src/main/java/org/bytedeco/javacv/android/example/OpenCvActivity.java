package org.bytedeco.javacv.android.example;

import android.app.Activity;
import android.os.AsyncTask;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v4.view.GestureDetectorCompat;
import android.util.Log;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Toast;

import org.bytedeco.javacpp.opencv_calib3d;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacv.android.example.utils.StorageHelper;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Size;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.javacpp.opencv_core.CV_32F;
import static org.bytedeco.javacpp.opencv_core.CV_32FC1;
import static org.bytedeco.javacpp.opencv_core.CV_64FC1;
import static org.bytedeco.javacpp.opencv_core.CV_8UC1;
import static org.bytedeco.javacpp.opencv_core.CV_8UC3;
import static org.bytedeco.javacpp.opencv_core.LINE_8;
import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_core.setIdentity;
import static org.bytedeco.javacpp.opencv_imgproc.CV_AA;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2RGB;
import static org.bytedeco.javacpp.opencv_imgproc.CV_FONT_HERSHEY_PLAIN;
import static org.bytedeco.javacpp.opencv_imgproc.CV_FONT_HERSHEY_SCRIPT_SIMPLEX;
import static org.bytedeco.javacpp.opencv_imgproc.CV_RGB2BGR;
import static org.bytedeco.javacpp.opencv_imgproc.CV_RGB2BGRA;
import static org.bytedeco.javacpp.opencv_imgproc.cvInitFont;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;

/**
 * Created by hunghd on 4/10/17.
 */

public class OpenCvActivity extends Activity implements CvCameraPreview.CvCameraViewListener
        , GestureDetector.OnGestureListener
{

    final String TAG = "OpenCvActivity";

    private CascadeClassifier faceDetector;
    private CascadeClassifier personDetector;
    private int absoluteFaceSize = 0;
    private CvCameraPreview cameraView;
    private opencv_imgproc opencv_imgproc_;
    private opencv_imgproc.CvFont mCvFont;


    private GestureDetectorCompat mDetector;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_opencv);

        cameraView = (CvCameraPreview) findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this);


//        mDetector = new GestureDetectorCompat(this,cameraView);
        mDetector = new GestureDetectorCompat(this,this);
        cameraView.setGestureDetector(mDetector);


        new AsyncTask<Void, Void, Void>() {
            @Override
            protected Void doInBackground(Void... voids) {
                faceDetector    = StorageHelper.loadClassifierCascade(OpenCvActivity.this, R.raw.frontalface);
                personDetector  = StorageHelper.loadClassifierCascade(OpenCvActivity.this, R.raw.cascade_g_person);
                opencv_imgproc_ = new opencv_imgproc();

                mCvFont = new opencv_imgproc.CvFont();
                opencv_imgproc.cvInitFont(mCvFont,CV_FONT_HERSHEY_SCRIPT_SIMPLEX,1.0,1.0,0,1,CV_AA);
                return null;
            }
        }.execute();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        absoluteFaceSize = (int) (width * 0.32f);
    }

    @Override
    public void onCameraViewStopped() {

    }


    private int textLineNumberOnThisFrame = 0;
    private void displayText(Mat rgbaMat, String text) {
        //@Namespace("cv") public static native void putText( @ByVal Mat img, @Str String text, @ByVal Point org, int fontFace, double fontScale, @ByVal Scalar color );
        textLineNumberOnThisFrame++;
        opencv_core.Point pointForText = new opencv_core.Point (20,textLineNumberOnThisFrame*30);
        opencv_imgproc.putText(rgbaMat, text, pointForText, CV_FONT_HERSHEY_PLAIN, 1.0, opencv_core.Scalar.WHITE, 1, LINE_8, false);
    }

    int lastFrameCols = 0;
    int lastFrameRows = 0;
    private void resetForNewDisplay(Mat rgbaMat) {
        textLineNumberOnThisFrame=0;
        lastFrameCols = rgbaMat.cols();
        lastFrameRows = rgbaMat.rows();
    }


    private int neighborhoodDiameter = 5;
    private int sigmaColour = -80;
    private int sigmaSpace = sigmaColour;
    private int width = 0;
    private int height = 0;

    private int frameNum = 0;

    @Override
    public Mat onCameraFrame(Mat rgbaMat) {
        Log.i(TAG,"onCameraFrame: start");

        if(frameNum <=1) {
            Log.i(TAG, "onCameraFrame: before opencv_calib3d.solvePnPRansac");
            double[] worlddata = new double[] {
                    622.0000d, 643.5000d,  90.0000d,
                    655.0000d, 643.5000d,    0.000d,
                    638.0000d, 643.5000d, 115.0000d,
                    648.0000d, 643.5000d, 115.0000d,
                    648.0000d, 636.5000d, 105.0000d,
                    638.0000d, 636.5000d, 105.0000d,
                    763.0000d, 643.5000d, 116.0000d,
                    770.5000d, 643.5000d, 116.0000d,
                    770.5000d, 643.5000d, 104.5000d,
                    785.0000d, 560.0000d, 105.0000d,
                    785.0000d, 410.0000d, 105.0000d,
                    710.0000d, 410.0000d, 105.0000d,
                    549.0000d, 643.5000d,  80.0000d,
                    549.0000d, 568.5000d,  80.0000d,
                    399.0000d, 568.5000d,  80.0000d,
                    535.5000d, 466.0000d,  79.5000d,
                    535.5000d, 316.0000d,  79.5000d};
            Mat objectPoints   = new Mat(worlddata.length/3, 3, CV_64FC1); // 10 measurements, 3 dimensions, 1 channel: could also have Mat(10,1,3)
            objectPoints.getDoubleBuffer().put(worlddata);
            Log.i(TAG, "onCameraFrame: objectPoints = "+matToString(objectPoints));
            Log.i(TAG, "onCameraFrame: objectPoints.toString()="+objectPoints.toString());
            Log.i(TAG, "onCameraFrame: objectPoints.size().toString()="+objectPoints.size().toString());
            // i.e. a 10-element vector with each element a 3d measurement/point :
            // Nx3 1-channel or 1xN/Nx1 3-channel
            // HeadPose example uses coordinates based on the head model centre - relative to camera focal axis - [ x=right , y=up , z=depth/distance=toward-camera]
            List<opencv_core.Point2f> imagePointVector = new ArrayList<>();
            double[] pixeldata = new double[]{
                    1538d, 589d,
                    1723d, 1013d,
                    1622d, 454d,
                    1673d, 452d,
                    1676d, 511d,
                    1625d, 512d,
                    2290d, 433d,
                    2328d, 433d,
                    2323d, 493d,
                    2549d, 615d,
                    3009d, 1003d,
                    2352d, 1016d,
                    1158d, 639d,
                    1112d, 774d,
                     175d, 790d,
                     974d, 1041d,
                     616d, 1808d};
            Mat imagePoints    = new Mat(worlddata.length/3, 2, CV_64FC1);
            imagePoints.getDoubleBuffer().put(pixeldata);
            Log.i(TAG, "onCameraFrame: imagePoints = "+matToString(imagePoints));
            imagePoints.getDoubleBuffer().toString();
            Log.i(TAG, "onCameraFrame: imagePoints.toString()="+imagePoints.toString());
            Log.i(TAG, "onCameraFrame: imagePoints.size().toString()="+imagePoints.size().toString());
            imagePoints.size().toString();
//            org.bytedeco.javacpp.Indexer indexer = imagePoints.createIndexer();
//            indexer.put(row,col,val)
//            Mat imagePoints    = new Mat(10, 2, CV_64FC1); // pixel coordinates, as [ u=x=right, v=y=down ]
//            imagePoints.push_back(new Mat(opencv_core.Point2f(1538.0f,589.0f)));
            Mat cameraMatrix   = new Mat(3, 3, CV_64FC1);
            cameraMatrix.getDoubleBuffer().put(new double[]{
                    2873.9d,       0.0d,    1624.4d,
                       0.0d,    2867.8d,     921.6d,
                       0.0d,       0.0d,       1.0d});
            Log.i(TAG, "onCameraFrame: cameraMatrix = "+matToString(cameraMatrix));
            double[] distCoeffsArray = new double[]{ 0.0748d, -0.1524d, 0.0887d , 0.0d , 0.0d };
            Mat distCoeffs     = new Mat(distCoeffsArray.length, 1, CV_64FC1 );
            Log.i(TAG, "onCameraFrame: distCoeffs = "+matToString(distCoeffs));
            distCoeffs.getDoubleBuffer().put(distCoeffsArray);
            Mat rVec_Estimated = new Mat(3, 3, CV_64FC1); // Output rotation   _vector_ : world-to-camera
            Mat tVec_Estimated = new Mat(3, 1, CV_64FC1); // Output translation vector  : world-to-camera

            //        @param cameraMatrix Input camera matrix \f$A = \vecthreethree
            //              {fx}{0}{cx}
            //              {0}{fy}{cy}
            //              {0}{0}{1}\f$ .
            //        @param distCoeffs Input vector of distortion coefficients
            //             \f$( k_1, k_2, p_1, p_2  [, k_3  [, k_4, k_5, k_6  [, s_1, s_2, s_3, s_4  [, \tau_x, \tau_y] ] ] ] )\f$ of
            //        4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
            //        assumed.
            //        @param rvec Output rotation vector (see Rodrigues ) that, together with tvec , brings points from
            //        the model coordinate system to the camera coordinate system.
            //        @param tvec Output translation vector.
            //        @Namespace("cv") public static native @Cast("bool") boolean solvePnPRansac( @ByVal Mat objectPoints, @ByVal Mat imagePoints,
            //                                  @ByVal Mat cameraMatrix, @ByVal Mat distCoeffs,
            //                                  @ByVal Mat rvec, @ByVal Mat tvec,
            //                                  @Cast("bool") boolean useExtrinsicGuess/*=false*/, int iterationsCount/*=100*/,
            //                                  float reprojectionError/*=8.0*/, double confidence/*=0.99*/,
            //                                  @ByVal(nullValue = "cv::OutputArray(cv::noArray())") Mat inliers, int flags/*=cv::SOLVEPNP_ITERATIVE*/ );
            opencv_calib3d.solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs, rVec_Estimated, tVec_Estimated);

            Log.i(TAG, "onCameraFrame: rVec_Estimated = "+rVec_Estimated);
            Log.i(TAG, "onCameraFrame: rVec_Estimated.size().width()= "+rVec_Estimated.size().width()+", .height()="+rVec_Estimated.size().height());
            Log.i(TAG, "onCameraFrame: rVec_Estimated = "+matToString(rVec_Estimated));

            Mat rotationMatrix = new Mat(3, 3, CV_64FC1);
            opencv_calib3d.Rodrigues(rVec_Estimated,rotationMatrix);
            Log.i(TAG, "onCameraFrame: rotationMatrix = "+matToString(rotationMatrix));

            opencv_core.MatExpr rotationMatrixTransposeTemp = rotationMatrix.t();
            Mat rotationMatrixTranspose = new Mat(1,1,CV_64FC1);
            rotationMatrixTranspose.put(rotationMatrixTransposeTemp);
            Log.i(TAG, "onCameraFrame: rotationMatrixTranspose = "+matToString(rotationMatrixTranspose));

//            Mat rotationMatrixInverse      = rotationMatrixTranspose;
            Mat negOne = new Mat(3,3,CV_64FC1); setIdentity(negOne, new opencv_core.Scalar(-1.0d));
            Log.i(TAG, "onCameraFrame: negOne = "+matToString(negOne));
            negOne.getDoubleBuffer().put(new double[]{
                -1.0d,  0.0d,  0.0d,
                 0.0d, -1.0d,  0.0d,
                 0.0d,  0.0d, -1.0d });
            Log.i(TAG, "onCameraFrame: negOne = "+matToString(negOne));

            Log.i(TAG, "onCameraFrame: tVec_Estimated = "+tVec_Estimated);
            Log.i(TAG, "onCameraFrame: tVec_Estimated = "+matToString(tVec_Estimated));

            Mat one = new Mat(3,3,CV_64FC1); setIdentity(one, new opencv_core.Scalar(1.0d));
            Log.i(TAG, "onCameraFrame: one = "+matToString(one));
//            negOne.getDoubleBuffer().put(new double[]{
//                    1.0d,  0.0d,  0.0d,
//                    0.0d, 1.0d,  0.0d,
//                    0.0d,  0.0d, 1.0d });
            Log.i(TAG, "onCameraFrame: one = "+matToString(one));
            opencv_core.MatExpr translationInverseTemp_One = one.mul(rotationMatrixTranspose);
            Log.i(TAG, "onCameraFrame: translationInverseTemp_One.asMat() = "+matToString(translationInverseTemp_One.asMat()));
            translationInverseTemp_One = opencv_core.multiply(rotationMatrixTranspose,one);
            Log.i(TAG, "onCameraFrame: translationInverseTemp_One.asMat() = "+matToString(translationInverseTemp_One.asMat()));

            opencv_core.MatExpr translationInverseTemp = opencv_core.multiply(rotationMatrixTranspose,negOne); // element-wise
            Mat dummy_1 = translationInverseTemp.asMat();
            Log.i(TAG, "onCameraFrame: translationInverseTemp.asMat() = "+matToString(translationInverseTemp.asMat()));
            Log.i(TAG, "onCameraFrame: translationInverseTemp.size() width="+translationInverseTemp.size().width()+", height= "+translationInverseTemp.size().height());
            Log.i(TAG, "onCameraFrame: translationInverseTemp = "+translationInverseTemp);
            opencv_core.MatExpr translationInverseTempTemp = opencv_core.multiply(translationInverseTemp,tVec_Estimated);
            Mat dummy_2 = translationInverseTempTemp.asMat();
            Log.i(TAG, "onCameraFrame: translationInverseTempTemp.asMat() = "+matToString(translationInverseTempTemp.asMat()));
            Log.i(TAG, "onCameraFrame: translationInverseTempTemp.size() width="+translationInverseTempTemp.size().width()+", height= "+translationInverseTempTemp.size().height());
            Log.i(TAG, "onCameraFrame: translationInverseTempTemp = "+translationInverseTempTemp);
            Mat translationInverse = new Mat(1,1,CV_64FC1);
            Log.i(TAG, "onCameraFrame: translationInverse = "+translationInverse);
            translationInverse.put(translationInverseTempTemp);
            Log.i(TAG, "onCameraFrame: translationInverse = "+matToString(translationInverse));

            Mat cameraPoseInWorldCoords = new Mat(4, 4, CV_64FC1);
                // As in Python, start is an inclusive left boundary of the range and end is an exclusive right boundary of the range. Such a half-opened interval ...
                // 0..2, 0..2 inclusive
//            Mat camPoseTopLeft = cameraPoseInWorldCoords.apply(new opencv_core.Range(0,3), new opencv_core.Range(0,3));
//            Log.i(TAG, "onCameraFrame: camPoseTopLeft before = "+matToString(camPoseTopLeft));
//            camPoseTopLeft.getDoubleBuffer().put(rotationMatrixTranspose.getDoubleBuffer());
//            Log.i(TAG, "onCameraFrame: camPoseTopLeft = "+matToString(camPoseTopLeft));
//            Log.i(TAG, "onCameraFrame: cameraPoseInWorldCoords = "+matToString(cameraPoseInWorldCoords));  // goes sequentially, not in the shape of top left - how is this useful ??
            cameraPoseInWorldCoords.getDoubleBuffer().put(0, rotationMatrixTranspose.getDoubleBuffer().get(0)); // row one, all cols
            cameraPoseInWorldCoords.getDoubleBuffer().put(1, rotationMatrixTranspose.getDoubleBuffer().get(1)); // row one, all cols
            cameraPoseInWorldCoords.getDoubleBuffer().put(2, rotationMatrixTranspose.getDoubleBuffer().get(2)); // row one, all cols
            cameraPoseInWorldCoords.getDoubleBuffer().put(4, rotationMatrixTranspose.getDoubleBuffer().get(3));
            cameraPoseInWorldCoords.getDoubleBuffer().put(5, rotationMatrixTranspose.getDoubleBuffer().get(4));
            cameraPoseInWorldCoords.getDoubleBuffer().put(6, rotationMatrixTranspose.getDoubleBuffer().get(5));
            cameraPoseInWorldCoords.getDoubleBuffer().put(8, rotationMatrixTranspose.getDoubleBuffer().get(6));
            cameraPoseInWorldCoords.getDoubleBuffer().put(9, rotationMatrixTranspose.getDoubleBuffer().get(7));
            cameraPoseInWorldCoords.getDoubleBuffer().put(10, rotationMatrixTranspose.getDoubleBuffer().get(8));
            Log.i(TAG, "onCameraFrame: cameraPoseInWorldCoords after inserting rotation = "+matToString(cameraPoseInWorldCoords));

                // 0..2,3..3 inclusive
//            Mat camPoseRightCol = cameraPoseInWorldCoords.apply(new opencv_core.Range(0,3), new opencv_core.Range(3,4));
//            Log.i(TAG, "onCameraFrame: camPoseRightCol before = "+matToString(camPoseRightCol));
//            camPoseRightCol.getDoubleBuffer().put(translationInverse.getDoubleBuffer());
//            Log.i(TAG, "onCameraFrame: camPoseRightCol = "+matToString(camPoseRightCol));
//            Log.i(TAG, "onCameraFrame: cameraPoseInWorldCoords = "+matToString(cameraPoseInWorldCoords));  // goes sequentially, not in the shape of top left - how is this useful ??
            cameraPoseInWorldCoords.getDoubleBuffer().put(3, translationInverse.getDoubleBuffer().get(0));
            cameraPoseInWorldCoords.getDoubleBuffer().put(7, translationInverse.getDoubleBuffer().get(1));
            cameraPoseInWorldCoords.getDoubleBuffer().put(11, translationInverse.getDoubleBuffer().get(2));
            Log.i(TAG, "onCameraFrame: cameraPoseInWorldCoords after inserting translation = "+matToString(cameraPoseInWorldCoords));

// from here
            double[] homogeneousPadding = new double[]{0.0d, 0.0d, 0.0d, 1.0d};
//                // 3..3,0..3 inclusive
//            cameraPoseInWorldCoords.apply(new opencv_core.Range(3,4), new opencv_core.Range(0,4)).put(new Mat(homogeneousPadding));
//            Log.i(TAG, "onCameraFrame: cameraPoseInWorldCoords = "+matToString(cameraPoseInWorldCoords));
            cameraPoseInWorldCoords.getDoubleBuffer().put(12, homogeneousPadding[0]);
            cameraPoseInWorldCoords.getDoubleBuffer().put(13, homogeneousPadding[1]);
            cameraPoseInWorldCoords.getDoubleBuffer().put(14, homogeneousPadding[2]);
            cameraPoseInWorldCoords.getDoubleBuffer().put(15, homogeneousPadding[3]);
            Log.i(TAG, "onCameraFrame: cameraPoseInWorldCoords after padding = "+matToString(cameraPoseInWorldCoords));

            Log.i(TAG, "onCameraFrame: after opencv_calib3d.solvePnPRansac");
        }

        resetForNewDisplay(rgbaMat);
//        if (personDetector != null) {
//            Mat grayMat = new Mat(rgbaMat.rows(), rgbaMat.cols());
//            Size grayMatSize = grayMat.size();
//
//            // @Namespace("cv") public static native void cvtColor( @ByVal Mat src, @ByVal Mat dst, int code );
//            cvtColor(rgbaMat, grayMat, CV_BGR2GRAY);
//
//            if (opencv_imgproc_ != null) {
//                //  http://bytedeco.org/javacpp-presets/opencv/apidocs/
//                //  https://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html#bilateralfilter
//                //	bilateralFilter(opencv_core.Mat src, opencv_core.Mat dst, int d, double sigmaColor, double sigmaSpace)
//                Mat grayMatPrefiltered = new Mat(rgbaMat.rows(), rgbaMat.cols());
//                grayMatPrefiltered = grayMat.clone();
//                int neighborhoodDiameter = 5;
//                int sigmaColour = 80;
//                int sigmaSpace = sigmaColour;
//                if (grayMatSize.width() > grayMatSize.height()) {
//                    sigmaColour = grayMatSize.height();
//                    sigmaSpace = sigmaColour;
//                } else {
//                    sigmaColour = grayMatSize.width();
//                    sigmaSpace = sigmaColour;
//                }
//                opencv_imgproc_.bilateralFilter(grayMatPrefiltered, grayMat, neighborhoodDiameter, sigmaColour, sigmaSpace);
//                displayText(rgbaMat, "bilat:n=" + neighborhoodDiameter + ",sc=" + sigmaColour + ",ss=" + sigmaSpace);
//            }
//        }
        if (faceDetector != null) {
            Mat grayMat = new Mat(rgbaMat.rows(), rgbaMat.cols());
            Size grayMatSize = grayMat.size();

                // @Namespace("cv") public static native void cvtColor( @ByVal Mat src, @ByVal Mat dst, int code );
            cvtColor(rgbaMat, grayMat, CV_BGR2GRAY);

            Mat rgbMat = new Mat(rgbaMat.rows(), rgbaMat.cols(), CV_8UC3);
            cvtColor(rgbaMat, rgbMat, CV_BGR2RGB);

            if(opencv_imgproc_ != null) {
                    //  http://bytedeco.org/javacpp-presets/opencv/apidocs/
                    //  https://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html#bilateralfilter
                    //	bilateralFilter(opencv_core.Mat src, opencv_core.Mat dst, int d, double sigmaColor, double sigmaSpace)

                int rgbaMatColourSpace = rgbaMat.type();
                Log.i("OpenCvActivity","rgbaMatColourSpace = "+rgbaMatColourSpace);

                Mat grayMatPrefiltered = new Mat(rgbaMat.rows(), rgbaMat.cols());
                Mat rgbaMatPrefiltered = rgbMat.clone();
                Mat rgbaMatPostfiltered = rgbMat.clone();
//                cvtColor(rgbaMatPrefiltered, rgbaMatPrefiltered, CV_8UC3);
//                cvtColor(rgbaMatPostfiltered, rgbaMatPostfiltered, CV_8UC3);


                if(sigmaColour<=0) {
                    if (grayMatSize.width() > grayMatSize.height()) {
                        sigmaColour = grayMatSize.height();
                        sigmaSpace = sigmaColour;
                    } else {
                        sigmaColour = grayMatSize.width();
                        sigmaSpace = sigmaColour;
                    }
                }

                Log.i("OpenCvActivity","grayMatPrefiltered: "+grayMatPrefiltered.channels()+", "+grayMatPrefiltered.elemSize()+", "+grayMatPrefiltered.elemSize1());
                Log.i("OpenCvActivity","rgbaMatPrefiltered: "+rgbaMatPrefiltered.channels()+", "+rgbaMatPrefiltered.elemSize()+", "+rgbaMatPrefiltered.elemSize1());

                Log.i("OpenCvActivity","src.type() == CV_8UC1 : "+(rgbaMatPrefiltered.type() == CV_8UC1)+" | src.type() == CV_8UC3 : "+(rgbaMatPrefiltered.type() == CV_8UC3)+" | src.data != dst.data : "+(rgbaMatPrefiltered.data() != rgbaMatPostfiltered.data()));

//                opencv_imgproc_.bilateralFilter(grayMatPrefiltered, grayMat, neighborhoodDiameter, sigmaColour, sigmaSpace);
                opencv_imgproc_.bilateralFilter(rgbaMatPrefiltered, rgbaMatPostfiltered, neighborhoodDiameter, sigmaColour, sigmaSpace);


                cvtColor(rgbaMatPostfiltered, rgbaMat, CV_RGB2BGRA);
                displayText(rgbaMat,"bilat:n="+neighborhoodDiameter+",sc="+sigmaColour+",ss="+sigmaSpace);

                grayMatPrefiltered.release();
//                rgbMat.release();
                rgbaMatPostfiltered.release();
                rgbaMatPrefiltered.release();
            }

            RectVector faces = new RectVector();
            faceDetector.detectMultiScale(grayMat, faces, 1.25f, 3, 1,
                    new Size(absoluteFaceSize, absoluteFaceSize),
                    new Size(4 * absoluteFaceSize, 4 * absoluteFaceSize));
            Log.i("OpenCvActivity","onCameraFrame: found "+faces.size()+" faces.");
            for(int face_num = 0; face_num < faces.size(); face_num++) {
                int x = faces.get(face_num).x();
                int y = faces.get(face_num).y();
                int w = faces.get(face_num).width();
                int h = faces.get(face_num).height();
                rectangle(rgbaMat, new Point(x, y), new Point(x + w, y + h), opencv_core.Scalar.GREEN, 2, LINE_8, 0);
            }

//            displayText(rgbaMat,this.lastUserTouchAction);


            grayMat.release();
        } else {
            Log.i("OpenCvActivity","onCameraFrame: faceDetector is null");
        }
        Log.i("OpenCvActivity","onCameraFrame: end");

//        return rgbaMatPostfiltered;
        return rgbaMat;
    }

    private String matToString(Mat rotationMatrixTranspose) {
        String string = "";
        for (int rowNum_ = 0; rowNum_<rotationMatrixTranspose.size().height(); rowNum_++) {   // rows
            string = string+"\n";
            for(int colNum_ = 0; colNum_<rotationMatrixTranspose.size().width(); colNum_++) { // columns
                string = string + rotationMatrixTranspose.getDoubleBuffer().get( rowNum_*rotationMatrixTranspose.size().width() + colNum_) + ",";
            }
        }
        return string;
    }


//    @Override
//    public boolean onTouch(View v, MotionEvent event) {
//        double xOffset = (cameraView.getWidth() - lastFrameCols) / 2;
//        double yOffset = (cameraView.getHeight() - lastFrameRows) / 2;
//        double openCVCol = (double)(event).getX() - xOffset;
//        double openCVRow = (double)(event).getY() - yOffset;
//
//        Log.i(TAG, "Touch image coordinates: (" + openCVCol + ", " + openCVRow + ")");
//
//        return false;// don't need subsequent touch events
//
//    }


    private String lastUserTouchAction="none";

    @Override
    public boolean onDown(MotionEvent e) {
        Log.i("OpenCvActivity","GestureDetector - onDown - Will");
        lastUserTouchAction="onDown";
        return true;
    }

    @Override
    public void onShowPress(MotionEvent e) {
        Log.i("OpenCvActivity","GestureDetector - onShowPress - Will");
        lastUserTouchAction="onShowPress";

    }

    @Override
    public boolean onSingleTapUp(MotionEvent e) {
        Log.i("OpenCvActivity","GestureDetector - onSingleTapUp - Will");
        lastUserTouchAction="onSingleTapUp";
        return true;
    }

    @Override
    public boolean onScroll(MotionEvent e1, MotionEvent e2, float distanceX, float distanceY) {
        Log.i("OpenCvActivity","GestureDetector - onScroll - Will");
        lastUserTouchAction="onScroll:dx="+distanceX+",dy="+distanceY;
        return true;
    }

    @Override
    public void onLongPress(MotionEvent e) {
        Log.i("OpenCvActivity","GestureDetector - onLongPress - Will");
        lastUserTouchAction="onLongPress";

    }


    private static final int SWIPE_THRESHOLD = 100;
    private static final int SWIPE_VELOCITY_THRESHOLD = 100;


    @Override
    public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY) {
        Log.i("OpenCvActivity","GestureDetector - onFling - Will");
        lastUserTouchAction="onFling:velx="+velocityX+",vely="+velocityY;


        try {
            float diffY = e2.getY() - e1.getY();
            float diffX = e2.getX() - e1.getX();
            if (Math.abs(diffX) > Math.abs(diffY)) {
                if (Math.abs(diffX) > SWIPE_THRESHOLD && Math.abs(velocityX) > SWIPE_VELOCITY_THRESHOLD) {
                    if (e2.getY() < 900) {
                        if (diffX > 0) {
                            onSwipeRightTop();
                        } else {
                            onSwipeLeftTop();
                        }
                    } else {
                        if (diffX > 0) {
                            onSwipeRight();
                        } else {
                            onSwipeLeft();
                        }
                    }
                }
            } else {
                if (Math.abs(diffY) > SWIPE_THRESHOLD && Math.abs(velocityY) > SWIPE_VELOCITY_THRESHOLD) {
                    if (diffY > 0) {
                        onSwipeBottom();
                    } else {
                        onSwipeTop();
                    }
                }
            }
        } catch (Exception exception) {
            exception.printStackTrace();
        }
        return true;
    }

    public void onSwipeTop() {
        Toast.makeText(this, "top", Toast.LENGTH_SHORT).show();
        neighborhoodDiameter = neighborhoodDiameter*2;
    }
    public void onSwipeBottom() {
        Toast.makeText(this, "bottom", Toast.LENGTH_SHORT).show();
        neighborhoodDiameter = neighborhoodDiameter/2;
    }


    public void onSwipeRight() {
        Toast.makeText(this, "right", Toast.LENGTH_SHORT).show();
        sigmaColour = sigmaColour*2;
    }
    public void onSwipeLeft() {
        Toast.makeText(this, "left", Toast.LENGTH_SHORT).show();
        sigmaColour = sigmaColour/2;

    }
    public void onSwipeRightTop() {
        Toast.makeText(this, "right", Toast.LENGTH_SHORT).show();
        sigmaSpace = sigmaSpace*2;
    }
    public void onSwipeLeftTop() {
        Toast.makeText(this, "left", Toast.LENGTH_SHORT).show();
        sigmaSpace = sigmaSpace/2;

    }

}

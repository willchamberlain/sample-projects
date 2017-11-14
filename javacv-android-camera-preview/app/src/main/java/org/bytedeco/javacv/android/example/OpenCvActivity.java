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

import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacv.android.example.utils.StorageHelper;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Size;

import static org.bytedeco.javacpp.opencv_core.LINE_8;
import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_imgproc.CV_AA;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.CV_FONT_HERSHEY_PLAIN;
import static org.bytedeco.javacpp.opencv_imgproc.CV_FONT_HERSHEY_SCRIPT_SIMPLEX;
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


    @Override
    public Mat onCameraFrame(Mat rgbaMat) {
        Log.i("OpenCvActivity","onCameraFrame: start");
        resetForNewDisplay(rgbaMat);
        if (faceDetector != null) {
            Mat grayMat = new Mat(rgbaMat.rows(), rgbaMat.cols());
            Size grayMatSize = grayMat.size();

                // @Namespace("cv") public static native void cvtColor( @ByVal Mat src, @ByVal Mat dst, int code );
            cvtColor(rgbaMat, grayMat, CV_BGR2GRAY);

            if(opencv_imgproc_ != null) {
                    //  http://bytedeco.org/javacpp-presets/opencv/apidocs/
                    //  https://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html#bilateralfilter
                    //	bilateralFilter(opencv_core.Mat src, opencv_core.Mat dst, int d, double sigmaColor, double sigmaSpace)
                Mat grayMatPrefiltered = new Mat(rgbaMat.rows(), rgbaMat.cols());
                grayMatPrefiltered = grayMat.clone();
                int neighborhoodDiameter = 5;
                int sigmaColour = 80;
                int sigmaSpace = sigmaColour;
                if(grayMatSize.width() > grayMatSize.height()) {
                    sigmaColour = grayMatSize.height();
                    sigmaSpace = sigmaColour;
                } else {
                    sigmaColour = grayMatSize.width();
                    sigmaSpace = sigmaColour;
                }
                opencv_imgproc_.bilateralFilter(grayMatPrefiltered, grayMat, neighborhoodDiameter, sigmaColour, sigmaSpace);
                displayText(rgbaMat,"bilat:n="+neighborhoodDiameter+",sc="+sigmaColour+",ss="+sigmaSpace);
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

            displayText(rgbaMat,this.lastUserTouchAction);


            grayMat.release();
        } else {
            Log.i("OpenCvActivity","onCameraFrame: faceDetector is null");
        }
        Log.i("OpenCvActivity","onCameraFrame: end");

        return rgbaMat;
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
                    if (diffX > 0) {
                        onSwipeRight();
                    } else {
                        onSwipeLeft();
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
    }
    public void onSwipeRight() {
        Toast.makeText(this, "right", Toast.LENGTH_SHORT).show();
    }
    public void onSwipeLeft() {
        Toast.makeText(this, "left", Toast.LENGTH_SHORT).show();
    }
    public void onSwipeBottom() {
        Toast.makeText(this, "bottom", Toast.LENGTH_SHORT).show();
    }
}

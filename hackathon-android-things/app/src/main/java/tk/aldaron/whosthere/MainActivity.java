package tk.aldaron.whosthere;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.TensorFlowLite;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Collection;
import java.util.List;

import tk.aldaron.whosthere.classifier.Recognition;
import tk.aldaron.whosthere.classifier.TensorFlowHelper;


/**
 * Skeleton of an Android Things activity.
 *
 * Android Things peripheral APIs are accessible through the class
 * PeripheralManagerService. For example, the snippet below will open a GPIO pin and
 * set it to HIGH:
 *
 * <pre>{@code
 * PeripheralManagerService service = new PeripheralManagerService();
 * mLedGpio = service.openGpio("BCM6");
 * mLedGpio.setDirection(Gpio.DIRECTION_OUT_INITIALLY_LOW);
 * mLedGpio.setValue(true);
 * }</pre>
 *
 * For more complex peripherals, look for an existing user-space driver, or implement one if none
 * is available.
 *
 * @see <a href="https://github.com/androidthings/contrib-drivers#readme">https://github.com/androidthings/contrib-drivers#readme</a>
 *
 */
public class MainActivity extends Activity {
    private static final String TAG = "MainActivity";

    /** Tensor flow data files*/
    private static final String LABELS_FILE = "labels.txt";
    private static final String MODEL_FILE = "mobilenet_quant_v1_224.tflite";

    private Interpreter mTensorFlowLite;
    private List<String> mLabels;

    /** Camera image capture size */
    private static final int PREVIEW_IMAGE_WIDTH = 640;
    private static final int PREVIEW_IMAGE_HEIGHT = 480;

    /** Image dimensions required by TF model */
    private static final int TF_INPUT_IMAGE_WIDTH = 224;
    private static final int TF_INPUT_IMAGE_HEIGHT = 224;
    /** Dimensions of model inputs. */
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 3;

    private CameraHandler mCameraHandler;
    private Button mCameraButton;
    private ImagePreprocessor mImagePreprocessor;
    private ImageView mImage;


    /**
     * Initialize the classifier that will be used to process images.
     */
    private void initClassifier() {
        try {
            mTensorFlowLite = new Interpreter(TensorFlowHelper.loadModelFile(this, MODEL_FILE));
            mLabels = TensorFlowHelper.readLabels(this, LABELS_FILE);
        } catch (IOException e) {
            Log.w(TAG, "Unable to initialize TensorFlow Lite.", e);
        }
    }

    /**
     * Clean up the resources used by the classifier.
     */
    private void destroyClassifier() {
        mTensorFlowLite.close();
    }

    private void doRecognize(Bitmap image) {
        byte[][] confidencePerLabel = new byte[1][mLabels.size()];
        int [] intValues = new int[TF_INPUT_IMAGE_WIDTH*TF_INPUT_IMAGE_HEIGHT];
        ByteBuffer imgData = ByteBuffer.allocateDirect(DIM_BATCH_SIZE*TF_INPUT_IMAGE_WIDTH*TF_INPUT_IMAGE_HEIGHT*DIM_PIXEL_SIZE);

        imgData.order(ByteOrder.nativeOrder());

        TensorFlowHelper.convertBitmapToByteBuffer(image, intValues, imgData);

        mTensorFlowLite.run(imgData,confidencePerLabel);
        Recognition result = TensorFlowHelper.getBestResults(confidencePerLabel, mLabels);

        onPhotoRecognitionReady(result);
    }

    private void onPhotoRecognitionReady(Recognition result){
        // send result.toString somewhere
    }

    
    /**
     * Initialize the camera
     */
    private void initCamera() {
        mImagePreprocessor = new ImagePreprocessor(
                PREVIEW_IMAGE_WIDTH, PREVIEW_IMAGE_HEIGHT,
                TF_INPUT_IMAGE_WIDTH, TF_INPUT_IMAGE_HEIGHT);
        mCameraHandler = CameraHandler.getInstance();
        mCameraHandler.initializeCamera(this,
                PREVIEW_IMAGE_WIDTH, PREVIEW_IMAGE_HEIGHT, null,
                new ImageReader.OnImageAvailableListener() {
                    @Override
                    public void onImageAvailable(ImageReader imageReader) {
                        Bitmap bitmap = mImagePreprocessor.preprocessImage(imageReader.acquireNextImage());
                        onPhotoReady(bitmap);
                    }
                });
    }

    /**
     * Clean up the resources used by the camera
     */
    private void closeCamera() {
        mCameraHandler.shutDown();

    }
    /**
     * Image capture process complete
     */
    private void onPhotoReady(Bitmap bitmap) {
        mImage.setImageBitmap(bitmap);
        doRecognize(bitmap);
    }

    /**
     * load the image that will be used in the classification process
     */
    private void loadPhoto() {
        mCameraHandler.takePicture();
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mCameraButton = findViewById(R.id.button);
        mImage = findViewById(R.id.camera);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        initCamera();
        initClassifier();
        mCameraButton.setOnClickListener(new View.OnClickListener() {
             @Override
             public void onClick(View v) {
                 loadPhoto();
             }
         }

        );

    }

}

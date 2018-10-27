package tk.aldaron.whosthere.classifier;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Helper functions for the TensorFlow image classifier.
 */
public class TensorFlowHelper {

    /**
     * Memory-map the model file in Assets.
     */
    public static MappedByteBuffer loadModelFile(Context context, String modelFile)
            throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelFile);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public static List<String> readLabels(Context context, String labelsFile) {
        AssetManager assetManager = context.getAssets();
        ArrayList<String> result = new ArrayList<>();
        try (InputStream is = assetManager.open(labelsFile);
             BufferedReader br = new BufferedReader(new InputStreamReader(is))) {
            String line;
            while ((line = br.readLine()) != null) {
                result.add(line);
            }
            return result;
        } catch (IOException ex) {
            throw new IllegalStateException("Cannot read labels from " + labelsFile);
        }
    }

    /**
     * Find the best classifications.
      */
    public static Recognition getBestResults(byte[][] labelProbArray, List<String> labelList) {
        Recognition topResult = null;
        for (int i = 0; i < labelList.size(); ++i) {
            Recognition r = new Recognition( String.valueOf(i),
                    labelList.get(i), (labelProbArray[0][i] & 0xff) / 255.0f);
            if (r.getConfidence() > 0) {
                if (topResult == null){
                    topResult = r;
                }else if (r.getConfidence() > topResult.getConfidence()){
                    topResult = r;
                }
            }
        }
        return topResult;
    }

    /** Writes Image data into a {@code ByteBuffer}. */
    public static void convertBitmapToByteBuffer(Bitmap bitmap, int[] intValues, ByteBuffer imgData) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0,
                bitmap.getWidth(), bitmap.getHeight());
        // Encode the image pixels into a byte buffer representation matching the expected
        // input of the Tensorflow model
        int pixel = 0;
        for (int i = 0; i < bitmap.getWidth(); ++i) {
            for (int j = 0; j < bitmap.getHeight(); ++j) {
                final int val = intValues[pixel++];
                imgData.put((byte) ((val >> 16) & 0xFF));
                imgData.put((byte) ((val >> 8) & 0xFF));
                imgData.put((byte) (val & 0xFF));
            }
        }
    }
}

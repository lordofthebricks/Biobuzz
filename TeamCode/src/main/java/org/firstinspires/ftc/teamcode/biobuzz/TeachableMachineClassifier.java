package org.firstinspires.ftc.teamcode.biobuzz;

import android.content.Context;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Runs an image-classification model exported from Google Teachable Machine
 * (Image Project -> Export Model -> TensorFlow Lite -> Floating point or Quantized).
 *
 * The export zip contains a .tflite model and labels.txt. Files are looked up first in
 * {@value #SDCARD_MODEL_DIR} on the Control Hub, then in TeamCode/src/main/assets/.
 *
 * Not thread-safe: call {@link #classify} from one thread only (the vision thread).
 */
public class TeachableMachineClassifier implements Closeable {

    public static final String SDCARD_MODEL_DIR = "/sdcard/FIRST/tflitemodels/";

    private final Interpreter interpreter;
    private final List<String> labels;
    private final int inputWidth;
    private final int inputHeight;
    private final boolean quantizedInput;
    private final boolean quantizedOutput;
    private final ByteBuffer inputBuffer;
    private final ByteBuffer outputBuffer;
    private final byte[] pixelBytes;

    public TeachableMachineClassifier(Context context, String modelFile, String labelsFile, int numThreads)
            throws IOException {
        byte[] modelBytes = readFile(context, modelFile);
        ByteBuffer model = ByteBuffer.allocateDirect(modelBytes.length).order(ByteOrder.nativeOrder());
        model.put(modelBytes);
        model.rewind();

        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(numThreads);
        interpreter = new Interpreter(model, options);
        labels = parseLabels(new String(readFile(context, labelsFile), "UTF-8"));

        Tensor in = interpreter.getInputTensor(0);   // [1, height, width, 3]
        int[] inShape = in.shape();
        inputHeight = inShape[1];
        inputWidth = inShape[2];
        quantizedInput = in.dataType() == DataType.UINT8;

        Tensor out = interpreter.getOutputTensor(0); // [1, numClasses]
        int[] outShape = out.shape();
        int numClasses = outShape[outShape.length - 1];
        quantizedOutput = out.dataType() == DataType.UINT8;

        if (numClasses != labels.size()) {
            throw new IOException("Model has " + numClasses + " classes but " + labelsFile
                    + " has " + labels.size() + " labels");
        }

        inputBuffer = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3 * (quantizedInput ? 1 : 4))
                .order(ByteOrder.nativeOrder());
        outputBuffer = ByteBuffer.allocateDirect(numClasses * (quantizedOutput ? 1 : 4))
                .order(ByteOrder.nativeOrder());
        pixelBytes = new byte[inputWidth * inputHeight * 3];
    }

    public int getInputWidth()  { return inputWidth; }
    public int getInputHeight() { return inputHeight; }
    public List<String> getLabels() { return labels; }

    /**
     * Classifies one image.
     *
     * @param rgb an RGB, CV_8UC3 Mat that is exactly {@link #getInputWidth()} x {@link #getInputHeight()}
     * @return one score (0..1) per label, in the same order as {@link #getLabels()}
     */
    public float[] classify(Mat rgb) {
        if (rgb.cols() != inputWidth || rgb.rows() != inputHeight || rgb.type() != CvType.CV_8UC3) {
            throw new IllegalArgumentException("Expected " + inputWidth + "x" + inputHeight + " RGB image");
        }
        rgb.get(0, 0, pixelBytes);

        inputBuffer.rewind();
        if (quantizedInput) {
            inputBuffer.put(pixelBytes);
        } else {
            // Teachable Machine float models expect pixels normalized to [-1, 1]
            for (byte b : pixelBytes) {
                inputBuffer.putFloat((b & 0xFF) / 127.5f - 1f);
            }
        }
        inputBuffer.rewind();
        outputBuffer.rewind();

        interpreter.run(inputBuffer, outputBuffer);

        outputBuffer.rewind();
        float[] scores = new float[labels.size()];
        for (int i = 0; i < scores.length; i++) {
            scores[i] = quantizedOutput ? (outputBuffer.get() & 0xFF) / 255f : outputBuffer.getFloat();
        }
        return scores;
    }

    @Override
    public void close() {
        interpreter.close();
    }

    /** labels.txt lines look like "0 Pollen"; strip the leading index. */
    private static List<String> parseLabels(String text) throws IOException {
        List<String> result = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new java.io.StringReader(text));
        String line;
        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;
            int space = line.indexOf(' ');
            if (space > 0 && line.substring(0, space).matches("\\d+")) {
                line = line.substring(space + 1).trim();
            }
            result.add(line);
        }
        return Collections.unmodifiableList(result);
    }

    private static byte[] readFile(Context context, String name) throws IOException {
        File onSdCard = new File(SDCARD_MODEL_DIR, name);
        InputStream stream = onSdCard.exists()
                ? new FileInputStream(onSdCard)
                : context.getAssets().open(name);
        try {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buf = new byte[16 * 1024];
            int n;
            while ((n = stream.read(buf)) > 0) {
                out.write(buf, 0, n);
            }
            return out.toByteArray();
        } finally {
            stream.close();
        }
    }
}

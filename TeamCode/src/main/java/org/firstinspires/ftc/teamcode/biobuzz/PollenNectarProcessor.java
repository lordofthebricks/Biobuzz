package org.firstinspires.ftc.teamcode.biobuzz;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import org.firstinspires.ftc.robotcore.internal.camera.calibration.CameraCalibration;
import org.firstinspires.ftc.vision.VisionProcessor;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * VisionPortal processor that finds pollen / nectar with a Teachable Machine model.
 *
 * The field background changes every match (robots, scored pieces, people, lighting), so the
 * model is never shown whole scenes. Instead:
 *
 *   1. PROPOSE  A cheap OpenCV color filter finds candidate blobs. FTC field tiles are gray, so
 *               by default "anything clearly colored" is a candidate. Tighten the HSV ranges in
 *               {@link Settings#candidateRanges} once you know the game pieces' colors.
 *   2. CROP     Each candidate is cut out as a tight square crop (plus a little padding), so the
 *               object fills most of the image and the background barely matters.
 *   3. CLASSIFY The Teachable Machine model decides Pollen / Nectar / Other for each crop.
 *               "Other" only has to cover things that pass the color filter (robots, alliance
 *               tape, field elements), which is a far smaller set than "the whole field".
 *
 * Crops can be saved to {@value #CAPTURE_DIR} (see {@link #requestCapture()} and
 * {@link Settings#autoCaptureEveryNFrames}) so the model can be retrained on exactly what the
 * robot sees during real gameplay.
 *
 * Each sighting reports:
 *   x : -1 (left edge of image) .. 0 (center) .. +1 (right edge)  -> which way to turn
 *   y :  0 (top of image) .. 1 (bottom of image), bottom edge of the object -> how close it is
 * With the camera tilted down toward the floor, objects lower in the image are closer.
 */
public class PollenNectarProcessor implements VisionProcessor {

    public static final String CAPTURE_DIR = "/sdcard/FIRST/biobuzz-captures/";

    /** One HSV range (OpenCV scale: H 0-180, S 0-255, V 0-255). */
    public static class HsvRange {
        public final Scalar low, high;

        public HsvRange(double hLow, double sLow, double vLow, double hHigh, double sHigh, double vHigh) {
            low = new Scalar(hLow, sLow, vLow);
            high = new Scalar(hHigh, sHigh, vHigh);
        }
    }

    public static class Settings {
        /** File names from the Teachable Machine TensorFlow Lite export. */
        public String modelFile = "model_unquant.tflite";
        public String labelsFile = "labels.txt";

        /**
         * Pixels inside ANY of these ranges are candidates. The default accepts every clearly
         * colored pixel (gray tiles, white lines and black shadows are rejected). Replace with the
         * game pieces' actual colors to cut down on candidates, e.g. yellow: (15,100,100)-(35,255,255).
         */
        public List<HsvRange> candidateRanges = new ArrayList<>(Collections.singletonList(
                new HsvRange(0, 90, 60, 180, 255, 255)));
        /** Ignore blobs smaller than this fraction of the frame (noise, far-away clutter). */
        public double minCandidateArea = 0.002;
        /** Ignore blobs bigger than this fraction of the frame (walls, robots filling the view). */
        public double maxCandidateArea = 0.5;
        /** Classify at most this many candidates per frame (largest first). One model run each. */
        public int maxCandidates = 6;
        /** Extra margin around each blob before cropping, as a fraction of its size. */
        public double cropPadding = 0.25;

        /** A crop must score at least this for its top label. */
        public float minConfidence = 0.75f;
        /** ...and beat the second-best label by at least this much (rejects "unsure" crops). */
        public float minMargin = 0.3f;
        /** Labels that mean "not a target" (case-insensitive). */
        public String[] backgroundLabels = {"Other", "Background", "Nothing"};

        /** If > 0, save every candidate crop every N frames to {@value #CAPTURE_DIR}. */
        public int autoCaptureEveryNFrames = 0;
        /** Stop auto-capturing after this many files so storage doesn't fill up. */
        public int maxAutoCaptures = 1500;

        public int numThreads = 4;
    }

    /** One classified candidate blob. */
    public static class Detection {
        public final Rect blob;   // tight bounding box of the colored blob
        public final Rect crop;   // square region that was classified
        public final float[] scores;
        public final String label;
        public final float confidence;
        public final boolean isTarget;

        Detection(Rect blob, Rect crop, float[] scores, String label, float confidence, boolean isTarget) {
            this.blob = blob;
            this.crop = crop;
            this.scores = scores;
            this.label = label;
            this.confidence = confidence;
            this.isTarget = isTarget;
        }
    }

    /** The best detection of one label in one frame, in normalized image coordinates. */
    public static class Sighting {
        public final String label;
        public final float confidence;
        /** Horizontal center, -1 (left) .. +1 (right). */
        public final double x;
        /** Bottom edge of the object, 0 (top / far) .. 1 (bottom / near). */
        public final double y;
        /** Object area as a fraction of the frame (bigger = closer). */
        public final double area;
        /** How many detections of this label were in the frame. */
        public final int count;

        Sighting(String label, float confidence, double x, double y, double area, int count) {
            this.label = label;
            this.confidence = confidence;
            this.x = x;
            this.y = y;
            this.area = area;
            this.count = count;
        }

        @Override
        public String toString() {
            return String.format(Locale.US, "%s %.0f%% x=%+.2f y=%.2f area=%.3f n=%d",
                    label, confidence * 100, x, y, area, count);
        }
    }

    /** Everything found in one processed frame. Immutable, safe to read from the OpMode thread. */
    public static class Result {
        public final long timestampNanos;
        public final int frameWidth, frameHeight;
        public final List<Detection> detections;
        public final Map<String, Sighting> sightings;

        Result(long timestampNanos, int frameWidth, int frameHeight,
               List<Detection> detections, Map<String, Sighting> sightings) {
            this.timestampNanos = timestampNanos;
            this.frameWidth = frameWidth;
            this.frameHeight = frameHeight;
            this.detections = Collections.unmodifiableList(detections);
            this.sightings = Collections.unmodifiableMap(sightings);
        }

        public double ageMillis() {
            return (System.nanoTime() - timestampNanos) / 1e6;
        }

        /** Most confident sighting among the wanted labels (case-insensitive), or null. */
        public Sighting best(Collection<String> wantedLabels) {
            Sighting best = null;
            for (Sighting s : sightings.values()) {
                if (!containsIgnoreCase(wantedLabels, s.label)) continue;
                if (best == null || s.confidence > best.confidence) best = s;
            }
            return best;
        }
    }

    private final Settings settings;
    private final TeachableMachineClassifier classifier;
    private final Mat rgb = new Mat();
    private final Mat hsv = new Mat();
    private final Mat mask = new Mat();
    private final Mat rangeMask = new Mat();
    private final Mat resized = new Mat();
    private final Mat bgr = new Mat();
    private final Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));

    private volatile Result latest;
    private volatile boolean captureRequested;
    private volatile int autoCaptureEveryNFrames;
    private volatile int capturesSaved;
    private boolean closed;
    private long frameCount;

    private Paint candidatePaint, targetPaint, textPaint;

    public PollenNectarProcessor(Context context, Settings settings) throws IOException {
        this.settings = settings;
        this.autoCaptureEveryNFrames = settings.autoCaptureEveryNFrames;
        this.classifier = new TeachableMachineClassifier(
                context, settings.modelFile, settings.labelsFile, settings.numThreads);
    }

    public List<String> getLabels() {
        return classifier.getLabels();
    }

    /** Most recent result, or null if no frame has been processed yet. */
    public Result getLatestResult() {
        return latest;
    }

    /** Save every candidate crop (and the full frame) from the next processed frame. */
    public void requestCapture() {
        captureRequested = true;
    }

    /** Save every candidate crop every N frames (0 = off). Safe to call while running. */
    public void setAutoCapture(int everyNFrames) {
        autoCaptureEveryNFrames = everyNFrames;
    }

    public int getAutoCapture() {
        return autoCaptureEveryNFrames;
    }

    public int getCapturesSaved() {
        return capturesSaved;
    }

    @Override
    public void init(int width, int height, CameraCalibration calibration) {
        candidatePaint = new Paint();
        candidatePaint.setStyle(Paint.Style.STROKE);
        candidatePaint.setColor(Color.GRAY);
        candidatePaint.setStrokeWidth(3);

        targetPaint = new Paint();
        targetPaint.setStyle(Paint.Style.STROKE);
        targetPaint.setStrokeWidth(8);

        textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setAntiAlias(true);
    }

    @Override
    public synchronized Object processFrame(Mat frame, long captureTimeNanos) {
        if (closed) return null;
        frameCount++;

        // VisionPortal delivers RGBA frames; normalize to RGB for the model.
        if (frame.channels() == 4) {
            Imgproc.cvtColor(frame, rgb, Imgproc.COLOR_RGBA2RGB);
        } else {
            frame.copyTo(rgb);
        }
        int width = rgb.cols(), height = rgb.rows();
        double frameArea = (double) width * height;

        List<Rect> candidates = findCandidates(frameArea);

        boolean manualCapture = captureRequested;
        int every = autoCaptureEveryNFrames;
        boolean autoCapture = every > 0
                && frameCount % every == 0
                && capturesSaved < settings.maxAutoCaptures;
        String captureStamp = (manualCapture || autoCapture)
                ? String.format(Locale.US, "%d_%d", System.currentTimeMillis(), frameCount) : null;
        if (manualCapture) {
            captureRequested = false;
            saveImage(rgb, captureStamp + "_frame.png");
        }

        Size modelSize = new Size(classifier.getInputWidth(), classifier.getInputHeight());
        List<String> labels = classifier.getLabels();
        List<Detection> detections = new ArrayList<>();
        for (int i = 0; i < candidates.size(); i++) {
            Rect blob = candidates.get(i);
            Rect crop = squareCrop(blob, width, height);
            Mat sub = rgb.submat(crop);
            Imgproc.resize(sub, resized, modelSize, 0, 0, Imgproc.INTER_AREA);
            sub.release();

            if (captureStamp != null) {
                saveImage(resized, captureStamp + "_crop" + i + ".png");
            }
            detections.add(classify(blob, crop, classifier.classify(resized), labels));
        }

        Result result = new Result(System.nanoTime(), width, height, detections,
                buildSightings(detections, width, height));
        latest = result;
        return result;
    }

    /** Color-threshold the frame and return candidate blob boxes, largest first. */
    private List<Rect> findCandidates(double frameArea) {
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV);
        mask.create(hsv.size(), org.opencv.core.CvType.CV_8UC1);
        mask.setTo(new Scalar(0));
        for (HsvRange range : settings.candidateRanges) {
            Core.inRange(hsv, range.low, range.high, rangeMask);
            Core.bitwise_or(mask, rangeMask, mask);
        }
        // Remove speckle, then fill small holes so one object becomes one blob
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel);
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        hierarchy.release();

        List<Rect> boxes = new ArrayList<>();
        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area >= settings.minCandidateArea * frameArea && area <= settings.maxCandidateArea * frameArea) {
                boxes.add(Imgproc.boundingRect(contour));
            }
            contour.release();
        }
        Collections.sort(boxes, new Comparator<Rect>() {
            @Override
            public int compare(Rect a, Rect b) {
                return Double.compare(b.area(), a.area());
            }
        });
        return boxes.size() > settings.maxCandidates ? boxes.subList(0, settings.maxCandidates) : boxes;
    }

    /** Square, padded region around a blob, clamped to the frame. Teachable Machine models are square. */
    private Rect squareCrop(Rect blob, int frameW, int frameH) {
        int side = (int) Math.round(Math.max(blob.width, blob.height) * (1 + 2 * settings.cropPadding));
        side = Math.max(8, Math.min(side, Math.min(frameW, frameH)));
        int cx = blob.x + blob.width / 2, cy = blob.y + blob.height / 2;
        int x = Math.max(0, Math.min(cx - side / 2, frameW - side));
        int y = Math.max(0, Math.min(cy - side / 2, frameH - side));
        return new Rect(x, y, side, side);
    }

    private Detection classify(Rect blob, Rect crop, float[] scores, List<String> labels) {
        int best = 0, second = -1;
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] > scores[best]) {
                second = best;
                best = i;
            } else if (second < 0 || scores[i] > scores[second]) {
                second = i;
            }
        }
        float margin = scores[best] - (second >= 0 ? scores[second] : 0);
        String label = labels.get(best);
        boolean isTarget = !isBackground(label)
                && scores[best] >= settings.minConfidence
                && margin >= settings.minMargin;
        return new Detection(blob, crop, scores, label, scores[best], isTarget);
    }

    private Map<String, Sighting> buildSightings(List<Detection> detections, int width, int height) {
        Map<String, Detection> bestByLabel = new HashMap<>();
        Map<String, Integer> counts = new HashMap<>();
        for (Detection d : detections) {
            if (!d.isTarget) continue;
            Integer n = counts.get(d.label);
            counts.put(d.label, n == null ? 1 : n + 1);
            Detection prev = bestByLabel.get(d.label);
            // Prefer the closest one (lowest in the image), then the most confident
            if (prev == null || bottom(d) > bottom(prev)
                    || (bottom(d) == bottom(prev) && d.confidence > prev.confidence)) {
                bestByLabel.put(d.label, d);
            }
        }
        Map<String, Sighting> sightings = new HashMap<>();
        for (Detection d : bestByLabel.values()) {
            double x = (d.blob.x + d.blob.width / 2.0) / width * 2 - 1;
            double y = (double) bottom(d) / height;
            double area = d.blob.area() / ((double) width * height);
            sightings.put(d.label, new Sighting(d.label, d.confidence, x, y, area, counts.get(d.label)));
        }
        return sightings;
    }

    private static int bottom(Detection d) {
        return d.blob.y + d.blob.height;
    }

    private boolean isBackground(String label) {
        for (String bg : settings.backgroundLabels) {
            if (bg.equalsIgnoreCase(label)) return true;
        }
        return false;
    }

    private void saveImage(Mat rgbImage, String fileName) {
        File dir = new File(CAPTURE_DIR);
        if (!dir.exists() && !dir.mkdirs()) return;
        Imgproc.cvtColor(rgbImage, bgr, Imgproc.COLOR_RGB2BGR);  // imwrite expects BGR
        if (Imgcodecs.imwrite(new File(dir, fileName).getPath(), bgr)) {
            capturesSaved++;
        }
    }

    @Override
    public void onDrawFrame(Canvas canvas, int onscreenWidth, int onscreenHeight,
                            float scaleBmpPxToCanvasPx, float scaleCanvasDensity, Object userContext) {
        if (!(userContext instanceof Result)) return;
        Result result = (Result) userContext;
        float s = scaleBmpPxToCanvasPx;
        textPaint.setTextSize(16 * scaleCanvasDensity);

        for (Detection d : result.detections) {
            Paint paint = candidatePaint;
            if (d.isTarget) {
                targetPaint.setColor(colorFor(d.label));
                paint = targetPaint;
            }
            canvas.drawRect(d.blob.x * s, d.blob.y * s,
                    (d.blob.x + d.blob.width) * s, (d.blob.y + d.blob.height) * s, paint);
            canvas.drawText(String.format(Locale.US, "%s %.0f%%", d.label, d.confidence * 100),
                    d.blob.x * s, d.blob.y * s - 6 * scaleCanvasDensity, textPaint);
        }
    }

    private static int colorFor(String label) {
        String l = label.toLowerCase(Locale.US);
        if (l.contains("pollen")) return Color.YELLOW;
        if (l.contains("nectar")) return Color.CYAN;
        return Color.MAGENTA;
    }

    /** Release the TFLite interpreter. Call after closing the VisionPortal. */
    public synchronized void close() {
        closed = true;
        classifier.close();
        for (Mat m : new Mat[]{rgb, hsv, mask, rangeMask, resized, bgr, kernel}) {
            m.release();
        }
    }

    static boolean containsIgnoreCase(Collection<String> values, String value) {
        for (String v : values) {
            if (v.equalsIgnoreCase(value)) return true;
        }
        return false;
    }
}

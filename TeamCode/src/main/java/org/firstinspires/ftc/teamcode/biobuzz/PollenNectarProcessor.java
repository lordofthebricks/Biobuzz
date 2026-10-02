package org.firstinspires.ftc.teamcode.biobuzz;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import org.firstinspires.ftc.robotcore.internal.camera.calibration.CameraCalibration;
import org.firstinspires.ftc.vision.VisionProcessor;
import org.opencv.core.Core;
import org.opencv.core.CvType;
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
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * VisionPortal processor that finds pollen (yellow) and our alliance's nectar (red or blue) with
 * a Teachable Machine model, and reports the opposing alliance's nectar as hazards to avoid.
 *
 * The field background changes every match (robots, scored pieces, people, lighting), so the
 * model is never shown whole scenes. Instead:
 *
 *   1. PROPOSE  OpenCV color masks find yellow, red and blue blobs. FTC tiles are gray, so
 *               this throws away almost all of the background before the model runs.
 *   2. CROP     Each blob is cut out as a tight square crop, so the object fills the image and
 *               whatever is behind it barely matters.
 *   3. CLASSIFY The Teachable Machine model decides Pollen / Nectar / Other for each crop.
 *               "Other" only has to cover yellow/red/blue things that aren't game pieces
 *               (alliance tape, bumpers, robot parts).
 *   4. ALLIANCE The blob's color decides what a detection means:
 *                 Pollen + yellow             -> TARGET
 *                 Nectar + our alliance color -> TARGET
 *                 Nectar + opposing color     -> HAZARD (never collect; steer around it)
 *                 anything else               -> ignored
 *
 * Crops can be saved to {@value #CAPTURE_DIR} (see {@link #requestCapture()} and
 * {@link Settings#autoCaptureEveryNFrames}) so the model can be retrained on exactly what the
 * robot sees during real gameplay.
 *
 * Positions are normalized:
 *   x : -1 (left edge of image) .. 0 (center) .. +1 (right edge)  -> which way to turn
 *   y :  0 (top of image) .. 1 (bottom of image), bottom edge of the object -> how close it is
 * With the camera tilted down toward the floor, objects lower in the image are closer.
 */
public class PollenNectarProcessor implements VisionProcessor {

    public static final String CAPTURE_DIR = "/sdcard/FIRST/biobuzz-captures/";

    public enum Alliance { RED, BLUE }

    public enum PieceColor { YELLOW, RED, BLUE }

    public enum Role { TARGET, HAZARD, NONE }

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

        /** Teachable Machine class names (case-insensitive). Any other class means "not a game piece". */
        public String pollenLabel = "Pollen";
        public String nectarLabel = "Nectar";

        public Alliance alliance = Alliance.RED;

        /**
         * HSV ranges for each piece color. Check them with the Vision Test OpMode under your
         * venue's lighting: every game piece should get a box.
         */
        public List<HsvRange> yellowRanges = new ArrayList<>(Collections.singletonList(
                new HsvRange(15, 100, 100, 35, 255, 255)));
        // Red wraps around the end of the hue circle, so it needs two ranges
        public List<HsvRange> redRanges = new ArrayList<>(Arrays.asList(
                new HsvRange(0, 120, 70, 8, 255, 255),
                new HsvRange(170, 120, 70, 180, 255, 255)));
        public List<HsvRange> blueRanges = new ArrayList<>(Collections.singletonList(
                new HsvRange(95, 120, 60, 130, 255, 255)));

        /** Also look for the opposing alliance's nectar so the robot can avoid it. */
        public boolean detectHazards = true;

        /** Ignore blobs smaller than this fraction of the frame (noise, far-away clutter). */
        public double minCandidateArea = 0.002;
        /** Ignore blobs bigger than this fraction of the frame (walls, robots filling the view). */
        public double maxCandidateArea = 0.5;
        /** Classify at most this many candidates per frame (largest first). One model run each. */
        public int maxCandidates = 6;
        /** Extra margin around each blob before cropping, as a fraction of its size. */
        public double cropPadding = 0.25;

        /** A target crop must score at least this for its top label... */
        public float minConfidence = 0.75f;
        /** ...and beat the second-best label by at least this much (rejects "unsure" crops). */
        public float minMargin = 0.3f;
        /**
         * An opposing-color blob is a hazard if its Nectar score is at least this. Deliberately
         * lower than minConfidence: wrongly avoiding something is cheap, a possession penalty isn't.
         */
        public float hazardConfidence = 0.4f;

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
        public final PieceColor color;
        public final float[] scores;
        public final String label;      // model's top label
        public final float confidence;  // model's top score
        public final Role role;

        Detection(Rect blob, Rect crop, PieceColor color, float[] scores, String label, float confidence, Role role) {
            this.blob = blob;
            this.crop = crop;
            this.color = color;
            this.scores = scores;
            this.label = label;
            this.confidence = confidence;
            this.role = role;
        }

        public boolean isTarget() {
            return role == Role.TARGET;
        }
    }

    /** One detected object in normalized image coordinates. */
    public static class Sighting {
        public final String label;
        public final PieceColor color;
        public final float confidence;
        /** Horizontal center, -1 (left) .. +1 (right). */
        public final double x;
        /** Bottom edge of the object, 0 (top / far) .. 1 (bottom / near). */
        public final double y;
        /** Half the object's width, in the same units as x. */
        public final double halfWidth;
        /** Object area as a fraction of the frame (bigger = closer). */
        public final double area;

        Sighting(String label, PieceColor color, float confidence, double x, double y, double halfWidth, double area) {
            this.label = label;
            this.color = color;
            this.confidence = confidence;
            this.x = x;
            this.y = y;
            this.halfWidth = halfWidth;
            this.area = area;
        }

        @Override
        public String toString() {
            return String.format(Locale.US, "%s(%s) %.0f%% x=%+.2f y=%.2f area=%.3f",
                    label, color, confidence * 100, x, y, area);
        }
    }

    /** Everything found in one processed frame. Immutable, safe to read from the OpMode thread. */
    public static class Result {
        public final long timestampNanos;
        public final int frameWidth, frameHeight;
        public final Alliance alliance;
        public final List<Detection> detections;
        /** Closest target per label (Pollen, Nectar). Only our alliance's nectar appears here. */
        public final Map<String, Sighting> sightings;
        /** Every opposing-alliance nectar seen. Never collect these. */
        public final List<Sighting> hazards;

        Result(long timestampNanos, int frameWidth, int frameHeight, Alliance alliance,
               List<Detection> detections, Map<String, Sighting> sightings, List<Sighting> hazards) {
            this.timestampNanos = timestampNanos;
            this.frameWidth = frameWidth;
            this.frameHeight = frameHeight;
            this.alliance = alliance;
            this.detections = Collections.unmodifiableList(detections);
            this.sightings = Collections.unmodifiableMap(sightings);
            this.hazards = Collections.unmodifiableList(hazards);
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

    private static class Candidate {
        final Rect blob;
        final PieceColor color;

        Candidate(Rect blob, PieceColor color) {
            this.blob = blob;
            this.color = color;
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
    private volatile Alliance alliance;
    private volatile boolean captureRequested;
    private volatile int autoCaptureEveryNFrames;
    private volatile int capturesSaved;
    private boolean closed;
    private long frameCount;

    private Paint candidatePaint, boxPaint, textPaint;

    public PollenNectarProcessor(Context context, Settings settings) throws IOException {
        this.settings = settings;
        this.alliance = settings.alliance;
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

    public Alliance getAlliance() {
        return alliance;
    }

    /** Change alliance while running (e.g. from the Vision Test OpMode). */
    public void setAlliance(Alliance alliance) {
        this.alliance = alliance;
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

        boxPaint = new Paint();
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(8);

        textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setAntiAlias(true);
    }

    @Override
    public synchronized Object processFrame(Mat frame, long captureTimeNanos) {
        if (closed) return null;
        frameCount++;
        Alliance ourAlliance = alliance;
        PieceColor ourColor = ourAlliance == Alliance.RED ? PieceColor.RED : PieceColor.BLUE;
        PieceColor theirColor = ourAlliance == Alliance.RED ? PieceColor.BLUE : PieceColor.RED;

        // VisionPortal delivers RGBA frames; normalize to RGB for the model.
        if (frame.channels() == 4) {
            Imgproc.cvtColor(frame, rgb, Imgproc.COLOR_RGBA2RGB);
        } else {
            frame.copyTo(rgb);
        }
        int width = rgb.cols(), height = rgb.rows();
        double frameArea = (double) width * height;
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV);

        List<Candidate> candidates = new ArrayList<>();
        addCandidates(candidates, settings.yellowRanges, PieceColor.YELLOW, frameArea);
        addCandidates(candidates, colorRanges(ourColor), ourColor, frameArea);
        if (settings.detectHazards) {
            addCandidates(candidates, colorRanges(theirColor), theirColor, frameArea);
        }
        Collections.sort(candidates, new Comparator<Candidate>() {
            @Override
            public int compare(Candidate a, Candidate b) {
                return Double.compare(b.blob.area(), a.blob.area());
            }
        });
        if (candidates.size() > settings.maxCandidates) {
            candidates = candidates.subList(0, settings.maxCandidates);
        }

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
            Candidate c = candidates.get(i);
            Rect crop = squareCrop(c.blob, width, height);
            Mat sub = rgb.submat(crop);
            Imgproc.resize(sub, resized, modelSize, 0, 0, Imgproc.INTER_AREA);
            sub.release();

            if (captureStamp != null) {
                saveImage(resized, captureStamp + "_" + c.color.name().toLowerCase(Locale.US) + i + ".png");
            }
            detections.add(classify(c, crop, classifier.classify(resized), labels, ourColor));
        }

        List<Sighting> hazards = new ArrayList<>();
        for (Detection d : detections) {
            if (d.role == Role.HAZARD) hazards.add(toSighting(d, settings.nectarLabel, width, height));
        }

        Result result = new Result(System.nanoTime(), width, height, ourAlliance, detections,
                closestTargets(detections, width, height), hazards);
        latest = result;
        return result;
    }

    private List<HsvRange> colorRanges(PieceColor color) {
        switch (color) {
            case YELLOW: return settings.yellowRanges;
            case RED:    return settings.redRanges;
            default:     return settings.blueRanges;
        }
    }

    /** Threshold {@link #hsv} with the given ranges and add each blob of the right size. */
    private void addCandidates(List<Candidate> out, List<HsvRange> ranges, PieceColor color, double frameArea) {
        mask.create(hsv.size(), CvType.CV_8UC1);
        mask.setTo(new Scalar(0));
        for (HsvRange range : ranges) {
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

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area >= settings.minCandidateArea * frameArea && area <= settings.maxCandidateArea * frameArea) {
                out.add(new Candidate(Imgproc.boundingRect(contour), color));
            }
            contour.release();
        }
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

    private Detection classify(Candidate c, Rect crop, float[] scores, List<String> labels, PieceColor ourColor) {
        int best = 0, second = -1;
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] > scores[best]) {
                second = best;
                best = i;
            } else if (second < 0 || scores[i] > scores[second]) {
                second = i;
            }
        }
        String label = labels.get(best);
        float margin = scores[best] - (second >= 0 ? scores[second] : 0);
        boolean confident = scores[best] >= settings.minConfidence && margin >= settings.minMargin;

        Role role = Role.NONE;
        if (c.color == PieceColor.YELLOW) {
            if (confident && label.equalsIgnoreCase(settings.pollenLabel)) role = Role.TARGET;
        } else if (c.color == ourColor) {
            if (confident && label.equalsIgnoreCase(settings.nectarLabel)) role = Role.TARGET;
        } else {
            // Opposing color: be cautious, a lower bar is enough to avoid it
            int nectar = indexOfIgnoreCase(labels, settings.nectarLabel);
            if (nectar >= 0 && scores[nectar] >= settings.hazardConfidence) role = Role.HAZARD;
        }
        return new Detection(c.blob, crop, c.color, scores, label, scores[best], role);
    }

    /** Closest (lowest in the image) target of each label. */
    private Map<String, Sighting> closestTargets(List<Detection> detections, int width, int height) {
        Map<String, Detection> closest = new HashMap<>();
        for (Detection d : detections) {
            if (d.role != Role.TARGET) continue;
            Detection prev = closest.get(d.label);
            if (prev == null || bottom(d) > bottom(prev)
                    || (bottom(d) == bottom(prev) && d.confidence > prev.confidence)) {
                closest.put(d.label, d);
            }
        }
        Map<String, Sighting> sightings = new HashMap<>();
        for (Detection d : closest.values()) {
            sightings.put(d.label, toSighting(d, d.label, width, height));
        }
        return sightings;
    }

    private static Sighting toSighting(Detection d, String label, int width, int height) {
        double x = (d.blob.x + d.blob.width / 2.0) / width * 2 - 1;
        double y = (double) bottom(d) / height;
        double halfWidth = (double) d.blob.width / width;
        double area = d.blob.area() / ((double) width * height);
        return new Sighting(label, d.color, d.confidence, x, y, halfWidth, area);
    }

    private static int bottom(Detection d) {
        return d.blob.y + d.blob.height;
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
            String text = String.format(Locale.US, "%s %.0f%%", d.label, d.confidence * 100);
            if (d.role != Role.NONE) {
                boxPaint.setColor(colorFor(d.color));
                paint = boxPaint;
            }
            if (d.role == Role.HAZARD) text = "AVOID " + text;
            float left = d.blob.x * s, top = d.blob.y * s;
            float right = (d.blob.x + d.blob.width) * s, bottom = (d.blob.y + d.blob.height) * s;
            canvas.drawRect(left, top, right, bottom, paint);
            if (d.role == Role.HAZARD) {  // cross it out
                canvas.drawLine(left, top, right, bottom, paint);
                canvas.drawLine(left, bottom, right, top, paint);
            }
            canvas.drawText(text, left, top - 6 * scaleCanvasDensity, textPaint);
        }
    }

    private static int colorFor(PieceColor color) {
        switch (color) {
            case YELLOW: return Color.YELLOW;
            case RED:    return Color.RED;
            default:     return Color.BLUE;
        }
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
        return indexOfIgnoreCase(values, value) >= 0;
    }

    private static int indexOfIgnoreCase(Collection<String> values, String value) {
        int i = 0;
        for (String v : values) {
            if (v.equalsIgnoreCase(value)) return i;
            i++;
        }
        return -1;
    }
}

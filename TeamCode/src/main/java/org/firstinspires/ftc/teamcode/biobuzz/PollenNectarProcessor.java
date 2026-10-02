package org.firstinspires.ftc.teamcode.biobuzz;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import org.firstinspires.ftc.robotcore.internal.camera.calibration.CameraCalibration;
import org.firstinspires.ftc.vision.VisionProcessor;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * VisionPortal processor that finds pollen / nectar with a Teachable Machine model.
 *
 * Teachable Machine models only answer "what is in this picture?", not "where?". To get a
 * location, each camera frame is split into a grid (default 3 columns x 2 rows) and every
 * cell is classified separately. Cells that score above {@link Settings#minConfidence} for a
 * label are combined into a confidence-weighted centroid, which gives:
 *   x : -1 (left edge of image) .. 0 (center) .. +1 (right edge)  -> which way to turn
 *   y :  0 (top of image) .. 1 (bottom of image)                   -> how close it is
 * With the camera tilted down toward the floor, objects lower in the image are closer.
 */
public class PollenNectarProcessor implements VisionProcessor {

    public static class Settings {
        /** File names from the Teachable Machine TensorFlow Lite export. */
        public String modelFile = "model_unquant.tflite";
        public String labelsFile = "labels.txt";
        /** Grid size. More columns = finer steering but slower (one inference per cell). */
        public int columns = 3;
        public int rows = 2;
        /** A cell must score at least this for a label to count as a sighting. */
        public float minConfidence = 0.75f;
        /** Labels that mean "nothing interesting here" (case-insensitive). */
        public String[] backgroundLabels = {"Background", "Nothing", "Field"};
        public int numThreads = 4;
    }

    /** Classification of one grid cell. */
    public static class Cell {
        public final int column, row;
        public final Rect rect;
        public final float[] scores;
        public final String bestLabel;
        public final float bestScore;

        Cell(int column, int row, Rect rect, float[] scores, List<String> labels) {
            this.column = column;
            this.row = row;
            this.rect = rect;
            this.scores = scores;
            int best = 0;
            for (int i = 1; i < scores.length; i++) {
                if (scores[i] > scores[best]) best = i;
            }
            this.bestLabel = labels.get(best);
            this.bestScore = scores[best];
        }
    }

    /** Where a label was seen in one frame. */
    public static class Sighting {
        public final String label;
        /** Highest single-cell score for this label. */
        public final float confidence;
        /** Horizontal position, -1 (left) .. +1 (right). */
        public final double x;
        /** Vertical position, 0 (top / far) .. 1 (bottom / near). */
        public final double y;
        /** Number of grid cells that contained this label. */
        public final int cellCount;

        Sighting(String label, float confidence, double x, double y, int cellCount) {
            this.label = label;
            this.confidence = confidence;
            this.x = x;
            this.y = y;
            this.cellCount = cellCount;
        }

        @Override
        public String toString() {
            return String.format("%s %.0f%% x=%+.2f y=%.2f cells=%d", label, confidence * 100, x, y, cellCount);
        }
    }

    /** Everything found in one processed frame. Immutable, safe to read from the OpMode thread. */
    public static class Result {
        public final long timestampNanos;
        public final int frameWidth, frameHeight;
        public final List<Cell> cells;
        public final Map<String, Sighting> sightings;

        Result(long timestampNanos, int frameWidth, int frameHeight, List<Cell> cells, Map<String, Sighting> sightings) {
            this.timestampNanos = timestampNanos;
            this.frameWidth = frameWidth;
            this.frameHeight = frameHeight;
            this.cells = Collections.unmodifiableList(cells);
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
    private final Mat resized = new Mat();
    private volatile Result latest;
    private volatile boolean closed;

    private Paint gridPaint, hitPaint, textPaint;

    public PollenNectarProcessor(Context context, Settings settings) throws IOException {
        this.settings = settings;
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

    @Override
    public void init(int width, int height, CameraCalibration calibration) {
        gridPaint = new Paint();
        gridPaint.setStyle(Paint.Style.STROKE);
        gridPaint.setColor(Color.WHITE);
        gridPaint.setStrokeWidth(2);

        hitPaint = new Paint();
        hitPaint.setStyle(Paint.Style.STROKE);
        hitPaint.setStrokeWidth(8);

        textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setAntiAlias(true);
    }

    @Override
    public synchronized Object processFrame(Mat frame, long captureTimeNanos) {
        if (closed) return null;

        // VisionPortal frames may be RGB or RGBA depending on SDK version; normalize to RGB.
        if (frame.channels() == 4) {
            Imgproc.cvtColor(frame, rgb, Imgproc.COLOR_RGBA2RGB);
        } else {
            frame.copyTo(rgb);
        }

        int width = rgb.cols(), height = rgb.rows();
        int cellW = width / settings.columns, cellH = height / settings.rows;
        Size modelSize = new Size(classifier.getInputWidth(), classifier.getInputHeight());
        List<String> labels = classifier.getLabels();

        List<Cell> cells = new ArrayList<>();
        for (int r = 0; r < settings.rows; r++) {
            for (int c = 0; c < settings.columns; c++) {
                Rect rect = new Rect(c * cellW, r * cellH, cellW, cellH);
                Mat sub = rgb.submat(rect);
                Imgproc.resize(sub, resized, modelSize, 0, 0, Imgproc.INTER_AREA);
                sub.release();
                cells.add(new Cell(c, r, rect, classifier.classify(resized), labels));
            }
        }

        Result result = new Result(System.nanoTime(), width, height, cells, findSightings(cells, labels));
        latest = result;
        return result;
    }

    private Map<String, Sighting> findSightings(List<Cell> cells, List<String> labels) {
        Map<String, Sighting> sightings = new HashMap<>();
        for (int i = 0; i < labels.size(); i++) {
            String label = labels.get(i);
            if (isBackground(label)) continue;

            double weight = 0, sumX = 0, sumY = 0;
            float max = 0;
            int count = 0;
            for (Cell cell : cells) {
                float score = cell.scores[i];
                if (score < settings.minConfidence) continue;
                double cx = (cell.column + 0.5) / settings.columns * 2 - 1; // -1..1
                double cy = (cell.row + 0.5) / settings.rows;               //  0..1
                weight += score;
                sumX += score * cx;
                sumY += score * cy;
                max = Math.max(max, score);
                count++;
            }
            if (count > 0) {
                sightings.put(label, new Sighting(label, max, sumX / weight, sumY / weight, count));
            }
        }
        return sightings;
    }

    private boolean isBackground(String label) {
        for (String bg : settings.backgroundLabels) {
            if (bg.equalsIgnoreCase(label)) return true;
        }
        return false;
    }

    @Override
    public void onDrawFrame(Canvas canvas, int onscreenWidth, int onscreenHeight,
                            float scaleBmpPxToCanvasPx, float scaleCanvasDensity, Object userContext) {
        if (!(userContext instanceof Result)) return;
        Result result = (Result) userContext;
        float s = scaleBmpPxToCanvasPx;
        textPaint.setTextSize(18 * scaleCanvasDensity);

        for (Cell cell : result.cells) {
            float left = cell.rect.x * s, top = cell.rect.y * s;
            float right = (cell.rect.x + cell.rect.width) * s, bottom = (cell.rect.y + cell.rect.height) * s;
            boolean hit = !isBackground(cell.bestLabel) && cell.bestScore >= settings.minConfidence;
            if (hit) {
                hitPaint.setColor(colorFor(cell.bestLabel));
                canvas.drawRect(left + 4, top + 4, right - 4, bottom - 4, hitPaint);
            } else {
                canvas.drawRect(left, top, right, bottom, gridPaint);
            }
            canvas.drawText(String.format("%s %.0f%%", cell.bestLabel, cell.bestScore * 100),
                    left + 8, top + 24 * scaleCanvasDensity, textPaint);
        }

        // Crosshair at each sighting's centroid
        for (Sighting sighting : result.sightings.values()) {
            float cx = (float) ((sighting.x + 1) / 2 * result.frameWidth) * s;
            float cy = (float) (sighting.y * result.frameHeight) * s;
            hitPaint.setColor(colorFor(sighting.label));
            canvas.drawCircle(cx, cy, 20 * scaleCanvasDensity, hitPaint);
        }
    }

    private static int colorFor(String label) {
        String l = label.toLowerCase();
        if (l.contains("pollen")) return Color.YELLOW;
        if (l.contains("nectar")) return Color.CYAN;
        return Color.MAGENTA;
    }

    /** Release the TFLite interpreter. Call after closing the VisionPortal. */
    public synchronized void close() {
        closed = true;
        classifier.close();
        rgb.release();
        resized.release();
    }

    static boolean containsIgnoreCase(Collection<String> values, String value) {
        for (String v : values) {
            if (v.equalsIgnoreCase(value)) return true;
        }
        return false;
    }
}

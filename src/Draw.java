import ANN.NeuralNetwork;
import utils.LabelEncoder;
import utils.ModelIO;
import utils.TrainedModel;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;

public class Draw {

    // FOR ILLUSTRATION PURPOSES ONLY, NOT PART OF THE ASSIGNMENT
    // IT PROVIDES REAL TIME VISUALIZATION OF THE PROJECT

    // Draw: hand, bicycle, butterfly, computer, eye, eyeglasses, spider, face, key, T-shirt.

    // ===== Logical (NN) size =====
    private static final int DRAW_SIZE = 56;    // <-- CHANGE THIS
    private static final int WINDOW_SIZE = 672; // stays constant
    private static final int NN_SIZE = 28;      // fixed

    // ===== Visual scale (each pixel becomes SCALE×SCALE) =====
    private static final int SCALE = WINDOW_SIZE / DRAW_SIZE;

    private static BufferedImage drawImage;
    private static NeuralNetwork neuralNetwork;
    private static LabelEncoder labelEncoder;
    private static JLabel predictionLabel;

    public static void main(String[] args) {
        SwingUtilities.invokeLater(Draw::createUI);
    }

    private static void createUI() {

        // ===== Load Model =====
        try {
            TrainedModel model = ModelIO.load("models/ann.bin");
            neuralNetwork = model.network();
            labelEncoder = model.encoder();
        } catch (Exception e) {
            throw new RuntimeException("Failed to load model", e);
        }

        // ===== Logical Image (28×28) =====
        drawImage = new BufferedImage(
                DRAW_SIZE,
                DRAW_SIZE,
                BufferedImage.TYPE_BYTE_GRAY
        );
        clearImage();

        // ===== Window =====
        JFrame frame = new JFrame("ANN Pixel Classifier");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new BorderLayout());

        // ===== Drawing Panel =====
        JPanel drawPanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2 = (Graphics2D) g;

                // IMPORTANT: show blocky pixels
                g2.setRenderingHint(
                        RenderingHints.KEY_INTERPOLATION,
                        RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR
                );

                g2.drawImage(
                        drawImage,
                        0, 0,
                        WINDOW_SIZE, WINDOW_SIZE,
                        null
                );
            }

            @Override
            public Dimension getPreferredSize() {
                return new Dimension(WINDOW_SIZE, WINDOW_SIZE);
            }
        };

        // ===== Mouse Drawing =====
        MouseAdapter mouse = new MouseAdapter() {

            int lastPX = -1, lastPY = -1;

            @Override
            public void mousePressed(MouseEvent e) {
                lastPX = e.getX() / SCALE;
                lastPY = e.getY() / SCALE;
                drawBrush(lastPX, lastPY);
            }

            @Override
            public void mouseDragged(MouseEvent e) {
                int px = e.getX() / SCALE;
                int py = e.getY() / SCALE;

                drawLine(lastPX, lastPY, px, py);

                lastPX = px;
                lastPY = py;

                drawPanel.repaint();
                classify();
            }

            private void drawLine(int x0, int y0, int x1, int y1) {
                int dx = x1 - x0;
                int dy = y1 - y0;

                int steps = Math.max(Math.abs(dx), Math.abs(dy));
                if (steps == 0) {
                    drawBrush(x0, y0);
                    return;
                }

                for (int i = 0; i <= steps; i++) {
                    int x = x0 + dx * i / steps;
                    int y = y0 + dy * i / steps;
                    drawBrush(x, y);
                }
            }

            private void drawBrush(int cx, int cy) {
                int radius = Math.max(1, DRAW_SIZE / 28);

                for (int dy = -radius; dy <= radius; dy++) {
                    for (int dx = -radius; dx <= radius; dx++) {
                        int nx = cx + dx;
                        int ny = cy + dy;

                        if (nx < 0 || ny < 0 || nx >= DRAW_SIZE || ny >= DRAW_SIZE) continue;

                        double dist = Math.sqrt(dx * dx + dy * dy);
                        if (dist <= radius) {
                            int value = (int) (255 * (1.0 - dist / radius));
                            int current = drawImage.getRaster().getSample(nx, ny, 0);
                            drawImage.getRaster().setSample(
                                    nx, ny, 0,
                                    Math.min(255, current + value)
                            );
                        }
                    }
                }
            }
        };

        drawPanel.addMouseListener(mouse);
        drawPanel.addMouseMotionListener(mouse);

        // ===== Controls =====
        predictionLabel = new JLabel("Draw something");
        predictionLabel.setFont(new Font("Arial", Font.BOLD, 18));

        JButton clearBtn = new JButton("Clear");
        clearBtn.addActionListener(e -> {
            clearImage();
            predictionLabel.setText("Draw something");
            drawPanel.repaint();
        });

        JPanel bottom = new JPanel();
        bottom.add(predictionLabel);
        bottom.add(clearBtn);

        frame.add(drawPanel, BorderLayout.CENTER);
        frame.add(bottom, BorderLayout.SOUTH);

        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    // ===== Classification =====
    private static void classify() {
        double[] input = downsampleTo28();
        int prediction = neuralNetwork.classify(input);
        String label = labelEncoder.decode(prediction);
        predictionLabel.setText("Prediction: " + label);
    }

    // ===== Exact NN Input (No Resizing) =====
    private static double[] downsampleTo28() {
        double[] input = new double[NN_SIZE * NN_SIZE];

        int block = DRAW_SIZE / NN_SIZE; // must be integer (e.g. 56/28 = 2)

        for (int y = 0; y < NN_SIZE; y++) {
            for (int x = 0; x < NN_SIZE; x++) {

                int sum = 0;

                for (int dy = 0; dy < block; dy++) {
                    for (int dx = 0; dx < block; dx++) {
                        int px = x * block + dx;
                        int py = y * block + dy;
                        sum += drawImage.getRaster().getSample(px, py, 0);
                    }
                }

                input[y * NN_SIZE + x] = (sum / (double) (block * block)) / 255.0;
            }
        }
        return input;
    }

    private static void clearImage() {
        Graphics2D g = drawImage.createGraphics();
        g.setColor(Color.BLACK);
        g.fillRect(0, 0, DRAW_SIZE, DRAW_SIZE);
        g.dispose();
    }
}

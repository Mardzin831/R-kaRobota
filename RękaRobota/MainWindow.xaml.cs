using Numpy;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Ink;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;


namespace RękaRobota
{
    public partial class MainWindow : Window
    {
        public int rounds = 100000;
        public double learn_const = 0.1;
        public double hand_size = 50;
        public double ERROR = 0;
        public int goodCounter = 0;

        public static int[] layers = new int[] { 2, 9,9, 2 };
        public static int layers_amount = layers.Length;

        public double[][][] weights = new double[layers_amount][][];
        public double[][] bias = new double[layers_amount][];

        public double[][] before_activation = new double[layers_amount][];
        public double[][] after_activation = new double[layers_amount][];
        public double[][] delta = new double[layers_amount][];
        public double[][] second_delta = new double[layers_amount][];

        public Random randA = new Random();
        public Random randW = new Random();
        public DrawingAttributes attributes = new DrawingAttributes();
        public double alfa, beta;
        public MainWindow()
        {
            InitializeComponent();
            attributes.StylusTip = StylusTip.Rectangle;
            attributes.Width = 2;
            attributes.Height = 2;

            weights[0] = new double[][] { new double[] { 0, 0 } };
            bias[0] = new double[] { 0, 0 };
            before_activation[0] = new double[] { 0, 0 };

            // losowanie wag
            for (int k = 0; k < layers_amount - 1; k++)
            {
                weights[k + 1] = new double[layers[k]][];
                bias[k + 1] = new double[layers[k + 1]];
                for (int i = 0; i < layers[k]; i++)
                {
                    weights[k + 1][i] = new double[layers[k + 1]];
                    for (int j = 0; j < layers[k + 1]; j++)
                    {
                        weights[k + 1][i][j] = randW.NextDouble() - 0.5;
                    }
                }
                for (int i = 0; i < layers[k + 1]; i++)
                {
                    bias[k + 1][i] = randW.NextDouble() - 0.5;
                }
            }
            //PrintWeights();
            Debug.WriteLine("");
    
            //double[][] t = DotProduct(new double[][] { new double[] { 1 }, new double[] { 2 } }
            //    , new double[][] { new double[] { 1, 2 } });
            //for (int i = 0; i < t.Length; i++)
            //{
            //    Debug.WriteLine("");
            //    for (int j = 0; j < t[0].Length; j++)
            //    {
            //        Debug.Write(t[i][j] + " ");
            //    }
            //}
            
        }
        public void PrintWeights()
        {
            for (int k = 0; k < layers_amount - 1; k++)
            {
                for (int i = 0; i < layers[k]; i++)
                {
                    for (int j = 0; j < layers[k + 1]; j++)
                    {
                        Debug.Write(weights[k + 1][i][j] + " ");
                    }
                    Debug.WriteLine("");
                }
                Debug.WriteLine("");
            }
            Debug.WriteLine("------------------------------------");
            //Debug.WriteLine("---------------Transposed------------------");
            //double[][][] tmp = new double[layers_amount][][];
            //for (int k = 0; k < layers_amount - 1; k++)
            //{
            //    tmp[k + 1] = Transpose(weights[k + 1]);

            //    for (int i = 0; i < layers[k + 1]; i++)
            //    {
            //        for (int j = 0; j < layers[k]; j++)
            //        {
            //            Debug.Write(tmp[k + 1][i][j] + " ");
            //        }
            //        Debug.WriteLine("");
            //    }
            //    Debug.WriteLine("");
            //}
        }
        public double[][] Reshape(double[] tmp, int size1, int size2)
        {
            double[][] result = new double[size1][];

            for (int i = 0; i < size1; i++)
            {
                result[i] = new double[size2];
                for (int j = 0; j < size2; j++)
                {
                    result[i][j] = tmp[i * size2 + j];
                }
            }

            return result;
        }

        public double[][] DotProduct(double[][] tmp1, double[][] tmp2)
        {
            int size1 = tmp1.Length;
            int size2 = tmp2[0].Length;
            int size3 = tmp2.Length;
            double[][] result = new double[size1][];
            
            for (int i = 0; i < size1; i++)
            {
                result[i] = new double[size2];
                for (int j = 0; j < size2; j++)
                {
                    for (int k = 0; k < size3; k++)
                    {
                        result[i][j] += tmp1[i][k] * tmp2[k][j];
                    }
                }
            }
            return result;
        }

        public double[][] Transpose(double[][] tmpArray)
        {
            var result = tmpArray
                .SelectMany(inner => inner.Select((item, index) => new { item, index }))
                .GroupBy(i => i.index, i => i.item)
                .Select(g => g.ToArray())
                .ToArray();

            return result;
        }

        // liczenie delty i aktualizacja wag
        public void BackPropagation(double[] sample)
        {
            double[] SS = (double[]) SecondSigmoid(after_activation[layers_amount - 1]).Clone();
            delta[layers_amount - 1] = new double[2];
            delta[layers_amount - 1][0] = (after_activation[layers_amount - 1][0] - sample[0]) * SS[0];
            delta[layers_amount - 1][1] = (after_activation[layers_amount - 1][1] - sample[1]) * SS[1];

            // delty od przedostatniej do pierwszej
            for (int i = layers_amount - 2; i >= 0; i--)
            {
                double[][] tmpd = new double[1][] { delta[i + 1] };

                second_delta[i] = (double[]) DotProduct(tmpd, Transpose(weights[i + 1]))[0].Clone();
                delta[i] = new double[layers[i]];

                double[] tmpss = (double[]) SecondSigmoid(after_activation[i]).Clone();

                for (int j = 0; j < layers[i]; j++)
                {
                    delta[i][j] = second_delta[i][j] * tmpss[j];
                }
            }
            // aktualizacja wag
            for (int k = 1; k < layers_amount; k++)
            {
                for (int i = 0; i < layers[k - 1]; i++)
                {
                    double[][] tmpd = (double[][]) DotProduct(Reshape(after_activation[k - 1], layers[k - 1], 1),
                            Reshape(delta[k], 1, layers[k])).Clone();

                    for (int j = 0; j < layers[k]; j++)
                    {
                        weights[k][i][j] -= learn_const * tmpd[i][j];
                    }
                }
                for (int i = 0; i < layers[k]; i++)
                {
                    bias[k][i] -= learn_const * delta[k][i];
                }
            }
        }
        public void StepForward()
        {
            // zapamiętanie wcześniejszej wartości dla propagacji wstecznej
            for (int i = 1; i < layers_amount; i++)
            {
                double[][] tmpa = new double[1][] { after_activation[i - 1] };
                double[][] tmp = (double[][]) DotProduct(tmpa, weights[i]).Clone();
                before_activation[i] = new double[layers[i]];
                for (int j = 0; j < layers[i]; j++)
                {
                    before_activation[i][j] = tmp[0][j] + bias[i][j];
                }
                after_activation[i] = (double[]) Sigmoid(before_activation[i]).Clone();
            }
        }

        // predykcja kąta ramienia dla podanych współrzędnych
        public double[] Predict(double[] input)
        {
            // umieszczenie input w warstwie wejściowej
            after_activation[0] = (double[]) input.Clone();
            after_activation[0][0] /= hand_size * 2.0;
            after_activation[0][1] /= hand_size * 2.0;

            PrintWeights();
            //Debug.WriteLine("-----before------");
            // zdobycie przewidywanych kątów
            for (int i = 1; i < layers_amount; i++)
            {
                //Debug.WriteLine("");
                double[][] tmpa = new double[1][] { after_activation[i - 1] };
                double[][] tmp = (double[][]) DotProduct(tmpa, weights[i]).Clone();
                before_activation[i] = new double[layers[i]];
                for (int j = 0; j < layers[i]; j++)
                {
                    //Debug.WriteLine("");
                    before_activation[i][j] = tmp[0][j] + bias[i][j];
                    //Debug.Write(before_activation[i][j] + " ");
                }
                after_activation[i] = (double[]) Sigmoid(before_activation[i]).Clone();

            }
            //Debug.WriteLine("\n------");
            //Debug.WriteLine(" last after " + after_activation[layers_amount - 1][0] + " " + after_activation[layers_amount - 1][1]);
            //Debug.WriteLine(UnnormalizeDegrees(after_activation[layers_amount - 1])[0] + " " + UnnormalizeDegrees(after_activation[layers_amount - 1])[1]);
            // odnormalizowane wartości kątów
            return UnnormalizeDegrees(after_activation[layers_amount - 1]);
        }

        private void TrainClick(object sender, RoutedEventArgs e)
        {
            Train();
            //PrintWeights();

            //foreach (double[] q in before_activation)
            //{
            //    Debug.WriteLine("++++++\n");
            //    foreach (double r in q)
            //    {
            //        Debug.Write(r + " ");
            //    }
            //}
        }
        public void Train()
        {
            for (int i = 0; i < rounds; i++)
            {
                double[] sample = (double[]) RandRadians().Clone();
                double[] trueValues = (double[]) Input(sample).Clone();
                // wyliczenie wejścia
                
                after_activation[0] = (double[]) trueValues.Clone();
               
                // normalizacja
                after_activation[0][0] /= hand_size * 2.0;
                after_activation[0][1] /= hand_size * 2.0;
                sample = (double[]) NormalizeDegrees(sample).Clone();
                
                StepForward();
                // poprawianie delt
                BackPropagation(sample);
                
                if((i + 1) % 2000 == 0)
                {
                    //Debug.WriteLine(after_activation[0][0] + " " + after_activation[0][1]);
                    double[] predicted = (double[]) Input(UnnormalizeDegrees(after_activation[layers_amount - 1])).Clone();

                    double errorT = Math.Abs(-Math.Sqrt(Math.Pow(predicted[0] - trueValues[0], 2) + Math.Pow(predicted[1] - trueValues[1], 2)));

                    Debug.WriteLine("Trening prawdziwe punkty: " + trueValues[0] + " " + trueValues[1]);
                    Debug.WriteLine("Trening przewidywane punkty: " + predicted[0] + " " + predicted[1]);
                    Debug.WriteLine("Trening ERROR: " + errorT);
                    Debug.WriteLine("----------------");

                    if(errorT < 20)
                    {
                        break;
                    }
                }
            }
        }
        // funkcja aktywacji
        public double[] Sigmoid(double[] x)
        {
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = 1 / (1 + Math.Exp(-x[i]));
            }
            return x;
        }
        // pochodna funkcji aktywacji
        public double[] SecondSigmoid(double[] x)
        {
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = x[i] * (1 - x[i]);
            }
            return x;
        }
        // kąty w radianach (0 - pi)
        public double[] RandRadians()
        {
            double[] degrees = new double[] { Math.PI * randA.NextDouble(), Math.PI * randA.NextDouble() };
            return degrees;
        }
        // normalizacja stopni (0.1 - 0.9)
        public double[] NormalizeDegrees(double[] degrees)
        {
            double[] norm_degrees = new double[2];
            norm_degrees[0] = 0.8 * degrees[0] / Math.PI + 0.1;
            norm_degrees[1] = 0.8 * degrees[1] / Math.PI + 0.1;

            return norm_degrees;
        }
        // cofa normalizacje (0 - pi)
        public double[] UnnormalizeDegrees(double[] degrees)
        {
            double[] unnorm_degrees = new double[2];
            unnorm_degrees[0] = (degrees[0] - 0.1) * Math.PI / 0.8;
            unnorm_degrees[1] = (degrees[1] - 0.1) * Math.PI / 0.8;
            
            return unnorm_degrees;
        }
        // końcowe współrzędne ręki robota
        public double[] Input(double[] angles)
        {
            double xa, ya, xb, yb;
   
            xa = hand_size * Math.Sin(angles[0]) + 100;
            ya = -hand_size * Math.Cos(angles[0]) + 130;

            xb = xa + (-100 + xa) * Math.Cos(angles[1]) + (ya - 130) * Math.Sin(angles[1]);
            yb = ya + (100 - xa) * Math.Sin(angles[1]) + (ya - 130) * Math.Cos(angles[1]);

            //Debug.WriteLine(xb + " | " + yb);

            return new double[] { xb, yb };
        }
        private new void MouseUp(object sender, MouseButtonEventArgs e)
        {
            double xa, ya, xb, yb, clickedX, clickedY;

            clickedX = drawSpace.Strokes.First().StylusPoints.First().X;
            clickedY = drawSpace.Strokes.First().StylusPoints.First().Y;

            //alfa = randA.NextDouble() * Math.PI;
            //beta = randA.NextDouble() * Math.PI;
            double[] predicted_degrees = (double[]) Predict(new double[] { clickedX, clickedY }).Clone();
            
            alfa = predicted_degrees[0];
            beta = predicted_degrees[1];
            Debug.WriteLine("\nKąty wyjściowe: " + alfa + " " + beta);

            StylusPoint p0, p1, p2;

            List<Stroke> l = new List<Stroke>();
            StylusPointCollection points = new StylusPointCollection();

            xa = hand_size * Math.Sin(alfa) + 100;
            ya = -hand_size * Math.Cos(alfa) + 130;

            xb = xa + (-100 + xa) * Math.Cos(beta) + (ya - 130) * Math.Sin(beta);
            yb = ya + (100 - xa) * Math.Sin(beta) + (ya - 130) * Math.Cos(beta);

            p0 = new StylusPoint(100, 130);
            p1 = new StylusPoint(xa, ya);
            p2 = new StylusPoint(xb, yb);

            points.Add(p0);
            points.Add(p1);
            points.Add(p2);

            Stroke s = new Stroke(points, attributes);
            l.Add(s);
            drawSpace.Strokes.Add(s);

            // odległość między przewidywaną a prawdziwą wartością(błąd)
            double error = Math.Abs(-Math.Sqrt(Math.Pow(xb - clickedX, 2) + Math.Pow(yb - clickedY, 2)));

            Debug.WriteLine("Punkt prawdziwy: X: " + clickedX + " Y: " + clickedY);
            Debug.WriteLine("Punkt przewidywany: " + Input(predicted_degrees)[0] + " " + Input(predicted_degrees)[1]);
            Debug.WriteLine("ERROR: " + error);
            Debug.WriteLine("------------------");

        }

        private new void MouseDown(object sender, MouseButtonEventArgs e)
        {
            drawSpace.Strokes.Clear();
        }
    }
}

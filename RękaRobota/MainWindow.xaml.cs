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
        public int rounds = 500000;
        public double learn_const = 0.1;
        public double hand_size = 50;

        public static int[] layers = new int[] { 2, 3, 4, 2 };
        public static int layers_amount = layers.Count();

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

            weights[0] = new double[][] { new double[] { 0 } };
            bias[0] = new double[] { 0 };

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
                        weights[k + 1][i][j] = 2.0 * randW.NextDouble() - 1.0;
                    }
                }  
                for (int i = 0; i < layers[k + 1]; i++)
                {
                    bias[k + 1][i] = 2.0 * randW.NextDouble() - 1.0;
                }
            }
            PrintWeights();
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

        public double[] DotProduct(double[][] tmpW, double[] tmpD, int size1, int size2)
        {
            double[] result = new double[size1];
            //Debug.WriteLine(size1 + " " + size2);
            //Debug.WriteLine(tmpW.Length + " " + tmpD.Length);
       
            for (int i = 0; i < size1; i++)
            {
                for (int j = 0; j < size2; j++)
                {   
                    result[i] += tmpD[j] * tmpW[j][i];
                    //Debug.WriteLine(result[i]);
                }
            }
            
            return result;
        }
        public double[] DotProduct2(double[][] tmpW, double[][] tmpD, int size1, int size2)
        {
            double[] result = new double[size2];
            //Debug.WriteLine(size1 + " " + size2);
            //Debug.WriteLine(tmpW.Length + " " + tmpD.Length);

            for (int i = 0; i < size2; i++)
            {
                for (int j = 0; j < size1; j++)
                {
                    result[i] += tmpW[j][0] * tmpD[0][i];
                    //Debug.WriteLine(result[i]);
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
            double[] SS = SecondSigmoid(after_activation[layers_amount - 1]);
            delta[layers_amount - 1] = new double[2];
            delta[layers_amount - 1][0] = (after_activation[layers_amount - 1][0] - sample[0]) * SS[0];
            delta[layers_amount - 1][1] = (after_activation[layers_amount - 1][1] - sample[1]) * SS[1];

            // delty od przedostatniej do pierwszej
            for (int i = layers_amount - 2; i >= 0; i--)
            {
                second_delta[i] = DotProduct(Transpose(weights[i + 1]), delta[i + 1], layers[i], layers[i + 1]);
                delta[i] = new double[layers[i]];
                for (int j = 0; j < layers[i]; j++)
                {
                    delta[i][j] = second_delta[i][j] * SecondSigmoid(after_activation[i])[j];
                } 
            }
            // aktualizacja wag
            for (int k = 0; k < layers_amount - 1; k++)
            {
                for (int i = 0; i < layers[k]; i++)
                {
                    for (int j = 0; j < layers[k + 1]; j++)
                    {
                        weights[k + 1][i][j] -= learn_const * DotProduct2(Reshape(after_activation[k], layers[k], 1), 
                            Reshape(delta[k + 1], 1, layers[k + 1]), layers[k], layers[k + 1])[j];
                    }
                }
                for (int i = 0; i < layers[k + 1]; i++)
                {
                    bias[k + 1][i] -= learn_const * delta[k + 1][i];
                }
            }
        }
        public void StepForward()
        {
            // zapamiętanie wcześniejszej wartości dla propagacji wstecznej
            for (int i = 1; i < layers_amount; i++)
            {
                double[] tmp = DotProduct(weights[i], after_activation[i - 1], layers[i], layers[i - 1]);
                before_activation[i] = new double[layers[i]];
                for (int j = 0; j < layers[i]; j++)
                {
                    before_activation[i][j] = tmp[j] + bias[i][j];
                }
                after_activation[i] = Sigmoid(before_activation[i]);
            }
        }

        // predykcja kąta ramienia dla podanych współrzędnych
        public double[] Predict(double[] input)
        {
            // umieszczenie input w warstwie wejściowej
            after_activation[0] = input;

            // zdobycie przewidywanych kątów
            for (int i = 1; i < layers_amount; i++)
            {
                double[] tmp = DotProduct(weights[i], after_activation[i - 1], layers[i], layers[i - 1]);
                before_activation[i] = new double[layers[i]];
                for (int j = 0; j < layers[i]; j++)
                {
                    before_activation[i][j] = tmp[j] + bias[i][j];
                }
                after_activation[i] = Sigmoid(before_activation[i]);
            }

            // odnormalizowane wartości kątów
            return UnnormalizeDegrees(after_activation[layers_amount - 1]);
        }

        private void TrainClick(object sender, RoutedEventArgs e)
        {
            Train();
            PrintWeights();
        }
        public void Train()
        {
            for(int i = 0; i < rounds; i++)
            {
                double[] sample = RandRadians();
                // wyliczenie wejścia
                after_activation[0] = Input(sample);
                // normalizacja
                after_activation[0][0] /= (hand_size * 2);
                after_activation[0][1] /= (hand_size * 2);
                sample = NormalizeDegrees(sample);

                StepForward();

                // poprawianie delt
                BackPropagation(sample);
            }
        }
        // funkcja aktywacji
        public double[] Sigmoid(double[] x)
        {
            for(int i = 0; i < x.Count(); i++)
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
            double[] degrees = new double[] { randA.NextDouble() * 180, randA.NextDouble() * 180 };
            return degrees;
        }
        // normalizacja stopni (0.1 - 0.9)
        public double[] NormalizeDegrees(double[] degrees)
        {
            double[] norm_degrees = new double[degrees.Count()];
            foreach (double tmp in degrees)
            {
                norm_degrees.Append(0.8 * tmp / Math.PI + 0.1);
            }
            return norm_degrees;
        }
        // cofa normalizacje (0 - 180)
        public double[] UnnormalizeDegrees(double[] degrees)
        {
            double[] unnorm_degrees = new double[degrees.Count()];
            foreach (double tmp in degrees)
            {
                unnorm_degrees.Append((tmp - 0.1) * Math.PI / 0.8);
            }
            return unnorm_degrees;
        }
        // końcowe współrzędne ręki robota
        public double[] Input(double[] angles)
        {
            double xa, ya, xb, yb;

            xa = hand_size * Math.Sin(angles[0]);
            ya = -hand_size * Math.Cos(angles[0]) + 130;

            xb = xa + (0 + xa) * Math.Cos(angles[1]) + (ya - 130) * Math.Sin(angles[1]);
            yb = ya + (0 - xa) * Math.Sin(angles[1]) + (ya - 130) * Math.Cos(angles[1]);

            return new double[] { xb, yb };
        }
        private new void MouseUp(object sender, MouseButtonEventArgs e)
        {
            double xa, ya, xb, yb;
            alfa = randA.NextDouble() * Math.PI;
            beta = randA.NextDouble() * Math.PI;
            StylusPoint p0, p1, p2;

            List<Stroke> l = new List<Stroke>();
            StylusPointCollection points = new StylusPointCollection();

            xa = hand_size * Math.Sin(alfa);
            ya = -hand_size * Math.Cos(alfa) + 130;

            xb = xa + (0 + xa) * Math.Cos(beta) + (ya - 130) * Math.Sin(beta);
            yb = ya + (0 - xa) * Math.Sin(beta) + (ya - 130) * Math.Cos(beta);

            p0 = new StylusPoint(0, 130);
            p1 = new StylusPoint(xa, ya);
            p2 = new StylusPoint(xb, yb);

            points.Add(p0);
            points.Add(p1);
            points.Add(p2);

            Stroke s = new Stroke(points, attributes);
            l.Add(s);
            drawSpace.Strokes.Add(s);
        }

        private new void MouseDown(object sender, MouseButtonEventArgs e)
        {
            drawSpace.Strokes.Clear();
        }
    }
}

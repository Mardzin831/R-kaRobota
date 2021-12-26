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
        public int rounds = 2;//500000;
        public double learn_const = 0.1;
        public double hand_size = 70;

        public static List<int> layers = new List<int>() { 2, 4, 4, 2 };
        public static int layers_amount = layers.Count();

        public List<List<double>> weights = new List<List<double>>();
        public List<List<double>> bias = new List<List<double>>();

        public List<List<double>> before_activation = new List<List<double>>(new List<double>[layers_amount]);
        public List<List<double>> after_activation = new List<List<double>>(new List<double>[layers_amount]);
        public List<List<double>> delta = new List<List<double>>(new List<double>[layers_amount]);
        public List<List<double>> second_delta = new List<List<double>>(new List<double>[layers_amount]);
 
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

            List<double> tmpW = new List<double>() { 0 };
            List<double> tmpB = new List<double>() { 0 };
            weights.Add(tmpW);
            bias.Add(tmpB);

            // losowanie wag
            for (int i = 0; i < layers_amount - 1; i++)
            {
                tmpW = new List<double>();
                tmpB = new List<double>();
                for (int j = 0; j < layers[i] * layers[i + 1]; j++)
                {
                    tmpW.Add(2.0 * randW.NextDouble() - 1.0);
                }
                for (int k = 0; k < layers[i + 1]; k++)
                {
                    tmpB.Add(2.0 * randW.NextDouble() - 1.0);
                }
                weights.Add(tmpW);
                bias.Add(tmpB);
            }

            for (int i = 0; i < weights.Count(); i++)
            {
                for (int j = 0; j < weights[i].Count(); j++)
                {
                    Debug.Write(weights[i][j] + " ");
                }
                Debug.WriteLine("");
            }
            Debug.WriteLine("---------------------");
            var tw = Transpose(weights);
            for (int i = 0; i < tw.Count(); i++)
            { 
                for (int j = 0; j < tw[i].Count(); j++)
                {
                    Debug.Write(tw[i][j] + " ");
                }
                Debug.WriteLine("");
            }
        }
        public List<List<double>> Transpose(List<List<double>> tmpList)
        {
            var result = tmpList
                .SelectMany(inner => inner.Select((item, index) => new { item, index }))
                .GroupBy(i => i.index, i => i.item)
                .Select(g => g.ToList())
                .ToList();

            return result;
        }

        // liczenie delty i aktualizacja wag
        public void BackPropagation(List<double> sample)
        {
            List<double> SS = new List<double>(SecondSigmoid(after_activation[-1]));
            delta[-1][0] = (after_activation[-1][0] - sample[0]) * SS[0];
            delta[-1][1] = (after_activation[-1][1] - sample[1]) * SS[1];

            // delty od przedostatniej do pierwszej
            for (int i = layers_amount - 2; i >= 0; i--)
            {
                for (int j = 0; j >= 0; i--)
                {
                    second_delta[i] = delta[i - 1].dot(weights[i - 1].T);
                }
                
            }
        }
        public void StepForward()
        {
            for(int i = 1; i < layers_amount; i++)
            {
                after_activation[i] = Sigmoid(before_activation[i]);
            }
        }
        
        private void TrainClick(object sender, RoutedEventArgs e)
        {
            Train();
        }
        public void Train()
        {
            for(int i = 0; i < rounds; i++)
            {
                List<double> sample = new List<double>(RandRadians());

                after_activation[0] = Input(sample);
                after_activation[0][0] /= (hand_size * 2);
                after_activation[0][1] /= (hand_size * 2);
                sample = NormalizeDegrees(sample);

                //StepForward();

                // poprawianie delt
                //BackPropagation(sample);
            }
        }
        // funkcja aktywacji
        public List<double> Sigmoid(List<double> x)
        {
            for(int i = 0; i < x.Count(); i++)
            {
                x[i] = 1 / (1 + Math.Exp(-x[i]));
            }
            return x;
        }
        // pochodna funkcji aktywacji
        public List<double> SecondSigmoid(List<double> x)
        {
            for (int i = 0; i < x.Count(); i++)
            {
                x[i] = x[i] * (1 - x[i]);
            }
            return x;
        }
        // kąty w radianach (0 - pi)
        public List<double> RandRadians()
        {
            List<double> degrees = new List<double>() { randA.NextDouble() * 180, randA.NextDouble() * 180 };
            return degrees;
        }
        // normalizacja stopni (0.1 - 0.9)
        public List<double> NormalizeDegrees(List<double> degrees)
        {
            List<double> norm_degrees = new List<double>();
            foreach (double tmp in degrees)
            {
                norm_degrees.Add(0.8 * tmp / Math.PI + 0.1);
            }
            return norm_degrees;
        }
        // cofa normalizacje (0 - 180)
        public List<double> UnnormalizeDegrees(List<double> degrees)
        {
            List<double> unnorm_degrees = new List<double>();
            foreach (double tmp in degrees)
            {
                unnorm_degrees.Add((tmp - 0.1) * Math.PI / 0.8);
            }
            return unnorm_degrees;
        }
        // końcowe współrzędne ręki robota
        public List<double> Input(List<double> angles)
        {
            double xa, ya, xb, yb;

            xa = hand_size * Math.Sin(angles[0]);
            ya = -hand_size * Math.Cos(angles[0]) + 130;

            xb = xa + (0 + xa) * Math.Cos(angles[1]) + (ya - 130) * Math.Sin(angles[1]);
            yb = ya + (0 - xa) * Math.Sin(angles[1]) + (ya - 130) * Math.Cos(angles[1]);

            return new List<double>(){ xb, yb };
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
